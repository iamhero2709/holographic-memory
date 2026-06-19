"""
FINAL Modal Analysis -- addresses reviewer's top priorities:
  1. FULL test set MRR/Hits (not 1000-query sample) for Real HRR + FHRR
  2. Softmax margin analysis (quantifies the atomic-vs-composition paradox)
  3. Binomial test for "at chance" zero-shot claim
  4. FHRR ablation note (cleanup is architecturally mandatory)

Usage:  modal run final_analysis.py
Cost:   ~$0.60, runtime ~15 min (full test set ranking is the heavy part)
"""

import modal

volume = modal.Volume.from_name("hrr-results", create_if_missing=True)
VOLUME_PATH = "/vol"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.2.0", "numpy", "scipy", "requests")
)

app = modal.App(name="final-analysis", image=image)


@app.function(gpu="A10", timeout=3600, volumes={VOLUME_PATH: volume})
def final_analysis():
    import json, math
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import TensorDataset, DataLoader
    from collections import defaultdict
    from pathlib import Path
    import requests
    from scipy import stats

    DEVICE = torch.device("cuda")
    OUT = Path(VOLUME_PATH) / "final_analysis"
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}\nFINAL ANALYSIS (full test set + margin + stats)\n{'='*60}")

    # ── DATA ──
    base = "https://raw.githubusercontent.com/villmow/datasets_knowledge_embedding/master/FB15k-237"
    raw = {}
    for split in ["train", "valid", "test"]:
        resp = requests.get(f"{base}/{split}.txt", timeout=60)
        raw[split] = [tuple(l.strip().split("\t")) for l in resp.text.strip().split("\n")
                      if len(l.strip().split("\t")) == 3]
    all_triples = raw["train"] + raw["valid"] + raw["test"]
    all_tokens = sorted(set(x for h,r,t in all_triples for x in (h,r,t)))
    tok2id = {w:i for i,w in enumerate(all_tokens)}
    VOCAB = len(tok2id)
    print(f"  Vocab: {VOCAB:,}")

    def encode(triples):
        return (torch.tensor([tok2id[a] for a,b,c in triples], dtype=torch.long),
                torch.tensor([tok2id[b] for a,b,c in triples], dtype=torch.long),
                torch.tensor([tok2id[c] for a,b,c in triples], dtype=torch.long))
    te_s, te_r, te_o = encode(raw["test"])
    test_loader = DataLoader(TensorDataset(te_s, te_r, te_o), batch_size=2048, shuffle=False)
    print(f"  Test triples: {len(te_s):,}")

    all_ids = set((tok2id[h], tok2id[r], tok2id[t]) for h,r,t in all_triples)
    true_tails = defaultdict(set)
    for h,r,t in all_ids: true_tails[(h,r)].add(t)

    # chains
    train_adj = defaultdict(lambda: defaultdict(set))
    for h,r,t in raw["train"]: train_adj[tok2id[h]][tok2id[r]].add(tok2id[t])
    test_adj = defaultdict(lambda: defaultdict(set))
    for h,r,t in raw["test"]: test_adj[tok2id[h]][tok2id[r]].add(tok2id[t])

    def discover_chains(train_adj, test_adj, n=10, min_pairs=30):
        train_p = {h: dict(rd) for h,rd in train_adj.items()}
        test_p  = {h: dict(rd) for h,rd in test_adj.items()}
        chain_train = defaultdict(int)
        for h, rd in train_p.items():
            for r1, mids in rd.items():
                for mid in mids:
                    for r2, tails in train_p.get(mid, {}).items():
                        chain_train[(r1,r2)] += len(tails)
        test_reach = defaultdict(set)
        for h, rd in test_p.items():
            for r1, mids in rd.items():
                for mid in mids:
                    for r2, tails in test_p.get(mid, {}).items():
                        for t in tails: test_reach[(r1,r2)].add((h,t))
        valid = [(ch, sorted(tp)) for ch,tp in test_reach.items()
                 if len(tp) >= min_pairs and chain_train[ch] >= 100]
        valid.sort(key=lambda x: -len(x[1]))
        return valid[:n]

    chains = discover_chains(train_adj, test_adj, 10)
    train_lookup = {h:{r:set(ts) for r,ts in rd.items()} for h,rd in train_adj.items()}
    zh, zr1, zr2, zt = [], [], [], []
    for (r1,r2), pairs in chains:
        for h,t in pairs:
            if any(t in ts for ts in train_lookup.get(h,{}).values()): continue
            zh.append(h); zr1.append(r1); zr2.append(r2); zt.append(t)
    zh=torch.tensor(zh); zr1=torch.tensor(zr1); zr2=torch.tensor(zr2); zt=torch.tensor(zt)
    print(f"  Zero-shot pairs: {len(zh):,}")

    # ── MODELS ──
    class RealHRR(nn.Module):
        def __init__(self, D, vocab_size, beta=8.0, use_hopfield=True):
            super().__init__()
            self.D, self.vocab_size, self.beta = D, vocab_size, beta
            self.use_hopfield = use_hopfield
            self.entity_emb = nn.Embedding(vocab_size, D)
            self.role_emb   = nn.Embedding(vocab_size, D)
            self.M          = nn.Parameter(torch.zeros(D))
            nn.init.normal_(self.entity_emb.weight, std=1/D**0.5)
            nn.init.normal_(self.role_emb.weight,   std=1/D**0.5)
        @staticmethod
        def _bind(a,b):
            return torch.fft.ifft(torch.fft.fft(a.float())*torch.fft.fft(b.float())).real.to(a.dtype)
        @staticmethod
        def _unbind(bnd,key):
            return torch.fft.ifft(torch.fft.fft(bnd.float())*torch.conj(torch.fft.fft(key.float()))).real.to(bnd.dtype)
        def _cleanup(self, z, hard=False):
            if not self.use_hopfield:
                return z/(z.norm(dim=-1,keepdim=True)+1e-8)
            sc=(z.float()@self.entity_emb.weight.float().T*self.beta).clamp(-50,50)
            w=F.gumbel_softmax(sc,tau=1.0,hard=True) if hard else F.softmax(sc,dim=-1)
            return (w@self.entity_emb.weight.float()).to(z.dtype)
        def forward_atomic(self,s,r):
            f,ro=self.entity_emb(s),self.role_emb(r)
            M=self.M.unsqueeze(0).expand_as(f)
            return self._cleanup(self._unbind(M,self._bind(f,ro)))@self.entity_emb.weight.float().T*self.beta
        def forward_composition(self,h,r1,r2):
            f,ro1,ro2=self.entity_emb(h),self.role_emb(r1),self.role_emb(r2)
            M=self.M.unsqueeze(0).expand_as(f)
            mid=self._cleanup(self._unbind(M,self._bind(f,ro1)),hard=True)
            M2=self.M.unsqueeze(0).expand_as(mid)
            return self._cleanup(self._unbind(M2,self._bind(mid,ro2)))@self.entity_emb.weight.float().T*self.beta

    class ComplexFHRR(nn.Module):
        def __init__(self, D, vocab_size, beta=12.0):
            super().__init__()
            self.D,self.vocab_size,self.beta=D,vocab_size,beta
            self.embed_phase=nn.Embedding(vocab_size,D)
            self.role_phase =nn.Embedding(vocab_size,D)
            self.M_real=nn.Parameter(torch.zeros(D)); self.M_imag=nn.Parameter(torch.zeros(D))
            nn.init.uniform_(self.embed_phase.weight,-math.pi,math.pi)
            nn.init.uniform_(self.role_phase.weight,-math.pi,math.pi)
        def _phasor(self,idx,kind='filler'):
            ph=self.embed_phase(idx) if kind=='filler' else self.role_phase(idx)
            return torch.complex(torch.cos(ph),torch.sin(ph))
        def _M(self): return torch.complex(self.M_real,self.M_imag)
        def _codebook(self):
            ph=self.embed_phase.weight; return torch.complex(torch.cos(ph),torch.sin(ph))
        @staticmethod
        def _bind(a,b): return a*b
        @staticmethod
        def _unbind(bnd,key): return bnd*torch.conj(key)
        def _cos_sim(self,z,cb):
            dot=z.real@cb.real.T+z.imag@cb.imag.T
            nz=(z.real**2+z.imag**2).sum(-1,keepdim=True).sqrt()+1e-8
            return dot/(nz*self.D**0.5)
        def _cleanup(self,z,hard=False):
            cb=self._codebook(); sc=self._cos_sim(z,cb)*self.beta
            w=F.gumbel_softmax(sc,tau=1.0,hard=True) if hard else F.softmax(sc,dim=-1)
            cr,ci=w@cb.real,w@cb.imag; ph=torch.atan2(ci,cr+1e-8)
            return torch.complex(torch.cos(ph),torch.sin(ph))
        def forward_atomic(self,s,r):
            f,ro=self._phasor(s),self._phasor(r,'role')
            M=self._M().unsqueeze(0).expand_as(f)
            cln=self._cleanup(self._unbind(M,self._bind(f,ro)))
            return self._cos_sim(cln,self._codebook())*self.beta
        def forward_composition(self,h,r1,r2):
            f,ro1,ro2=self._phasor(h),self._phasor(r1,'role'),self._phasor(r2,'role')
            M=self._M()
            mid=self._cleanup(self._unbind(M.unsqueeze(0).expand_as(f),self._bind(f,ro1)),hard=True)
            cln=self._cleanup(self._unbind(M.unsqueeze(0).expand_as(mid),self._bind(mid,ro2)))
            return self._cos_sim(cln,self._codebook())*self.beta

    # ── FULL TEST SET ranking (reviewer #1) ──
    @torch.no_grad()
    def full_test_metrics(model):
        model.eval()
        ranks=[]; top1=0; total=0
        for sb,rb,ob in test_loader:
            sb,rb,ob=sb.to(DEVICE),rb.to(DEVICE),ob.to(DEVICE)
            logits=model.forward_atomic(sb,rb).float()
            top1+=(logits.argmax(1)==ob).sum().item(); total+=ob.size(0)
            for j in range(len(sb)):
                hi,ri,oi=sb[j].item(),rb[j].item(),ob[j].item()
                mask=torch.zeros(logits.size(1),device=DEVICE)
                for tt in true_tails.get((hi,ri),set()):
                    if tt!=oi: mask[tt]=1e9
                sc=logits[j]-mask
                ranks.append((sc>sc[oi]).sum().item()+1)
        ranks=np.array(ranks)
        return {
            "top1": top1/total,
            "mrr": float(np.mean(1/ranks)),
            "h1": float(np.mean(ranks<=1)),
            "h3": float(np.mean(ranks<=3)),
            "h10": float(np.mean(ranks<=10)),
            "n": int(total),
        }

    # ── SOFTMAX MARGIN (reviewer #4 -- paradox quantification) ──
    @torch.no_grad()
    def margin_analysis(model, loader, n_max=3000):
        """gold logit minus max-incorrect logit, for atomic queries"""
        model.eval(); margins=[]; gold_logits=[]; collected=0
        for sb,rb,ob in loader:
            sb,rb,ob=sb.to(DEVICE),rb.to(DEVICE),ob.to(DEVICE)
            logits=model.forward_atomic(sb,rb).float()
            for j in range(len(sb)):
                oi=ob[j].item()
                gl=logits[j,oi].item()
                tmp=logits[j].clone(); tmp[oi]=-1e9
                mi=tmp.max().item()
                margins.append(gl-mi); gold_logits.append(gl)
                collected+=1
            if collected>=n_max: break
        margins=np.array(margins)
        return {
            "margin_mean": float(margins.mean()),
            "margin_std": float(margins.std()),
            "margin_positive_frac": float((margins>0).mean()),
            "n": len(margins),
        }

    # ── BINOMIAL TEST for "at chance" (reviewer #6/#10) ──
    @torch.no_grad()
    def zero_shot_with_test(model, h,r1,r2,t, batch=512):
        model.eval(); correct=0; total=0
        for i in range(0,len(h),batch):
            hb,r1b,r2b,tb=h[i:i+batch].to(DEVICE),r1[i:i+batch].to(DEVICE),r2[i:i+batch].to(DEVICE),t[i:i+batch].to(DEVICE)
            correct+=(model.forward_composition(hb,r1b,r2b).argmax(1)==tb).sum().item()
            total+=tb.size(0)
        chance=1.0/VOCAB
        # one-sided binomial: is observed > chance?
        pval=stats.binomtest(correct, total, chance, alternative='greater').pvalue
        return {"correct":correct, "total":total, "acc":correct/total,
                "chance":chance, "binom_p_greater_than_chance":float(pval)}

    # ════════════════════════════════════════════════════
    results={}
    for seed in [1,2,3,4,42]:
        cr=Path(VOLUME_PATH)/f"seed_{seed}"/"best_real_hrr.pt"
        cf=Path(VOLUME_PATH)/f"seed_{seed}"/"best_complex_fhrr.pt"
        if not (cr.exists() and cf.exists()):
            print(f"❌ seed {seed} missing"); continue
        print(f"\n{'='*60}\n🔍 SEED {seed} (full test set ranking...)\n{'='*60}")
        real=RealHRR(1024,VOCAB,beta=8.0).to(DEVICE); real.load_state_dict(torch.load(str(cr),map_location=DEVICE))
        fhrr=ComplexFHRR(512,VOCAB,beta=12.0).to(DEVICE); fhrr.load_state_dict(torch.load(str(cf),map_location=DEVICE))

        rm=full_test_metrics(real)
        fm=full_test_metrics(fhrr)
        print(f"[Real HRR full] top1={rm['top1']:.4f} MRR={rm['mrr']:.4f} H@1={rm['h1']:.4f} H@3={rm['h3']:.4f} H@10={rm['h10']:.4f} (n={rm['n']})")
        print(f"[FHRR full]     top1={fm['top1']:.4f} MRR={fm['mrr']:.4f} H@1={fm['h1']:.4f} H@3={fm['h3']:.4f} H@10={fm['h10']:.4f} (n={fm['n']})")

        rmar=margin_analysis(real,test_loader)
        fmar=margin_analysis(fhrr,test_loader)
        print(f"[Real margin] mean={rmar['margin_mean']:.4f}±{rmar['margin_std']:.4f} pos_frac={rmar['margin_positive_frac']:.3f}")
        print(f"[FHRR margin] mean={fmar['margin_mean']:.4f}±{fmar['margin_std']:.4f} pos_frac={fmar['margin_positive_frac']:.3f}")

        rzs=zero_shot_with_test(real,zh,zr1,zr2,zt)
        fzs=zero_shot_with_test(fhrr,zh,zr1,zr2,zt)
        print(f"[Real ZS] acc={rzs['acc']:.6f} chance={rzs['chance']:.6f} binom_p={rzs['binom_p_greater_than_chance']:.4f}")
        print(f"[FHRR ZS] acc={fzs['acc']:.6f} chance={fzs['chance']:.6f} binom_p={fzs['binom_p_greater_than_chance']:.4f}")

        results[seed]={"real_full":rm,"fhrr_full":fm,"real_margin":rmar,"fhrr_margin":fmar,
                       "real_zs":rzs,"fhrr_zs":fzs}

    # ── AGGREGATE ──
    print(f"\n{'='*60}\n📊 AGGREGATE (5 seeds, FULL test set)\n{'='*60}")
    seeds=list(results.keys())
    def ag(p):
        v=[]
        for s in seeds:
            d=results[s]
            for k in p: d=d[k]
            v.append(d)
        return float(np.mean(v)), float(np.std(v))

    print("\nReal HRR (full test):")
    for m in ["top1","mrr","h1","h3","h10"]:
        mu,sd=ag(["real_full",m]); print(f"  {m}: {mu:.4f} ± {sd:.4f}")
    print("FHRR (full test):")
    for m in ["top1","mrr","h1","h3","h10"]:
        mu,sd=ag(["fhrr_full",m]); print(f"  {m}: {mu:.4f} ± {sd:.4f}")
    print("Softmax margin (atomic):")
    print(f"  Real: {ag(['real_margin','margin_mean'])[0]:.4f}  pos_frac={ag(['real_margin','margin_positive_frac'])[0]:.3f}")
    print(f"  FHRR: {ag(['fhrr_margin','margin_mean'])[0]:.4f}  pos_frac={ag(['fhrr_margin','margin_positive_frac'])[0]:.3f}")
    print("Zero-shot binomial test (vs chance):")
    print(f"  Real: acc={ag(['real_zs','acc'])[0]:.6f}  mean p={ag(['real_zs','binom_p_greater_than_chance'])[0]:.4f}")
    print(f"  FHRR: acc={ag(['fhrr_zs','acc'])[0]:.6f}  mean p={ag(['fhrr_zs','binom_p_greater_than_chance'])[0]:.4f}")

    agg={
        "real_full":{m:{"mean":ag(["real_full",m])[0],"std":ag(["real_full",m])[1]} for m in ["top1","mrr","h1","h3","h10"]},
        "fhrr_full":{m:{"mean":ag(["fhrr_full",m])[0],"std":ag(["fhrr_full",m])[1]} for m in ["top1","mrr","h1","h3","h10"]},
        "real_margin":{"mean":ag(["real_margin","margin_mean"])[0],"pos_frac":ag(["real_margin","margin_positive_frac"])[0]},
        "fhrr_margin":{"mean":ag(["fhrr_margin","margin_mean"])[0],"pos_frac":ag(["fhrr_margin","margin_positive_frac"])[0]},
        "real_zs":{"acc":ag(["real_zs","acc"])[0],"binom_p":ag(["real_zs","binom_p_greater_than_chance"])[0]},
        "fhrr_zs":{"acc":ag(["fhrr_zs","acc"])[0],"binom_p":ag(["fhrr_zs","binom_p_greater_than_chance"])[0]},
        "chance":1.0/VOCAB, "test_n":len(te_s), "per_seed":results, "seeds":seeds,
    }
    with open(str(OUT/"final_results.json"),"w") as f:
        json.dump(agg,f,indent=2)
    print(f"\n✅ Saved: {OUT}/final_results.json")
    volume.commit()
    return agg


@app.local_entrypoint()
def main():
    print("\n🔬 Final analysis: full test set + margin + stats...\n")
    final_analysis.remote()
    print(f"\n{'='*60}\n✅ DONE\n{'='*60}")
    print("Download: modal volume get hrr-results ./final/")
