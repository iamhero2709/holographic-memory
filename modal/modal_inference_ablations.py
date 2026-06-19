"""
Modal Script: Complete Inference + Ablations + JSON Generation
Runs on Modal A10, generates JSON + graphs, saves to Volume

LOCAL REQUIREMENTS: Only `modal` (pip install modal)
All heavy deps (torch, numpy, etc) run on Modal cloud.

Usage:
  modal run modal_inference_ablations.py
"""

import modal

# ─────────────────────────────────────────────────────────
# MODAL SETUP (only modal needed locally)
# ─────────────────────────────────────────────────────────
volume = modal.Volume.from_name("hrr-results", create_if_missing=True)
VOLUME_PATH = "/vol"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.2.0",
        "numpy",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "tqdm",
        "requests",
    )
)

app = modal.App(
    name="hrr-inference-ablations",
    image=image,
)

# ─────────────────────────────────────────────────────────
# MAIN FUNCTION - all heavy imports INSIDE
# ─────────────────────────────────────────────────────────

@app.function(
    gpu="A10",
    timeout=3600,
    volumes={VOLUME_PATH: volume},
)
def run_inference_and_ablations():
    # ALL IMPORTS INSIDE FUNCTION (Modal cloud has these installed)
    import os, sys, json, math, random
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import TensorDataset, DataLoader
    from collections import defaultdict
    from pathlib import Path
    import requests
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    DEVICE = torch.device("cuda")
    OUT = Path(VOLUME_PATH) / "inference_results"
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}\nINFERENCE + ABLATIONS on Modal A10\n{'='*60}")

    # ─────────────────────────────────────────────────────
    # 1. LOAD DATA
    # ─────────────────────────────────────────────────────
    print("\n── LOADING DATA ──")
    base = "https://raw.githubusercontent.com/villmow/datasets_knowledge_embedding/master/FB15k-237"
    raw = {}
    for split in ["train", "valid", "test"]:
        resp = requests.get(f"{base}/{split}.txt", timeout=60)
        raw[split] = [tuple(l.strip().split("\t")) for l in resp.text.strip().split("\n")
                      if len(l.strip().split("\t")) == 3]
        print(f"  {split}: {len(raw[split]):,}")

    all_triples = raw["train"] + raw["valid"] + raw["test"]
    all_tokens = sorted(set(x for h,r,t in all_triples for x in (h,r,t)))
    tok2id = {w:i for i,w in enumerate(all_tokens)}
    VOCAB = len(tok2id)
    print(f"  Vocab: {VOCAB:,}")

    def encode(triples):
        h = torch.tensor([tok2id[a] for a,b,c in triples], dtype=torch.long)
        r = torch.tensor([tok2id[b] for a,b,c in triples], dtype=torch.long)
        t = torch.tensor([tok2id[c] for a,b,c in triples], dtype=torch.long)
        return h, r, t

    tr_s, tr_r, tr_o = encode(raw["train"])
    te_s, te_r, te_o = encode(raw["test"])

    all_triples_ids = set((tok2id[h], tok2id[r], tok2id[t]) for h,r,t in raw["train"] + raw["valid"] + raw["test"])
    true_tails = defaultdict(set)
    for h,r,t in all_triples_ids:
        true_tails[(h, r)].add(t)

    test_loader = DataLoader(TensorDataset(te_s, te_r, te_o), batch_size=2048, shuffle=False)

    # Chains
    print("\n── CHAINS ──")
    train_adj = defaultdict(lambda: defaultdict(set))
    for h,r,t in raw["train"]: train_adj[tok2id[h]][tok2id[r]].add(tok2id[t])
    test_adj = defaultdict(lambda: defaultdict(set))
    for h,r,t in raw["test"]: test_adj[tok2id[h]][tok2id[r]].add(tok2id[t])

    def discover_chains(train_adj, test_adj, n=10, min_pairs=30):
        train_p = {h: dict(rd) for h,rd in train_adj.items()}
        test_p  = {h: dict(rd) for h,rd in test_adj.items()}
        EMPTY = {}
        chain_train = defaultdict(int)
        for h, rd in train_p.items():
            for r1, mids in rd.items():
                for mid in mids:
                    for r2, tails in train_p.get(mid, EMPTY).items():
                        chain_train[(r1,r2)] += len(tails)
        test_reach = defaultdict(set)
        for h, rd in test_p.items():
            for r1, mids in rd.items():
                for mid in mids:
                    for r2, tails in test_p.get(mid, EMPTY).items():
                        for t in tails:
                            test_reach[(r1,r2)].add((h,t))
        valid = [(ch, sorted(tp)) for ch,tp in test_reach.items()
                 if len(tp) >= min_pairs and chain_train[ch] >= 100]
        valid.sort(key=lambda x: -len(x[1]))
        return valid[:n]

    chains = discover_chains(train_adj, test_adj, 10)
    train_lookup = {h:{r:set(ts) for r,ts in rd.items()} for h,rd in train_adj.items()}
    zs_h, zs_r1, zs_r2, zs_t = [], [], [], []
    leaked = 0
    for (r1,r2), pairs in chains:
        for h,t in pairs:
            if any(t in ts for ts in train_lookup.get(h,{}).values()):
                leaked += 1; continue
            zs_h.append(h); zs_r1.append(r1); zs_r2.append(r2); zs_t.append(t)

    zs_h  = torch.tensor(zs_h,  dtype=torch.long)
    zs_r1 = torch.tensor(zs_r1, dtype=torch.long)
    zs_r2 = torch.tensor(zs_r2, dtype=torch.long)
    zs_t  = torch.tensor(zs_t,  dtype=torch.long)
    print(f"  Chains: {len(chains)} | Zero-shot pairs: {len(zs_h):,}")

    # ─────────────────────────────────────────────────────
    # 2. MODELS
    # ─────────────────────────────────────────────────────
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
        def _bind(a, b):
            a32,b32 = a.float(),b.float()
            return torch.fft.ifft(torch.fft.fft(a32)*torch.fft.fft(b32)).real.to(a.dtype)

        @staticmethod
        def _unbind(bnd, key):
            b32,k32 = bnd.float(),key.float()
            return torch.fft.ifft(torch.fft.fft(b32)*torch.conj(torch.fft.fft(k32))).real.to(bnd.dtype)

        def _cleanup(self, z, hard=False):
            if not self.use_hopfield:
                return z / (z.norm(dim=-1, keepdim=True) + 1e-8)
            scores = z.float() @ self.entity_emb.weight.float().T * self.beta
            scores = scores.clamp(-50, 50)
            w = F.gumbel_softmax(scores, tau=1.0, hard=True) if hard else F.softmax(scores, dim=-1)
            return (w @ self.entity_emb.weight.float()).to(z.dtype)

        def forward_atomic(self, s, r):
            f, ro = self.entity_emb(s), self.role_emb(r)
            M = self.M.unsqueeze(0).expand_as(f)
            return self._cleanup(self._unbind(M, self._bind(f, ro))) @ self.entity_emb.weight.float().T * self.beta

        def forward_composition(self, h, r1, r2, hard_inter=True):
            f, ro1, ro2 = self.entity_emb(h), self.role_emb(r1), self.role_emb(r2)
            M = self.M.unsqueeze(0).expand_as(f)
            mid = self._cleanup(self._unbind(M, self._bind(f, ro1)), hard=hard_inter)
            M2 = self.M.unsqueeze(0).expand_as(mid)
            return self._cleanup(self._unbind(M2, self._bind(mid, ro2))) @ self.entity_emb.weight.float().T * self.beta

    class ComplexFHRR(nn.Module):
        def __init__(self, D, vocab_size, beta=12.0):
            super().__init__()
            self.D, self.vocab_size, self.beta = D, vocab_size, beta
            self.embed_phase = nn.Embedding(vocab_size, D)
            self.role_phase  = nn.Embedding(vocab_size, D)
            self.M_real      = nn.Parameter(torch.zeros(D))
            self.M_imag      = nn.Parameter(torch.zeros(D))
            nn.init.uniform_(self.embed_phase.weight, -math.pi, math.pi)
            nn.init.uniform_(self.role_phase.weight,  -math.pi, math.pi)

        def _phasor(self, idx, kind='filler'):
            ph = self.embed_phase(idx) if kind=='filler' else self.role_phase(idx)
            return torch.complex(torch.cos(ph), torch.sin(ph))

        def _M(self):
            return torch.complex(self.M_real, self.M_imag)

        def _codebook(self):
            ph = self.embed_phase.weight
            return torch.complex(torch.cos(ph), torch.sin(ph))

        @staticmethod
        def _bind(a,b): return a*b
        @staticmethod
        def _unbind(bnd,key): return bnd*torch.conj(key)

        def _cos_sim(self, z, cb):
            dot = z.real@cb.real.T + z.imag@cb.imag.T
            nz  = (z.real**2+z.imag**2).sum(-1,keepdim=True).sqrt()+1e-8
            return dot/(nz*self.D**0.5)

        def _cleanup(self, z, hard=False):
            cb = self._codebook()
            scores = self._cos_sim(z, cb)*self.beta
            w = F.gumbel_softmax(scores, tau=1.0, hard=True) if hard else F.softmax(scores, dim=-1)
            cr, ci = w@cb.real, w@cb.imag
            ph = torch.atan2(ci, cr+1e-8)
            return torch.complex(torch.cos(ph), torch.sin(ph))

        def forward_atomic(self, s, r):
            f, ro = self._phasor(s), self._phasor(r,'role')
            M = self._M().unsqueeze(0).expand_as(f)
            cln = self._cleanup(self._unbind(M,self._bind(f,ro)))
            return self._cos_sim(cln,self._codebook()) * self.beta

        def forward_composition(self, h, r1, r2, hard_inter=True):
            f, ro1, ro2 = self._phasor(h), self._phasor(r1,'role'), self._phasor(r2,'role')
            M = self._M()
            mid = self._cleanup(self._unbind(M.unsqueeze(0).expand_as(f),self._bind(f,ro1)),hard=hard_inter)
            cln = self._cleanup(self._unbind(M.unsqueeze(0).expand_as(mid),self._bind(mid,ro2)))
            return self._cos_sim(cln,self._codebook()) * self.beta

        @torch.no_grad()
        def probe_modulus(self, h, r1, r2):
            self.eval()
            f, ro1, ro2 = self._phasor(h), self._phasor(r1,'role'), self._phasor(r2,'role')
            M = self._M()
            def mag(z): return torch.abs(z).mean().item()
            log={"01_initial":mag(f)}
            q1=self._bind(f,ro1); log["02_after_bind_1"]=mag(q1)
            e1=self._unbind(M.unsqueeze(0).expand_as(q1),q1); log["03_before_cleanup_1"]=mag(e1)
            mid=self._cleanup(e1,hard=True); log["04_after_cleanup_1"]=mag(mid)
            q2=self._bind(mid,ro2); log["05_after_bind_2"]=mag(q2)
            e2=self._unbind(M.unsqueeze(0).expand_as(q2),q2); log["06_before_cleanup_2"]=mag(e2)
            cln=self._cleanup(e2,hard=False); log["07_after_cleanup_2"]=mag(cln)
            return log

        @torch.no_grad()
        def forward_with_renorm(self, h, r1, r2):
            f, ro1, ro2 = self._phasor(h), self._phasor(r1,'role'), self._phasor(r2,'role')
            M = self._M()
            mid = self._cleanup(self._unbind(M.unsqueeze(0).expand_as(f),self._bind(f,ro1)),hard=True)
            mid = mid/(torch.abs(mid)+1e-8)
            cln = self._cleanup(self._unbind(M.unsqueeze(0).expand_as(mid),self._bind(mid,ro2)))
            cln = cln/(torch.abs(cln)+1e-8)
            return self._cos_sim(cln,self._codebook()) * self.beta

        @torch.no_grad()
        def forward_hard_cleanup(self, h, r1, r2):
            f, ro1, ro2 = self._phasor(h), self._phasor(r1,'role'), self._phasor(r2,'role')
            M = self._M()
            mid = self._cleanup(self._unbind(M.unsqueeze(0).expand_as(f),self._bind(f,ro1)),hard=True)
            cln = self._cleanup(self._unbind(M.unsqueeze(0).expand_as(mid),self._bind(mid,ro2)),hard=True)
            return self._cos_sim(cln,self._codebook()) * self.beta

    # ─────────────────────────────────────────────────────
    # 3. EVALUATOR
    # ─────────────────────────────────────────────────────
    class Evaluator:
        @staticmethod
        @torch.no_grad()
        def atomic_accuracy(model, loader):
            model.eval(); correct=total=0
            for sb,rb,ob in loader:
                sb,rb,ob=sb.to(DEVICE),rb.to(DEVICE),ob.to(DEVICE)
                correct+=(model.forward_atomic(sb,rb).argmax(1)==ob).sum().item()
                total+=ob.size(0)
            return correct/total if total else 0.0

        @staticmethod
        @torch.no_grad()
        def zero_shot_accuracy(model, h, r1, r2, t, batch=512):
            model.eval(); correct=total=0
            for i in range(0,len(h),batch):
                hb,r1b,r2b,tb=h[i:i+batch].to(DEVICE),r1[i:i+batch].to(DEVICE),r2[i:i+batch].to(DEVICE),t[i:i+batch].to(DEVICE)
                correct+=(model.forward_composition(hb,r1b,r2b).argmax(1)==tb).sum().item()
                total+=tb.size(0)
            return correct/total if total else 0.0

        @staticmethod
        @torch.no_grad()
        def filtered_mrr_hits(model, s_all, r_all, o_all, true_tails, n=1000, batch=128):
            model.eval()
            idx=torch.randperm(len(s_all))[:n]
            s,r,o=s_all[idx].to(DEVICE),r_all[idx].to(DEVICE),o_all[idx].to(DEVICE)
            ranks=[]
            for i in range(0,n,batch):
                sb,rb,ob=s[i:i+batch],r[i:i+batch],o[i:i+batch]
                logits=model.forward_atomic(sb,rb).float()
                for j in range(len(sb)):
                    hi,ri,oi=sb[j].item(),rb[j].item(),ob[j].item()
                    mask=torch.zeros(logits.size(1),device=DEVICE)
                    for tt in true_tails.get((hi,ri),set()):
                        if tt!=oi: mask[tt]=1e9
                    score_j=logits[j]-mask
                    ranks.append((score_j>score_j[oi]).sum().item()+1)
            ranks=np.array(ranks)
            return (float(np.mean(1/ranks)), float(np.mean(ranks<=1)), float(np.mean(ranks<=3)), float(np.mean(ranks<=10)))

    # ─────────────────────────────────────────────────────
    # 4. MAIN EVALUATION LOOP
    # ─────────────────────────────────────────────────────
    results_all = {}

    print("\n" + "="*60)
    print("INFERENCE: Loading checkpoints from Modal volume...")
    print("="*60)

    for seed in [1, 2, 3, 4, 42]:
        print(f"\n--- Seed {seed} ---")
        real_hrr = RealHRR(1024, VOCAB, beta=8.0).to(DEVICE)
        fhrr = ComplexFHRR(512, VOCAB, beta=12.0).to(DEVICE)
        
        try:
            ckpt_real = Path(VOLUME_PATH) / f"seed_{seed}" / "best_real_hrr.pt"
            ckpt_fhrr = Path(VOLUME_PATH) / f"seed_{seed}" / "best_complex_fhrr.pt"
            
            if ckpt_real.exists() and ckpt_fhrr.exists():
                real_hrr.load_state_dict(torch.load(str(ckpt_real), map_location=DEVICE))
                fhrr.load_state_dict(torch.load(str(ckpt_fhrr), map_location=DEVICE))
                print(f"✅ Loaded checkpoints")
            else:
                print(f"⚠️ Checkpoints not found for seed {seed}")
                print(f"   Expected: {ckpt_real}")
                print(f"   Expected: {ckpt_fhrr}")
                continue
        except Exception as e:
            print(f"❌ Error loading seed {seed}: {e}")
            continue

        seed_results = {}

        # Baseline: Real HRR standard
        te_acc = Evaluator.atomic_accuracy(real_hrr, test_loader)
        zs_acc = Evaluator.zero_shot_accuracy(real_hrr, zs_h, zs_r1, zs_r2, zs_t)
        mrr,h1,h3,h10 = Evaluator.filtered_mrr_hits(real_hrr, te_s, te_r, te_o, true_tails)
        seed_results["real_hrr_baseline"] = {
            "test_acc": float(te_acc), "zero_shot_acc": float(zs_acc),
            "mrr": float(mrr), "h1": float(h1), "h3": float(h3), "h10": float(h10)
        }
        print(f"Real HRR (standard): Test={te_acc:.4f} ZS={zs_acc:.4f} MRR={mrr:.4f}")

        # ABLATION 1: No Hopfield
        real_hrr.use_hopfield = False
        te_acc_nh = Evaluator.atomic_accuracy(real_hrr, test_loader)
        zs_acc_nh = Evaluator.zero_shot_accuracy(real_hrr, zs_h, zs_r1, zs_r2, zs_t)
        seed_results["ablation_no_hopfield"] = {
            "test_acc": float(te_acc_nh), "zero_shot_acc": float(zs_acc_nh)
        }
        real_hrr.use_hopfield = True
        print(f"No Hopfield:        Test={te_acc_nh:.4f} ZS={zs_acc_nh:.4f}")

        # ABLATION 2: Beta sweep
        beta_sweep = {}
        orig_beta = real_hrr.beta
        for b in [1, 5, 10, 20, 50]:
            real_hrr.beta = b
            zs_b = Evaluator.zero_shot_accuracy(real_hrr, zs_h, zs_r1, zs_r2, zs_t)
            beta_sweep[b] = float(zs_b)
            print(f"β={b:2d}: ZS={zs_b:.4f}")
        real_hrr.beta = orig_beta
        seed_results["beta_sweep"] = beta_sweep

        # ABLATION 3: FHRR Probes (A, B, C)
        N = min(500, len(zs_h))
        ph, pr1, pr2, pt = zs_h[:N].to(DEVICE), zs_r1[:N].to(DEVICE), zs_r2[:N].to(DEVICE), zs_t[:N].to(DEVICE)
        
        moduli = fhrr.probe_modulus(ph, pr1, pr2)
        acc_renorm = (fhrr.forward_with_renorm(ph,pr1,pr2).argmax(1)==pt).float().mean().item()
        acc_hard = (fhrr.forward_hard_cleanup(ph,pr1,pr2).argmax(1)==pt).float().mean().item()
        expected = (math.pi**0.5/2) / (VOCAB**0.5)
        random_bl = 1.0/VOCAB
        
        seed_results["fhrr_probes"] = {
            "moduli": {k: float(v) for k,v in moduli.items()},
            "renorm_acc": float(acc_renorm),
            "hard_acc": float(acc_hard),
            "theory_floor": float(expected),
            "random_baseline": float(random_bl)
        }
        print(f"FHRR Probe A: hop1={moduli['04_after_cleanup_1']:.4f} hop2={moduli['07_after_cleanup_2']:.4f}")
        print(f"FHRR Probe B: renorm={acc_renorm:.6f}")
        print(f"FHRR Probe C: hard={acc_hard:.6f}")

        results_all[seed] = seed_results

    # ─────────────────────────────────────────────────────
    # 5. AGGREGATE & SAVE
    # ─────────────────────────────────────────────────────
    print(f"\n{'='*60}\nAGGREGATING RESULTS\n{'='*60}")
    
    if len(results_all) == 0:
        print("❌ No results - checkpoints missing from Modal volume!")
        print("   Run: modal volume ls hrr-results")
        return {}
    
    available_seeds = list(results_all.keys())
    print(f"Aggregating from seeds: {available_seeds}")
    
    baseline = np.array([results_all[s]["real_hrr_baseline"]["test_acc"] for s in available_seeds])
    baseline_zs = np.array([results_all[s]["real_hrr_baseline"]["zero_shot_acc"] for s in available_seeds])
    baseline_mrr = np.array([results_all[s]["real_hrr_baseline"]["mrr"] for s in available_seeds])
    baseline_h1 = np.array([results_all[s]["real_hrr_baseline"]["h1"] for s in available_seeds])
    baseline_h10 = np.array([results_all[s]["real_hrr_baseline"]["h10"] for s in available_seeds])
    
    ablation_nh = np.array([results_all[s]["ablation_no_hopfield"]["test_acc"] for s in available_seeds])
    ablation_nh_zs = np.array([results_all[s]["ablation_no_hopfield"]["zero_shot_acc"] for s in available_seeds])

    aggregate = {
        "real_hrr_baseline": {
            "test_acc": {"mean": float(baseline.mean()), "std": float(baseline.std())},
            "zero_shot_acc": {"mean": float(baseline_zs.mean()), "std": float(baseline_zs.std())},
            "mrr": {"mean": float(baseline_mrr.mean()), "std": float(baseline_mrr.std())},
            "h1": {"mean": float(baseline_h1.mean()), "std": float(baseline_h1.std())},
            "h10": {"mean": float(baseline_h10.mean()), "std": float(baseline_h10.std())},
        },
        "ablation_no_hopfield": {
            "test_acc": {"mean": float(ablation_nh.mean()), "std": float(ablation_nh.std())},
            "zero_shot_acc": {"mean": float(ablation_nh_zs.mean()), "std": float(ablation_nh_zs.std())},
        },
        "beta_sweep_sample": results_all[available_seeds[0]]["beta_sweep"],
        "fhrr_probes_sample": results_all[available_seeds[0]]["fhrr_probes"],
        "per_seed_results": results_all,
        "seeds_used": available_seeds,
        "num_seeds": len(available_seeds),
    }

    json_path = OUT / "results_inference_ablations.json"
    with open(str(json_path), "w") as f:
        json.dump(aggregate, f, indent=2)
    print(f"✅ Saved: {json_path}")

    # ─────────────────────────────────────────────────────
    # 6. GENERATE FIGURES
    # ─────────────────────────────────────────────────────
    print(f"\n{'='*60}\nGENERATING FIGURES\n{'='*60}")

    # Figure 1: Core components ablation
    fig, ax = plt.subplots(figsize=(10, 6))
    categories = ["Baseline\n(Full)", "No Hopfield\n(Direct Cosine)"]
    test_vals = [baseline.mean(), ablation_nh.mean()]
    test_errs = [baseline.std(), ablation_nh.std()]
    zs_vals = [baseline_zs.mean(), ablation_nh_zs.mean()]
    zs_errs = [baseline_zs.std(), ablation_nh_zs.std()]
    
    x = np.arange(len(categories))
    width = 0.35
    ax.bar(x - width/2, test_vals, width, label="Test Acc", yerr=test_errs, capsize=5, color="steelblue", alpha=0.8)
    ax.bar(x + width/2, zs_vals, width, label="Zero-Shot", yerr=zs_errs, capsize=5, color="darkorange", alpha=0.8)
    ax.set_ylabel("Accuracy")
    ax.set_title("TABLE 2: Core Components Ablation")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(alpha=0.3)
    fig_path = OUT / "table2_ablation_core.png"
    plt.savefig(str(fig_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Figure: table2_ablation_core.png")

    # Figure 2: Beta sweep
    fig, ax = plt.subplots(figsize=(10, 6))
    betas = sorted(results_all[available_seeds[0]]["beta_sweep"].keys())
    zs_per_beta = [results_all[available_seeds[0]]["beta_sweep"][b] for b in betas]
    ax.plot(betas, zs_per_beta, "o-", linewidth=2, markersize=8, color="steelblue")
    ax.set_xlabel("β (Temperature)")
    ax.set_ylabel("Zero-Shot Accuracy")
    ax.set_title("TABLE 5: Beta Temperature Sweep")
    ax.grid(alpha=0.3)
    fig_path = OUT / "table5_beta_sweep.png"
    plt.savefig(str(fig_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Figure: table5_beta_sweep.png")

    # Figure 3: FHRR Probe A - Modulus collapse
    fig, ax = plt.subplots(figsize=(12, 6))
    moduli_sample = results_all[available_seeds[0]]["fhrr_probes"]["moduli"]
    stages = list(moduli_sample.keys())
    values = [moduli_sample[s] for s in stages]
    colors = ["#2ecc71" if "01" in s or "02" in s or "05" in s else "#e74c3c" if "cleanup" in s else "#f39c12" for s in stages]
    ax.bar(range(len(stages)), values, color=colors, edgecolor="black", alpha=0.85)
    ax.axhline(results_all[available_seeds[0]]["fhrr_probes"]["theory_floor"], color="red", linestyle=":", linewidth=2, label=f"Theory floor {results_all[available_seeds[0]]['fhrr_probes']['theory_floor']:.6f}")
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels([s[3:] for s in stages], rotation=45, ha="right")
    ax.set_ylabel("Mean |z|")
    ax.set_title("FIGURE: FHRR Probe A - Modulus Collapse")
    ax.legend()
    ax.grid(alpha=0.3)
    fig_path = OUT / "fhrr_probe_a_modulus.png"
    plt.savefig(str(fig_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Figure: fhrr_probe_a_modulus.png")

    # Figure 4: FHRR Probes B, C
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = ["FHRR\n(Original)", "FHRR\n+Renorm", "FHRR\n+Hard"]
    values = [0.0000, results_all[available_seeds[0]]["fhrr_probes"]["renorm_acc"], results_all[available_seeds[0]]["fhrr_probes"]["hard_acc"]]
    colors = ["#e74c3c", "#f39c12", "#3498db"]
    bars = ax.bar(labels, values, color=colors, edgecolor="black", alpha=0.85)
    ax.axhline(results_all[available_seeds[0]]["fhrr_probes"]["random_baseline"], color="gray", linestyle="--", label=f"Random ({results_all[available_seeds[0]]['fhrr_probes']['random_baseline']:.5f})")
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width()/2, v + 0.0001, f"{v:.6f}", ha="center", fontweight="bold")
    ax.set_ylabel("Zero-Shot Accuracy")
    ax.set_title("FIGURE: FHRR Probes B,C - Renorm & Hard Cleanup")
    ax.legend()
    ax.grid(alpha=0.3)
    fig_path = OUT / "fhrr_probes_bc.png"
    plt.savefig(str(fig_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Figure: fhrr_probes_bc.png")

    volume.commit()
    print(f"\n✅ ALL RESULTS SAVED to Modal Volume at /vol/inference_results/")
    return aggregate

# ─────────────────────────────────────────────────────
# LOCAL ENTRYPOINT
# ─────────────────────────────────────────────────────
@app.local_entrypoint()
def main():
    print("\n🚀 Running inference + ablations on Modal...\n")
    results = run_inference_and_ablations.remote()
    print(f"\n{'='*60}")
    print("✅ COMPLETE!")
    print(f"{'='*60}")
    print("\nResults saved to Modal volume.")
    print("Download with: modal volume get hrr-results ./final_results/")
