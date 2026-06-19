"""
Experiment 1 — Holographic Memory on Modal A10G
  • Logs every metric to Weights & Biases (wandb)
  • Saves results + figures + checkpoints to Modal Volume
  • Run 5 seeds in parallel with one command

SETUP (laptop, one time):
  pip install modal wandb
  modal setup          # login with GitHub  (already done ✅)
  wandb login          # already done ✅

RUN (from your laptop terminal):
  # Single seed (test):
  modal run modal_experiment1.py --seed 42

  # All 5 seeds in PARALLEL (paper run):
  modal run modal_experiment1.py --all-seeds
"""

import os
import sys
import modal

# ─────────────────────────────────────────────────────────────
# 1. MODAL INFRASTRUCTURE
# ─────────────────────────────────────────────────────────────

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
        "urllib3",
        "wandb",
    )
)

def _wandb_secret():
    """Simplified W&B secret detection."""
    try:
        return modal.Secret.from_name("wandb-secret")
    except Exception:
        return modal.Secret.from_dict({
            "WANDB_API_KEY":  os.environ.get("WANDB_API_KEY",  ""),
            "WANDB_ENTITY":   os.environ.get("WANDB_ENTITY",   ""),
            "WANDB_PROJECT":  os.environ.get("WANDB_PROJECT",  "holographic-memory"),
        })

app = modal.App(
    name="holographic-memory-exp1",
    image=image,
    secrets=[_wandb_secret()],
)


# ─────────────────────────────────────────────────────────────
# 2. THE TRAINING FUNCTION
# ─────────────────────────────────────────────────────────────

@app.function(
    gpu="A10G",
    timeout=10800,
    volumes={VOLUME_PATH: volume},
)
def run_experiment(seed: int = 42):
    import os, json, random, math
    from collections import defaultdict
    from pathlib import Path

    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import TensorDataset, DataLoader
    from tqdm import tqdm
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import wandb

    # ── W&B init ─────────────────────────────────────────────
    wb_project = os.environ.get("WANDB_PROJECT", "holographic-memory")
    wb_entity  = os.environ.get("WANDB_ENTITY",  None)

    run = wandb.init(
        project=wb_project,
        entity=wb_entity,
        name=f"exp1-seed{seed}",
        group="experiment1",
        tags=["real_hrr", "fhrr", "fb15k237", f"seed{seed}"],
        config={
            "seed": seed,
            "D_real": 1024,
            "D_complex": 512,
            "beta": 8.0,
            "beta_complex": 12.0,
            "epochs": 2000,
            "lr": 1e-3,
            "lr_min": 1e-5,
            "lambda_inv": 0.2,
            "grad_clip": 1.0,
            "batch_size": 2048,
            "patience": 10,
            "eval_every": 50,
            "dataset": "FB15k-237",
            "gpu": "A10G",
        },
    )

    OUT   = Path(VOLUME_PATH) / f"seed_{seed}"
    FIGS  = OUT / "figures"
    OUT.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)

    class Cfg:
        D_real        = 1024
        D_complex     = 512
        beta          = 8.0
        beta_complex  = 12.0
        epochs        = 2000
        lr            = 1e-3
        lr_min        = 1e-5
        lambda_inv    = 0.2
        grad_clip     = 1.0
        batch_size    = 2048
        patience      = 10
        use_amp       = True
        hebbian_chunk = 4096
        betas_sweep   = [1, 5, 10, 20, 50]
        eval_every    = 50
        mrr_samples   = 1000
        n_chains      = 10

    cfg = Cfg()

    def set_seed(s):
        random.seed(s); np.random.seed(s); torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)

    set_seed(seed)
    DEVICE = torch.device("cuda")
    print(f"\n{'='*60}\nSeed={seed} | GPU={torch.cuda.get_device_name(0)}\n{'='*60}")

    # ─────────────────────────────────────────────────────────
    # DATA
    # ─────────────────────────────────────────────────────────
    print("\n── DATA ────────────────────────────────────────────────")
    base = "https://raw.githubusercontent.com/villmow/datasets_knowledge_embedding/master/FB15k-237"
    raw  = {}
    
    # Robust network logic
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))

    for split in ["train", "valid", "test"]:
        resp = session.get(f"{base}/{split}.txt", timeout=60)
        resp.raise_for_status()
        raw[split] = [tuple(l.strip().split("\t")) for l in resp.text.strip().split("\n")
                      if len(l.strip().split("\t")) == 3]
        print(f"  {split:5s}: {len(raw[split]):>7,}")

    all_triples = raw["train"] + raw["valid"] + raw["test"]
    all_tokens  = sorted(set(x for h,r,t in all_triples for x in (h,r,t)))
    tok2id      = {w:i for i,w in enumerate(all_tokens)}
    id2tok      = {i:w for w,i in tok2id.items()}
    VOCAB       = len(tok2id)
    print(f"  vocab : {VOCAB:,}")

    def encode(triples):
        h = torch.tensor([tok2id[a] for a,b,c in triples], dtype=torch.long)
        r = torch.tensor([tok2id[b] for a,b,c in triples], dtype=torch.long)
        t = torch.tensor([tok2id[c] for a,b,c in triples], dtype=torch.long)
        return h, r, t

    tr_s, tr_r, tr_o = encode(raw["train"])
    va_s, va_r, va_o = encode(raw["valid"])
    te_s, te_r, te_o = encode(raw["test"])

    true_tails = defaultdict(set)
    for h,r,t in all_triples:
        true_tails[(tok2id[h], tok2id[r])].add(tok2id[t])

    def make_loader(s, r, o, shuffle):
        # num_workers=0 as data is entirely in memory
        return DataLoader(TensorDataset(s,r,o), batch_size=cfg.batch_size,
                          shuffle=shuffle, num_workers=0, pin_memory=True)

    train_loader = make_loader(tr_s, tr_r, tr_o, True)
    val_loader   = make_loader(va_s, va_r, va_o, False)
    test_loader  = make_loader(te_s, te_r, te_o, False)

    # ─────────────────────────────────────────────────────────
    # CHAIN EXTRACTION
    # ─────────────────────────────────────────────────────────
    print("\n── CHAINS ───────────────────────────────────────────────")
    train_adj = defaultdict(lambda: defaultdict(set))
    for h,r,t in raw["train"]: train_adj[tok2id[h]][tok2id[r]].add(tok2id[t])
    test_adj = defaultdict(lambda: defaultdict(set))
    for h,r,t in raw["test"]: test_adj[tok2id[h]][tok2id[r]].add(tok2id[t])

    def discover_chains(train_adj, test_adj, n=10, min_pairs=30):
        train_p = {h: dict(rd) for h,rd in train_adj.items()}
        test_p  = {h: dict(rd) for h,rd in test_adj.items()}
        EMPTY   = {}
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

    chains = discover_chains(train_adj, test_adj, cfg.n_chains)
    if not chains: raise RuntimeError("No chains found")
    print(f"  Found {len(chains)} chains")
    
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
    print(f"  Zero-shot pairs: {len(zs_h):,}  |  leaked (removed): {leaked}")

    wandb.config.update({"vocab_size": VOCAB, "n_zeroshot_pairs": len(zs_h), "n_chains": len(chains)})

    # ─────────────────────────────────────────────────────────
    # MODELS
    # ─────────────────────────────────────────────────────────
    class RealHRR(nn.Module):
        def __init__(self, D, vocab_size, beta=8.0):
            super().__init__()
            self.D, self.vocab_size, self.beta = D, vocab_size, beta
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
            scores = z.float() @ self.entity_emb.weight.float().T * self.beta
            scores = scores.clamp(-50, 50)
            w = F.gumbel_softmax(scores, tau=1.0, hard=True) if hard else F.softmax(scores, dim=-1)
            return (w @ self.entity_emb.weight.float()).to(z.dtype)

        def hebbian_init(self, s, r, o, chunk=4096):
            target = self.D ** 0.5
            with torch.no_grad():
                acc = torch.zeros(self.D, device=self.M.device)
                for i in range(0, len(s), chunk):
                    sb, rb, ob = s[i:i+chunk].to(self.M.device), r[i:i+chunk].to(self.M.device), o[i:i+chunk].to(self.M.device)
                    acc += self._bind(self._bind(self.entity_emb(sb), self.role_emb(rb)), self.entity_emb(ob)).sum(0)
                self.M.data = acc / acc.norm().clamp(1e-8) * target
            print(f"  Hebbian ||M|| = {self.M.norm():.2f} (target≈{target:.1f})")

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
            """Cache removed - computed directly on the fly."""
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

        def hebbian_init(self, s, r, o, scale=0.01, chunk=4096):
            target = self.D ** 0.5
            with torch.no_grad():
                ar = torch.zeros(self.D, device=self.M_real.device)
                ai = torch.zeros(self.D, device=self.M_real.device)
                for i in range(0, len(s), chunk):
                    sb, rb, ob = s[i:i+chunk].to(self.M_real.device), r[i:i+chunk].to(self.M_real.device), o[i:i+chunk].to(self.M_real.device)
                    facts = self._bind(self._bind(self._phasor(sb),self._phasor(rb,'role')),self._phasor(ob))
                    ar+=facts.real.sum(0); ai+=facts.imag.sum(0)
                norm = (ar.norm()**2 + ai.norm()**2).sqrt().clamp(min=1e-8)
                self.M_real.data = ar / norm * target
                self.M_imag.data = ai / norm * target
            print(f"  FHRR Hebbian ||M_real|| = {self.M_real.norm():.4f}  (target≈{target:.1f})")

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

    # ─────────────────────────────────────────────────────────
    # EVALUATOR
    # ─────────────────────────────────────────────────────────
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

    # ─────────────────────────────────────────────────────────
    # TRAINER with W&B logging
    # ─────────────────────────────────────────────────────────
    def train_model(model, name, use_amp=True):
        model = model.to(DEVICE)
        opt   = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.epochs, cfg.lr_min)
        ce    = nn.CrossEntropyLoss()

        try:
            scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        except Exception:
            scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        prefix    = name.lower().replace(" ","_")
        history   = []
        best_val  = 0.0
        best_ep   = patience_ctr = 0
        ckpt_path = str(OUT / f"best_{prefix}.pt")

        pbar = tqdm(range(cfg.epochs), desc=f"▸ {name}", ncols=100)
        for ep in pbar:
            model.train()
            ep_loss=ep_corr=ep_total=0

            for sb,rb,ob in train_loader:
                sb,rb,ob=sb.to(DEVICE),rb.to(DEVICE),ob.to(DEVICE)
                opt.zero_grad(set_to_none=True)

                ctx = torch.autocast("cuda", enabled=scaler.is_enabled())
                with ctx:
                    logits  = model.forward_atomic(sb,rb)
                    loss_ce = ce(logits,ob)

                    if ep%2==0:
                        B  = min(256,VOCAB)
                        fi = torch.randint(0,VOCAB,(B,),device=DEVICE)
                        ri = torch.randint(0,VOCAB,(B,),device=DEVICE)
                        rf = torch.randint(0,VOCAB,(B,),device=DEVICE)
                        if isinstance(model, RealHRR):
                            f=model.entity_emb(fi); ro=model.role_emb(ri)
                            bnd=model._bind(f,ro); f_r=model._unbind(bnd,ro)
                            pos=F.cosine_similarity(f_r,f,dim=-1).mean()
                            neg=F.cosine_similarity(f_r,model.entity_emb(rf),dim=-1).mean()
                        else:
                            f=model._phasor(fi); ro=model._phasor(ri,'role')
                            bnd=model._bind(f,ro); f_r=model._unbind(bnd,ro)
                            def cc(a,b):
                                d=(a.real*b.real+a.imag*b.imag).sum(-1)
                                na=(a.real**2+a.imag**2).sum(-1).sqrt()+1e-8
                                nb=(b.real**2+b.imag**2).sum(-1).sqrt()+1e-8
                                return (d/(na*nb)).mean()
                            pos=cc(f_r,f); neg=cc(f_r,model._phasor(rf))
                        loss_inv=(1-pos)+neg
                        loss=loss_ce+cfg.lambda_inv*loss_inv
                    else:
                        loss=loss_ce

                # Catch NaN and log gracefully
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"\n⚠️ NaN/Inf loss at epoch {ep+1}, skipping batch.", file=sys.stderr)
                    wandb.log({f"{prefix}/nan_events": 1})
                    opt.zero_grad(set_to_none=True); continue

                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(opt); scaler.update()

                ep_loss +=loss.item()
                ep_corr +=(logits.argmax(1)==ob).sum().item()
                ep_total+=ob.size(0)

            sched.step()
            train_acc = ep_corr/ep_total
            avg_loss  = ep_loss/len(train_loader)

            # ── Eval + Early Stopping + W&B log ────────────────────────────────
            if (ep+1)%cfg.eval_every==0 or ep==cfg.epochs-1:
                val_acc = Evaluator.atomic_accuracy(model, val_loader)
                zs_acc  = Evaluator.zero_shot_accuracy(model, zs_h, zs_r1, zs_r2, zs_t)

                history.append({"epoch":ep+1,"train_acc":train_acc,
                                 "val_acc":val_acc,"zs_acc":zs_acc,"loss":avg_loss})

                wandb.log({
                    f"{prefix}/epoch"     : ep+1,
                    f"{prefix}/loss"      : avg_loss,
                    f"{prefix}/train_acc" : train_acc,
                    f"{prefix}/val_acc"   : val_acc,
                    f"{prefix}/zs_acc"    : zs_acc,
                    f"{prefix}/lr"        : opt.param_groups[0]["lr"],
                })

                pbar.set_postfix({"loss":f"{avg_loss:.3f}", "val":f"{val_acc:.2%}", "zs": f"{zs_acc:.2%}", "bV": f"{best_val:.2%}"})
                if (ep+1)%200==0:
                    tqdm.write(f"  ── ep {ep+1:>4d} | loss {avg_loss:.4f} | val {val_acc:.2%} | zs {zs_acc:.2%}")

                # Early stopping based on validation accuracy
                if val_acc > best_val:
                    best_val, best_ep, patience_ctr = val_acc, ep+1, 0
                    torch.save(model.state_dict(), ckpt_path)
                    wandb.log({f"{prefix}/best_val_acc": best_val, f"{prefix}/best_ep": best_ep})
                else:
                    patience_ctr += 1
                    if patience_ctr >= cfg.patience:
                        tqdm.write(f"  ⏹ {name} early stop @ ep {ep+1} (best val {best_val:.4f} @ ep {best_ep})")
                        break

        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        return history, model

    # ─────────────────────────────────────────────────────────
    # TRAIN: Real HRR
    # ─────────────────────────────────────────────────────────
    print("\n── Real HRR D=1024 ──────────────────────────────────────")
    real_hrr = RealHRR(cfg.D_real, VOCAB, cfg.beta).to(DEVICE)
    real_hrr.hebbian_init(tr_s, tr_r, tr_o, chunk=cfg.hebbian_chunk)
    hist_real, real_hrr = train_model(real_hrr, "Real HRR", use_amp=True)

    te_acc = Evaluator.atomic_accuracy(real_hrr, test_loader)
    zs_acc = Evaluator.zero_shot_accuracy(real_hrr, zs_h, zs_r1, zs_r2, zs_t)
    mrr,h1,h3,h10 = Evaluator.filtered_mrr_hits(real_hrr, te_s, te_r, te_o, true_tails, cfg.mrr_samples)

    wandb.log({"real_hrr/final_test_acc":te_acc, "real_hrr/final_zs_acc":zs_acc,
               "real_hrr/mrr_filtered":mrr, "real_hrr/hits1":h1,
               "real_hrr/hits3":h3, "real_hrr/hits10":h10})
    print(f"  Test acc : {te_acc:.4f} | Zero-shot : {zs_acc:.4f}")
    print(f"  MRR      : {mrr:.4f}  | H@1/3/10  : {h1:.4f}/{h3:.4f}/{h10:.4f}")

    # ─────────────────────────────────────────────────────────
    # TRAIN: Complex FHRR
    # ─────────────────────────────────────────────────────────
    print("\n── Complex FHRR D=512 ───────────────────────────────────")
    fhrr = ComplexFHRR(cfg.D_complex, VOCAB, cfg.beta_complex).to(DEVICE)
    fhrr.hebbian_init(tr_s, tr_r, tr_o, chunk=cfg.hebbian_chunk)
    hist_fhrr, fhrr = train_model(fhrr, "Complex FHRR", use_amp=False)

    fhrr_te  = Evaluator.atomic_accuracy(fhrr, test_loader)
    fhrr_zs  = Evaluator.zero_shot_accuracy(fhrr, zs_h, zs_r1, zs_r2, zs_t)
    wandb.log({"complex_fhrr/final_test_acc":fhrr_te, "complex_fhrr/final_zs_acc":fhrr_zs})
    print(f"  Test acc : {fhrr_te:.4f} | Zero-shot : {fhrr_zs:.4f}")

    # ─────────────────────────────────────────────────────────
    # FHRR PROBES
    # ─────────────────────────────────────────────────────────
    print("\n── FHRR Probes ──────────────────────────────────────────")
    N   = min(500, len(zs_h))
    ph, pr1, pr2, pt = zs_h[:N].to(DEVICE), zs_r1[:N].to(DEVICE), zs_r2[:N].to(DEVICE), zs_t[:N].to(DEVICE)
    fhrr.eval()

    moduli    = fhrr.probe_modulus(ph, pr1, pr2)
    acc_renorm= (fhrr.forward_with_renorm(ph,pr1,pr2).argmax(1)==pt).float().mean().item()
    acc_hard  = (fhrr.forward_hard_cleanup(ph,pr1,pr2).argmax(1)==pt).float().mean().item()
    expected  = (math.pi**0.5/2) / (VOCAB**0.5)
    random_bl = 1.0/VOCAB

    for stage,val in moduli.items(): wandb.log({f"fhrr_probe/modulus_{stage}":val})
    wandb.log({"fhrr_probe/renorm_acc":acc_renorm, "fhrr_probe/hard_acc":acc_hard,
               "fhrr_probe/theory_floor":expected, "fhrr_probe/random_baseline":random_bl})

    print(f"  Modulus hop1 after cleanup : {moduli['04_after_cleanup_1']:.4f}")
    print(f"  Modulus hop2 after cleanup : {moduli['07_after_cleanup_2']:.4f}")
    print(f"  Theory floor               : {expected:.4f}")
    print(f"  + Renorm acc               : {acc_renorm:.4f}")
    print(f"  + Hard cleanup acc         : {acc_hard:.4f}")

    # ─────────────────────────────────────────────────────────
    # BETA SWEEP
    # ─────────────────────────────────────────────────────────
    print("\n── β Sweep ──────────────────────────────────────────────")
    orig_beta = real_hrr.beta
    beta_results = {}
    for b in cfg.betas_sweep:
        real_hrr.beta = b
        acc = Evaluator.zero_shot_accuracy(real_hrr, zs_h, zs_r1, zs_r2, zs_t)
        beta_results[b] = acc
        wandb.log({f"beta_sweep/beta_{b}":acc})
        print(f"  β={b:2d} → zs={acc:.4f}")
    real_hrr.beta = orig_beta

    # ─────────────────────────────────────────────────────────
    # SAVE & PLOT
    # ─────────────────────────────────────────────────────────
    results_dict = {
        "seed": seed, "vocab_size": VOCAB, "n_zeroshot_pairs": len(zs_h), "n_chains": len(chains),
        "real_hrr": {"test_acc":te_acc,"zero_shot_acc":zs_acc,"mrr_filtered":mrr,"hits1":h1,"hits3":h3,"hits10":h10},
        "complex_fhrr": {"test_acc":fhrr_te,"zero_shot_acc":fhrr_zs},
        "fhrr_probes": {"modulus_stages":moduli,"renorm_acc":acc_renorm,"hard_acc":acc_hard,"theory_floor":expected,"random_baseline":random_bl},
        "beta_sweep": beta_results, "history_real": hist_real, "history_fhrr": hist_fhrr,
    }
    with open(OUT/"results.json","w") as f: json.dump(results_dict, f, indent=2)
    print(f"\n  Saved JSON → {OUT}/results.json")

    print("\n── Figures ──────────────────────────────────────────────")
    fig, axes = plt.subplots(1,3,figsize=(18,5))
    def ph(ax, hist, lbl, col, k): ax.plot([h["epoch"] for h in hist],[h[k] for h in hist],lw=2,color=col,label=lbl)
    ph(axes[0],hist_real,"Real HRR","steelblue","val_acc"); ph(axes[0],hist_fhrr,"FHRR","darkorange","val_acc")
    axes[0].set(title="Val Atomic Accuracy",xlabel="Epoch",ylabel="Accuracy"); axes[0].legend(); axes[0].grid(alpha=.3)
    ph(axes[1],hist_real,"Real HRR","steelblue","zs_acc"); ph(axes[1],hist_fhrr,"FHRR","darkorange","zs_acc")
    axes[1].set(title="Zero-Shot Accuracy",xlabel="Epoch",ylabel="Accuracy"); axes[1].legend(); axes[1].grid(alpha=.3)
    ph(axes[2],hist_real,"Real HRR","steelblue","loss"); ph(axes[2],hist_fhrr,"FHRR","darkorange","loss")
    axes[2].set(title="Training Loss",xlabel="Epoch",ylabel="Loss"); axes[2].legend(); axes[2].grid(alpha=.3)
    plt.tight_layout(); p=str(FIGS/"1_training_curves.png")
    plt.savefig(p,dpi=150,bbox_inches="tight"); plt.close()
    wandb.log({"figures/training_curves":wandb.Image(p)}); print(f"  1_training_curves.png")

    fig,ax=plt.subplots(figsize=(10,5))
    st=list(moduli.keys()); vl=list(moduli.values())
    cols=["#2ecc71" if "01" in s or "02" in s or "05" in s else "#e74c3c" if "cleanup" in s else "#f39c12" for s in st]
    ax.bar(range(len(st)),vl,color=cols,edgecolor="black",alpha=.85)
    ax.axhline(1.0,color="gray",ls="--",label="|z|=1"); ax.axhline(expected,color="red",ls=":",lw=2,label=f"Theory floor {expected:.4f}")
    ax.set_xticks(range(len(st))); ax.set_xticklabels([s[3:] for s in st],rotation=40,ha="right")
    ax.set(title="FHRR Probe A: Modulus Collapse",ylabel="Mean |z|"); ax.legend(); ax.grid(alpha=.3)
    plt.tight_layout(); p=str(FIGS/"2_modulus_collapse.png")
    plt.savefig(p,dpi=150,bbox_inches="tight"); plt.close()
    wandb.log({"figures/modulus_collapse":wandb.Image(p)}); print(f"  2_modulus_collapse.png")

    fig,ax=plt.subplots(figsize=(8,5))
    lbls=["FHRR\n(orig)","FHRR\n+Renorm","FHRR\n+Hard","Real HRR"]; vs=[fhrr_zs,acc_renorm,acc_hard,zs_acc]
    cs=["#e74c3c","#f39c12","#3498db","#2ecc71"]
    bars=ax.bar(lbls,vs,color=cs,edgecolor="black",alpha=.85); ax.axhline(random_bl,color="gray",ls="--",label=f"Random ({random_bl:.5f})")
    for b,v in zip(bars,vs): ax.text(b.get_x()+b.get_width()/2,v+.002,f"{v:.4f}",ha="center",fontsize=11,fontweight="bold")
    ax.set(title="FHRR Collapse Mechanism",ylabel="Zero-Shot Accuracy"); ax.legend(); ax.grid(alpha=.3)
    plt.tight_layout(); p=str(FIGS/"3_probe_comparison.png")
    plt.savefig(p,dpi=150,bbox_inches="tight"); plt.close()
    wandb.log({"figures/probe_comparison":wandb.Image(p)}); print(f"  3_probe_comparison.png")

    fig,ax=plt.subplots(figsize=(8,5))
    ax.plot(list(beta_results.keys()),list(beta_results.values()),"o-",lw=2,color="steelblue",ms=8)
    for b,a in beta_results.items(): ax.annotate(f"{a:.3f}",(b,a),textcoords="offset points",xytext=(0,10),ha="center",fontsize=9)
    ax.set(title="β Temperature Sweep",xlabel="β",ylabel="Zero-Shot Accuracy"); ax.grid(alpha=.3)
    plt.tight_layout(); p=str(FIGS/"4_beta_sweep.png")
    plt.savefig(p,dpi=150,bbox_inches="tight"); plt.close()
    wandb.log({"figures/beta_sweep":wandb.Image(p)}); print(f"  4_beta_sweep.png")

    fig,ax=plt.subplots(figsize=(10,5))
    cats=["Real HRR\nTest","Real HRR\nZero-Shot","FHRR\nTest","FHRR\nZero-Shot"]; vs=[te_acc,zs_acc,fhrr_te,fhrr_zs]
    cs=["#2980b9","#27ae60","#e67e22","#e74c3c"]
    bars=ax.bar(cats,vs,color=cs,edgecolor="black",alpha=.85,width=.5); ax.axhline(random_bl,color="gray",ls="--",label=f"Random ({random_bl:.5f})")
    for b,v in zip(bars,vs): ax.text(b.get_x()+b.get_width()/2,v+.003,f"{v:.4f}",ha="center",fontsize=11,fontweight="bold")
    ax.set(title=f"Experiment 1 Results — Seed {seed}",ylabel="Accuracy"); ax.legend(); ax.grid(alpha=.3)
    plt.tight_layout(); p=str(FIGS/"5_main_results.png")
    plt.savefig(p,dpi=150,bbox_inches="tight"); plt.close()
    wandb.log({"figures/main_results":wandb.Image(p)}); print(f"  5_main_results.png")

    volume.commit()
    print(f"\n  ✅ All files committed to Modal Volume 'hrr-results/seed_{seed}/'")
    wandb.finish()
    
    print(f"\n{'='*60}\nEXPERIMENT 1 COMPLETE — Seed {seed}")
    print(f"  W&B  : wandb.ai / {wb_entity or 'your-account'} / {wb_project}")
    print(f"  Vol  : modal volume get hrr-results / ./local_results/\n{'='*60}")
    return results_dict


# ─────────────────────────────────────────────────────────────
# 3. LOCAL ENTRYPOINTS
# ─────────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(seed: int = 42, all_seeds: bool = False):
    if all_seeds:
        seeds = [42, 1, 2, 3, 4]
        print(f"🚀 Launching {len(seeds)} seeds in PARALLEL on A10G...\n   Seeds: {seeds}\n   Watch live: wandb.ai (check your project)\n")
        for i, result in enumerate(run_experiment.map(seeds)):
            s = seeds[i]
            print(f"\n✅ Seed {s} done:\n   Real HRR  zs={result['real_hrr']['zero_shot_acc']:.4f}  mrr={result['real_hrr']['mrr_filtered']:.4f}\n   FHRR      zs={result['complex_fhrr']['zero_shot_acc']:.4f}")
    else:
        print(f"🚀 Launching seed={seed} on A10G...")
        result = run_experiment.remote(seed)
        print(f"\n✅ Done!\n   Real HRR  zs={result['real_hrr']['zero_shot_acc']:.4f}  mrr={result['real_hrr']['mrr_filtered']:.4f}\n   FHRR      zs={result['complex_fhrr']['zero_shot_acc']:.4f}")
        print(f"\n   Download: modal volume get hrr-results / ./local_results/")
