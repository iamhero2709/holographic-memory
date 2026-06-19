"""
Phase Error Propagation Analysis + FHRR Full Metrics
Addresses reviewer points 4 & 6:
  - Measure phase error variance: 1-hop vs 2-hop (does it cross pi/2?)
  - Report FHRR MRR / Hits@K (not just accuracy)
  - Per-hop degradation quantification

Usage:  modal run phase_error_analysis.py
Cost:   ~$0.60, runtime ~12 min
"""

import modal

volume = modal.Volume.from_name("hrr-results", create_if_missing=True)
VOLUME_PATH = "/vol"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.2.0", "numpy", "matplotlib", "requests")
)

app = modal.App(name="phase-error-analysis", image=image)


@app.function(gpu="A10", timeout=2400, volumes={VOLUME_PATH: volume})
def analyze():
    import json, math
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
    OUT = Path(VOLUME_PATH) / "phase_analysis"
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}\nPHASE ERROR PROPAGATION ANALYSIS\n{'='*60}")

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
    # Build chains WITH intermediate entity (needed for phase error at hop 1)
    train_lookup = {h:{r:set(ts) for r,ts in rd.items()} for h,rd in train_adj.items()}
    # We need (h, r1, mid, r2, t) tuples - find a valid mid for each (h,t) pair
    train_full = defaultdict(lambda: defaultdict(set))
    for h,r,t in raw["train"]:
        train_full[tok2id[h]][tok2id[r]].add(tok2id[t])

    zh, zr1, zmid, zr2, zt = [], [], [], [], []
    for (r1,r2), pairs in chains:
        for h,t in pairs:
            if any(t in ts for ts in train_lookup.get(h,{}).values()): continue
            # find an intermediate m: (h,r1,m) in train AND (m,r2,t) in train
            mids = train_full.get(h,{}).get(r1, set())
            found_mid = None
            for m in mids:
                if t in train_full.get(m,{}).get(r2, set()):
                    found_mid = m; break
            if found_mid is None and mids:
                found_mid = next(iter(mids))  # fallback: any first-hop target
            if found_mid is None: continue
            zh.append(h); zr1.append(r1); zmid.append(found_mid); zr2.append(r2); zt.append(t)

    zh   = torch.tensor(zh,   dtype=torch.long)
    zr1  = torch.tensor(zr1,  dtype=torch.long)
    zmid = torch.tensor(zmid, dtype=torch.long)
    zr2  = torch.tensor(zr2,  dtype=torch.long)
    zt   = torch.tensor(zt,   dtype=torch.long)
    print(f"  Zero-shot pairs with known intermediate: {len(zh):,}")

    # ── MODEL ──
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
        def _phase_of(self, idx, kind='filler'):
            return self.embed_phase(idx) if kind=='filler' else self.role_phase(idx)
        def _M(self): return torch.complex(self.M_real, self.M_imag)
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
            return torch.complex(torch.cos(ph), torch.sin(ph)), ph
        def forward_atomic(self, s, r):
            f, ro = self._phasor(s), self._phasor(r,'role')
            M = self._M().unsqueeze(0).expand_as(f)
            cln,_ = self._cleanup(self._unbind(M,self._bind(f,ro)))
            return self._cos_sim(cln,self._codebook()) * self.beta

    def atomic_metrics(model, loader, true_tails, n_rank=1000):
        """Returns top1 acc, MRR, H@1, H@3, H@10"""
        model.eval(); c=t=0
        with torch.no_grad():
            for sb,rb,ob in loader:
                sb,rb,ob=sb.to(DEVICE),rb.to(DEVICE),ob.to(DEVICE)
                c+=(model.forward_atomic(sb,rb).argmax(1)==ob).sum().item(); t+=ob.size(0)
        top1 = c/t if t else 0.0
        # filtered ranking on sample
        idx = torch.randperm(len(te_s))[:n_rank]
        s,r,o = te_s[idx].to(DEVICE), te_r[idx].to(DEVICE), te_o[idx].to(DEVICE)
        ranks=[]
        with torch.no_grad():
            for i in range(0,n_rank,128):
                sb,rb,ob=s[i:i+128],r[i:i+128],o[i:i+128]
                logits=model.forward_atomic(sb,rb).float()
                for j in range(len(sb)):
                    hi,ri,oi=sb[j].item(),rb[j].item(),ob[j].item()
                    mask=torch.zeros(logits.size(1),device=DEVICE)
                    for tt in true_tails.get((hi,ri),set()):
                        if tt!=oi: mask[tt]=1e9
                    sc=logits[j]-mask
                    ranks.append((sc>sc[oi]).sum().item()+1)
        ranks=np.array(ranks)
        return top1, float(np.mean(1/ranks)), float(np.mean(ranks<=1)), float(np.mean(ranks<=3)), float(np.mean(ranks<=10))

    def wrap_to_pi(x):
        """wrap angle difference to [-pi, pi]"""
        return torch.atan2(torch.sin(x), torch.cos(x))

    # ════════════════════════════════════════════════════
    # PER-SEED ANALYSIS
    # ════════════════════════════════════════════════════
    report = {}

    for seed in [1, 2, 3, 4, 42]:
        ckpt = Path(VOLUME_PATH) / f"seed_{seed}" / "best_complex_fhrr.pt"
        if not ckpt.exists():
            print(f"❌ No checkpoint seed {seed}"); continue
        print(f"\n{'='*60}\n🔍 SEED {seed}\n{'='*60}")

        model = ComplexFHRR(512, VOCAB, beta=12.0).to(DEVICE)
        model.load_state_dict(torch.load(str(ckpt), map_location=DEVICE))
        model.eval()

        sr = {}

        # ── FHRR FULL METRICS (reviewer point 6) ──
        top1, mrr, h1, h3, h10 = atomic_metrics(model, test_loader, true_tails)
        print(f"[FHRR atomic] top1={top1:.4f} MRR={mrr:.4f} H@1={h1:.4f} H@3={h3:.4f} H@10={h10:.4f}")
        sr["fhrr_metrics"] = {"top1": top1, "mrr": mrr, "h1": h1, "h3": h3, "h10": h10}

        # ── PHASE ERROR PROPAGATION (reviewer point 4) ──
        N = min(800, len(zh))
        h_   = zh[:N].to(DEVICE)
        r1_  = zr1[:N].to(DEVICE)
        mid_ = zmid[:N].to(DEVICE)
        r2_  = zr2[:N].to(DEVICE)
        t_   = zt[:N].to(DEVICE)

        with torch.no_grad():
            M = model._M()
            # ---- HOP 1 ----
            f   = model._phasor(h_)
            ro1 = model._phasor(r1_, 'role')
            q1  = model._bind(f, ro1)
            e1  = model._unbind(M.unsqueeze(0).expand_as(q1), q1)
            mid_clean, mid_phase = model._cleanup(e1, hard=True)   # predicted mid phasor + phase

            # TRUE intermediate phase
            true_mid_phase = model._phase_of(mid_)                  # [N, D]
            # phase error at hop 1 (wrapped)
            err1 = wrap_to_pi(mid_phase - true_mid_phase)           # [N, D]
            abs_err1 = err1.abs()

            # ---- HOP 2 ----
            ro2 = model._phasor(r2_, 'role')
            q2  = model._bind(mid_clean, ro2)
            e2  = model._unbind(M.unsqueeze(0).expand_as(q2), q2)
            tgt_clean, tgt_phase = model._cleanup(e2, hard=False)   # predicted target phase

            true_tgt_phase = model._phase_of(t_)                    # [N, D]
            err2 = wrap_to_pi(tgt_phase - true_tgt_phase)
            abs_err2 = err2.abs()

            # statistics
            mae1 = abs_err1.mean().item()
            mae2 = abs_err2.mean().item()
            std1 = err1.std().item()
            std2 = err2.std().item()
            # fraction of components with |error| > pi/2 (cosine becomes uninformative)
            frac1 = (abs_err1 > math.pi/2).float().mean().item()
            frac2 = (abs_err2 > math.pi/2).float().mean().item()
            # circular variance: 1 - |mean(e^{i*err})|  (0=coherent, 1=uniform)
            cvar1 = 1 - torch.abs(torch.complex(torch.cos(err1), torch.sin(err1)).mean()).item()
            cvar2 = 1 - torch.abs(torch.complex(torch.cos(err2), torch.sin(err2)).mean()).item()

        print(f"[Phase error] HOP1: MAE={mae1:.4f} std={std1:.4f} frac>pi/2={frac1:.3f} cvar={cvar1:.4f}")
        print(f"[Phase error] HOP2: MAE={mae2:.4f} std={std2:.4f} frac>pi/2={frac2:.3f} cvar={cvar2:.4f}")
        print(f"  → uniform random baseline: MAE≈{math.pi/2:.4f}, frac>pi/2=0.5, cvar≈1.0")

        sr["phase_error"] = {
            "hop1": {"mae": mae1, "std": std1, "frac_over_pi2": frac1, "circ_var": cvar1},
            "hop2": {"mae": mae2, "std": std2, "frac_over_pi2": frac2, "circ_var": cvar2},
            "uniform_mae": math.pi/2, "uniform_frac": 0.5, "uniform_cvar": 1.0
        }
        report[seed] = sr

    # ════════════════════════════════════════════════════
    # AGGREGATE
    # ════════════════════════════════════════════════════
    print(f"\n{'='*60}\n📊 AGGREGATE\n{'='*60}")
    seeds = list(report.keys())

    def agg(path):
        vals = []
        for s in seeds:
            d = report[s]
            for k in path: d = d[k]
            vals.append(d)
        return float(np.mean(vals)), float(np.std(vals))

    fhrr_mrr  = agg(["fhrr_metrics","mrr"])
    fhrr_h1   = agg(["fhrr_metrics","h1"])
    fhrr_h3   = agg(["fhrr_metrics","h3"])
    fhrr_h10  = agg(["fhrr_metrics","h10"])
    fhrr_top1 = agg(["fhrr_metrics","top1"])

    mae1 = agg(["phase_error","hop1","mae"]);  mae2 = agg(["phase_error","hop2","mae"])
    frac1= agg(["phase_error","hop1","frac_over_pi2"]); frac2= agg(["phase_error","hop2","frac_over_pi2"])
    cv1  = agg(["phase_error","hop1","circ_var"]); cv2 = agg(["phase_error","hop2","circ_var"])

    print(f"\nFHRR full metrics (5 seeds):")
    print(f"  top1={fhrr_top1[0]:.4f}±{fhrr_top1[1]:.4f}  MRR={fhrr_mrr[0]:.4f}±{fhrr_mrr[1]:.4f}")
    print(f"  H@1={fhrr_h1[0]:.4f}  H@3={fhrr_h3[0]:.4f}  H@10={fhrr_h10[0]:.4f}")
    print(f"\nPhase error propagation (5 seeds):")
    print(f"  HOP1: MAE={mae1[0]:.4f}  frac>pi/2={frac1[0]:.3f}  circ_var={cv1[0]:.4f}")
    print(f"  HOP2: MAE={mae2[0]:.4f}  frac>pi/2={frac2[0]:.3f}  circ_var={cv2[0]:.4f}")
    print(f"  (uniform: MAE={math.pi/2:.4f}, frac=0.5, cvar=1.0)")

    aggregate = {
        "fhrr_metrics": {
            "top1": {"mean": fhrr_top1[0], "std": fhrr_top1[1]},
            "mrr":  {"mean": fhrr_mrr[0],  "std": fhrr_mrr[1]},
            "h1":   {"mean": fhrr_h1[0],   "std": fhrr_h1[1]},
            "h3":   {"mean": fhrr_h3[0],   "std": fhrr_h3[1]},
            "h10":  {"mean": fhrr_h10[0],  "std": fhrr_h10[1]},
        },
        "phase_error": {
            "hop1": {"mae": mae1, "frac_over_pi2": frac1, "circ_var": cv1},
            "hop2": {"mae": mae2, "frac_over_pi2": frac2, "circ_var": cv2},
            "uniform_mae": math.pi/2,
        },
        "per_seed": report,
        "seeds": seeds,
    }
    with open(str(OUT / "phase_error_results.json"), "w") as f:
        json.dump(aggregate, f, indent=2)
    print(f"\n✅ Saved: phase_error_results.json")

    # ════════════════════════════════════════════════════
    # FIGURE: phase error growth across hops
    # ════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: MAE per hop with pi/2 threshold
    ax = axes[0]
    hops = [1, 2]
    maes = [mae1[0], mae2[0]]
    errs = [mae1[1], mae2[1]]
    ax.errorbar(hops, maes, yerr=errs, marker='o', markersize=12, linewidth=2.5,
                capsize=8, color="#2c3e50", label="Mean abs. phase error")
    ax.axhline(math.pi/2, color="#e74c3c", linestyle="--", linewidth=2,
               label=r"$\pi/2$ (cosine uninformative)")
    ax.axhline(math.pi/2, color="#e74c3c", linestyle="--", linewidth=2, alpha=0)
    ax.fill_between([0.5, 2.5], math.pi/2, math.pi, alpha=0.12, color="#e74c3c",
                    label="decorrelated regime")
    ax.set_xticks([1, 2]); ax.set_xticklabels(["Hop 1\n(intermediate)", "Hop 2\n(target)"])
    ax.set_xlim(0.7, 2.3)
    ax.set_ylabel("Mean absolute phase error (rad)")
    ax.set_title("Phase Error Propagation Across Hops")
    ax.legend(loc="center right", fontsize=9)
    ax.grid(alpha=0.3)

    # Right: circular variance + fraction beyond pi/2
    ax = axes[1]
    x = np.arange(2); w = 0.35
    cvars = [cv1[0], cv2[0]]; cverr = [cv1[1], cv2[1]]
    fracs = [frac1[0], frac2[0]]; ferr = [frac1[1], frac2[1]]
    ax.bar(x - w/2, cvars, w, yerr=cverr, capsize=6, label="Circular variance",
           color="#3498db", edgecolor="black")
    ax.bar(x + w/2, fracs, w, yerr=ferr, capsize=6, label=r"Frac. $|\Delta\phi| > \pi/2$",
           color="#e67e22", edgecolor="black")
    ax.axhline(1.0, color="gray", linestyle=":", label="uniform (cvar=1)")
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.6, label="uniform (frac=0.5)")
    ax.set_xticks(x); ax.set_xticklabels(["Hop 1", "Hop 2"])
    ax.set_ylabel("Value")
    ax.set_title("Phase Coherence Degradation")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(OUT / "phase_error_propagation.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: phase_error_propagation.png")

    volume.commit()
    return aggregate


@app.local_entrypoint()
def main():
    print("\n🔬 Running phase error propagation analysis...\n")
    analyze.remote()
    print(f"\n{'='*60}\n✅ DONE\n{'='*60}")
    print("Download: modal volume get hrr-results ./phase_results/")
