"""
FHRR Collapse VERIFICATION Script
Determines whether probe result (hop1=hop2=1.0) is genuine or artifact.

Runs 5 diagnostic checks:
  CHECK 1: Is FHRR checkpoint actually trained? (Hebbian M norm)
  CHECK 2: Single-hop sanity (does atomic work at all?)
  CHECK 3: Per-component modulus distribution (not just mean)
  CHECK 4: Phase coherence tracking (the REAL suspected cause)
  CHECK 5: Fresh untrained model comparison (control)

Usage:
  modal run verify_fhrr_collapse.py

Cost: ~$0.50, runtime ~10 min
"""

import modal

volume = modal.Volume.from_name("hrr-results", create_if_missing=True)
VOLUME_PATH = "/vol"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.2.0", "numpy", "matplotlib", "requests")
)

app = modal.App(name="verify-fhrr-collapse", image=image)


@app.function(gpu="A10", timeout=2400, volumes={VOLUME_PATH: volume})
def verify():
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
    OUT = Path(VOLUME_PATH) / "verification"
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}\nFHRR COLLAPSE VERIFICATION\n{'='*60}")

    # ── DATA ──
    print("\n── LOADING DATA ──")
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

    # chains for zero-shot
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
    zh  = torch.tensor(zh, dtype=torch.long)
    zr1 = torch.tensor(zr1, dtype=torch.long)
    zr2 = torch.tensor(zr2, dtype=torch.long)
    zt  = torch.tensor(zt, dtype=torch.long)
    print(f"  Zero-shot pairs: {len(zh):,}")

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

    def atomic_acc(model, loader):
        model.eval(); c=t=0
        with torch.no_grad():
            for sb,rb,ob in loader:
                sb,rb,ob=sb.to(DEVICE),rb.to(DEVICE),ob.to(DEVICE)
                c+=(model.forward_atomic(sb,rb).argmax(1)==ob).sum().item(); t+=ob.size(0)
        return c/t if t else 0.0

    def zs_acc(model, h,r1,r2,t, batch=512):
        model.eval(); c=tot=0
        with torch.no_grad():
            for i in range(0,len(h),batch):
                hb,r1b,r2b,tb=h[i:i+batch].to(DEVICE),r1[i:i+batch].to(DEVICE),r2[i:i+batch].to(DEVICE),t[i:i+batch].to(DEVICE)
                c+=(model.forward_composition(hb,r1b,r2b).argmax(1)==tb).sum().item(); tot+=tb.size(0)
        return c/tot if tot else 0.0

    # ════════════════════════════════════════════════════
    # VERIFICATION CHECKS PER SEED
    # ════════════════════════════════════════════════════
    report = {}

    for seed in [1, 2, 3, 4, 42]:
        print(f"\n{'='*60}\n🔍 SEED {seed}\n{'='*60}")
        ckpt = Path(VOLUME_PATH) / f"seed_{seed}" / "best_complex_fhrr.pt"
        if not ckpt.exists():
            print(f"❌ No checkpoint for seed {seed}"); continue

        model = ComplexFHRR(512, VOCAB, beta=12.0).to(DEVICE)
        state = torch.load(str(ckpt), map_location=DEVICE)
        model.load_state_dict(state)
        model.eval()

        seed_report = {}

        # ── CHECK 1: Is checkpoint actually trained? ──
        # Untrained M_real/M_imag = 0. Hebbian init makes ||M|| large.
        M = model._M()
        M_norm = torch.abs(M).norm().item()
        M_real_norm = model.M_real.norm().item()
        M_imag_norm = model.M_imag.norm().item()
        # Phase embeddings: untrained = uniform [-pi,pi], trained = shifted
        phase_std = model.embed_phase.weight.std().item()
        print(f"\n[CHECK 1] Checkpoint Trained?")
        print(f"  ||M|| (complex)  = {M_norm:.4f}   (untrained≈0, trained≈large)")
        print(f"  ||M_real||       = {M_real_norm:.4f}")
        print(f"  ||M_imag||       = {M_imag_norm:.4f}")
        print(f"  phase emb std    = {phase_std:.4f}   (uniform≈1.814)")
        is_trained = M_norm > 1.0
        print(f"  → {'✅ TRAINED' if is_trained else '❌ UNTRAINED/EMPTY M!'}")
        seed_report["check1_trained"] = {
            "M_norm": M_norm, "M_real_norm": M_real_norm,
            "M_imag_norm": M_imag_norm, "phase_std": phase_std,
            "is_trained": bool(is_trained)
        }

        # ── CHECK 2: Single-hop sanity ──
        atom = atomic_acc(model, test_loader)
        zs = zs_acc(model, zh, zr1, zr2, zt)
        print(f"\n[CHECK 2] Single-hop vs Two-hop")
        print(f"  Atomic (1-hop)   = {atom:.4f}   (should be >> random {1/VOCAB:.6f})")
        print(f"  Zero-shot (2-hop)= {zs:.6f}")
        print(f"  → 1-hop {'works' if atom > 0.05 else 'BROKEN'}, "
              f"2-hop {'works' if zs > 0.01 else 'fails'}")
        seed_report["check2_sanity"] = {"atomic": atom, "zero_shot": zs}

        # ── CHECK 3: Per-component modulus DISTRIBUTION (not just mean) ──
        N = min(500, len(zh))
        h_, r1_, r2_ = zh[:N].to(DEVICE), zr1[:N].to(DEVICE), zr2[:N].to(DEVICE)
        with torch.no_grad():
            f = model._phasor(h_); ro1 = model._phasor(r1_,'role'); ro2 = model._phasor(r2_,'role')
            M = model._M()
            # hop 1
            q1 = model._bind(f, ro1)
            e1 = model._unbind(M.unsqueeze(0).expand_as(q1), q1)
            mid = model._cleanup(e1, hard=True)
            mid_mod = torch.abs(mid)            # [N, D]
            # hop 2
            q2 = model._bind(mid, ro2)
            e2 = model._unbind(M.unsqueeze(0).expand_as(q2), q2)
            cln = model._cleanup(e2, hard=False)
            cln_mod = torch.abs(cln)            # [N, D]
        print(f"\n[CHECK 3] Modulus DISTRIBUTION (per-component, not just mean)")
        print(f"  hop1 |z|: mean={mid_mod.mean():.4f} std={mid_mod.std():.4f} "
              f"min={mid_mod.min():.4f} max={mid_mod.max():.4f}")
        print(f"  hop2 |z|: mean={cln_mod.mean():.4f} std={cln_mod.std():.4f} "
              f"min={cln_mod.min():.4f} max={cln_mod.max():.4f}")
        # If cleanup outputs unit phasors by construction (cos/sin), |z| is ALWAYS 1.
        # That means modulus probe is MEANINGLESS for this architecture!
        note = ("⚠️ Cleanup outputs cos/sin phasors → |z|=1 BY CONSTRUCTION. "
                "Modulus probe cannot detect collapse here.")
        if abs(cln_mod.mean().item() - 1.0) < 0.01:
            print(f"  → {note}")
        seed_report["check3_modulus_dist"] = {
            "hop1": {"mean": mid_mod.mean().item(), "std": mid_mod.std().item(),
                     "min": mid_mod.min().item(), "max": mid_mod.max().item()},
            "hop2": {"mean": cln_mod.mean().item(), "std": cln_mod.std().item(),
                     "min": cln_mod.min().item(), "max": cln_mod.max().item()},
            "note": note if abs(cln_mod.mean().item()-1.0) < 0.01 else "modulus varies"
        }

        # ── CHECK 4: PHASE COHERENCE (the REAL diagnostic) ──
        # Compare cleaned vector's phase to TRUE target's phase.
        # If phase scrambles, cos-sim to true target drops to ~0.
        with torch.no_grad():
            t_true = model._phasor(zt[:N].to(DEVICE))  # true target phasors
            # cosine similarity between hop2 cleaned and true target
            def phasor_cos(a, b):
                dot = (a.real*b.real + a.imag*b.imag).sum(-1)
                na = (a.real**2+a.imag**2).sum(-1).sqrt()+1e-8
                nb = (b.real**2+b.imag**2).sum(-1).sqrt()+1e-8
                return (dot/(na*nb)).mean().item()
            # baseline: random target phasor cos-sim
            rand_idx = torch.randint(0, VOCAB, (N,), device=DEVICE)
            t_rand = model._phasor(rand_idx)
            sim_true = phasor_cos(cln, t_true)
            sim_rand = phasor_cos(cln, t_rand)
            # also: intermediate hop accuracy (did hop1 find right mid-entity?)
            # we don't have ground-truth mid, so skip; track phase variance instead
            phase_var_hop1 = torch.atan2(mid.imag, mid.real).std().item()
            phase_var_hop2 = torch.atan2(cln.imag, cln.real).std().item()
        print(f"\n[CHECK 4] PHASE COHERENCE (real diagnostic)")
        print(f"  cos(hop2_clean, TRUE target)   = {sim_true:.4f}")
        print(f"  cos(hop2_clean, RANDOM target) = {sim_rand:.4f}")
        print(f"  → signal vs noise: {sim_true:.4f} vs {sim_rand:.4f}")
        if abs(sim_true - sim_rand) < 0.02:
            print(f"  → ✅ CONFIRMED: phase scrambled — cleaned vector no closer "
                  f"to true than random. THIS is the collapse mechanism.")
        else:
            print(f"  → Cleaned vector retains some signal toward true target.")
        print(f"  phase std hop1={phase_var_hop1:.4f} hop2={phase_var_hop2:.4f}")
        seed_report["check4_phase_coherence"] = {
            "sim_true": sim_true, "sim_rand": sim_rand,
            "signal_gap": sim_true - sim_rand,
            "phase_scrambled": bool(abs(sim_true - sim_rand) < 0.02),
            "phase_std_hop1": phase_var_hop1, "phase_std_hop2": phase_var_hop2
        }

        # ── CHECK 5: UNTRAINED CONTROL ──
        # Fresh model (random init) — does it ALSO get ~0 zero-shot?
        # If trained == untrained on zero-shot, training didn't help composition at all.
        fresh = ComplexFHRR(512, VOCAB, beta=12.0).to(DEVICE)
        fresh.eval()
        fresh_atom = atomic_acc(fresh, test_loader)
        fresh_zs = zs_acc(fresh, zh, zr1, zr2, zt)
        print(f"\n[CHECK 5] UNTRAINED CONTROL")
        print(f"  Untrained atomic   = {fresh_atom:.4f}  (trained: {atom:.4f})")
        print(f"  Untrained zero-shot= {fresh_zs:.6f}  (trained: {zs:.6f})")
        print(f"  → Training {'HELPED atomic' if atom > fresh_atom + 0.01 else 'did not help atomic'}, "
              f"{'helped' if zs > fresh_zs + 0.005 else 'did NOT help'} zero-shot")
        seed_report["check5_control"] = {
            "untrained_atomic": fresh_atom, "untrained_zs": fresh_zs,
            "trained_atomic": atom, "trained_zs": zs
        }

        report[seed] = seed_report

    # ════════════════════════════════════════════════════
    # VERDICT
    # ════════════════════════════════════════════════════
    print(f"\n{'='*60}\n📋 FINAL VERDICT\n{'='*60}")

    if len(report) == 0:
        print("❌ No FHRR checkpoints found.")
        return {}

    seeds = list(report.keys())
    trained_all = all(report[s]["check1_trained"]["is_trained"] for s in seeds)
    atomic_works = all(report[s]["check2_sanity"]["atomic"] > 0.05 for s in seeds)
    scrambled_all = all(report[s]["check4_phase_coherence"]["phase_scrambled"] for s in seeds)
    avg_sim_true = np.mean([report[s]["check4_phase_coherence"]["sim_true"] for s in seeds])
    avg_sim_rand = np.mean([report[s]["check4_phase_coherence"]["sim_rand"] for s in seeds])

    print(f"\n1. Checkpoints trained?     {'✅ YES' if trained_all else '❌ NO (incomplete!)'}")
    print(f"2. Single-hop works?        {'✅ YES' if atomic_works else '❌ NO (broken model)'}")
    print(f"3. Modulus probe valid?     ❌ NO — cleanup forces |z|=1 by construction")
    print(f"4. Phase scrambled?         {'✅ YES' if scrambled_all else '⚠️ PARTIAL'}")
    print(f"   avg cos(true)={avg_sim_true:.4f}  avg cos(rand)={avg_sim_rand:.4f}")

    print(f"\n{'─'*60}")
    if trained_all and atomic_works and scrambled_all:
        verdict = ("✅ FHRR COLLAPSE VERIFIED via PHASE COHERENCE.\n"
                   "   - Checkpoints are properly trained (||M|| large, atomic works)\n"
                   "   - Modulus probe was the WRONG tool (|z|=1 by construction)\n"
                   "   - REAL mechanism: cleaned hop-2 vector is no closer to the\n"
                   "     true target than a random entity → phase information lost.\n"
                   "   - This is a publishable, correct mechanistic finding.")
    elif trained_all and atomic_works and not scrambled_all:
        verdict = ("⚠️ MIXED: trained + atomic works, but cleaned vector retains\n"
                   "   some signal toward true target. Failure may be partial /\n"
                   "   bottleneck rather than full scramble. Report honestly.")
    elif not trained_all:
        verdict = ("❌ CHECKPOINTS INCOMPLETE: FHRR training stopped too early,\n"
                   "   ||M|| not properly initialized. The 0% zero-shot may be an\n"
                   "   artifact of undertraining, NOT a fundamental collapse.\n"
                   "   → You CANNOT claim FHRR fundamentally fails. Need full retrain.")
    else:
        verdict = ("⚠️ Atomic itself is broken — model not usable. Investigate.")
    print(verdict)
    print(f"{'─'*60}")

    report["_verdict"] = verdict
    report["_summary"] = {
        "trained_all": bool(trained_all),
        "atomic_works": bool(atomic_works),
        "phase_scrambled": bool(scrambled_all),
        "avg_sim_true": float(avg_sim_true),
        "avg_sim_rand": float(avg_sim_rand),
    }

    # save
    with open(str(OUT / "verification_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n✅ Saved: {OUT}/verification_report.json")

    # ── FIGURE: phase coherence across seeds ──
    fig, ax = plt.subplots(figsize=(10,6))
    x = np.arange(len(seeds))
    sim_t = [report[s]["check4_phase_coherence"]["sim_true"] for s in seeds]
    sim_r = [report[s]["check4_phase_coherence"]["sim_rand"] for s in seeds]
    w = 0.35
    ax.bar(x-w/2, sim_t, w, label="cos(clean, TRUE)", color="#2ecc71", edgecolor="black")
    ax.bar(x+w/2, sim_r, w, label="cos(clean, RANDOM)", color="#e74c3c", edgecolor="black")
    ax.set_xticks(x); ax.set_xticklabels([f"seed {s}" for s in seeds])
    ax.set_ylabel("Phasor cosine similarity")
    ax.set_title("FHRR Phase Coherence: cleaned hop-2 vs true target\n(overlap = phase scrambled = collapse confirmed)")
    ax.legend(); ax.grid(alpha=0.3)
    plt.savefig(str(OUT / "phase_coherence_verification.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: phase_coherence_verification.png")

    volume.commit()
    return report


@app.local_entrypoint()
def main():
    print("\n🔍 Verifying FHRR collapse mechanism...\n")
    verify.remote()
    print(f"\n{'='*60}\n✅ VERIFICATION COMPLETE\n{'='*60}")
    print("Download: modal volume get hrr-results ./verify_results/")
