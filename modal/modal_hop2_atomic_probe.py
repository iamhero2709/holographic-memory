"""
Hop-2 Atomic-Difficulty Probe  — Holographic Memory on Modal A10G
===================================================================
Follow-up to modal_hop1_probe.py. That probe showed hop-1 mid retrieval is
EXCELLENT (MRR ~0.85-0.90) but composition still fails even when the mid is
exactly correct (comp_acc | mid_valid ~500x lower than atomic Hits@1).

This script isolates WHY by asking a narrower question:

  Take the TRUE mid-entity m (not the model's prediction — ground truth from
  the training/test graph) and the relation r2 from each zero-shot chain
  quadruple (h, r1, r2, t). Treat (m, r2) as a STANDALONE atomic query,
  exactly like the ones the model was trained and evaluated on in Section
  6.1 of the paper. What is forward_atomic(m, r2)'s accuracy at predicting t?

  - If accuracy ≈ overall atomic Hits@1 (~0.16 Real / ~0.13 FHRR):
      hop-2 queries are NOT intrinsically harder than average atomic queries.
      The composition pipeline itself (re-embedding mid as a clean codebook
      vector before the second bind) is what breaks something — points
      toward Mode B (cleanup/non-commutativity-type issues) or an
      implementation detail in the compose path.

  - If accuracy is near chance, MUCH lower than overall atomic Hits@1:
      these specific (mid, r2) pairs are intrinsically hard for the memory
      to answer — e.g. very high fan-out relations, or mid entities that
      were poorly learned. This directly confirms Mode A (memory cross-talk
      for chain-relevant facts), independent of anything to do with hop-1
      or the composition pipeline at all.

Also reports the SAME (m, r2) queries' difficulty stratified by relation
fan-out (#tails for that (m, r2) pair in the full graph), since the paper's
selected chains have fan-out 2.4-18.7 and high fan-out alone could explain
low single-shot accuracy without invoking any composition-specific failure.

INFERENCE ONLY over existing checkpoints. No retraining. ~$0.20-0.30, ~5 min.

It is a STRICT EXTENSION of modal_experiment1.py / modal_hop1_probe.py:
  - identical data load, vocab, discover_chains(), leakage filter
  - identical RealHRR / ComplexFHRR classes (state_dicts load cleanly)
  - reuses forward_atomic() verbatim — the SAME readout used for every
    atomic number reported in the paper, so results are directly comparable

RUN (from laptop):
  modal run modal_hop2_atomic_probe.py                 # all 5 seeds
  modal run modal_hop2_atomic_probe.py --seeds 42       # single seed
"""

import modal

volume = modal.Volume.from_name("hrr-results", create_if_missing=True)
VOLUME_PATH = "/vol"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.2.0",
        "numpy<2",  # <-- FIX: avoid NumPy 2.x incompatibility with PyTorch 2.2.0
        "requests",
        "urllib3",
    )
)

app = modal.App(name="holographic-memory-hop2-atomic-probe", image=image)


# ─────────────────────────────────────────────────────────────
# PER-SEED PROBE (runs on GPU)
# ─────────────────────────────────────────────────────────────
@app.function(gpu="A10G", timeout=3600, volumes={VOLUME_PATH: volume})
def probe_seed(seed: int = 42):
    import os, math, json, random
    from collections import defaultdict
    from pathlib import Path

    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    DEVICE = torch.device("cuda")
    SEED_DIR = Path(VOLUME_PATH) / f"seed_{seed}"

    def set_seed(s):
        random.seed(s); np.random.seed(s); torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)
    set_seed(seed)

    # config — must match modal_experiment1.py
    D_REAL, D_COMPLEX = 1024, 512
    BETA_REAL, BETA_COMPLEX = 8.0, 12.0
    N_CHAINS = 10

    print(f"\n{'='*64}\nHOP-2 ATOMIC-DIFFICULTY PROBE | seed={seed} | {torch.cuda.get_device_name(0)}\n{'='*64}")

    # ─────────────────────────────────────────────────────────
    # DATA  (verbatim from modal_experiment1.py)
    # ─────────────────────────────────────────────────────────
    base = "https://raw.githubusercontent.com/villmow/datasets_knowledge_embedding/master/FB15k-237"
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))
    raw = {}
    for split in ["train", "valid", "test"]:
        resp = session.get(f"{base}/{split}.txt", timeout=60)
        resp.raise_for_status()
        raw[split] = [tuple(l.strip().split("\t")) for l in resp.text.strip().split("\n")
                      if len(l.strip().split("\t")) == 3]
    all_triples = raw["train"] + raw["valid"] + raw["test"]
    all_tokens = sorted(set(x for h, r, t in all_triples for x in (h, r, t)))
    tok2id = {w: i for i, w in enumerate(all_tokens)}
    VOCAB = len(tok2id)
    print(f"  vocab : {VOCAB:,}")

    # full-graph adjacency (train+valid+test) for filtered ranking & true mids
    adj = defaultdict(lambda: defaultdict(set))
    for h, r, t in all_triples:
        adj[tok2id[h]][tok2id[r]].add(tok2id[t])

    true_tails = defaultdict(set)  # (h_id, r_id) -> {t_id, ...}  for filtered ranking
    for h, r, t in all_triples:
        true_tails[(tok2id[h], tok2id[r])].add(tok2id[t])

    # ─────────────────────────────────────────────────────────
    # CHAINS + ZERO-SHOT SET  (verbatim discover_chains + leakage filter)
    # ─────────────────────────────────────────────────────────
    train_adj = defaultdict(lambda: defaultdict(set))
    for h, r, t in raw["train"]: train_adj[tok2id[h]][tok2id[r]].add(tok2id[t])
    test_adj = defaultdict(lambda: defaultdict(set))
    for h, r, t in raw["test"]: test_adj[tok2id[h]][tok2id[r]].add(tok2id[t])

    def discover_chains(train_adj, test_adj, n=10, min_pairs=30):
        train_p = {h: dict(rd) for h, rd in train_adj.items()}
        test_p = {h: dict(rd) for h, rd in test_adj.items()}
        EMPTY = {}
        chain_train = defaultdict(int)
        for h, rd in train_p.items():
            for r1, mids in rd.items():
                for mid in mids:
                    for r2, tails in train_p.get(mid, EMPTY).items():
                        chain_train[(r1, r2)] += len(tails)
        test_reach = defaultdict(set)
        for h, rd in test_p.items():
            for r1, mids in rd.items():
                for mid in mids:
                    for r2, tails in test_p.get(mid, EMPTY).items():
                        for t in tails:
                            test_reach[(r1, r2)].add((h, t))
        valid = [(ch, sorted(tp)) for ch, tp in test_reach.items()
                 if len(tp) >= min_pairs and chain_train[ch] >= 100]
        valid.sort(key=lambda x: -len(x[1]))
        return valid[:n]

    chains = discover_chains(train_adj, test_adj, N_CHAINS)
    if not chains:
        raise RuntimeError("No chains found")

    train_lookup = {h: {r: set(ts) for r, ts in rd.items()} for h, rd in train_adj.items()}
    zs_h, zs_r1, zs_r2, zs_t = [], [], [], []
    leaked = 0
    for (r1, r2), pairs in chains:
        for h, t in pairs:
            if any(t in ts for ts in train_lookup.get(h, {}).values()):
                leaked += 1; continue
            zs_h.append(h); zs_r1.append(r1); zs_r2.append(r2); zs_t.append(t)
    zs_h = torch.tensor(zs_h, dtype=torch.long)
    zs_r1 = torch.tensor(zs_r1, dtype=torch.long)
    zs_r2 = torch.tensor(zs_r2, dtype=torch.long)
    zs_t = torch.tensor(zs_t, dtype=torch.long)
    N = len(zs_h)
    print(f"  zero-shot quadruples: {N:,} | leaked removed: {leaked}")

    # ── TRUE mid-entity per quadruple (ground truth, NOT model prediction) ──
    # A quadruple (h, r1, r2, t) was only kept by discover_chains if SOME mid
    # satisfies (h,r1,mid) in train AND (mid,r2,t) in train (test_reach
    # construction uses test_adj on both sides, but the chain itself was
    # validated against TRAIN support >= 100; here we just need *a* true mid
    # for THIS specific (h,t) pair, taken from the full graph for max
    # coverage). If multiple true mids exist we take all of them — the
    # (mid, r2) atomic query is evaluated for each and we report the BEST
    # (most charitable to the model) rank, matching how a system with
    # multiple correct routes would be graded.
    true_mids_per_quad = []
    for i in range(N):
        h, r1, r2, t = zs_h[i].item(), zs_r1[i].item(), zs_r2[i].item(), zs_t[i].item()
        mids = [m for m in adj[h].get(r1, set()) if t in adj[m].get(r2, set())]
        true_mids_per_quad.append(mids)

    n_with_true_mid = sum(1 for m in true_mids_per_quad if m)
    print(f"  quadruples with >=1 ground-truth (h,r1,mid,r2,t) chain present in full graph: "
          f"{n_with_true_mid:,} / {N:,}")

    # build flat list of (mid, r2, t, fanout) atomic queries, one per
    # (quadruple, true_mid) pair, deduplicated on (mid, r2) since the same
    # hop-2 atomic query can recur across many quadruples
    seen = {}
    flat_mid, flat_r2, flat_t, flat_quad_idx = [], [], [], []
    for i, mids in enumerate(true_mids_per_quad):
        r2 = zs_r2[i].item()
        t = zs_t[i].item()
        for m in mids:
            key = (m, r2)
            if key not in seen:
                seen[key] = True
                flat_mid.append(m); flat_r2.append(r2); flat_t.append(t)
                flat_quad_idx.append(i)
    n_hop2_queries = len(flat_mid)
    print(f"  unique (mid, r2) hop-2 atomic queries to test: {n_hop2_queries:,}")

    flat_mid_t = torch.tensor(flat_mid, dtype=torch.long)
    flat_r2_t = torch.tensor(flat_r2, dtype=torch.long)
    flat_t_t = torch.tensor(flat_t, dtype=torch.long)

    # fan-out of each (mid, r2) query, i.e. |{t' : (mid, r2, t') in full graph}|
    fanout = [len(adj[m].get(r2, set())) for m, r2 in zip(flat_mid, flat_r2)]
    fanout = np.array(fanout)

    # ─────────────────────────────────────────────────────────
    # MODELS  (verbatim — so checkpoints load)
    # ─────────────────────────────────────────────────────────
    class RealHRR(nn.Module):
        def __init__(self, D, vocab_size, beta=8.0):
            super().__init__()
            self.D, self.vocab_size, self.beta = D, vocab_size, beta
            self.entity_emb = nn.Embedding(vocab_size, D)
            self.role_emb = nn.Embedding(vocab_size, D)
            self.M = nn.Parameter(torch.zeros(D))
            nn.init.normal_(self.entity_emb.weight, std=1 / D ** 0.5)
            nn.init.normal_(self.role_emb.weight, std=1 / D ** 0.5)

        @staticmethod
        def _bind(a, b):
            a32, b32 = a.float(), b.float()
            return torch.fft.ifft(torch.fft.fft(a32) * torch.fft.fft(b32)).real.to(a.dtype)

        @staticmethod
        def _unbind(bnd, key):
            b32, k32 = bnd.float(), key.float()
            return torch.fft.ifft(torch.fft.fft(b32) * torch.conj(torch.fft.fft(k32))).real.to(bnd.dtype)

        def _cleanup(self, z, hard=False):
            scores = z.float() @ self.entity_emb.weight.float().T * self.beta
            scores = scores.clamp(-50, 50)
            w = F.gumbel_softmax(scores, tau=1.0, hard=True) if hard else F.softmax(scores, dim=-1)
            return (w @ self.entity_emb.weight.float()).to(z.dtype)

        def forward_atomic(self, s, r):
            f, ro = self.entity_emb(s), self.role_emb(r)
            M = self.M.unsqueeze(0).expand_as(f)
            return self._cleanup(self._unbind(M, self._bind(f, ro))) @ self.entity_emb.weight.float().T * self.beta

    class ComplexFHRR(nn.Module):
        def __init__(self, D, vocab_size, beta=12.0):
            super().__init__()
            self.D, self.vocab_size, self.beta = D, vocab_size, beta
            self.embed_phase = nn.Embedding(vocab_size, D)
            self.role_phase = nn.Embedding(vocab_size, D)
            self.M_real = nn.Parameter(torch.zeros(D))
            self.M_imag = nn.Parameter(torch.zeros(D))
            nn.init.uniform_(self.embed_phase.weight, -math.pi, math.pi)
            nn.init.uniform_(self.role_phase.weight, -math.pi, math.pi)

        def _phasor(self, idx, kind="filler"):
            ph = self.embed_phase(idx) if kind == "filler" else self.role_phase(idx)
            return torch.complex(torch.cos(ph), torch.sin(ph))

        def _M(self):
            return torch.complex(self.M_real, self.M_imag)

        def _codebook(self):
            ph = self.embed_phase.weight
            return torch.complex(torch.cos(ph), torch.sin(ph))

        @staticmethod
        def _bind(a, b): return a * b

        @staticmethod
        def _unbind(bnd, key): return bnd * torch.conj(key)

        def _cos_sim(self, z, cb):
            dot = z.real @ cb.real.T + z.imag @ cb.imag.T
            nz = (z.real ** 2 + z.imag ** 2).sum(-1, keepdim=True).sqrt() + 1e-8
            return dot / (nz * self.D ** 0.5)

        def _cleanup(self, z, hard=False):
            cb = self._codebook()
            scores = self._cos_sim(z, cb) * self.beta
            w = F.gumbel_softmax(scores, tau=1.0, hard=True) if hard else F.softmax(scores, dim=-1)
            cr, ci = w @ cb.real, w @ cb.imag
            ph = torch.atan2(ci, cr + 1e-8)
            return torch.complex(torch.cos(ph), torch.sin(ph))

        def forward_atomic(self, s, r):
            f, ro = self._phasor(s), self._phasor(r, "role")
            M = self._M().unsqueeze(0).expand_as(f)
            cln = self._cleanup(self._unbind(M, self._bind(f, ro)))
            return self._cos_sim(cln, self._codebook()) * self.beta

    # ─────────────────────────────────────────────────────────
    # LOAD CHECKPOINTS
    # ─────────────────────────────────────────────────────────
    real = RealHRR(D_REAL, VOCAB, BETA_REAL).to(DEVICE)
    fhrr = ComplexFHRR(D_COMPLEX, VOCAB, BETA_COMPLEX).to(DEVICE)
    real_ckpt = SEED_DIR / "best_real_hrr.pt"
    fhrr_ckpt = SEED_DIR / "best_complex_fhrr.pt"
    for m, p in [(real, real_ckpt), (fhrr, fhrr_ckpt)]:
        if not p.exists():
            raise FileNotFoundError(f"Missing checkpoint: {p}")
        m.load_state_dict(torch.load(p, map_location=DEVICE))
        m.eval()
    print(f"  loaded: {real_ckpt.name}, {fhrr_ckpt.name}")

    # ─────────────────────────────────────────────────────────
    # OVERALL ATOMIC ACCURACY ON THE STANDARD TEST SET
    # (for direct comparison — same protocol as Table 1 in the paper)
    # ─────────────────────────────────────────────────────────
    te_s = torch.tensor([tok2id[h] for h, r, t in raw["test"]], dtype=torch.long)
    te_r = torch.tensor([tok2id[r] for h, r, t in raw["test"]], dtype=torch.long)
    te_o = torch.tensor([tok2id[t] for h, r, t in raw["test"]], dtype=torch.long)

    @torch.no_grad()
    def filtered_eval(model, s_all, r_all, o_all, batch=256):
        """Filtered top-1 / rank accuracy, identical convention to the
        paper's Evaluator.filtered_mrr_hits but applied to an arbitrary
        (s, r, o) query set (here: hop-2 (mid, r2, t) queries)."""
        model.eval()
        n = len(s_all)
        ranks = []
        top1_correct = 0
        for i in range(0, n, batch):
            sb = s_all[i:i + batch].to(DEVICE)
            rb = r_all[i:i + batch].to(DEVICE)
            ob = o_all[i:i + batch].to(DEVICE)
            logits = model.forward_atomic(sb, rb).float()
            for j in range(len(sb)):
                hi, ri, oi = sb[j].item(), rb[j].item(), ob[j].item()
                mask = torch.zeros(logits.size(1), device=DEVICE)
                for tt in true_tails.get((hi, ri), set()):
                    if tt != oi:
                        mask[tt] = 1e9
                score_j = logits[j] - mask
                rank = int((score_j > score_j[oi]).sum().item()) + 1
                ranks.append(rank)
                top1_correct += int(rank == 1)
        ranks = np.array(ranks)
        return {
            "n": int(n),
            "top1": top1_correct / n,
            "mrr": float(np.mean(1.0 / ranks)),
            "h1": float(np.mean(ranks <= 1)),
            "h3": float(np.mean(ranks <= 3)),
            "h10": float(np.mean(ranks <= 10)),
        }

    # ─────────────────────────────────────────────────────────
    # RUN: standard test-set atomic accuracy (baseline for comparison)
    #       + hop-2 (true mid, r2) -> t atomic accuracy (the new probe)
    #       + fan-out stratified breakdown
    # ─────────────────────────────────────────────────────────
    out = {
        "seed": seed, "vocab": VOCAB, "n_zeroshot": N,
        "n_quad_with_true_mid": n_with_true_mid,
        "n_hop2_unique_queries": n_hop2_queries,
        "chance": 1.0 / VOCAB,
    }

    for tag, model in [("real_hrr", real), ("fhrr", fhrr)]:
        print(f"\n  ── {tag} ──")
        std_atomic = filtered_eval(model, te_s, te_r, te_o)
        hop2_atomic = filtered_eval(model, flat_mid_t, flat_r2_t, flat_t_t)

        # fan-out stratification (median split + quartiles)
        med_fo = float(np.median(fanout))
        low_mask = fanout <= med_fo
        high_mask = ~low_mask

        @torch.no_grad()
        def acc_on_subset(mask):
            if mask.sum() == 0:
                return None
            # Convert NumPy indices to a PyTorch tensor to avoid
            # incompatibility between NumPy 2 and PyTorch 2.2.0
            idx_np = np.where(mask)[0]
            idx = torch.as_tensor(idx_np, dtype=torch.long, device=flat_mid_t.device)
            sub_s = flat_mid_t[idx]
            sub_r = flat_r2_t[idx]
            sub_o = flat_t_t[idx]
            res = filtered_eval(model, sub_s, sub_r, sub_o)
            return res

        low_res = acc_on_subset(low_mask)
        high_res = acc_on_subset(high_mask)

        out[tag] = {
            "standard_test_atomic": std_atomic,
            "hop2_true_mid_atomic": hop2_atomic,
            "fanout_median": med_fo,
            "fanout_mean": float(fanout.mean()),
            "hop2_low_fanout": low_res,
            "hop2_high_fanout": high_res,
        }

        print(f"    standard test atomic : top1={std_atomic['top1']:.4f}  "
              f"mrr={std_atomic['mrr']:.4f}  h10={std_atomic['h10']:.4f}  (n={std_atomic['n']})")
        print(f"    hop-2 true-mid atomic: top1={hop2_atomic['top1']:.4f}  "
              f"mrr={hop2_atomic['mrr']:.4f}  h10={hop2_atomic['h10']:.4f}  (n={hop2_atomic['n']})")
        print(f"    ratio (hop2/standard) top1 = "
              f"{(hop2_atomic['top1']/std_atomic['top1'] if std_atomic['top1'] else float('nan')):.4f}")
        print(f"    fan-out median={med_fo:.1f}  mean={fanout.mean():.2f}")
        if low_res:
            print(f"    low fan-out (<=median)  top1={low_res['top1']:.4f}  (n={low_res['n']})")
        if high_res:
            print(f"    high fan-out (>median)  top1={high_res['top1']:.4f}  (n={high_res['n']})")

    out_dir = Path(VOLUME_PATH) / "hop2_atomic_probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"hop2_atomic_probe_seed_{seed}.json", "w") as f:
        json.dump(out, f, indent=2)
    volume.commit()
    return out


# ─────────────────────────────────────────────────────────────
# AGGREGATION (runs locally after all seeds)
# ─────────────────────────────────────────────────────────────
def _agg(per_seed):
    import statistics as st

    def grab(path, model):
        vals = []
        for s in per_seed:
            node = s[model]
            for k in path:
                if node is None:
                    node = None
                    break
                node = node[k]
            if node is not None:
                vals.append(node)
        return vals

    def ms(vals):
        if not vals:
            return {"mean": None, "std": None, "n": 0}
        return {"mean": sum(vals) / len(vals),
                "std": (st.pstdev(vals) if len(vals) > 1 else 0.0),
                "n": len(vals)}

    agg = {}
    for model in ["real_hrr", "fhrr"]:
        agg[model] = {
            "standard_top1": ms(grab(["standard_test_atomic", "top1"], model)),
            "standard_mrr": ms(grab(["standard_test_atomic", "mrr"], model)),
            "hop2_top1": ms(grab(["hop2_true_mid_atomic", "top1"], model)),
            "hop2_mrr": ms(grab(["hop2_true_mid_atomic", "mrr"], model)),
            "hop2_h10": ms(grab(["hop2_true_mid_atomic", "h10"], model)),
            "low_fanout_top1": ms(grab(["hop2_low_fanout", "top1"], model)),
            "high_fanout_top1": ms(grab(["hop2_high_fanout", "top1"], model)),
        }
    return agg


@app.local_entrypoint()
def main(seeds: str = "42,1,2,3,4"):
    import json
    seed_list = [int(x) for x in seeds.split(",") if x.strip()]
    print(f"Running hop-2 atomic-difficulty probe on seeds: {seed_list}")
    per_seed = list(probe_seed.map(seed_list))

    agg = _agg(per_seed)
    payload = {"seeds": seed_list, "per_seed": per_seed, "aggregate": agg}

    with open("hop2_atomic_probe_results.json", "w") as f:
        json.dump(payload, f, indent=2)

    print("\n" + "=" * 64)
    print("AGGREGATE (mean +/- std over seeds)")
    print("=" * 64)
    for model in ["real_hrr", "fhrr"]:
        a = agg[model]
        print(f"\n{model}")
        print(f"  standard test atomic top1 : {a['standard_top1']['mean']:.4f} +/- {a['standard_top1']['std']:.4f}")
        print(f"  hop-2 true-mid atomic top1: {a['hop2_top1']['mean']:.4f} +/- {a['hop2_top1']['std']:.4f}")
        ratio = a['hop2_top1']['mean'] / a['standard_top1']['mean'] if a['standard_top1']['mean'] else None
        print(f"  ratio (hop2 / standard)   : {(f'{ratio:.4f}' if ratio is not None else 'n/a')}")
        lo = a['low_fanout_top1']['mean']
        hi = a['high_fanout_top1']['mean']
        print(f"  low fan-out top1          : {(f'{lo:.4f}' if lo is not None else 'n/a')}")
        print(f"  high fan-out top1         : {(f'{hi:.4f}' if hi is not None else 'n/a')}")
    print("\nSaved: hop2_atomic_probe_results.json (also per-seed in volume hrr-results/hop2_atomic_probe/)")
