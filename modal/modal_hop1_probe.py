"""
Hop-1 Mid-Entity Retrieval Probe  — Holographic Memory on Modal A10G
====================================================================
Answers the reviewer question: "Is two-hop failure already failing at HOP 1
(the model never recovers the right mid-entity), or does it fail at HOP 2
(mid is right but the re-bind destroys it)?"

This is INFERENCE ONLY over the existing 5-seed checkpoints in the
Modal Volume `hrr-results`. No retraining. ~$0.30-0.50, ~10 min.

It is a STRICT EXTENSION of modal_experiment1.py:
  - identical data load, vocab, discover_chains(), and leakage filter
    => identical 69,855 zero-shot quadruples
  - identical RealHRR / ComplexFHRR classes (so state_dicts load cleanly)
  - hop-1 mid ranking reuses the SAME forward_atomic readout the paper uses

Outputs per seed and aggregated (mean +/- std over seeds), saved to
  hrr-results/hop1_probe/hop1_probe_results.json
and also printed.

RUN (from laptop):
  modal run modal_hop1_probe.py                 # all 5 seeds
  modal run modal_hop1_probe.py --seeds 42      # single seed
"""

import modal

volume = modal.Volume.from_name("hrr-results", create_if_missing=True)
VOLUME_PATH = "/vol"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.2.0",
        "numpy",
        "requests",
        "urllib3",
    )
)

app = modal.App(name="holographic-memory-hop1-probe", image=image)


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

    print(f"\n{'='*64}\nHOP-1 PROBE | seed={seed} | {torch.cuda.get_device_name(0)}\n{'='*64}")

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

    # full-graph adjacency (train+valid+test) for filtered ranking & chain-consistent mids
    adj = defaultdict(lambda: defaultdict(set))
    for h, r, t in all_triples:
        adj[tok2id[h]][tok2id[r]].add(tok2id[t])

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
    print(f"  zero-shot pairs: {N:,} | leaked removed: {leaked}")

    # gold mid sets
    #   (a) atomic-style: any r1-neighbour of h  -> "does the model know a valid mid"
    #   (b) chain-consistent: r1-neighbour of h that also reaches gold t via r2
    gold_mids_atomic = []   # list[set] aligned with zs
    gold_mids_chain = []    # list[set]
    for i in range(N):
        h, r1, r2, t = zs_h[i].item(), zs_r1[i].item(), zs_r2[i].item(), zs_t[i].item()
        r1_neigh = adj[h].get(r1, set())
        gold_mids_atomic.append(r1_neigh)
        chain_ok = {m for m in r1_neigh if t in adj[m].get(r2, set())}
        gold_mids_chain.append(chain_ok)

    # ─────────────────────────────────────────────────────────
    # MODELS  (verbatim from modal_experiment1.py — so checkpoints load)
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

        # ---- deterministic intermediate (no gumbel) for the conditional analysis ----
        @torch.no_grad()
        def hop1_raw_unbind(self, h, r1):
            """e1 = M (-) (h (*) r1), the noisy mid estimate BEFORE cleanup."""
            f, ro1 = self.entity_emb(h), self.role_emb(r1)
            M = self.M.unsqueeze(0).expand_as(f)
            return self._unbind(M, self._bind(f, ro1))

        @torch.no_grad()
        def propagated_mid_idx(self, h, r1):
            """Entity a deterministic hard cleanup would select = argmax(e1 . codebook)."""
            e1 = self.hop1_raw_unbind(h, r1)
            return (e1.float() @ self.entity_emb.weight.float().T).argmax(1)

        @torch.no_grad()
        def compose_deterministic(self, h, r1, r2):
            """forward_composition but with argmax mid (no gumbel)."""
            mid_idx = self.propagated_mid_idx(h, r1)
            mid_emb = self.entity_emb(mid_idx)
            ro2 = self.role_emb(r2)
            M = self.M.unsqueeze(0).expand_as(mid_emb)
            final = self._cleanup(self._unbind(M, self._bind(mid_emb, ro2)))
            return final @ self.entity_emb.weight.float().T * self.beta, mid_idx

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

        @torch.no_grad()
        def hop1_raw_unbind(self, h, r1):
            f, ro1 = self._phasor(h), self._phasor(r1, "role")
            M = self._M().unsqueeze(0).expand_as(f)
            return self._unbind(M, self._bind(f, ro1))

        @torch.no_grad()
        def propagated_mid_idx(self, h, r1):
            e1 = self.hop1_raw_unbind(h, r1)
            return self._cos_sim(e1, self._codebook()).argmax(1)

        @torch.no_grad()
        def compose_deterministic(self, h, r1, r2):
            mid_idx = self.propagated_mid_idx(h, r1)
            mid = self._phasor(mid_idx)                 # re-embed selected entity as clean phasor
            ro2 = self._phasor(r2, "role")
            M = self._M().unsqueeze(0).expand_as(mid)
            cln = self._cleanup(self._unbind(M, self._bind(mid, ro2)))
            return self._cos_sim(cln, self._codebook()) * self.beta, mid_idx

    # ─────────────────────────────────────────────────────────
    # LOAD CHECKPOINTS
    # ─────────────────────────────────────────────────────────
    real = RealHRR(D_REAL, VOCAB, BETA_REAL).to(DEVICE)
    fhrr = ComplexFHRR(D_COMPLEX, VOCAB, BETA_COMPLEX).to(DEVICE)
    real_ckpt = SEED_DIR / "best_real_hrr.pt"
    fhrr_ckpt = SEED_DIR / "best_complex_fhrr.pt"
    for m, p in [(real, real_ckpt), (fhrr, fhrr_ckpt)]:
        if not p.exists():
            raise FileNotFoundError(f"Missing checkpoint: {p}  (run modal_experiment1.py first)")
        m.load_state_dict(torch.load(p, map_location=DEVICE))
        m.eval()
    print(f"  loaded: {real_ckpt.name}, {fhrr_ckpt.name}")

    # ─────────────────────────────────────────────────────────
    # METRIC HELPERS
    # ─────────────────────────────────────────────────────────
    @torch.no_grad()
    def hop1_mid_retrieval(model, batch=256):
        """Filtered MRR / Hits over UNIQUE (h, r1) pairs in the zs set.
        Gold = atomic r1-neighbours of h (full graph). Mirrors the paper's
        filtered atomic protocol, applied to the first hop."""
        # unique (h, r1) pairs
        seen = {}
        order = []
        for i in range(N):
            key = (zs_h[i].item(), zs_r1[i].item())
            if key not in seen:
                seen[key] = i
                order.append(key)
        ranks = []
        keys = order
        for b in range(0, len(keys), batch):
            chunk = keys[b:b + batch]
            hb = torch.tensor([k[0] for k in chunk], device=DEVICE)
            rb = torch.tensor([k[1] for k in chunk], device=DEVICE)
            logits = model.forward_atomic(hb, rb).float()
            for j, (h, r1) in enumerate(chunk):
                gold = adj[h].get(r1, set())
                if not gold:
                    continue
                row = logits[j].clone()
                # filtered: remove all gold mids except the best-ranked one
                gold_list = list(gold)
                gold_scores = row[gold_list]
                best_gold = gold_list[int(torch.argmax(gold_scores).item())]
                mask = torch.zeros_like(row)
                for g in gold:
                    if g != best_gold:
                        mask[g] = 1e9
                row = row - mask
                rank = int((row > row[best_gold]).sum().item()) + 1
                ranks.append(rank)
        ranks = np.array(ranks) if ranks else np.array([VOCAB])
        return {
            "n_pairs": int(len(ranks)),
            "mrr": float(np.mean(1.0 / ranks)),
            "h1": float(np.mean(ranks <= 1)),
            "h3": float(np.mean(ranks <= 3)),
            "h10": float(np.mean(ranks <= 10)),
            "median_rank": float(np.median(ranks)),
        }

    @torch.no_grad()
    def composition_conditional(model, batch=256):
        """Deterministic composition + the killer conditional:
        P(comp correct | propagated mid valid) vs P(comp correct | mid invalid)."""
        comp_correct = 0
        mid_valid = 0
        mid_valid_chain = 0
        corr_given_valid = 0
        corr_given_invalid = 0
        n_valid = 0
        n_invalid = 0
        for b in range(0, N, batch):
            hb = zs_h[b:b + batch].to(DEVICE)
            r1b = zs_r1[b:b + batch].to(DEVICE)
            r2b = zs_r2[b:b + batch].to(DEVICE)
            tb = zs_t[b:b + batch].to(DEVICE)
            logits, mid_idx = model.compose_deterministic(hb, r1b, r2b)
            pred = logits.argmax(1)
            for j in range(hb.size(0)):
                gi = b + j
                m = mid_idx[j].item()
                is_valid = m in gold_mids_atomic[gi]
                is_chain = m in gold_mids_chain[gi]
                is_comp = (pred[j].item() == tb[j].item())
                comp_correct += int(is_comp)
                mid_valid += int(is_valid)
                mid_valid_chain += int(is_chain)
                if is_valid:
                    n_valid += 1; corr_given_valid += int(is_comp)
                else:
                    n_invalid += 1; corr_given_invalid += int(is_comp)
        return {
            "comp_acc_deterministic": comp_correct / N,
            "frac_mid_valid_atomic": mid_valid / N,
            "frac_mid_valid_chain": mid_valid_chain / N,
            "comp_acc_given_mid_valid": (corr_given_valid / n_valid) if n_valid else None,
            "comp_acc_given_mid_invalid": (corr_given_invalid / n_invalid) if n_invalid else None,
            "n_mid_valid": n_valid,
            "n_mid_invalid": n_invalid,
        }

    # ─────────────────────────────────────────────────────────
    # RUN
    # ─────────────────────────────────────────────────────────
    out = {"seed": seed, "vocab": VOCAB, "n_zeroshot": N, "chance": 1.0 / VOCAB}
    for tag, model in [("real_hrr", real), ("fhrr", fhrr)]:
        print(f"\n  ── {tag} ──")
        mid = hop1_mid_retrieval(model)
        cond = composition_conditional(model)
        out[tag] = {"hop1_mid": mid, "composition": cond}
        print(f"    hop1 mid retrieval : MRR={mid['mrr']:.4f}  H@1={mid['h1']:.4f}  "
              f"H@10={mid['h10']:.4f}  median_rank={mid['median_rank']:.0f}")
        print(f"    propagated mid valid (atomic) : {cond['frac_mid_valid_atomic']:.4f}")
        print(f"    propagated mid valid (chain)  : {cond['frac_mid_valid_chain']:.4f}")
        print(f"    comp acc (deterministic)      : {cond['comp_acc_deterministic']:.2e}  "
              f"(chance {1.0/VOCAB:.2e})")
        cgv = cond['comp_acc_given_mid_valid']
        cgi = cond['comp_acc_given_mid_invalid']
        print(f"    comp acc | mid VALID          : "
              f"{(f'{cgv:.4f}' if cgv is not None else 'n/a')}  (n={cond['n_mid_valid']})")
        print(f"    comp acc | mid INVALID        : "
              f"{(f'{cgi:.2e}' if cgi is not None else 'n/a')}  (n={cond['n_mid_invalid']})")

    out_dir = Path(VOLUME_PATH) / "hop1_probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"hop1_probe_seed_{seed}.json", "w") as f:
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
            "hop1_mid_mrr": ms(grab(["hop1_mid", "mrr"], model)),
            "hop1_mid_h1": ms(grab(["hop1_mid", "h1"], model)),
            "hop1_mid_h10": ms(grab(["hop1_mid", "h10"], model)),
            "hop1_mid_median_rank": ms(grab(["hop1_mid", "median_rank"], model)),
            "frac_mid_valid_atomic": ms(grab(["composition", "frac_mid_valid_atomic"], model)),
            "frac_mid_valid_chain": ms(grab(["composition", "frac_mid_valid_chain"], model)),
            "comp_acc_deterministic": ms(grab(["composition", "comp_acc_deterministic"], model)),
            "comp_acc_given_mid_valid": ms(grab(["composition", "comp_acc_given_mid_valid"], model)),
            "comp_acc_given_mid_invalid": ms(grab(["composition", "comp_acc_given_mid_invalid"], model)),
        }
    return agg


@app.local_entrypoint()
def main(seeds: str = "42,1,2,3,4"):
    import json
    seed_list = [int(x) for x in seeds.split(",") if x.strip()]
    print(f"Running hop-1 probe on seeds: {seed_list}")
    per_seed = list(probe_seed.map(seed_list))

    agg = _agg(per_seed)
    payload = {"seeds": seed_list, "per_seed": per_seed, "aggregate": agg}

    # write aggregate to volume too
    with open("hop1_probe_results.json", "w") as f:
        json.dump(payload, f, indent=2)

    print("\n" + "=" * 64)
    print("AGGREGATE (mean +/- std over seeds)")
    print("=" * 64)
    for model in ["real_hrr", "fhrr"]:
        a = agg[model]
        print(f"\n{model}")
        print(f"  hop-1 mid MRR            : {a['hop1_mid_mrr']['mean']:.4f} +/- {a['hop1_mid_mrr']['std']:.4f}")
        print(f"  hop-1 mid Hits@1         : {a['hop1_mid_h1']['mean']:.4f} +/- {a['hop1_mid_h1']['std']:.4f}")
        print(f"  hop-1 mid Hits@10        : {a['hop1_mid_h10']['mean']:.4f} +/- {a['hop1_mid_h10']['std']:.4f}")
        print(f"  propagated mid valid     : {a['frac_mid_valid_atomic']['mean']:.4f} +/- {a['frac_mid_valid_atomic']['std']:.4f}")
        cgv = a['comp_acc_given_mid_valid']['mean']
        cgi = a['comp_acc_given_mid_invalid']['mean']
        print(f"  comp acc | mid VALID     : {(f'{cgv:.4f}' if cgv is not None else 'n/a')}")
        print(f"  comp acc | mid INVALID   : {(f'{cgi:.2e}' if cgi is not None else 'n/a')}")
    print("\nSaved: hop1_probe_results.json  (also per-seed in volume hrr-results/hop1_probe/)")
