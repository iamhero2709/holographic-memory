<div align="center">

# Holographic Memory for Zero-Shot Compositional Reasoning in Knowledge Graphs

**A Mechanistic Study of Where and Why It Fails**

[![Paper](https://img.shields.io/badge/paper-PDF-red)](paper/holographic_memory_paper_FINAL.tex)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](requirements.txt)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Results](https://img.shields.io/badge/results-5%20seeds-orange)](results/final_results.json)
[![GPU](https://img.shields.io/badge/GPU-A10G-important)](https://modal.com)

</div>

---

## Abstract

Knowledge graph embedding (KGE) models achieve strong single-hop link prediction but cannot answer *zero-shot compositional* queries. Holographic Reduced Representations (HRR) offer a theoretically appealing candidate through convolution-based binding.

We study **two holographic memory variants** -- real-valued HRR and phase-only Fourier HRR (FHRR) -- on **FB15k-237 over 5 random seeds**:

1. Both variants are **competitive atomic retrievers** (Real HRR MRR 0.358, FHRR MRR 0.350)
2. **Both fail at zero-shot composition** (accuracy at chance, binomial test p > 0.2)
3. **FHRR failure mechanism is phase decorrelation**, not modulus collapse

---

## Results

### Single-Hop (Atomic) Link Prediction

Full test set (20,466 queries), filtered metrics, mean +/- std over 5 seeds.

| Model | Top-1 | MRR | Hits@1 | Hits@3 | Hits@10 |
|-------|:-----:|:---:|:------:|:------:|:-------:|
| Real HRR (D=1024) | **0.158** +/- 0.001 | **0.358** +/- 0.002 | **0.267** +/- 0.002 | **0.392** +/- 0.003 | **0.540** +/- 0.003 |
| Complex FHRR (D=512) | 0.126 +/- 0.001 | 0.350 +/- 0.021 | 0.262 +/- 0.017 | 0.390 +/- 0.024 | 0.524 +/- 0.028 |
| RotatE (literature) | -- | 0.338 | 0.241 | 0.375 | 0.533 |

### Two-Hop Zero-Shot Composition

69,855 query pairs, leakage-controlled protocol.

| Model | Accuracy | p-value (vs chance) | Result |
|-------|:--------:|:-------------------:|:------:|
| Real HRR | 0.00017 +/- 0.00009 | 0.22 | at chance |
| Complex FHRR | 0.00003 +/- 0.00003 | 0.85 | at chance |
| Random baseline | 0.00007 | -- | -- |

### Ablation: Effect of Hopfield Cleanup (Real HRR)

Removing the Hopfield cleanup drops atomic Top-1 by ~48%.

| Configuration | Atomic Top-1 | Zero-Shot |
|:---|---:|:---:|
| Full model | 0.158 +/- 0.001 | 0.00017 +/- 0.00009 |
| Without Hopfield cleanup | 0.081 +/- 0.010 | 0.00030 +/- 0.00010 |

---

## Failure Mechanism

The FHRR failure is **phase decorrelation**, not modulus collapse:

| Probe | What It Tests | Result |
|-------|:---|---|
| **A -- Modulus tracking** | Does the signal magnitude decay across hops? | No -- |z| = 1.0000 at all stages (by construction) |
| **B -- Renormalization** | Does restoring unit modulus fix it? | No -- accuracy remains 0.0000 |
| **C -- Hard cleanup** | Does argmax (vs softmax) fix it? | No -- accuracy remains 0.0000 |
| **Phase coherence** | Is the phase correlated with the target? | No -- cosine similarity = -0.009 (same as random) |
| **Phase error** | Per-component error vs ground truth | pi/2 (indistinguishable from uniform) |

**Why atomic ranking survives:** Single-hop readout aggregates similarity across all dimensions, robust to per-component noise. Composition feeds the intermediate into a *bind* -- a per-component phase operation -- so it requires correct phase per dimension. The cleanup supplies aggregate ranking but not per-component phase coherence.

---

## Figures

<table>
<tr>
  <td width="50%"><img src="figures/phase_error_propagation.png" alt="Phase Error Propagation"/></td>
  <td width="50%"><img src="figures/fhrr_probe_a_modulus.png" alt="FHRR Modulus Probe"/></td>
</tr>
<tr>
  <td align="center"><b>Phase Error Propagation</b><br/>Mean abs phase error at hop 1 and hop 2<br/>sits at uniform limit (pi/2)</td>
  <td align="center"><b>Modulus Probe</b><br/>FHRR cleanup forces |z|=1 at every stage<br/>magnitude cannot explain the failure</td>
</tr>
<tr>
  <td width="50%"><img src="figures/fhrr_probes_bc.png" alt="FHRR Probes B and C"/></td>
  <td width="50%"><img src="figures/table5_beta_sweep.png" alt="Beta Sweep"/></td>
</tr>
<tr>
  <td align="center"><b>Renormalization + Hard Cleanup</b><br/>Neither intervention recovers accuracy</td>
  <td align="center"><b>Beta Temperature Sweep</b><br/>Accuracy flat at chance for all beta</td>
</tr>
</table>

<p align="center">
  <img src="figures/table2_ablation_core.png" alt="Core Ablation" width="70%"/>
  <br/>
  <b>Core Ablation:</b> Removing Hopfield cleanup halves atomic accuracy
</p>

---

## Reproducibility

```bash
# Setup
pip install torch numpy scikit-learn matplotlib seaborn tqdm requests wandb

# Run on Modal (A10G GPU)
modal run modal/modal_experiment1.py --seed 42      # single seed
modal run modal/modal_experiment1.py --all-seeds    # 5 seeds parallel
```

All results: `results/final_results.json`, `results/phase_error_results.json`, `results/results_inference_ablations.json`

---

## Project Structure

```
modal/                          Modal A10G experiment scripts
  modal_experiment1.py          Main experiment (5 seeds parallel)
  final_analysis.py             Aggregated analysis
  phase_error_analysis.py       Phase decorrelation measurement
  modal_inference_ablations.py  Ablation + probes + beta sweep
paper/
  holographic_memory_paper_FINAL.tex   Self-contained LaTeX (TikZ figures)
results/                        5-seed experiment outputs (JSON)
figures/                        Publication-quality figures (PNG)
README.md
requirements.txt
```

---

## Citation

```bibtex
@misc{holographic-memory-2026,
  title={Holographic Memory for Zero-Shot Compositional Reasoning
         in Knowledge Graphs: A Mechanistic Study of Where and Why It Fails},
  author={Kumar, Randhir},
  year={2026},
  note={Preprint}
}
```

## Acknowledgments

Thanks to [Modal](https://modal.com) for $30 in free GPU compute credits (A10G), which enabled all experiments.

---

<div align="center"><i>MIT License</i></div>
