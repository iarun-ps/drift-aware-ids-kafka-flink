# Experiment 4: Drift Adaptation Safeguard Ablation

## Objective
Evaluate how two safeguards affect AE + ADWIN adaptation:
1. Conservative reconstruction-score filtering.
2. Maximum 5% alert rate for accepting training minutes.

## Method
Three configurations were evaluated using chronological CIC-IDS2017 traffic:
- Full: Both safeguards enabled.
- No alert gate: Score filtering only.
- No score filter: Alert-rate safeguard only.

All configurations detected three drift events and approved three model updates.

## Detection Performance

| Configuration | PortScan F1 | DDoS F1 |
|---|---:|---:|
| Full | 0.9591 | 0.5136 |
| No alert gate | 0.9572 | 0.5133 |
| No score filter | 0.9327 | 0.4978 |

## Training Contamination

| Configuration | Training records | Attack records | Contamination |
|---|---:|---:|---:|
| Full | 847 | 0 | 0.0% |
| No alert gate | 876 | 0 | 0.0% |
| No score filter | 5904 | 3389 | 57.4% |

Without score filtering, Wednesday's adaptation buffer contained
3389 attack records out of 4096 (82.74%).

## Findings
- Score filtering prevented observed attack contamination in the tested updates.
- Removing score filtering reduced PortScan and DDoS F1 scores.
- Removing the alert-rate safeguard produced slightly lower F1 scores but no observed contamination.
- Removing safeguards allowed earlier model updates.
- Candidate validation approved the contaminated Wednesday update, demonstrating that its existing checks alone do not guarantee contamination-free training.

## Limitations
- Results are retrospective and specific to CIC-IDS2017.
- Attack labels were used only for auditing, not for adaptation decisions.
- Zero observed contamination does not guarantee safety on unseen traffic.
- Adaptation was evaluated offline; live Kafka/Flink model updating was not demonstrated.

## Reproducibility
Script: scripts/eval_adaptive_ablation.py

Run with --variant full, --variant no_alert_gate,
or --variant no_score_filter.

Results: results/exp4_*.csv
