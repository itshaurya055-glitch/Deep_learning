# Four Worlds Adversarial Fraud Detection Lab

## Project Summary

This repository implements a research-driven fraud detection laboratory for evaluating how machine learning detectors behave under adversarial payment attacks. The work uses the PaySim synthetic financial transaction dataset and expands the standard fraud-classification setting into a four-world adversarial evaluation protocol designed to study robustness under known and unknown attack patterns.

The project compares two detectors:

- XGBoost classifier for tabular transaction analysis
- LSTM detector for sequential account-level fraud behavior

It evaluates each detector against:

- static fraud campaigns
- random-action control attackers
- PPO-trained adaptive attackers

The lab is designed to answer a central research question:

> How robust are fraud detection models when attackers evolve their strategies and fraud patterns are no longer observed during training?

The project is implemented as a complete reproducible pipeline with data processing, feature engineering, model training, adversarial simulation, evaluation metrics, and result visualization.

---

## Abstract

Financial fraud detection systems are usually evaluated on static train/test splits, which do not capture how attackers continuously adapt their behavior. In production environments, fraud patterns evolve, novel campaign strategies emerge, and malicious actors exploit model blind spots. This project introduces a four-world adversarial evaluation framework for fraud detection on the PaySim dataset.

We construct a lab that compares model behavior under four conditions: known fraud with static attacks, known fraud with adaptive attacks, unknown fraud with static attacks, and unknown fraud with adaptive attacks. The system trains both XGBoost and LSTM detectors, simulates fraud campaigns using static and reinforcement-learning-based attackers, and measures detector robustness using campaign success rate, cumulative financial exposure, intervention timing, and F1-based performance degradation.

The work includes a hardened retraining mechanism, random-action control baselines, and validation-based calibration. The checked-in results are preliminary: the current CSV does not provide an auditable three-seed aggregate, parity-check log, or hardened comparison table, so no claim about hardened performance is made here.

---

## Why this project matters

Most fraud detection systems are tested only on stationary transaction distributions. This creates a critical gap: the model may appear highly accurate on current fraud examples, but fail under new adversarial patterns or shifting attacker strategies.

This project addresses that limitation by explicitly simulating the evolution of attack behavior. It does not only ask "Can the model detect fraud?" It asks:

- Can it detect known attacks?
- Can it detect adaptive attacks?
- Can it detect unknown patterns it has not seen before?
- How much does performance degrade as adversarial behavior changes?

These are central questions for real financial security systems.

---

## Novelty and research contribution

The novelty of this work lies in combining three ideas into one unified lab:

1. Four-world adversarial evaluation
2. Reward-driven adaptive attacker generation using PPO
3. Robustness measurement under both known and unknown fraud archetypes

### Compared with baseline fraud detection studies

Existing financial fraud studies usually focus on one or more of the following:

- static classification benchmark on a single split
- anomaly detection or outlier detection only
- rule-based or score-based fraud screening
- isolated fraud detection without adversarial simulation

In contrast, this project provides a systematic adversarial robustness framework with the following new contributions:

- It simulates both known and unknown fraud archetypes.
- It compares static and adaptive attacker settings side by side.
- It measures not only conventional fraud metrics, but also campaign-level robustness indicators such as campaign success rate and financial exposure.
- It includes control attackers to ensure that improvements are not caused by a weak or random baseline.
- It evaluates detector degradation between worlds, which directly measures robustness under adversarial drift.

### Research novelty statement

This project contributes to the literature by modeling fraud detection as a moving-target adversarial problem rather than a static classification problem. The key novelty is the use of a four-world protocol that quantifies how detector performance changes when attackers shift from familiar patterns to previously unseen and adaptive strategies.

This formulation is closer to real deployment conditions in financial systems, where adversarial actors continuously adapt in response to defensive measures.

---

## Related work and positioning

The project fits into the intersection of:

- fraud detection on tabular and sequential transaction data
- adversarial machine learning
- reinforcement learning for attack generation
- robust evaluation of security systems under distribution shift

### Comparison with representative methods

| Work | Date | Attacker type | Multi-step campaign | Unseen-fraud split | Defence |
| --- | --- | --- | --- | --- | --- |
| Mead et al. (UVA) | 2018 | RL | No | No | No |
| El-Awady | Jul 2021 | RL | No | No | No |
| Fursov et al. (KDD) | 2021 | Sequence perturbation | No | No | Yes |
| FRAUD-RLA | Feb 2025 | PPO | No (single transaction) | No | No |
| Catch Me If You Can (ICDE) | 2025 | Multi-agent | No | No | Yes (co-evolving) |
| FraudShield (arXiv) | Jul 2026 | None (static split) | No | No | Hybrid model |
| **This work** | **2026** | **Static, random and PPO** | **Yes** | **Yes** | **In progress; adversarial retraining implemented but not evidenced in the checked-in run** |

The dates and venue labels above should be verified against the final bibliography before submission. The comparison positions this project around the RL attacker and campaign-level evaluation; the LSTM remains a supporting detector rather than the headline contribution.

---

## System overview

The project is composed of the following core components:

1. Data loading and preprocessing
   - Loads PaySim transaction data
   - Samples legitimate transactions for class balance control
   - Engineers transaction-level and account-level features

2. Feature engineering
   - Balance drain indicators
   - Transaction type flags
   - New-beneficiary flags
   - Rolling counts and sums over recent transactions
   - Timing and velocity features

3. Detector training
   - XGBoost baseline model on tabular features
   - LSTM sequence model for account history
   - Threshold calibration using validation-aware logic

4. Adversarial campaign generation
   - Static known fraud archetypes
   - Static unknown fraud archetypes
   - PPO-based adaptive campaigns
   - Random-action baseline control for comparison

5. Simulation engine
   - Runs each campaign step-by-step
   - Blocks the campaign when the detector raises an alarm
   - Records cumulative exfiltration and intervention timing

6. Evaluation and reporting
   - Computes campaign success rate, financial exposure, F1, precision, recall, and degradation
   - Generates comparison tables and plots

---

## Dataset and preprocessing

### Dataset

The project uses the PaySim synthetic mobile-money transaction dataset, which is a realistic synthetic dataset commonly used in fraud detection research. It contains transaction events that reflect payment flows, withdrawal behavior, and account balance changes.

Fields used in the pipeline include:

- step
- type
- amount
- payer account and beneficiary account
- old and new balances of sender and receiver
- isFraud

### Preprocessing pipeline

The preprocessing pipeline in [src/data_processor.py](src/data_processor.py) performs the following steps:

- loads the raw PaySim CSV
- renames transaction fields to a UPI-style schema
- isolates fraud and legitimate rows
- samples legitimate rows to moderate class imbalance
- combines fraud and sampled legitimate transactions
- sorts by transaction step
- engineers behavioral features for fraud detection

### Feature engineering

The engineered feature set includes:

- amount
- oldbalanceOrg
- newbalanceOrig
- oldbalanceDest
- newbalanceDest
- type_TRANSFER
- type_CASH_OUT
- is_balance_drain
- is_new_beneficiary
- tx_count_last_N
- sum_amount_last_N
- distinct_dest_last_N
- steps_since_last_tx

These features encode the suspicious patterns that fraud attackers typically exploit, including rapid transfers, new destinations, unusual cash-out events, and sudden balance depletion.

---

## Model design

### 1. XGBoost detector

Implemented in [src/models/xgboost_detector.py](src/models/xgboost_detector.py).

The XGBoost model is trained on the engineered transaction features. It uses class balancing through scale_pos_weight and calibrates the prediction threshold using F1 on the training data. It is designed as a strong tabular baseline for fraud classification.

### 2. LSTM detector

Implemented in [src/models/lstm_detector.py](src/models/lstm_detector.py).

The LSTM model processes account-level transaction sequences and learns temporal fraud patterns that may not be obvious in a single-row feature representation. It maintains a short rolling history per payer account and calculates fraud probability using recent activity. In the current simulation, however, a campaign step can be evaluated with zero-padded history, so the LSTM does not always receive a genuine sequence. This is a limitation of the current comparison and a likely contributor to its weaker preliminary result.

### 3. PPO-based adversarial attacker

Implemented in [src/campaigns/rl_attacker.py](src/campaigns/rl_attacker.py).

The RL attacker is trained in a reward-driven environment where the attacker chooses actions designed to maximize fraud success while avoiding detection. It learns campaign patterns like amount, delay, and transfer structure that are optimized toward exfiltration goals.

---

## Four-world evaluation protocol

The lab evaluates detector robustness in four worlds:

### World 1: Known fraud + static attacker

- Standard known campaign patterns
- No adversarial agent adaptation
- Baseline performance reference

### World 2: Known fraud + adaptive attacker

- The same known fraud archetypes are attacked by a learned PPO policy
- Tests whether a static detector collapses when the attacker adapts

### World 3: Unknown fraud + static attacker

- Fraud patterns are unseen during training
- Measures transferability to novel, unknown fraud archetypes

### World 4: Unknown fraud + adaptive attacker

- The most challenging setting
- Combines distribution shift and optimization from the adversary

This protocol is the core research contribution of the project, because it provides a way to study robustness under adversarial evolution and unseen fraud behavior.

---

## Campaign generation and simulation

### Static campaign generator

Implemented in [src/campaigns/static_generator.py](src/campaigns/static_generator.py).

Known archetypes include:

- mule_cascade
- smurfing
- rapid_fire

Unknown archetypes include:

- slow_drip
- interleaved_noise

These campaigns model different adversarial patterns, ranging from large sudden transfers to complex multi-step fraud sequences.

### Simulation engine

Implemented in [src/simulation/simulator.py](src/simulation/simulator.py).

The simulation engine converts each fraud campaign into step-by-step transaction evaluation. For each transaction step:

- it transforms the step to detector features
- queries the detector
- checks whether the transaction is flagged
- blocks the campaign at the first alarm
- records cumulative exfiltration and successful vs blocked status

This makes the project closer to real defense evaluation than a simple offline static F1 score.

---

## Evaluation metrics

The evaluation module in [src/evaluation/metrics.py](src/evaluation/metrics.py) computes the following metrics:

- CSR: Campaign Success Rate
- CFE: Cumulative Financial Exposure
- IT: Intervention Timing
- F1 score
- Precision
- Recall
- AUC (where applicable)
- DD: detector degradation score between World 1 and later worlds

### Why these metrics matter

A detector is not only judged by fraud classification accuracy; it is judged by how well it prevents successful campaigns and how early it intervenes. In adversarial fraud settings, a model with high static F1 can still be operationally weak if it fails to stop aggressive campaigns before exfiltration succeeds.

---

## Demo and reproducibility

### Quick demo

The project provides a lightweight demo script:

- [demo_showcase.py](demo_showcase.py)
- [demo_showcase_presentation.py](demo_showcase_presentation.py)

These scripts are intended to present the project in a clear and compact way for interviews, student demos, or brief presentations.

### Full experiment

Run the full adversarial lab from the root directory:

```bash
python run_lab_v2.py --seeds 0 1 2 --ppo-steps 50000 --num-campaigns 50
```

Optional flag to disable hardened retraining:

```bash
python run_lab_v2.py --seeds 0 --ppo-steps 50000 --num-campaigns 50 --no-hardening
```

### Output artifacts

The project writes key outputs to the folder:

- [output](output)

This folder includes:

- result CSV files
- model files
- plot outputs
- comparison summaries

---

## Results and evidence status

The checked-in [output/four_worlds_results_v2.csv](output/four_worlds_results_v2.csv) is a preliminary run artifact. It contains one result row per detector/attacker/world combination, but no seed column, parity-check output, or hardened rows. Therefore it should not be presented as a verified three-seed result, and the near-zero F1 and high CSR values may reflect a feature-path or calibration mismatch rather than detector quality alone.

The visible preliminary values are:

| Detector | World | Attacker | CSR | CFE | IT | F1 | DD |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| XGBoost | World 1 | static | 0.6667 | 45666.7 | 4.33 | 0.1111 | 0.00 |
| XGBoost | World 2 | random_control | 1.0000 | 53000.0 | 6.67 | 0.0000 | 0.1111 |
| XGBoost | World 2 | PPO | 1.0000 | 60000.0 | 4.00 | 0.0000 | 0.1111 |
| XGBoost | World 3 | static | 1.0000 | 53333.3 | 10.00 | 0.0000 | 0.1111 |
| XGBoost | World 4 | random_control | 1.0000 | 53000.0 | 8.00 | 0.0000 | 0.1111 |
| XGBoost | World 4 | PPO | 1.0000 | 60000.0 | 4.00 | 0.0000 | 0.1111 |
| LSTM | World 1 | static | 1.0000 | 52333.3 | 4.33 | 0.0000 | 0.0000 |
| LSTM | World 2 | random_control | 1.0000 | 51333.3 | 5.67 | 0.0000 | 0.0000 |
| LSTM | World 2 | PPO | 1.0000 | 60000.0 | 4.00 | 0.0000 | 0.0000 |
| LSTM | World 3 | static | 1.0000 | 53333.3 | 10.00 | 0.0000 | 0.0000 |
| LSTM | World 4 | random_control | 1.0000 | 53000.0 | 8.00 | 0.0000 | 0.0000 |
| LSTM | World 4 | PPO | 1.0000 | 60000.0 | 4.00 | 0.0000 | 0.0000 |

These values are descriptive only. In particular, CSR values such as 0.6667 should not be interpreted as evidence from 50 campaigns without a run log that records the campaign count and seed. The F1 values near zero are an open validation issue and require a parity/calibration check before interpretation.

### Hardened versus baseline comparison

The strongest intended result is a baseline-versus-hardened comparison on World 3 and World 4. It is not available in the checked-in CSV: the file has no `*_Hardened` detector rows. The comparison is therefore **in progress** and must be populated only after hardened retraining, the parity check, and the three-seed run have been completed and archived.

| Detector | World | Attacker | Baseline F1 | Hardened F1 | Baseline CSR | Hardened CSR | Evidence |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| XGBoost | World 3 | static | pending verified rerun | pending | pending | pending | Not in checked-in artifact |
| XGBoost | World 4 | PPO | pending verified rerun | pending | pending | pending | Not in checked-in artifact |
| LSTM | World 3 | static | pending verified rerun | pending | pending | pending | Not in checked-in artifact |
| LSTM | World 4 | PPO | pending verified rerun | pending | pending | pending | Not in checked-in artifact |

The RL attacker is the headline contribution. The random-action control is required to show that PPO is adaptive rather than merely another source of random campaigns. The LSTM is retained as a supporting deep-learning detector and should be interpreted with the sequence-input limitation above.

The actual preliminary result files are stored in:

- [output/four_worlds_results_v2.csv](output/four_worlds_results_v2.csv)
- [output/four_worlds_results.csv](output/four_worlds_results.csv) (older result format)

The plots are stored in:

- [output/plots](output/plots)

---

## Graphs and comparison plots

This project generates comparison plots to visualize:

- detector F1 across worlds
- campaign success rate across attack modes
- cumulative financial exposure
- intervention timing
- degradation trends from World 1 to later worlds

Example plots in the project include:

- F1 comparison between XGBoost and LSTM
- attack-world comparisons for static vs adaptive adversaries
- cumulative exposure and intervention timing plots

These visualizations make the lab more readable and make it easier to turn the results into a research figure set for a conference paper.

---

## Implementation notes

### Code organization

- [run_lab_v2.py](run_lab_v2.py): main pipeline driver
- [src/config.py](src/config.py): project constants and hyperparameters
- [src/data_processor.py](src/data_processor.py): data ingestion and feature engineering
- [src/models/xgboost_detector.py](src/models/xgboost_detector.py): XGBoost detector
- [src/models/lstm_detector.py](src/models/lstm_detector.py): LSTM detector
- [src/campaigns/static_generator.py](src/campaigns/static_generator.py): known and unknown attack archetypes
- [src/campaigns/rl_attacker.py](src/campaigns/rl_attacker.py): PPO-based attacker
- [src/simulation/simulator.py](src/simulation/simulator.py): simulation engine
- [src/evaluation/metrics.py](src/evaluation/metrics.py): campaign and detector metrics
- [src/evaluation/visualizer.py](src/evaluation/visualizer.py): plotting logic

### Typical execution behavior

The lab prints:

- dataset loading information
- training logs for XGBoost and LSTM
- campaign generation summaries
- evaluation metrics by world
- final aggregated comparison table

This makes the project easy to debug and easy to demonstrate to collaborators or reviewers.

---

## Conference paper draft outline

The project is already structured in a way that can be expanded into an IEEE-style conference paper. A suitable paper draft organization is:

### Title

Adversarial Fraud Detection under Known and Unknown Attack Worlds: A Four-World Evaluation Study on PaySim

### 1. Introduction

- Motivation: fraud detection under adversarial drift
- Problem statement
- Research questions and contributions

### 2. Related Work

- Fraud detection using machine learning
- Sequence-based fraud models
- Adversarial learning and RL-based attack generation
- Robustness under distribution shift

### 3. Methodology

- Dataset and preprocessing
- Feature engineering
- Detectors and training procedure
- Four-world adversarial evaluation protocol
- Campaign generation and simulator design

### 4. Experimental Setup

- Train/test split
- World definitions
- Attack configurations
- Evaluation metrics

### 5. Results and Discussion

- Performance across worlds
- Comparison of known vs unknown attack performance
- Robustness degradation analysis
- ablation and control analysis

### 6. Conclusion

- Summary of findings
- Implications for real financial systems
- Future work

### IEEE-style formatting guidance

For a conference submission, the main paper should use:

- 2-column IEEE format
- concise abstract (150-250 words)
- formal sections and subsection numbering
- numbered equations or algorithm-like descriptions if needed
- tables and figures with captions
- a strong novelty and contribution paragraph
- a brief discussion of limitations and future work

This repository provides the raw material for that paper: the dataset pipeline, attack models, metrics, and result outputs.

---

## Current interpretation of results

The project is not designed to claim that a simple static detector is "good enough" in all scenarios. Instead, its main value is demonstrating that adversarial robustness is not a static property of a model; it depends heavily on how the attacker changes, how unfamiliar the attack is, and how aggressively the system must intervene.

This makes the lab especially useful for research discussion, class projects, applied AI experiments, and conference-level work on adversarial financial security.

---

## Future work

Possible extensions include:

- adding more advanced detectors such as GNNs or transformer-based account models
- evaluating more attack archetypes and perturbation strategies
- calibrating rewards and adversary policy for stronger optimization
- testing against real banking transaction streams or privacy-safe datasets
- integrating explainability and feature importance analysis
- turning this into a full IEEE conference paper with figures, ablation tables, and formal evaluation sections

---

## Summary

This project is a complete adversarial fraud detection laboratory built around the PaySim dataset. It is strong not only as an implementation, but also as a research framework: it captures adversarial drift, surprise attacks, RL-driven campaign generation, and detector robustness under realistic fraud evolution.

The repository is therefore suitable for:

- research discussion
- project presentation
- collaboration and code review
- conference paper drafting
- student or thesis demonstration

It demonstrates both the engineering implementation and the core research idea behind adversarially robust fraud detection.

The detector degradation formula is:

DD = F1(World 1) - F1(World 4)

This quantifies how much performance drops in the hardest adversarial world.

#### [src/evaluation/visualizer.py](src/evaluation/visualizer.py)
This module generates charts for the simulation results and saves them to the output folder.

It produces:

- campaign success rate comparison
- financial exposure comparison
- intervention timing comparison
- detector F1 comparison
- degradation curves across worlds

---

## Limitations and next steps

- The unknown archetypes are synthetic and were designed for this lab, not externally validated attacks.
- The PPO attacker uses a simplified 9-action space and requires comparison with the random-action control.
- PaySim is synthetic and is not a full production UPI transaction stream.
- The current checked-in artifact does not prove a three-seed run, parity success, or hardened improvement.
- The LSTM should receive real campaign history in a future run; otherwise its result must remain a supporting baseline rather than evidence of sequence-model robustness.

Before presenting final results, rerun the documented command, archive the seed-level outputs and parity logs, and populate the hardened World 3/World 4 table only from that verified run.
