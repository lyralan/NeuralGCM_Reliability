# Project Plan: Forecast Reliability Under Distribution Shift for NeuralGCM

## Project Title

**Improving Probabilistic Forecast Reliability of NeuralGCM Under
Distribution Shift**

## Motivation

NeuralGCM (Kochkov et al., Nature 2024) introduces a hybrid physics--ML
global circulation model capable of probabilistic forecasting. While
performance on standard benchmarks is strong, reliability and calibration
under distribution shift remain underexplored. The original paper shows
NeuralGCM-ENS achieves a global spread-skill ratio of ~1.0 (Extended
Data Fig. 2), but with significant regional variation --- both
underdispersion and overdispersion appear in specific areas and
variables. The follow-up precipitation paper (Yuval et al., Science
Advances 2026) explicitly notes that "calibrating the spread perfectly at
a local level is a nontrivial task" and requires further work.

Reliable uncertainty quantification is critical for:

-   Operational weather forecasting
-   Climate risk assessment
-   Decision-making under uncertainty

This project systematically evaluates and improves the probabilistic
reliability of NeuralGCM forecasts under realistic distribution shifts.

------------------------------------------------------------------------

## Core Research Questions

1.  How well calibrated are NeuralGCM probabilistic forecasts under
    distribution shift (held-out years, extreme events, longer lead
    times)?
2.  How does stochastic vs deterministic configuration affect
    reliability?
3.  Can lightweight post-hoc calibration improve reliability without full
    retraining?

------------------------------------------------------------------------

## Models & Checkpoints

| Model | Path | Use |
|-------|------|-----|
| NeuralGCM-ENS 1.4° stochastic | `v1/stochastic_1_4_deg.pkl` | Primary: 50-member ensemble, best overall weather skill |
| NeuralGCM 1.4° deterministic | `v1/deterministic_1_4_deg.pkl` | Comparison: stochastic vs deterministic (Q2) |
| NeuralGCM 2.8° deterministic | `v1/deterministic_2_8_deg.pkl` | Quick prototyping (fastest inference, lowest memory) |

All checkpoints from `gs://neuralgcm/models/`.

------------------------------------------------------------------------

## Technical Approach

### 1. Baseline Setup

-   Use official NeuralGCM open-source repository (legacy API:
    `PressureLevelModel`)
-   Run inference using pretrained checkpoints on Google Colab Pro (A100
    GPU)
-   Build evaluation pipeline leveraging:
    -   JAX + neuralgcm inference API (`encode`, `unroll`, `decode`)
    -   xarray for data handling
    -   NeuralGCM's existing `experimental/metrics/` module (CRPS,
        probabilistic losses, evaluators)
    -   ERA5 data via WeatherBench2 Zarr on Google Cloud Storage

### 2. Distribution Shift Experiments

Evaluate performance under:

-   **Temporal shift**: Held-out years (2020 WeatherBench holdout; 2019--2022
    holdout period from the precipitation paper) vs in-distribution years.
    NeuralGCM was trained on 2001--2017/2018 across all seasons, so the
    shift is temporal, not seasonal.
-   **Extreme event subsets**: Top-quantile temperature anomalies, wind
    speed, and precipitation extremes (Rx1day, 99.9th percentile). The
    precipitation paper shows remaining gaps in extreme tails.
-   **Lead-time degradation**: Short-range (1--3 day) vs medium-range
    (5--15 day) forecasts, where ensemble spread must grow to match
    increasing forecast error.
-   **Regional decomposition**: Tropics vs extratropics, land vs ocean,
    and specific regions where spread-skill ratio deviates from 1.0
    (Extended Data Fig. 2 of Nature paper).

### 3. Evaluation Metrics

Deterministic:

-   RMSE
-   Anomaly Correlation Coefficient (ACC)
-   Root-mean-squared bias (RMSB)

Probabilistic:

-   CRPS (Continuous Ranked Probability Score)
-   Energy score
-   Spread--skill ratio (target: 1.0)
-   Rank histograms (PIT histograms)
-   Brier score (0.95 quantile, following the precipitation paper)
-   Calibration error
-   Sharpness vs reliability tradeoff

------------------------------------------------------------------------

## Post-hoc Calibration Module

Implement lightweight calibration methods applied to ensemble outputs:

-   **Variance scaling** (global temperature-scaling of ensemble spread)
-   **Isotonic regression** on spread--skill relationship (per variable,
    per lead time)
-   **Regional spread rescaling** (learned per-gridpoint or per-region
    scaling factors)
-   **Small learned correction network** on predicted mean/variance
    (optional, if time permits)

Goal: Improve reliability without retraining the full NeuralGCM model.
Calibration parameters are fit on a held-out calibration set, evaluated
on a separate test set.

------------------------------------------------------------------------

## Compute Environment

-   **Platform**: Google Colab Pro, defaulting to **T4 GPU** (16 GB VRAM)
    to conserve compute units. All models (stochastic 1.4° at 1,011 MB,
    deterministic 1.4° at 1,100 MB, deterministic 2.8° at 255 MB) fit
    comfortably on T4. Reserve A100 for full-scale evaluation sweeps only.
-   **Inference budget**: NeuralGCM-ENS 1.4° takes ~12s/forecast on TPU
    v4; estimate ~30--60s per member on T4. For 732 WeatherBench2 init
    times with 50 members, ~15--30 hours per full evaluation sweep on T4.
-   **Total estimate**: ~50--100 T4 GPU-hours across all experiments
-   **Data**: ERA5 via
    `gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3`

------------------------------------------------------------------------

## 8-Week Execution Plan

### Week 1: Setup & Reproduce

-   Set up Colab environment (JAX + GPU, neuralgcm, dependencies)
-   Reproduce inference demo with 2.8° deterministic (fast) and 1.4°
    stochastic (primary)
-   Verify data loading pipeline (ERA5 Zarr, regridding)
-   Familiarize with NeuralGCM `experimental/metrics/` module

### Week 2--3: Baseline Calibration Analysis

-   Run NeuralGCM-ENS on WeatherBench2 2020 holdout initial conditions
-   Compute RMSE, ACC, RMSB (deterministic baselines)
-   Compute CRPS, spread--skill ratio, rank histograms, Brier score
-   Generate spatial maps of calibration quality (cf. Extended Data Fig.
    2)
-   Identify variables/regions/lead-times with worst calibration

### Week 4: Distribution Shift Experiments

-   Evaluate on held-out years vs in-distribution years
-   Subset analysis: extreme events (top-decile anomalies)
-   Lead-time degradation curves for spread--skill and CRPS
-   Regional decomposition (tropics/extratropics, land/ocean)
-   Document failure modes and calibration breakdown patterns

### Week 5--6: Post-hoc Calibration

-   Implement variance scaling, isotonic regression, regional rescaling
-   Split data: calibration fit set / test set
-   Compare calibrated vs uncalibrated across all shift conditions
-   Ablation: which calibration method works best where?

### Week 7: Ablation & Comparison

-   Stochastic vs deterministic reliability comparison (Q2)
-   Resolution sensitivity (2.8° vs 1.4°) if compute allows
-   Synthesize results into coherent narrative

### Week 8: Paper & Release

-   Write NeurIPS Climate Workshop submission draft
-   Generate publication-quality figures and ablation tables
-   Prepare reproducible Colab notebooks
-   Clean up and release public GitHub repository

------------------------------------------------------------------------

## Expected Deliverables

-   Reproducible evaluation framework (JAX + xarray, Colab notebooks)
-   Systematic reliability benchmarks under distribution shift
-   Post-hoc calibration module with documented performance gains
-   NeurIPS Climate Workshop submission draft
-   Public GitHub repository

------------------------------------------------------------------------

## Positioning for DeepMind Weather Internship

This project demonstrates:

-   Hybrid physics--ML modeling expertise (NeuralGCM architecture
    understanding)
-   Probabilistic forecasting evaluation and calibration research
-   JAX-based engineering competence on real-world models
-   Ability to identify and address open problems flagged in the
    literature
-   Climate-relevant ML research maturity

It aligns strongly with DeepMind's weather and climate modeling research
direction, and directly extends the NeuralGCM team's own identified
limitations.

------------------------------------------------------------------------

## Future Work (Out of Scope)

-   **Neural operator correction (FNO)**: Train lightweight FNO to map
    NeuralGCM state to corrected tendency or bias-corrected
    mean/variance. Standalone follow-up project.
-   Multi-resolution robustness
-   Data assimilation integration
-   Extreme-event-tail modeling (EVT-based)
-   Bayesian ensemble distillation
-   Climate change scenario stress-testing

------------------------------------------------------------------------

## Author

Ruoyu (Lyra) Lan\
PhD Student, MIT\
2026
