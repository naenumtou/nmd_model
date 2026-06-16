# Non-Maturity Deposit Models (NMD Models)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&style=for-the-badge)
![Pandas](https://img.shields.io/badge/pandas-Data%20Analysis-purple?logo=pandas&style=for-the-badge)
![NumPy](https://img.shields.io/badge/NumPy-Numerical-green?logo=numpy&style=for-the-badge)
![SciPy](https://img.shields.io/badge/SciPy-Scientific%20Computing-blue?logo=scipy&style=for-the-badge)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-blueviolet?style=for-the-badge&logo=Plotly&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-Visualization-3775a9?style=for-the-badge&logo=plotly&logoColor=white)
![MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

A complete end-to-end implementation of Non-Maturity Deposit (NMD) Models for IRRBB Measurement, behavioral analysis, and Asset-Liability Management (ALM) — built in Python across 10 sequential Jupyter notebooks.

<p align="center">
<img width="1536" height="1024" alt="สอนการพัฒนาแบบจำลอง Non-Maturity Deposit Models (NMD Models) ตั้งแต่ต้นจนจบ" src="https://github.com/user-attachments/assets/b49ff7f7-1234-484c-867b-2e2ecb2eb126" />
</p>

## Overview
### What is NMD and why does it matter?
**Non-Maturity Deposits (NMD)** are bank deposits with no contractual maturity date e.g., savings accounts, current accounts, and demand deposits. Customers can withdraw at any time, yet in practice most balances remain stable for years. This creates a fundamental challenge for banks:

- **Liquidity risk (ILAAP)**: If customers suddenly withdraw, the bank must have enough liquid assets to pay them.
- **Interest rate risk (IRRBB)**: The bank invests deposits at market rates but pays customers a lower "sticky" deposit rate. When interest rates change, both the value of investments and the cost of funding shift — but not at the same speed.

Under **BCBS 368** (*Interest Rate Risk in the Banking Book*, April 2016), banks are required to model NMD Behavior explicitly, quantify how sensitive their economic value is to interest rate movements, and report this exposure to regulators. This repository implements the complete NMD Modeling framework from raw data to IRRBB Disclosure tables with consideration of ILAAP integrated. 

## Project Structure
The project is built as **10 Sequential Jupyter Notebooks** (Notebook 01 - Notebook 10), each responsible for one modeling layer. Outputs from upstream notebooks flow as inputs into downstream ones via serialized model objects (`.pkl`).

```
Synthetic Data → Behavioral Models → Rate Models → Valuation → Hedging → IRRBB Report
     NB01            NB02–NB03        NB04–NB05    NB06–NB07  NB08–NB09     NB10
```

### What makes this implementation distinctive
- Fully reproducible — Every result can be traced back to Notebook 01's random seed.
- BCBS 368 Compliant — EVE Discount method, regulatory caps, and scenario shocks follow the standard.
- No OOP — All logic implemented as standalone functions with full docstrings.
- Modular — Each notebook can be modified or extended independently.

```
nmd_model/
├── model/                                        #Trainned model and parameters (pkl.)
│   ├── hazard_rate.pkl
│   ├── ci_model.pkl
│   ├── hp_model.pkl
│   ├── gbm_model.pkl
│   ├── ddm_model.pkl
│   ├── ddy_model.pkl
│   ├── beta_model.pkl
│   ├── threshold_model.pkl
│   ├── jvd_model.pkl
│   ├── threshold_model.pkl
│   ├── runoff_model.pkl
│   ├── replicating_weights.pkl
│   ├── yield_curve.pkl
│   └── dynamics_model.pkl
├── notebooks/
│   ├── 01_data_generation.ipynb
│   ├── 02_survival_decay.ipynb
│   ├── 03_stable_nonstable.ipynb
│   ├── 04_deposit_rate_model.ipynb
│   ├── 05_deposit_decay.ipynb
│   ├── 06_economic_theory.ipynb
│   ├── 07_nmd_floor.ipynb
│   ├── 08_structural_hedge.ipynb
│   ├── 09_wealth_allocation.ipynb
│   └── 10_irrbb_integration.ipynb
├── src/
│   ├── data_generator.py
│   ├── survival_analysis.py
│   ├── stable_nonstable_model.py
│   ├── deposit_rate_model.py
│   ├── deposit_decay_model.py
│   ├── economic_theory.py
│   ├── nmd_floor.py
│   ├── caterpillar.py
│   ├── wealth_allocation.py
│   ├── reporting.py
│   └── plot_function.py
├── data/          
│   ├── processed
│   └── raw/
|   └── └── nmd_data.parquet
├── requirements.txt
└── README.md
```

### Data Flow Architecture
Each notebook exports key outputs via `pickle` for downstream use. The diagram below shows the complete dependency chain.

```
NB01: Data Generation
  └─► nmd_data.parquet
        │
        ├─► NB02: Survival Decay
        │     └─► hazard_rate.pkl ──────────────────────────────► NB05
        │
        ├─► NB03: Stable/Non-Stable
        │     └─► stable_pct (CI method) ──────────────────────► NB05
        │
        ├─► NB04: Deposit Rate Model
        │     └─► γ=0.2072, α=0.0008, β₂=0.0797 ─────────────► NB05, NB06
        │
        ├─► NB05: Deposit Decay
        │     └─► core_balance=3,823 MB, WAL=2.85Y ───────────► NB06, NB08, NB09, NB10
        │     └─► IRRBB repricing buckets ─────────────────────► NB10
        │
        ├─► NB06: Economic Theory (EVE)
        │     └─► EVE=7,870 MB, ΔEVE=−1,062 MB ─────────────────► NB10
        │
        ├─► NB07: NMD Floor
        │     └─► Floor K=0%: 21 MB, K=d₀: 50 MB ──────────────► NB10
        │
        ├─► NB08: Structural Hedge
        │     └─► 5Y/5 tranches, NII=186 MB ────────────────────► NB09, NB10
        │
        └─► NB09: Wealth Allocation
              └─► w_cat=85%, NII=172 MB, LCR=173%, NSFR=135% ──► NB10
                  └─► Asset repricing profile ────────────────────► NB10

NB10: IRRBB Integration
  └─► 6 output tables:
        1. NMD Model Summary
        2. EVE Sensitivity Table
        3. NII Sensitivity Table
        4. Repricing Gap
        5. Net Repricing Gap (Asset − Liability)
        6. LCR / NSFR
```

## Project Details
### 1. Synthetic Data Generation
<p align="center">
<img width="1536" height="1024" alt="สอนการพัฒนาแบบจำลอง Non-Maturity Deposit Models (NMD Models) ตั้งแต่ต้นจนจบ" src="https://github.com/user-attachments/assets/8e3a4b45-50f2-4eed-9efb-c5ffed67f0e5" />
</p>

**Purpose:** Generate 150 months of realistic NMD Data capturing the joint dynamics of market rate, deposit rate, deposit balance, and CDS spread. The real NMD Data is proprietary and varies significantly across institutions. Synthetic data allows full reproducibility and enables stress scenario testing by simply changing seed parameters.

**The four variables are simulated jointly with realistic correlations**

| Variable | Model |
|---|---|
| Market rate `r_t` | AR(1) on changes |
| Deposit rate `d_t` | Asymmetric error correction |
| Balance `D_t` | Log-linear with macro drivers |
| CDS spread `cs_t` | Log AR(1) |

**Output:** `data/raw/nmd_data.parquet` — 150 rows × 4 columns

<p align="center">
<img width="1988" height="1376" alt="สอนการพัฒนาแบบจำลอง Non-Maturity Deposit Models (NMD Models) ตั้งแต่ต้นจนจบ" src="https://github.com/user-attachments/assets/46b3e615-88c8-40be-8ccf-e87d3e710d4d" />
</p>

### 2. Survival Decay Model
<p align="center">
<img width="1536" height="1024" alt="สอนการพัฒนาแบบจำลอง Non-Maturity Deposit Models (NMD Models) ตั้งแต่ต้นจนจบ" src="https://github.com/user-attachments/assets/fefa8aee-71c2-45fe-84e6-ea68b195db90" />
</p>

**Purpose:** Measure how quickly deposit balances run off each month using cohort analysis. The aggregated balance series hides vintage effects. Accounts opened during a low rate environment behave differently from those opened during rate hikes. Cohort analysis separates these effects precisely.

**Method — Discrete Hazard Rate:**
1. Build a balance matrix `B(t, c)` where `t` = observation month, `c` = cohort (origination month)
2. Compute monthly runoff rate per cohort: `RR(t, c) = 1 − B(t, c) / B(t−1, c)`
3. Average across all surviving cohorts at time `t`: `h(t) = mean(RR(t, c))`

**Why not Cox Regression?**
Cox PH is designed for binary events (closed vs. open). NMD has *partial runoff* that balance declines gradually with no single exit event. Discrete hazard rates capture this continuous erosion.

**Stressed runoff — two approaches:**

| Approach | Method | Use Case |
|---|---|---|
| Percentile | P95 of `h(t)` | ILAAP Base stress floor |
| MEV Regression | `h(t) = α + β₁·r_repo + β₂·unemployment + ε` | Macro scenario projection |

**Output:**
- `hazard_rate.pkl` — monthly hazard rates for first 24 months
- Average base hazard ≈ 3.65% per month

<p align="center">
<img width="1390" height="985" alt="สอนการพัฒนาแบบจำลอง Non-Maturity Deposit Models (NMD Models) ตั้งแต่ต้นจนจบ" src="https://github.com/user-attachments/assets/49ff41ae-caab-41e0-975c-91a4aeca48b9" />
</p>

### 3. Stable / Non-Stable Decomposition
<p align="center">
<img width="1536" height="1024" alt="สอนการพัฒนาแบบจำลอง Non-Maturity Deposit Models (NMD Models) ตั้งแต่ต้นจนจบ" src="https://github.com/user-attachments/assets/efbe6d3d-2316-42fc-9e7f-5c4158d38c99" />
</p>

**Purpose:** Decompose total balance into core and non-core (volatile) portions, subject to BCBS 368 regulatory caps. This decomposition matters because only the stable portion can be invested long term. Investing volatile balances in long duration assets creates liquidity risk.

#### 3.1 Confidence Interval
OLS regression of balance on time trend, using the lower 95% CI as the core floor:
```
stable_pct = mean(lower_CI) / mean(balance)
```
#### 3.2 Hodrick-Prescott
Decomposes balance into trend and cycle using λ = 1,600. The cycle component is non-core.
#### 3.3 Geometric Brownian Motion
Simulates worst-case paths using historical log-return volatility. Balances below the P5 quantile are non-core.
#### 3.4 Drawdowm Analysis
Maximum historical MoM or YoY decline percentage is treated as the non-corefraction.

**BCBS 368 Regulatory Caps:**
| Segment | Max Core % | Max WAL |
|---|---|---|
| Retail Transactional | 90% | 5.0Y |
| Retail Non-transactional | 70% | 4.5Y |
| Wholesale | 50% | 4.0Y |

**Output:** HP Method `stable_pct = 96.36%` on the synthetic data.

<p align="center">
<img width="1390" height="985" alt="สอนการพัฒนาแบบจำลอง Non-Maturity Deposit Models (NMD Models) ตั้งแต่ต้นจนจบ" src="https://github.com/user-attachments/assets/691b6eb9-4ca4-4969-a8cf-c316dec8e1e7" />
</p>

### 4. Deposit Rate Model
<p align="center">
<img width="1536" height="1024" alt="สอนการพัฒนาแบบจำลอง Non-Maturity Deposit Models (NMD Models) ตั้งแต่ต้นจนจบ" src="https://github.com/user-attachments/assets/7710d3d3-0453-4fe1-940e-1d0e889f589d" />
</p>

**Purpose:** Model how deposit rates respond to changes in market rates considering both in the short run and long run. The pass-through speed determines how quickly the bank's funding cost increases when rates rise. Slow pass-through means sticky deposit rates that wider spreads for longer where creating implicit duration risk.

#### 4.1 Beta Regression (Long-run Pass-through)
```
d = α + β·r   →   β = 0.2072
```
A 100bps rise in market rates produces only a 20.72bps rise in deposit rates in the long run.
#### 4.2 Threshold Model
Banks raise deposit rates slowly (λ⁺ = 0.15) but cut them quickly (λ⁻ = 0.45) — protecting margins on the way down. Also estimates bank spread `α = 0.0008`.
#### 4.3 Jarrow Ven Deventer
```
Δd_t = β₀ + β₁·t + β₂·Δr_t   →   β₂ = 0.0797
```
Only ~7.97% of any rate change passes through to deposit rates in the same month.

**Output:**
| Parameter | Value | Meaning |
|---|---|---|
| γ (gamma) | 0.2072 | Long-run pass-through |
| α (alpha) | 0.0008 | Bank spread |
| β₂ (beta_2) | 0.0797 | Short-term pass-through |

<p align="center">
<img width="1389" height="985" alt="สอนการพัฒนาแบบจำลอง Non-Maturity Deposit Models (NMD Models) ตั้งแต่ต้นจนจบ" src="https://github.com/user-attachments/assets/d483dd50-e5d4-4284-9458-1919f847b790" />
</p>

### 5. Deposit Decay Model
<p align="center">
<img width="1536" height="1024" alt="สอนการพัฒนาแบบจำลอง Non-Maturity Deposit Models (NMD Models) ตั้งแต่ต้นจนจบ" src="https://github.com/user-attachments/assets/c0a98820-d1ab-4628-9e35-c633e11d2747" />
</p>

**Purpose:** Build a forward 60-month(s) runoff portfolio profile of core deposits and producing by replicating portfolio for the IRRBB Repricing gap distribution.

**Core Balance Formula:**
```
Core Balance = Total Balance × stable_pct × (1 − β)
             = 5,039 MB × 96.36% × (1 − 0.2072) = 3,850 MB
```
- `stable_pct`: Behavioural stability (does not exhibit volatile withdrawal patterns)
- `(1 − β)`: Repricing stickiness (does not reprice immediately when market rates change)

#### 5.1 Historical Runoff Profile
```python
Maximise  WAL(seed)
s.t.      WAL ≤ 5.0Y            # BCBS 368 cap
          Core_remaining ≤ 90%  # BCBS 368 cap
```
Solved by `scipy.optimize.differential_evolution` with tolerance 1e-8.

#### 5.2 Replicating Portfolio
```
Minimise  std(X·w − margin − deposit_rate)
s.t.      Σwᵢ = 1,  WAL ≤ 5Y,  0 ≤ wᵢ ≤ 50%
```

**Output:** Weight Average Life (WAL) = **2.85 years**

<p align="center">
<img width="1380" height="986" alt="สอนการพัฒนาแบบจำลอง Non-Maturity Deposit Models (NMD Models) ตั้งแต่ต้นจนจบ" src="https://github.com/user-attachments/assets/b1fd0826-de7a-430b-814b-32d0b91a45b3" />
</p>

### 6. Economic Theory
<p align="center">
<img width="1536" height="1024" alt="สอนการพัฒนาแบบจำลอง Non-Maturity Deposit Models (NMD Models) ตั้งแต่ต้นจนจบ" src="https://github.com/user-attachments/assets/8de5608b-dde7-4d91-8bc2-ea50093f795a" />
</p>

**Purpose:** Measure the present value of the bank's future NMD Spread income, and quantify its sensitivity to interest rate shocks.

**Cash Flow Definition:** The shock is *subtracted from r_t* to isolate true NMD Sensitivity, not rate-level windfall. Without this, stressed EVE always exceeds base EVE and ΔEVE is always positive that economically meaningless.
```
CF_t = D_t × ((r_t − shock) − d_t)
```
**Four jointly simulated drivers (1,000 paths × 60 months):**
| Driver | Model | Key Feature |
|---|---|---|
| `r_t` | AR(1) on changes | Mean reverts to 5% |
| `d_t` | Error Correction Model | `Δd_t = β₂·Δr_t + κ·(d_{t-1} − γ·r_{t-1} − α)` |
| `D_t` | Log-linear balance | Falls when opportunity cost (r−d) or CDS rises |
| `CDS_t` | Log AR(1) | Always positive, mean reverts |

**Error Correction speed (κ = -0.0580):**
Only ~5.8% of any disequilibrium gap closes per month. After a +200bps shock, deposit rates take ~14 months to close half the gap — this slow adjustment creates the negative ΔEVE.

**EVE Calculation:**
```
EVE = mean(Σ CF_t / (1 + r₀/12)^t) over 1,000 paths
```

**EVE Sensitivity Results:**

| Scenario | EVE (MB) | ΔEVE (MB) |
|---|---|---|
| Base | 9,504 | — |
| +200bps | 8,290 | −1,214 |

The bank is **liability-sensitive**: rising rates compress the NMD spread and reduce franchise value.

<p align="center">
<img width="1390" height="985" alt="สอนการพัฒนาแบบจำลอง Non-Maturity Deposit Models (NMD Models) ตั้งแต่ต้นจนจบ" src="https://github.com/user-attachments/assets/e6cd34ac-9510-4fd8-b601-c65fac66d0e2" />
</p>

### 7. NMD Floor
<p align="center">
<img width="1536" height="1024" alt="สอนการพัฒนาแบบจำลอง Non-Maturity Deposit Models (NMD Models) ตั้งแต่ต้นจนจบ" src="https://github.com/user-attachments/assets/d894fdaf-8652-4912-9e38-ce87e81568f2" />
</p>

<p align="center">
<img width="690" height="985" alt="สอนการพัฒนาแบบจำลอง Non-Maturity Deposit Models (NMD Models) ตั้งแต่ต้นจนจบ" src="https://github.com/user-attachments/assets/7ff6135c-d968-4c02-8a1a-0fa6b8f641ce" />
</p>

### 8. Structual Hedge - Caterpillar
<p align="center">
<img width="1536" height="1024" alt="สอนการพัฒนาแบบจำลอง Non-Maturity Deposit Models (NMD Models) ตั้งแต่ต้นจนจบ" src="https://github.com/user-attachments/assets/0cb17ac8-4ec0-45e4-9ee4-9e25b4c2b86b" />
</p>

<p align="center">
<img width="1386" height="985" alt="สอนการพัฒนาแบบจำลอง Non-Maturity Deposit Models (NMD Models) ตั้งแต่ต้นจนจบ" src="https://github.com/user-attachments/assets/262c3aac-5d3e-4526-875f-07e923925990" />
</p>

### 9. Wealth Allocation Model
<p align="center">
<img width="1536" height="1024" alt="สอนการพัฒนาแบบจำลอง Non-Maturity Deposit Models (NMD Models) ตั้งแต่ต้นจนจบ" src="https://github.com/user-attachments/assets/1cf328fc-5080-4781-acb7-808aa0e84a22" />
</p>

<p align="center">
<img width="1390" height="985" alt="สอนการพัฒนาแบบจำลอง Non-Maturity Deposit Models (NMD Models) ตั้งแต่ต้นจนจบ" src="https://github.com/user-attachments/assets/48e3f64c-5637-4b42-b90f-46b6472c8b24" />
</p>

### 10. IRRBB Reporting
<p align="center">
<img width="1536" height="1024" alt="สอนการพัฒนาแบบจำลอง Non-Maturity Deposit Models (NMD Models) ตั้งแต่ต้นจนจบ" src="https://github.com/user-attachments/assets/8ed872c1-3f08-49ac-a07f-99300c41dcf8" />
</p>

<p align="center">
<img width="1389" height="985" alt="สอนการพัฒนาแบบจำลอง Non-Maturity Deposit Models (NMD Models) ตั้งแต่ต้นจนจบ" src="https://github.com/user-attachments/assets/f800c995-ee5c-4cf8-8bba-dd60e00014d1" />
</p>

## License
MIT · Built for learning purposes
