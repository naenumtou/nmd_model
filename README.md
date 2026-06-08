# Non-Maturity Deposit Models (NMD Models)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&style=for-the-badge)
![Pandas](https://img.shields.io/badge/pandas-Data%20Analysis-purple?logo=pandas&style=for-the-badge)
![NumPy](https://img.shields.io/badge/NumPy-Numerical-green?logo=numpy&style=for-the-badge)
![SciPy](https://img.shields.io/badge/SciPy-Scientific%20Computing-blue?logo=scipy&style=for-the-badge)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-blueviolet?style=for-the-badge&logo=Plotly&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-Visualization-3775a9?style=for-the-badge&logo=plotly&logoColor=white)
![MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)



## Overview
## Project Structure
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

## Project Details
### 1. Synthetic Data Generation
<p align="center">
<img width="1536" height="1024" alt="สอนการพัฒนาแบบจำลอง Non-Maturity Deposit Models (NMD Models) ตั้งแต่ต้นจนจบ" src="https://github.com/user-attachments/assets/8e3a4b45-50f2-4eed-9efb-c5ffed67f0e5" />
</p>

<p align="center">
<img width="1988" height="1376" alt="สอนการพัฒนาแบบจำลอง Non-Maturity Deposit Models (NMD Models) ตั้งแต่ต้นจนจบ" src="https://github.com/user-attachments/assets/46b3e615-88c8-40be-8ccf-e87d3e710d4d" />
</p>

### 2. Survival Decay Model
<p align="center">
<img width="1536" height="1024" alt="สอนการพัฒนาแบบจำลอง Non-Maturity Deposit Models (NMD Models) ตั้งแต่ต้นจนจบ" src="https://github.com/user-attachments/assets/fefa8aee-71c2-45fe-84e6-ea68b195db90" />
</p>

<p align="center">
<img width="1390" height="985" alt="สอนการพัฒนาแบบจำลอง Non-Maturity Deposit Models (NMD Models) ตั้งแต่ต้นจนจบ" src="https://github.com/user-attachments/assets/49ff41ae-caab-41e0-975c-91a4aeca48b9" />
</p>

#### 2.1 Non-Parametric
#### 2.2 Regression

### 3. Stable / Non-Stable Decomposition
<p align="center">
<img width="1536" height="1024" alt="สอนการพัฒนาแบบจำลอง Non-Maturity Deposit Models (NMD Models) ตั้งแต่ต้นจนจบ" src="https://github.com/user-attachments/assets/efbe6d3d-2316-42fc-9e7f-5c4158d38c99" />
</p>

<p align="center">
<img width="1390" height="985" alt="สอนการพัฒนาแบบจำลอง Non-Maturity Deposit Models (NMD Models) ตั้งแต่ต้นจนจบ" src="https://github.com/user-attachments/assets/691b6eb9-4ca4-4969-a8cf-c316dec8e1e7" />
</p>

#### 3.1 Confidence Interval
#### 3.2 Hodrick-Prescott
#### 3.3 Geometric Brownian Motion
#### 3.4 Drawdowm Analysis

### 4. Deposit Rate Model
<p align="center">
<img width="1389" height="985" alt="สอนการพัฒนาแบบจำลอง Non-Maturity Deposit Models (NMD Models) ตั้งแต่ต้นจนจบ" src="https://github.com/user-attachments/assets/d483dd50-e5d4-4284-9458-1919f847b790" />
</p>

#### 4.1 Beta Regression
#### 4.2 Threshold Model
#### 4.3 Jarrow Ven Deventer

### 5. Deposit Decay Model
#### 5.1 Historical Runoff Profile
#### 5.2 Replicating Portfolio

### 6. Economic Theory
#### 6.1 Dynamics Model Simulation
#### 6.2 Economic Value of Equity (EVE)

### 7. NMD Floor
### 8. Structual Hedge - Caterpillar
### 9. Wealth Allocation Model
### 10. IRRBB Reporting

## License
MIT · Built for learning purposes
