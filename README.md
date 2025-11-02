# Formative 3 — Probability Distributions, Bayesian Probability & Gradient Descent
  
**Group 9**

## Members
| Member | Email | Roles / Tasks |
|:--|:--|:--|
| **Justine Umuhoza** | u.justine@alustudent.com | Part 1 Dataset & Analysis · Part 3 Iteration 2 |
| **Sheryl Atieno Otieno** | s.otieno@alustudent.com | Part 4 Implementation · Part 3 Iteration 1 |
| **Ulrich Rukazambuga** | u.rukazambu@alustudent.com | Part 1 Code & Plots · Part 3 Iteration 4 |
| **Josue Byiringiro** | j.byiringir@alustudent.com | Part 2 Bayesian Model · Part 3 Iteration 3 |


## Overview

This project demonstrates applied understanding of:

1. **Probability Distributions** — implementing the *Bivariate Normal Distribution* from scratch.  
2. **Bayesian Probability** — applying Bayes’ Theorem to IMDb movie-review sentiment.  
3. **Gradient Descent** — manual derivation and Python implementation for linear regression.

All implementations follow the **DRY principle** via modular code in `src/`, and all plots / tables / outputs are generated reproducibly within Jupyter Notebooks.

---

## Repository Structure

```
Formative3-Group9/
│
├── data/
│   ├── IMDB Dataset.csv
│   └── measure1_smartphone_sens.csv
│
├── src/
│   ├── bvn.py            
│   ├── bayes_imdb.py     
│   └── gradient_manual.py 
│
├── notebooks/
│   ├── Part1_BVN.ipynb
│   ├── part2_Bayes.ipynb
│   ├── part4_gradient_descent.ipynb
│   └── Formativ3_Group9_Notebook.ipynb   #  Merged submission notebook
│
├── outputs/
│   ├── part1_bvn/        # Contour & 3D plots + PDF CSV
│   ├── part2_bayes/      # Bayes tables + posterior chart
│   └── part4_gradient/   # GD plots (m,b & MSE trends)
│
├── docs/
│   ├── Part3_Manual_Calculations.pdf   # Handwritten derivation & iterations
│   └── Contributions.pdf               # Team roles and evidence
│
├── requirements.txt
└── README.md
```

##  Part 1 — Bivariate Normal Distribution

**Goal:** Compute the *Bivariate Normal PDF* manually using two correlated variables from the smartphone-sensor dataset.

**Key Steps**
- Derived BVN formula from first principles (no `scipy.stats`).  
- Calculated mean (μ), covariance (Σ), and correlation (ρ).  
- Computed PDF for each data point and visualized on a grid.  
- Produced **Contour** and **3D Surface** plots.

**Deliverables**
- `/outputs/part1_bvn/contour.png`  
- `/outputs/part1_bvn/surface3d.png`  

**Insight:**  
ρ≈ 0 → nearly circular contour; higher ρ stretches the ellipse along its principal axis.

---

## Part 2 — Bayesian Probability (IMDb Reviews)

**Goal:** Estimate how strongly specific keywords predict *positive sentiment* using **Bayes’ Theorem**.

**Chosen Direction:**  P(Positive | keyword)  
**Keywords**
- Positive → `excellent`, `amazing`, `great`, `love`
- Negative → `bad`, `boring`, `terrible`, `awful`

**Outputs**
| Keyword | Prior P(Positive) | Likelihood P(keyword|Positive) | Marginal P(keyword) | Posterior P(Positive|keyword) |
|:--|:--:|:--:|:--:|:--:|
| excellent | 0.5 | 0.1069 | 0.0668 | 0.7998 |
| amazing  | 0.5 | 0.0638 | 0.0407 | 0.7843 |
| great   | 0.5 | 0.3198 | 0.2375 | 0.6732 |
| love    | 0.5 | 0.2149 | 0.1678 | 0.6403 |
| bad     | 0.5 | 0.1137 | 0.2280 | 0.2493 |
| boring  | 0.5 | 0.0235 | 0.0588 | 0.2001 |
| terrible | 0.5 | 0.0146 | 0.0510 | 0.1435 |
| awful   | 0.5 | 0.0108 | 0.0547 | 0.0990 |

**Deliverables**
- `/outputs/part2_bayes/bayes_table.csv`
- `/outputs/part2_bayes/posterior_bar.png`

**Interpretation:**  
Positive keywords yield high posteriors (≈ 0.64–0.80); negative ones low (≈ 0.10–0.25).  
Confirms Bayes’ Theorem captures sentiment direction using simple frequency probabilities.


## Part 3 & 4 — Gradient Descent (Manual + Code)

**Objective:** Fit a line \(y = mx + b\) via gradient descent.

### Part 3 — Manual Computation
- Initial m₀ = –1, b₀ = 1, α = 0.1, data = {(1, 3), (3, 6)}.  
- Derived ∂J/∂m and ∂J/∂b using chain rule.  
- Performed 4 iterations (manually – 1 per member).

| Iter | m | b | MSE (J) |
|:--:|:--:|:--:|:--:|
| 1 | 1.700 | 2.100 | 1.040 |
| 2 | 1.260 | 1.900 | 0.064 |
| 3 | 1.340 | 1.916 | 0.0348 |
| 4 | 1.3336 | 1.8968 | 0.0318 |

**Trend:**  
MSE drops each iteration, parameters move toward optimum (m*, b*) ≈ (1.5, 1.5).  
Stable learning rate ensures smooth convergence.

See details → [`docs/F3_Group9_Part3_Manual_Calculation.pdf`](docs/F3_Group9_Part3_Manual_Calculation.pdf)


### Part 4 — Python Implementation (+ SciPy Validation)
- Explicit loop showing each update of m and b.  
- Separate plots for:
  - Parameters (m, b) vs iteration  
  - Error (MSE) vs iteration  
- Verified with `scipy.optimize.minimize` (BFGS) → converged to (1.5, 1.5).

**Deliverables**
- `/outputs/part4_gradient/params_over_time.png`
- `/outputs/part4_gradient/mse_over_time.png`


##  Collaboration & Integrity
- Work divided by sections (Parts 1–4) + peer review.  
- Functions modularized in `src/` (imported into notebooks).  
- All code original and explainable by each member.  
- Academic Integrity Policy observed throughout.

 Full details → [`docs/contributions.pdf`](docs/contributions.pdf)


## Running the Project

### Clone & Set Up
```bash
git clone https://github.com/<your-username>/Formative3-Group9.git
cd Formative3-Group9
pip install -r requirements.txt
jupyter notebook notebooks/Formative3_Final.ipynb
```

### Requirements
```
numpy
pandas
matplotlib
scipy
tabulate
jupyter
nbformat
```


## Learning Reflection
- Understood how parameters (μ, Σ, ρ) shape joint distributions.  
- Practiced Bayesian reasoning on real text data without ML libraries.  
- Derived and implemented gradient descent mathematically and in code.  
- Strengthened skills in modular coding, visualization, and collaborative version control.

