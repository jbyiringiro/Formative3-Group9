Part 1 — Probability Distributions (Bivariate Normal)

Objective: Using a relevant dataset sourced online, compute the probability density values for each data point using the bivariate normal distribution formula. Implement this from scratch without using any statistical libraries in Python.

Requirements:

Implement the bivariate normal PDF manually.

Use only numpy, pandas, and matplotlib for numerical operations and visualisation.

Visualise results using:

Contour Plot — show PDF levels.

3D Plot — show surface of PDF.

Formula Reminder: 
𝑓
(
𝑥
)
=
1
2
𝜋
∣
Σ
∣
exp
⁡
(
−
1
2
(
𝑥
−
𝜇
)
𝑇
Σ
−
1
(
𝑥
−
𝜇
)
)
f(x)=
2π
∣Σ∣
	​

1
	​

exp(−
2
1
	​

(x−μ)
T
Σ
−1
(x−μ))


Deliverables:

part1_bivariate_pdf.py or notebook equivalent.

Contour and 3D plots saved as PNG files.

CSV file with PDF values for each data point.

Academic Integrity Disclaimer: This assignment is meant to assess your understanding and reasoning. You may use AI tools only to clarify concepts or check your logic, not to generate complete solutions or code.

Part 2 — Bayesian Probability (IMDb Movie Reviews)

Objective: Use the IMDb Movie Reviews Dataset to compute posterior probabilities for selected sentiment keywords using Bayes’ Theorem.

Steps:

Choose 2–4 positive keywords and 2–4 negative keywords.

Decide whether to calculate P(Positive | keyword) or P(Negative | keyword) — do not compute both.

Compute and present:

Prior: P(Positive)

Likelihood: P(keyword | Positive)

Marginal: P(keyword)

Posterior: P(Positive | keyword)

Implement Bayes’ theorem using basic Python operations only (no ML libraries).

Deliverables:

A small Markdown/CSV table showing computed probabilities for each keyword.

A clear explanation of your method and keyword reasoning.

Academic Integrity Disclaimer: Use AI tools only to clarify logic or confirm understanding, not to generate full answers or code.

Part 3 — Gradient Descent Manual Calculation

Objective: Manually compute three updates of gradient descent for parameters m and b in a simple linear regression model.

Given:

Linear equation: 
𝑦
=
𝑚
𝑥
+
𝑏
y=mx+b


Initial values: 
𝑚
0
,
𝑏
0
m
0
	​

,b
0
	​




Learning rate: 
𝛼
α


Data points: (x, y) pairs provided by the instructor.

Tasks:

Compute predictions 
𝑦
^
y
^
	​

 using current 
𝑚
m and 
𝑏
b.

Derive gradients of MSE cost function with respect to 
𝑚
m and 
𝑏
b.

Update 
𝑚
m and 
𝑏
b iteratively using gradient descent.

Each member performs at least one update.

Show all steps and intermediate results.

Deliverables:

Manual or handwritten PDF showing calculations.

Explanation of whether parameters are moving toward reducing the error.

Academic Integrity Disclaimer: Do not use AI-generated calculations; understand and perform all mathematical steps yourself.

Part 4 — Gradient Descent in Code

Objective: Convert the manual gradient descent process into Python code using SciPy (if necessary) and visualise the results.

Tasks:

Implement the gradient descent algorithm explicitly (show each iteration clearly).

Compute updated values of m, b, and error across iterations.

Visualise:

Plot 1: m & b vs iteration.

Plot 2: Error vs iteration.

Save results to CSV (iteration, m, b, mse).

Deliverables:

part4_gradient_descent.py or notebook equivalent.
Plots of parameter evolution and error trend.

Academic Integrity Disclaimer: This part measures your Python implementation skills. Use AI only for concept clarification, not to write full code solutions.
