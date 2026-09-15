# Medical Insurance Cost Statistical Dashboard

**Live Application URL:** [Insert your Streamlit Cloud link here for the +5 Bonus Marks!]

## 📌 Project Overview
This end-to-end data science project translates formal mathematical models and inferential statistical tests into an interactive Streamlit web application. It allows users to dynamically explore dataset characteristics, run hypothesis tests, and generate live OLS regression predictions alongside rigorous residual diagnostics.

## 📊 Dataset Summary
The application utilizes the **Medical Insurance Costs** dataset (1,338 observations). The primary objective is to analyze how different patient attributes influence total medical charges.
* **Continuous/Numerical Features:** `age`, `bmi`, `children`, `charges` (Target Variable).
* **Categorical Features:** `sex`, `smoker`, `region` (northeast, northwest, southeast, southwest).

## 🔬 Synthesis of Statistical Findings

**1. Exploratory Data Analysis (EDA)**
* The distribution of medical `charges` is heavily right-skewed, indicating a majority of patients incur lower costs, while a small subset incurs extreme costs.
* Bivariate scatter plots reveal a strong interaction between `bmi` and `smoker` status; high BMI significantly inflates medical charges primarily when the patient is also a smoker.

**2. Hypothesis Testing**
* **Two-Group Comparison (Smokers vs. Non-Smokers on Charges):** The Shapiro-Wilk test indicates the data violates normality assumptions. Utilizing the non-parametric Mann-Whitney U test, we reject the null hypothesis (H0) at α = 0.05. There is a statistically significant difference in medical charges between smokers and non-smokers.
* **One-Way ANOVA (Regional Charges):** Testing average charges across the four geographic regions yields a p-value indicating whether we reject or fail to reject the null hypothesis at α = 0.05 (users can dynamically compute this in Tab 2).

**3. OLS Regression Modeling & Diagnostics**
* **Model Fit:** The multiple linear regression model accounts for a substantial portion of the variance in medical charges, with `smoker_yes`, `age`, and `bmi` serving as the most significant positive predictors (p < 0.05).
* **Gauss-Markov Diagnostics:**
  * **Multicollinearity:** Variance Inflation Factor (VIF) scores for continuous variables (`age`, `bmi`, `children`) are well below 5 (approx ~1.01 to 1.09), indicating no problematic multicollinearity.
  * **Homoscedasticity:** The Residuals vs. Fitted plot displays a distinct funnel shape, indicating heteroscedasticity (variance of errors increases with the fitted values).
  * **Normality of Residuals:** The Q-Q plot shows deviation from the 45-degree line at the upper tail, indicating the residuals are not perfectly normally distributed, likely due to the right-skewness of the target variable.

## 🚀 How to Run the Application Locally

**1. Clone the repository and navigate to the project directory:**
```bash
git clone [Insert your GitHub Repo URL]
cd [Your Repository Name]