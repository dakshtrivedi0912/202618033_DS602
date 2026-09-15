import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

st.set_page_config(page_title="Insurance Cost Dashboard", layout="wide")

@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
    return pd.read_csv(url)

df = load_data()
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

tab1, tab2, tab3 = st.tabs(["Data Exploration", "Hypothesis Testing Lab", "Live Prediction & Diagnostics"])

# ==========================================
# TAB 1: DATA EXPLORATION
# ==========================================
with tab1:
    st.header("1. Exploratory Data Analysis")
    
    # Sidebar Filters (Specific to Tab 1 view)
    st.sidebar.header("Data Filters (Tab 1)")
    age_range = st.sidebar.slider("Select Age Range", int(df['age'].min()), int(df['age'].max()), (18, 64))
    smoker_filter = st.sidebar.multiselect("Select Smoker Status", df['smoker'].unique(), default=df['smoker'].unique())
    
    filtered_df = df[(df['age'] >= age_range[0]) & (df['age'] <= age_range[1]) & (df['smoker'].isin(smoker_filter))]
    
    # Descriptive Metrics
    st.subheader("Descriptive Metrics")
    desc_stats = filtered_df[numeric_cols].describe()
    desc_stats.loc['IQR'] = desc_stats.loc['75%'] - desc_stats.loc['25%']
    desc_stats.loc['skewness'] = filtered_df[numeric_cols].skew()
    desc_stats.loc['kurtosis'] = filtered_df[numeric_cols].kurt()
    st.dataframe(desc_stats)
    
    # Visual Exploration
    st.subheader("Visual Exploration")
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots()
        sns.histplot(filtered_df['charges'], kde=True, ax=ax)
        ax.set_title("Distribution of Medical Charges")
        st.pyplot(fig)
        
    with col2:
        fig, ax = plt.subplots()
        sns.scatterplot(x='bmi', y='charges', hue='smoker', data=filtered_df, ax=ax)
        ax.set_title("BMI vs Charges (colored by Smoker)")
        st.pyplot(fig)
        
    st.write("**Correlation Matrix**")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(filtered_df[numeric_cols].corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# ==========================================
# TAB 2: HYPOTHESIS TESTING LAB
# ==========================================
with tab2:
    st.header("2. Hypothesis Testing Lab")
    
    st.subheader("Hypothesis Test 1: Compare 2 Groups")
    cat_var = st.selectbox("Select Categorical Variable (2 groups)", ['smoker', 'sex'])
    num_var = st.selectbox("Select Continuous Metric", ['charges', 'bmi'])
    
    groups = df[cat_var].unique()
    group1 = df[df[cat_var] == groups[0]][num_var]
    group2 = df[df[cat_var] == groups[1]][num_var]
    
    st.write(f"**H0:** No difference in {num_var} between {groups[0]} and {groups[1]}.")
    st.write(f"**H1:** There is a significant difference.")
    
    # Assumptions Check
    stat1, p1 = stats.shapiro(group1)
    stat2, p2 = stats.shapiro(group2)
    stat_lev, p_lev = stats.levene(group1, group2)
    
    st.write(f"- Shapiro-Wilk (Normality): {groups[0]} p={p1:.4f}, {groups[1]} p={p2:.4f}")
    st.write(f"- Levene's (Equal Variance): p={p_lev:.4f}")
    
    # Test Execution
    if p1 > 0.05 and p2 > 0.05 and p_lev > 0.05:
        stat, p_val = stats.ttest_ind(group1, group2)
        test_used = "Two-Sample t-test"
    else:
        stat, p_val = stats.mannwhitneyu(group1, group2, alternative='two-sided')
        test_used = "Mann-Whitney U Test"
        
    st.write(f"**Test Applied:** {test_used} (p-value: {p_val:.4e})")
    if p_val < 0.05:
        st.success(f"**Conclusion:** Reject H0 at α=0.05. Significant difference exists.")
    else:
        st.info(f"**Conclusion:** Fail to Reject H0 at α=0.05.")
        
    st.divider()
    
    st.subheader("Hypothesis Test 2: One-Way ANOVA (3+ Groups)")
    st.write("**H0:** Average charges are the same across all 4 regions.")
    st.write("**H1:** At least one region has a significantly different average charge.")
    
    regions = [df[df['region'] == r]['charges'] for r in df['region'].unique()]
    stat_anova, p_anova = stats.f_oneway(*regions)
    
    st.write(f"**ANOVA p-value:** {p_anova:.4f}")
    if p_anova < 0.05:
        st.success("**Conclusion:** Reject H0 at α=0.05. Charges differ significantly across regions.")
    else:
        st.info("**Conclusion:** Fail to Reject H0 at α=0.05. No significant difference across regions.")

# ==========================================
# TAB 3: LIVE PREDICTION & DIAGNOSTICS
# ==========================================
with tab3:
    st.header("3. Statistical Modeling & Diagnostics")
    
    # Prepare Data for OLS
    df_encoded = pd.get_dummies(df, drop_first=True)
    X = df_encoded.drop('charges', axis=1)
    X = sm.add_constant(X)
    y = df_encoded['charges']
    
    # Fit OLS Model
    model = sm.OLS(y, X).fit()
    
    st.subheader("Model Formulation & Summary")
    st.write(f"$Y = \\beta_0 + \\beta_1(Age) + \\beta_2(BMI) + ... + \\varepsilon$")
    with st.expander("View Full OLS Summary"):
        st.text(model.summary().as_text())
        
    st.subheader("Live Prediction Tool")
    col_input1, col_input2, col_input3 = st.columns(3)
    
    with col_input1:
        input_age = st.number_input("Age", min_value=18, max_value=100, value=30)
        input_bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
    with col_input2:
        input_children = st.number_input("Children", min_value=0, max_value=10, value=0)
        input_sex = st.selectbox("Sex", ["male", "female"])
    with col_input3:
        input_smoker = st.selectbox("Smoker", ["yes", "no"])
        input_region = st.selectbox("Region", df['region'].unique())
        
    # Construct input vector matching the dummy variables
    input_dict = {
        'const': 1.0,
        'age': input_age,
        'bmi': input_bmi,
        'children': input_children,
        'sex_male': 1 if input_sex == 'male' else 0,
        'smoker_yes': 1 if input_smoker == 'yes' else 0,
        'region_northwest': 1 if input_region == 'northwest' else 0,
        'region_southeast': 1 if input_region == 'southeast' else 0,
        'region_southwest': 1 if input_region == 'southwest' else 0
    }
    
    input_df = pd.DataFrame([input_dict])
    predictions = model.get_prediction(input_df)
    pred_summary = predictions.summary_frame(alpha=0.05)
    
    st.write(f"**Predicted Charge:** ${pred_summary['mean'][0]:,.2f}")
    st.write(f"**95% Confidence Interval (Mean):** [${pred_summary['mean_ci_lower'][0]:,.2f}, ${pred_summary['mean_ci_upper'][0]:,.2f}]")
    st.write(f"**95% Prediction Interval (Observation):** [${pred_summary['obs_ci_lower'][0]:,.2f}, ${pred_summary['obs_ci_upper'][0]:,.2f}]")
    
    st.divider()
    
    st.subheader("Gauss-Markov Diagnostic Checks")
    col_diag1, col_diag2 = st.columns(2)
    
    residuals = model.resid
    fitted = model.fittedvalues
    
    with col_diag1:
        # Linearity & Homoscedasticity
        fig, ax = plt.subplots()
        sns.scatterplot(x=fitted, y=residuals, ax=ax, alpha=0.5)
        ax.axhline(0, color='red', linestyle='--')
        ax.set_title("Residuals vs Fitted (Homoscedasticity)")
        ax.set_xlabel("Fitted Values")
        ax.set_ylabel("Residuals")
        st.pyplot(fig)
        
    with col_diag2:
        # Normality of Residuals
        fig, ax = plt.subplots()
        sm.qqplot(residuals, line='45', fit=True, ax=ax)
        ax.set_title("Q-Q Plot (Normality)")
        st.pyplot(fig)
        
    # Multicollinearity (VIF)
    st.write("**Multicollinearity (VIF for Continuous Predictors)**")
    cont_X = df[['age', 'bmi', 'children']]
    cont_X_const = sm.add_constant(cont_X)
    vif_data = pd.DataFrame()
    vif_data["Feature"] = cont_X_const.columns
    vif_data["VIF"] = [variance_inflation_factor(cont_X_const.values, i) for i in range(cont_X_const.shape[1])]
    st.dataframe(vif_data[vif_data["Feature"] != "const"])