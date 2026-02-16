import pandas as pd
import statsmodels.api as sm
import matplotlib
# IMPORTANT: This line must be set BEFORE importing matplotlib.pyplot
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Data Loading and Cleaning
df = pd.read_csv('agents_data_ml.csv', sep=';')
df = df.dropna(axis=1, how='all')
df.columns = df.columns.str.strip()

# Cleaning numeric data: replacing commas with dots and converting to float
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].astype(str).str.replace(',', '.').astype(float)

# 2. Regression Model
# Target variable (y) and features (X)
y = df['accuracy_score']
X = df[['execution_time_seconds', 'cost_per_task_cents', 'task_complexity']]
# Adding a constant (intercept) for the linear equation
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()

# Print regression summary to console
print(model.summary())

# 3. VISUALIZATION 1: Predicted vs Actual
plt.figure(figsize=(10, 6))

# Plotting: Actual accuracy vs Predicted accuracy
plt.scatter(model.fittedvalues, y, alpha=0.5, color='royalblue', label='Observations')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', lw=2, label='Ideal Line')

plt.title(f'Regression Results: Predicted vs Actual\n(R² = {model.rsquared:.3f})', fontsize=14)
plt.xlabel('Predicted Accuracy', fontsize=12)
plt.ylabel('Actual Accuracy', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# SAVE PLOT 1
output_file = 'regression_analysis_plot.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\n✅ PLOT CREATED SUCCESSFULLY!")
print(f"Find the file '{output_file}' in your project directory (PyCharm file tree on the left).")

# ADDITIONAL VISUALIZATION
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# 1. Coefficients Plot (Feature Importance)
params = model.params.drop('const')
params.plot(kind='barh', ax=ax1, color='teal')
ax1.set_title('Impact of Factors on Accuracy (Coefficients)', fontsize=14)
ax1.set_xlabel('Coefficient Value')
ax1.axvline(0, color='black', lw=1)
ax1.grid(axis='x', linestyle='--', alpha=0.7)

# 2. Accuracy Distribution by Complexity (Boxplot)
sns.boxplot(x='task_complexity', y='accuracy_score', data=df, ax=ax2, palette='coolwarm')
ax2.set_title('Accuracy Distribution by Task Complexity', fontsize=14)
ax2.set_xlabel('Complexity Level (1-10)')
ax2.set_ylabel('Accuracy')

plt.tight_layout()
plt.savefig('factors_analysis.png')
print("\n✅ Additional analysis created: 'factors_analysis.png'")