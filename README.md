# Agentic AI Performance: Regression Analysis Workflow 🚀

This repository contains a machine learning workflow designed to quantify and predict the performance of Agentic AI systems based on resource allocation and task difficulty.

## 🧠 1. Research Hypotheses

We investigate the relationship between operational costs, execution speed, and the resulting quality of AI outputs.

*   **Core Hypothesis ($H_1$):** AI agent performance (measured by `accuracy_score`) is significantly improved by increasing resource allocation (higher `execution_time_seconds` and `cost_per_task_cents`), while being negatively impacted by the inherent difficulty of the work (`task_complexity`).
*   **Null Hypothesis ($H_0$):** Resource allocation (time and cost) has no statistically significant impact on accuracy compared to the dominant effect of task complexity.

---

## 🛠 2. Technical Project Description

### Objective
To build a transparent regression model that quantifies the trade-offs between **speed, cost, and quality** in agentic workflows.

### Methodology
1.  **Data Preprocessing:** The workflow cleans raw agent telemetry data, handles semicolon delimiters, and standardizes numeric formats (converting European-style commas to decimal points) to ensure mathematical compatibility.
2.  **Statistical Modeling:** We utilize **Ordinary Least Squares (OLS) Regression**. This model was chosen for its interpretability, allowing us to see the exact weight (coefficient) of each variable.
3.  **Validation:** The model's performance is validated using the **R-squared ($R^2$)** metric, which measures the proportion of variance for the target variable that's explained by the independent variables.

---

## 📊 3. Interpretation of Visual Results

The analysis generates two primary visualization files that illustrate the model's findings.

### A. Model Reliability
**File:** `regression_analysis_plot.png`

*   **Description:** This plot compares the model's "Predicted Accuracy" against the "Actual Accuracy" observed in the data.
*   **Result:** The high **$R^2 = 0.877$** indicates that **87.7%** of the fluctuations in accuracy are explained by our three features. The tight grouping of observations around the **"Ideal Line"** proves the model is highly reliable for forecasting and decision-making.

### B. Impact Factors and Distribution
**File:** `factors_analysis.png`

This visualization contains two sub-plots:

1.  **Impact of Factors (Coefficients):**
    *   **Task Complexity:** Shows a clear and dominant negative impact. As complexity rises, accuracy drops significantly.
    *   **Cost/Time:** While resources are present in the model, the statistical summary indicates they are far less influential than the task's inherent difficulty.
    
2.  **Distribution by Task Complexity (The "Staircase"):**
    *   **Description:** A boxplot showing the range of accuracy for every complexity level (1-10).
    *   **Result:** This chart confirms a **"Linear Decay."** It visualizes the **reliability gap**: at low complexity, accuracy is high and stable; at high complexity, accuracy is not only lower but also much more volatile (indicated by wider boxes and more outliers).

---

## 🏁 4. Final Conclusion
Based on the regression analysis, the **Null Hypothesis ($H_0$) is partially accepted**. 

Our data proves that **Task Complexity is the "Invisible Ceiling."** Simply increasing execution time or budget (API costs) does not compensate for high complexity. For developers and stakeholders, this suggests that future optimizations should focus on **decomposing complex tasks into simpler sub-tasks** rather than merely increasing computational resources.

---

## 💻 5. How to Run
1.  Clone the repository.
2.  Install dependencies:
    ```bash
    pip install pandas statsmodels matplotlib seaborn
    ```
3.  Ensure your data is named `agents_data_ml.csv` in the root directory.
4.  Run the script:
    ```bash
    python regression_workflow.py
    ```
