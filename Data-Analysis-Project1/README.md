# Data Analysis Project 1. Predicting Stock Returns Using Machine Learning

```python
Data-Analysis-Project1/
├── Data analysis project 1.pdf # The instruction of DAP1 from professor.
└── ...
```

## The Introduction of the Project

### Project Overview

The objective of this project is to develop **a machine learning-based model** that can predict **stock returns** based on **historical data and other relevant financial indicators**. The model should provide insights into future stock price movements, helping investors make informed decisions.

```
Input: Stock History Data and Indicators
Model: the Machine Learning-Based Model
Output: the Predict Stock Returns
```

### Grading and Submission

**At the group-level:** Focus on data collection and preprocessing, but have the model and prediction at the same time.

- **10% credits** will be granted based on the completion and degree of perfection.
- Submit the **program (code) used for data collection and preprocessing**.
- Submit a **short summary of data collection and preprocessing.**
- Submit any other thing that you are comfortable to share at the group-level.

**At the individual-level:** Focus on increasing the performance of model.

- **10% credits** will be granted based on the completion and machine learning performance.
- Submit the **program (code) used for individual analysis** (anything that your group did not share).
- Submit **a short summary** of (1) additional data sources that are not shared by the group (2) descriptions of feature selection (3) descriptions of model selection (4) report your model performance (5) economic interpretation on important features and its effects on stock return.

### Projuect Goals

- **Develop Predictive Models:** Create machine learning models to predict short-term and long-term stock returns.
- **Feature Engineering:** Identify and select relevant features that impact stock returns, such as historical prices, trading volume, macroeconomic indicators, and company-specific factors.
- **Backtesting and Validation:** Implement rigorous backtesting to evaluate model performance over historical data and validate its accuracy on out-of-sample data.
- **Deployment:** Deploy the model as an interactive tool or API for real-time predictions.



## Our steps in the Project

> 核心参考文献：Leippold, M., Wang, Q., & Zhou, W. (2022). Machine learning in the Chinese stock market. Journal of Financial Economics, 145(2), 64-82. [**文章解读**](https://zhuanlan.zhihu.com/p/519556712)
>
> 文章核心的 Motivation 是针对以下三点展开探讨
>
> - **Technical indicators emerging from collectivistic investment behavior** matter more for asset pricing than firm fundamentals.
> - Whether return predictability and portfolio performance are compromised for **SOEs where government signaling plays such a prominent role**.
> - Analyze **long-only portfolios**, which are more relevant from a practitioner’s viewpoint.

### Step 1. Data Collection

> 对于参考论文来说，因为没有直接可用的数据库，所以作者是自己组建的：
>
> - To this end, we obtain **daily and monthly total stock returns** for **all A-share stocks** listed on the Shanghai and Shenzhen stock ex-changes from the Wind Database, the largest financial data provider in China. 
> - The corresponding **quarterly financial statement data** are downloaded from the China Stock Market and Accounting Research (CSMAR) database.
> - Our data sample covers more than 3,900 A-share stocks traded from January 2000 to June 2020. 
> - Also, we obtain the yield rate for the **one-year government bon**d in China from CSMAR to proxy for the **risk-free rate**, which is necessary for calculating individual excess returns.
>
> **对于文中的因子，主要有如下三个部分**
>
> ***Part 1.*** 自己所构造的因子 (94 个特征)
>
> - To avoid outliers, we cross-sectionally rank all continuous stock-level characteristics period-by-period, and map them into the [−1, 1] interval following Kelly et al. (2019) and Gu et al. (2020). (横截面排序后正负一标准化，避免异常值)
> - In terms of data frequency, 22 stock-level characteristics are updated monthly, 51 are updated quarterly, 6 are updated semi-annually, and 15 are updated annually.
>
> ***Part 2.*** 80 个行业虚拟变量
>
> Include **80 industry dummies** based on the Guidelines for Industry Classification of Listed Companies issued by the China Securities Regulatory Commission (CSRC) in 2012. 
>
> ***Part 3.*** 11 个宏观经济预测因子
>
> - dividend price ratio (dp), dividend payout ratio (de), earnings price ratio (ep), book-to-market ratio (bm), net equity expansion (nits), stock variance (svar), term spread (tms), and inflation (infl). 
> - monthly turnover (mtr), M2 growth rate (m2gr), and international trade volume growth rate (itgr).







### Step 2. Data Preprocessing

- **Data Cleaning:** Handle missing data, outliers, and anomalies.
- **Label Generating:** Generating the label.
- **Normalization/Standardization:** Apply appropriate scaling to ensure model performance.
- **Feature Engineering:** Create additional features such as lagged returns, volatility measures, and sentiment analysis from news data.



### Step 3. Model Development

**Model Selection:** Evaluate different machine learning models, including but not limited to:

- **Linear Models:** Linear Regression, Lasso, Ridge
- **Tree-based Models:** Random Forest, Gradient Boosting, XGBoost
-  **Support Vector Machines (SVM)**
- **Neural Networks:** Recurrent Neural Networks (RNN), Long Short-Term Memory (LSTM)

**Training:** Train models using historical data with a focus on minimizing overfitting and improving generalization.

**Hyperparameter Tuning:** Use techniques like grid search or Bayesian optimization to find optimal parameters.



### Step 4. Model Evaluation

**Performance Metrics:** Evaluate model performance using metrics such as:

- **Mean Squared Error (MSE)**
- **Mean Absolute Error (MAE)**
- **R-squared (R²)**
- **Corrlation (Corr)**
- **Sharpe Ratio ?**