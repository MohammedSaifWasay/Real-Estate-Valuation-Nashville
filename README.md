# 🏡 Real Estate Valuation in Nashville: A Data-Driven Model Comparison

## 📌 Abstract
This project focuses on analyzing Nashville housing data to identify undervalued properties using various machine learning models. Through thorough preprocessing, feature engineering, and exploratory data analysis (EDA), we identify key factors influencing property valuation. Multiple regression models were trained and evaluated to determine the most effective method for predicting property values and uncovering profitable investment opportunities.

---

## 📖 Introduction

### 🎯 Project Objective
Assist a real estate firm in Nashville by:
- Identifying undervalued vs. overvalued properties.
- Building regression models to predict housing valuations.
- Recommending the most robust model based on performance.

### 💡 Target Variable
- `Sale Price Compared To Value` was converted to binary:
  - `1`: Undervalued  
  - `0`: Overvalued

---

## 🛠️ Methodology

### 🔧 Data Preprocessing
- Removed irrelevant columns (e.g., Parcel ID, Legal Reference).
- Imputed missing categorical values using mode, and numerical values using median.
- Engineered new features:
  - `Total Value` = Land Value + Building Value  
  - `Price per SqFt` = Total Value / Finished Area

### 📊 Exploratory Data Analysis (EDA)
- **Class Imbalance**: Most properties were overvalued.
- **Key Insights**:
  - Undervalued properties had lower Price per SqFt.
  - Strong correlation between `Finished Area`, `Land Value`, and `Total Value`.
  - Outlier properties with large areas but lower total values may be investment opportunities.

---

## 🤖 Models & Evaluation

### 📈 Models Compared
1. **Linear Regression**
2. **Decision Tree Regressor**
3. **Random Forest Regressor**
4. **Gradient Boosting Regressor**

### 📏 Evaluation Metrics
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score (Goodness of Fit)

### 🏆 Results Summary

| Model                   | MAE      | MSE       | RMSE     | R²    |
|------------------------|----------|-----------|----------|-------|
| Linear Regression       | ~0       | ~0        | ~0       | 1.00  |
| Decision Tree Regressor| 3087.5   | 3.2e+08   | 17940.6  | 0.996 |
| Random Forest Regressor| 2342.5   | 1.2e+09   | 35338.9  | 0.985 |
| Gradient Boosting       | 5933.7   | 5.8e+08   | 24078.7  | 0.993 |

---

## 📌 Key Takeaways

- **Gradient Boosting Regressor** delivered the best balance of accuracy and generalization, making it ideal for valuation prediction.
- **Price per SqFt** and **Finished Area** were major contributors to identifying undervalued homes.
- **Linear Regression** performed suspiciously well (likely overfitting); should be interpreted with caution.

---

## 💬 Recommendations

- Deploy **Gradient Boosting Regressor** to assess property deals.
- Focus on properties with:
  - High finished area but low total value.
  - Lower-than-average Price per SqFt.
- Future improvements:
  - Optimize model hyperparameters.
  - Include regional data and property condition.
  - Validate on external property datasets.

---

## 📚 References

- Friedman, J. H. (2001). *Greedy function approximation: A gradient boosting machine*. [Link](https://www.researchgate.net/publication/2424824)
- James, G., et al. (2013). *An Introduction to Statistical Learning*. [PDF](https://static1.squarespace.com/static/5ff2adbe3fe4fe33db902812/t/6009dd9fa7bc363aa822d2c7/1611259312432/ISLR+Seventh+Printing.pdf)
- Hyndman, R. J., & Koehler, A. B. (2006). *Forecast accuracy measures*. [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0169207006000239)
- Little, R. J. A., & Rubin, D. B. (2019). *Statistical Analysis with Missing Data*. Wiley.

---

## 🧠 Author  
**Mohammed Saif Wasay**  
*Data Analytics Graduate — Northeastern University*  
*Machine Learning Enthusiast | Passionate about turning data into insights*  

🔗 [Connect with me on LinkedIn](https://www.linkedin.com/in/mohammed-saif-wasay-4b3b64199/)

---
