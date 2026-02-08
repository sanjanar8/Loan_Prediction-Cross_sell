# Loan_Prediction-Cross_sell
An end-to-end ML pipeline for loan approval, customer similarity, product cross-sell, and profit prediction. Uses LightGBM for sanctioning, autoencoder + KNN for recommendations, and GLM for profitability, with complete preprocessing and inference workflow.
This system performs three major tasks:

1️⃣ Loan Approval Prediction (LGBM)

A LightGBM model predicts whether a customer's housing loan should be approved, using:
Advanced feature engineering
Target encoding
Label encoding for categorical fields
Outlier-robust scaling

2️⃣ Customer Embedding + KNN for Product Cross-Sell
An Autoencoder learns a 3-dimensional latent representation of each customer.
KNN is then used to retrieve similar customers and recommend likely financial products (e.g., credit cards, gold loans, auto loans).

3️⃣ Profit Estimation (GLM Model)
A Generalized Linear Model predicts expected customer profit:
Constant term added manually to match training structure
Categorical features label-encoded


