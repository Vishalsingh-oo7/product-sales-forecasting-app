# product-sales-forecasting-app
Built an end-to-end sales forecasting solution using LightGBM, XGBoost, Random Forest, ARIMA, SARIMA, and Prophet. Applied EDA, feature engineering, and time-series cross-validation. Used a weighted ensemble and confidence intervals to support retail demand planning.

---


## 📊 Project Overview

In the competitive retail industry, the ability to predict future sales accurately is crucial for operational and strategic planning. This project focuses on developing a predictive model that uses historical sales data from different stores to forecast sales for upcoming periods. Accurate sales forecasting helps in:

1.  **Inventory Management:** Optimizing stock levels to meet demand without overstocking.
2.  **Financial Planning:** Estimating future revenue and managing budgets effectively.
3.  **Marketing and Promotions:** Planning effective campaigns based on sales peaks and troughs.
4.  **Supply Chain Optimization:** Informing production schedules and logistics.
5.  **Strategic Decision Making:** Supporting broader business strategies like store expansions.

## 📦 Dataset Description

The dataset contains historical sales data with the following attributes:

*   `ID:` Unique identifier for each record.
*   `Store_id:` Unique identifier for each store (365 unique stores).
*   `Store_Type:` Categorization of the store (S1, S2, S3, S4).
*   `Location_Type:` Classification of the store's location (L1, L2, L3, L4, L5).
*   `Region_Code:` Geographical region where the store is located (R1, R2, R3, R4).
*   `Date:` The specific date of the record.
*   `Holiday:` Indicator of whether the date was a holiday (1: Yes, 0: No).
*   `Discount:` Indicates whether a discount was offered (Yes/No).
*   `#Order:` The number of orders received on the specified day.
*   `Sales:` Total sales amount for the store on the given day.

## 🛠️ Methodology and Analysis Steps

### 1. Data Loading and Initial Checks

*   **Libraries:** Imported `pandas`, `numpy`, `seaborn`, `matplotlib`. `pd.set_option('display.max_columns', None)` was used for better display.
*   **Data Loading:** `TRAIN.csv` (188,340 entries) and `TEST_FINAL.csv` (22,265 entries) were loaded into pandas DataFrames.
*   **Data Shape & Types:** The training data has 188,340 rows and 10 columns. Data types were inspected, with `Date` initially `object` and converted to `datetime64[ns]` later.
*   **Missing Values & Duplicates:** No missing values or duplicate rows were found, indicating a clean dataset.

### 2. Exploratory Data Analysis (EDA) and Hypothesis Testing

EDA provided a systematic approach to uncover trends and relationships, while hypothesis testing validated key assumptions.

*   **Descriptive Statistics:** Summarized numerical and categorical features, revealing insights like `Store_Type S1` being the most frequent (`88752` occurrences) and `Date 2019-05-31` having the highest frequency (`365` occurrences).
*   **Sales Distribution:** `Sales` exhibited a right-skewed distribution (Skewness Value: 1.249) with significant outliers, as visualized by histograms and box plots.
*   **Impact of Discounts on Sales:** A t-test (p-value = `0.0`) confirmed a statistically significant impact. Average sales on discount days (`49426.50`) were higher than on non-discount days (`37403.68`).
*   **Effect of Holidays on Sales:** A t-test (p-value = `0.0`) showed a significant impact. Interestingly, non-holiday sales (`43897.29`) were higher than holiday sales (`35451.88`).
*   **Sales Differences Across Store Types:** An ANOVA test (p-value = `0.0`) indicated significant differences. `S4` had the highest average sales (`59945.69`), and `S2` the lowest (`27530.83`).
*   **Regional Sales Variability:** A Kruskal-Wallis test (p-value = `0.0`) confirmed significant sales variability across regions.
*   **Correlation Between Orders and Sales:** A strong positive Spearman correlation (`0.938`, p-value = `0.0`) was found, highlighting `#Order` as a strong sales predictor.
*   **Outlier Treatment:** IQR Winsorization was applied to `Sales`, capping values between `0.0` and `84133.5`, creating `Sales_IQR_Capped` to handle extreme values and improve model stability. Zero-sales days (`0.01%` of data) were identified and preserved.

### 3. Feature Engineering

Extensive feature engineering was performed to enhance model performance:

*   **Calendar Features:** Extracted `Year`, `Month`, `Week`, `Day`, `DayOfWeek`, `Is_Weekend`, and `Quarter` from the `Date` column.
*   **Binary Conversion:** `Discount` (Yes/No) was converted to `1/0`.
*   **Target Variable:** `Sales_IQR_Capped` was chosen as the primary target due to its stabilized variance.
*   **Dropped Features:** `ID` was dropped as an identifier. `Store_id` was also removed due to its negligible correlation (`0.001`) with the target.

### 4. Model Training and Evaluation

Multiple regression models were trained and evaluated using training data and Time-Series Cross-Validation where appropriate. The primary evaluation metrics were MAE, RMSE, and R-squared.

*   **Linear Regression (Baseline):**
    *   **Training Performance:** MAE: `7767.87`, RMSE: `10396.93`, SMAPE: `19.76%`, R-squared: `0.6175`.
    *   **Insight:** Poor performance, indicating inadequacy for complex, non-linear sales dynamics.

*   **Random Forest Regressor:**
    *   **Training Performance:** MAE: `5206.81`, RMSE: `7278.85`, R-squared: `0.8125`.
    *   **Insight:** Strong training performance, but generalization to unseen data is uncertain without Time-Series CV.

*   **XGBoost Regressor:**
    *   **Training Performance:** MAE: `5802.48`, RMSE: `7908.96`, R-squared: `0.7787`.
    *   **Avg Time-Series CV Performance:** MAE: `8393.86`, RMSE: `11314.76`, R-squared: `0.5458`.
    *   **Insight:** Good training, but a notable drop in R-squared during CV, suggesting a generalization gap.

*   **LightGBM Regressor:**
    *   **Training Performance:** MAE: `6174.32`, RMSE: `8336.57`, R-squared: `0.7541`.
    *   **Avg Time-Series CV Performance:** MAE: `7833.09`, RMSE: `10693.22`, R-squared: `0.5963`.
    *   **Insight:** Achieved the **best balance between training accuracy and future generalization** among ML models.

*   **LSTM Model (Deep Learning):**
    *   **Training Performance:** MAE: `12175.02`, RMSE: `15463.98`, R-squared: `0.1412`.
    *   **Insight:** Very poor performance, indicating it was ineffective at capturing the main sales drivers in this dataset. High computational cost with low gain.

### 5. Traditional Time Series Models

*   **ARIMA:** Trained on aggregated daily sales data, forecasts generated for the test period. Provided a baseline for time-dependent patterns.
*   **SARIMA:** Trained with weekly seasonality (`(1,1,1,7)` order) to capture recurring weekly patterns. Forecasts generated for the test period.
*   **Prophet:** A robust time series forecasting model that handles trends, weekly, and yearly seasonality. Provided forecasts with confidence intervals. Key for ensemble's uncertainty estimation.

### 6. Ensemble Techniques

*   **Weighted Ensemble:** A weighted combination of LightGBM (`70%`) and Prophet (`30%`) predictions was created. This approach leverages LightGBM's superior predictive accuracy from feature-rich data and Prophet's ability to model temporal components and provide uncertainty estimates.
*   **Confidence Bands:** Prophet's confidence intervals (`yhat_lower`, `yhat_upper`) were used to generate ensemble confidence bands, providing reliable uncertainty estimates for the combined forecast.

## Overall Insights:

1.  **Data Quality and Preprocessing**: 
    *   The dataset is clean with no missing values or duplicates, providing a solid foundation for analysis.
    *   The target variable, `Sales`, exhibits a right-skewed distribution with significant outliers. This necessitated the use of `Sales_IQR_Capped` to stabilize variance and improve model robustness.
    *   Zero-sales days are rare (approximately 0.01% of the data), indicating that most entries represent active sales periods.

2.  **Key Influencers of Sales (EDA & Hypothesis Testing)**: 
    *   **Discounts** have a statistically significant positive impact on sales, making them a crucial demand-driving factor.
    *   **Holidays** also significantly affect sales, but their impact is not uniform, suggesting the need for nuanced modeling of different holiday types.
    *   **Store Type, Location Type, and Region Code** are vital determinants of sales performance, with significant differences observed across their categories. These categorical features play a critical role in revenue generation.
    *   There is a **strong positive correlation between the number of orders and sales**, confirming that order volume is a robust proxy for sales performance.

3.  **Feature Engineering Effectiveness**: 
    *   Extracting temporal features such as `Year`, `Month`, `Week`, `Day`, `DayOfWeek`, `Is_Weekend`, and `Quarter` from the `Date` column successfully captured seasonal and weekly patterns.
    *   The removal of `ID` and `Store_id` (due to low correlation with sales) helped reduce dimensionality without losing predictive power.
    *   Encoding categorical variables and scaling numerical features were essential steps to prepare the data for various machine learning models.

4.  **Model Performance Comparison**: 
    *   **Linear Regression** performed poorly (R² of 0.6175 on training data), indicating its inadequacy in capturing the complex, non-linear relationships and temporal dynamics inherent in sales data. Its residual analysis showed clear signs of underfitting and heteroscedasticity.
    *   **Tree-based Models (Random Forest, XGBoost, LightGBM)** significantly outperformed Linear Regression, demonstrating their ability to model non-linear relationships and feature interactions effectively.
        *   **Random Forest** achieved the highest training R² (0.8125), but without time-series cross-validation, its generalization to unseen future data is uncertain.
        *   **XGBoost** showed strong training performance (R² of 0.7787) but a notable drop in R² (to 0.5458) during time-series cross-validation, indicating a generalization gap. Residuals revealed heteroscedasticity at higher sales volumes and slight over-prediction for extreme sales.
        *   **LightGBM** demonstrated the best balance between training accuracy (R² of 0.7541) and future generalization, achieving the highest average time-series cross-validation R² (0.5963) and lowest RMSE (10693.22) among the tree-based models. This suggests LightGBM adapts better to changing future sales patterns.
    *   **Deep Learning Models (LSTM)** proved ineffective for this dataset, yielding very low explanatory power (R² of 0.1412) and high prediction errors. The high computational cost coupled with poor performance made it an inefficient choice, largely because sales drivers in this dataset are more feature-dependent (e.g., discounts, store type) than purely sequential.
    *   **Traditional Time-Series Models (ARIMA, SARIMA, Prophet)** are valuable for their explicit handling of trend and seasonality. Prophet, in particular, offered robust capabilities for capturing complex seasonal patterns and providing confidence intervals, even without incorporating external features directly.

5.  **Ensemble Approach for Robustness**: 
    *   A weighted ensemble combining LightGBM (70%) and Prophet (30%) was developed. This strategy leverages LightGBM's superior predictive accuracy from feature-rich data and Prophet's ability to model temporal components and provide uncertainty estimates, resulting in more stable and accurate forecasts with reliable confidence bands.

## Recommendations:

1.  **Primary Model for Deployment: LightGBM (with Ensemble)**:
    *   **Recommendation**: LightGBM should be the primary model for forecasting due to its superior generalization performance observed in time-series cross-validation. For production, the **weighted ensemble (70% LightGBM, 30% Prophet)** is highly recommended to combine predictive accuracy with robust trend/seasonality modeling and uncertainty quantification.
    *   **Actionable Steps**: Implement the trained LightGBM model and the ensemble strategy. Continuously monitor its performance against actual sales data.

2.  **Continuous Feature Engineering and Selection**:
    *   **Recommendation**: Regularly revisit and expand feature engineering efforts. While current temporal and categorical features are strong, further enhancements can be made.
    *   **Actionable Steps**:
        *   **Lagged Features**: Incorporate lagged sales and order values (e.g., sales from 1, 7, 30 days prior) to capture autoregressive patterns more explicitly within the ML models.
        *   **Interaction Features**: Explore interactions between key categorical variables (e.g., `Store_Type` x `Discount`, `Region_Code` x `Holiday`) to capture more nuanced effects.
        *   **Holiday Granularity**: Develop more granular holiday features, distinguishing between different types of holidays (e.g., national holidays, local festivals, long weekends) and their specific impacts on sales.
        *   **Promotional Effectiveness**: Create features that capture the intensity or duration of discounts if available, rather than just a binary "Discount" indicator.

3.  **Outlier Management Strategy**:
    *   **Recommendation**: Continue using the IQR-capping method (`Sales_IQR_Capped`) as a robust approach to handle extreme sales values, which helps stabilize model training and prevent overfitting to rare events.
    *   **Actionable Steps**: Ensure this preprocessing step is consistently applied to all new data prior to inference. Regularly re-evaluate outlier definitions as sales patterns evolve.

4.  **Optimizing External Factors (Discounts and Holidays)**:
    *   **Recommendation**: Leverage insights from the significant impact of discounts and holidays.
    *   **Actionable Steps**:
        *   **Discount Optimization**: Further analyze the optimal timing, frequency, and depth of discounts to maximize sales and profitability.
        *   **Holiday-Specific Planning**: Develop holiday-specific operational and marketing plans, understanding that sales behaviors differ significantly during these periods. Consider creating predictive segments for major holidays.

5.  **Monitoring and Retraining**:
    *   **Recommendation**: Implement a robust monitoring system for forecast accuracy and model drift. Sales patterns can change over time due to market dynamics, competitor actions, or external events.
    *   **Actionable Steps**:
        *   **Performance Dashboards**: Create dashboards to track key metrics (MAE, RMSE, SMAPE) for the ensemble model on new, unseen data.
        *   **Automated Retraining**: Set up an automated pipeline for periodic model retraining (e.g., monthly or quarterly) with the latest data to ensure forecasts remain relevant and accurate.

6.  **Exploration of External Data Sources**:
    *   **Recommendation**: Investigate the integration of external data that could influence sales.
    *   **Actionable Steps**: Consider adding features such as:
        *   **Marketing Spend**: Budget allocated to promotional campaigns.
        *   **Competitor Activity**: Sales or promotional data from key competitors.
        *   **Economic Indicators**: Include macroeconomic data such as inflation rates, consumer confidence indices, and local employment figures.
        *   **Weather Data**: For certain product categories, integrate local weather conditions, especially for seasonal goods.
        *   **Event Data**: Incorporate data on local events, festivals, or public gatherings that could drive increased foot traffic and sales.

7.  **Addressing Regional and Store-Type Heterogeneity**:
    *   **Recommendation**: Since sales vary significantly by `Store_Type`, `Location_Type`, and `Region_Code`, consider developing localized models or using hierarchical forecasting techniques if forecasts are required at finer granularities.
    *   **Actionable Steps**: Explore approaches like hierarchical time series or training separate models for distinct store/region clusters, potentially using transfer learning if data is sparse for certain segments.

## 💻 Technologies Used

*   Python 3.x
*   Pandas
*   NumPy
*   Matplotlib
*   Seaborn
*   Scikit-learn
*   XGBoost
*   LightGBM
*   TensorFlow/Keras (for LSTM)
*   Statsmodels (for ARIMA/SARIMA)
*   Prophet (by Meta)


## Project Structure

```
product-sales-forecasting/
├── model/
│   ├── sales_forecasting_model.pkl
│   ├── scaler.pkl
│   └── X_encoded_columns.pkl
├── sales_prediction_app.py
├── requirements.txt
└── README.md
```

## Setup Instructions

1.  **Clone the repository (if applicable)**:

    ```bash
    git clone <your-repo-link>
    cd product-sales-forecasting
    ```

2.  **Create a Python Virtual Environment** (recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scriptsctivate`
    ```

3.  **Install Dependencies**:

    Install all required Python packages using the `requirements.txt` file:

    ```bash
    pip install -r requirements.txt
    ```

4.  **Place Model Artifacts**:

    Ensure that the following files are present in the `model/` directory (or the same directory as `sales_prediction_app.py` if you adjust the paths):
    *   `sales_forecasting_model.pkl` (the trained machine learning model)
    *   `scaler.pkl` (the fitted StandardScaler object)
    *   `X_encoded_columns.pkl` (list of feature columns used during training)

    These files are typically generated during the training phase of the notebook.

## Running the Flask Application

1.  **Navigate to the project directory**:

    ```bash
    cd product-sales-forecasting
    ```

2.  **Run the Flask app**:

    ```bash
    python sales_prediction_app.py
    ```

    The application will start, typically on `http://127.0.0.1:5000/`.

## API Endpoints

### 1. Health Check

*   **Endpoint**: `/`
*   **Method**: `GET`
*   **Description**: Checks if the API is running.
*   **Response**: `{"status": "running", "message": "Sales Forecasting API is live"}`

### 2. Predict Sales

*   **Endpoint**: `/predict`
*   **Method**: `POST`
*   **Description**: Predicts sales for a given set of features.
*   **Request Body Example** (JSON):

    ```json
    {
        "Store_Type": "S1",
        "Location_Type": "L3",
        "Region_Code": "R1",
        "Date": "2019-06-01",
        "Holiday": 0,
        "Discount": "No",
        "Store_id": 171,
        "ID": "T1188341"
    }
    ```

    **Note**: `Store_id` and `ID` are optional and will be dropped during preprocessing.

*   **Response Example** (JSON):

    ```json
    {
        "predicted_sales": 45123.45
    }
    ```

