### Rainfall Prediction Using Machine Learning
### Overview
This project uses Machine Learning to predict rainfall based on historical rainfall data. A hybrid XGBoost regression approach was implemented by modeling residuals and combining them with the underlying trend prediction.
The project also includes an interactive Streamlit web application that allows users to generate rainfall predictions based on selected inputs.

### Objectives
Analyze historical rainfall patterns.
Perform data preprocessing and feature engineering.
Build a machine learning model for rainfall prediction.
Evaluate model performance using regression metrics.
Deploy the trained model through a Streamlit application.

### Technologies Used
1.Python
2.Pandas
NumPy
Scikit-learn
XGBoost
Matplotlib
Seaborn
Streamlit
Joblib
Jupyter Notebook / Google Colab

### Machine Learning Approach

The project uses a hybrid prediction approach:
A trend model is used to capture the underlying rainfall trend.
The residuals (difference between actual rainfall and predicted trend) are modeled using XGBoost Regressor.
The final rainfall prediction is calculated by combining the trend prediction and predicted residual.

*Features Used*
State
YEAR
Month
Target
Rainfall
Categorical variables such as State and Month are encoded using One-Hot Encoding, while YEAR is standardized using StandardScaler.

### Model Performance
The final hybrid XGBoost model achieved the following results on the test dataset:
Metric	Result
*R² Score* 0.8242
*Mean Squared Error* (MSE)	5437.91
*Mean Absolute Error* (MAE)	45.37

### Streamlit Application
The trained model is integrated into a Streamlit application that provides an interactive interface for rainfall prediction.

### How to Run
1. Install the required libraries
*pip install pandas,numpy,scikit-learn,xgboost,streamlit,joblib,matplotlib,seaborn*
2. Run the Streamlit application
*streamlit run app.py*
*The application will open in your browser.*

### Future Improvements
Improve model performance through hyperparameter tuning.
Include additional meteorological variables.
Use more recent weather and climate data.
Deploy the application on a cloud platform.
