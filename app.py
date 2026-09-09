import streamlit as st  
import pandas as pd  
import numpy as np  
import joblib  
from sklearn.linear_model import LinearRegression  
  
st.set_page_config(  
    page_title="Rainfall Dashboard",  
    page_icon="🌧️",  
    layout="wide"  
)  
  
st.markdown("""  
<style>  
.stApp {  
    background-color:#0e1117 !important;  
}  
  
header, [data-testid="stHeader"], [data-testid="stSidebar"] {  
    background-color:#0e1117 !important;  
}  
  
html, body, [class*="st-"], p, span, label, h1, h2, h3 {  
    color:white !important;  
}  
  
div[data-baseweb="select"] > div {  
    background-color:#1a1d24 !important;  
    border:1px solid #2a2d35 !important;  
}  
  
div[data-testid="stSelectbox"] div[data-baseweb="select"] span {  
    color:white !important;  
}  
  
ul[role="listbox"] {  
    background-color:#1a1d24 !important;  
}  
  
li[role="option"] {  
    color:white !important;  
    background-color:#1a1d24 !important;  
}  
  
li[role="option"]:hover {  
    background-color:#2a2d35 !important;  
}  
  
.stMetric {  
    background-color:#1a1d24 !important;  
    border:0.5px solid #2a2d35 !important;  
    border-radius:12px;  
    padding:12px;  
}  
  
button[data-baseweb="tab"] {  
    color:white !important;  
}  
  
div[data-testid="stAlert"] {  
    border-radius:8px;  
}  
</style>  
""", unsafe_allow_html=True)  
  
  
class TrendModel:  
  
    def fit(self, data):  
  
        self.models_ = {}  
  
        self.fallback_ = LinearRegression().fit(  
            data[["YEAR"]],  
            data["Rainfall"]  
        )  
  
        for (state, month), grp in data.groupby(["State", "Month"]):  
  
            if len(grp) >= 5:  
  
                lr = LinearRegression().fit(  
                    grp[["YEAR"]],  
                    grp["Rainfall"]  
                )  
  
                self.models_[(state, month)] = lr  
  
        return self  
  
    def predict(self, data):  
  
        preds = np.zeros(len(data))  
  
        for i, (_, row) in enumerate(data.iterrows()):  
  
            key = (row["State"], row["Month"])  
  
            model = self.models_.get(  
                key,  
                self.fallback_  
            )  
  
            preds[i] = model.predict(  
                [[row["YEAR"]]]  
            )[0]  
  
        return preds  
  
  
MONTHS = [  
    "JAN", "FEB", "MAR", "APR",  
    "MAY", "JUN", "JUL", "AUG",  
    "SEP", "OCT", "NOV", "DEC"  
]  
  
  
@st.cache_data  
def load_data():  
  
    return pd.read_csv(  
        "rainfall in india 1901-2015.csv"  
    )  
  
  
@st.cache_resource  
def load_model():  
  
    return joblib.load(  
        "rainfall_hybrid_model.pkl"  
    )  
  
  
df = load_data()  
bundle = load_model()  
  
  
def predict_rainfall(state, year, month):  
  
    input_df = pd.DataFrame({  
        "State": [state],  
        "YEAR": [year],  
        "Month": [month]  
    })  
  
    trend = bundle["trend_model"].predict(  
        input_df  
    )[0]  
  
    residual = bundle["residual_model"].predict(  
        input_df  
    )[0]  
  
    return trend + residual  
  
  
def get_value(state, year, month):  
    """  
    Returns actual rainfall if available in the dataset.  
    Otherwise returns hybrid model prediction.  
    """  
  
    if year <= 2015:  
  
        row = df[  
            (df["SUBDIVISION"] == state) &  
            (df["YEAR"] == year)  
        ]  
  
        if len(row) > 0 and pd.notna(row[month].values[0]):  
  
            return row[month].values[0], True  
  
    return predict_rainfall(  
        state,  
        year,  
        month  
    ), False  
  
  
# -----------------------------  
# SIDEBAR  
# -----------------------------  
  
st.sidebar.title("Dashboard controls")  
  
state = st.sidebar.selectbox(  
    "Select state",  
    sorted(df["SUBDIVISION"].unique())  
)  
  
  
# Forecast period: 2016-2045.  
year = st.sidebar.slider(  
    "Select year",  
    2026,  
    2045,  
    2026  
)  
  
month = st.sidebar.selectbox(  
    "Select month",  
    MONTHS  
)  
  
  
st.sidebar.markdown("---")  
  
with st.sidebar.expander("About this project"):  
  
    st.write(  
        "Trained on the IMD rainfall dataset (1901-2015). "  
        "Uses a hybrid model consisting of a linear trend component "  
        "and XGBoost to capture non-linear patterns. "  
      
    )  
  
  
# -----------------------------  
# MAIN DASHBOARD  
# -----------------------------  
  
st.title("🌧️ Rainfall Prediction Dashboard")  
  
st.caption(  
    "Machine learning based rainfall prediction using IMD dataset (1901-2015)"  
)  
  
  
st.subheader("Rainfall Prediction")  
  
  
value, is_actual = get_value(  
    state,  
    year,  
    month  
)  
  
  
  
# -----------------------------  
# METRICS  
# -----------------------------  
  
c1, c2, c3 = st.columns(3)  
  
c1.metric(  
    "State",  
    state  
)  
  
c2.metric(  
    "Year",  
    year  
)  
  
c3.metric(  
    "Rainfall (mm)",  
    f"{value:.2f}"  
)  
  
  
# -----------------------------  
# COMPARISON WITH 2015  
# -----------------------------  
  
baseline_row = df[  
    (df["SUBDIVISION"] == state) &  
    (df["YEAR"] == 2026)  
]  
  
  
if len(baseline_row) > 0 and pd.notna(  
    baseline_row[month].values[0]  
):  
  
    baseline = baseline_row[month].values[0]  
  
    change_pct = (  
        ((value - baseline) / baseline) * 100  
        if baseline != 0  
        else 0  
    )  
  
    st.caption(  
        f"{change_pct:+.1f}% vs. "  
        f"{month.title()} 2015 "  
        f"({baseline:.1f} mm)"  
    )