import yfinance as yf
import pandas as pd
import xgboost as xgb
import streamlit as st
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="OptionVision AI", layout="centered")
st.title("🔮 OptionVision AI – Stock Direction Predictor")

ticker = st.text_input("Enter Stock Ticker", value="AAPL").upper()
start_date = "2019-01-01"
end_date = datetime.today().strftime('%Y-%m-%d')

@st.cache_data
def load_data(ticker):
    df = yf.download(ticker, start=start_date, end=end_date)
    df['Return'] = df['Close'].pct_change()
    df['Direction'] = df['Return'].shift(-1) > 0
    df['MA10'] = df['Close'].rolling(10).mean()
    df['MA50'] = df['Close'].rolling(50).mean()
    df['Volatility'] = df['Return'].rolling(10).std()
    df = df.dropna()
    return df

if ticker:
    df = load_data(ticker)

    features = ['MA10', 'MA50', 'Volatility']
    X = df[features]
    y = df['Direction']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    st.metric("Model Accuracy", f"{accuracy * 100:.2f}%")

    latest_data = df[features].iloc[-1:]
    prediction = model.predict(latest_data)[0]
    proba = model.predict_proba(latest_data)[0]

    if prediction:
        st.success(f"📈 Prediction: Stock will go UP with {proba[1]*100:.2f}% confidence")
    else:
        st.error(f"📉 Prediction: Stock will go DOWN with {proba[0]*100:.2f}% confidence")

    with st.expander("📊 Show Raw Data"):
        st.dataframe(df.tail(50))
