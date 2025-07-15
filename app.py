
import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="OptionVision AI", layout="wide")

st.title("📈 OptionVision AI: Stock Prediction App")

# 🎯 User Inputs
ticker = st.text_input("Enter a stock ticker (e.g., AAPL):", value="AAPL")

start_date = st.date_input("Start date", value=datetime.today() - timedelta(days=180))
end_date = st.date_input("End date", value=datetime.today())

if st.button("Run Prediction"):
    try:
        df = yf.download(ticker, start=start_date, end=end_date)

        if df.empty:
            st.warning("⚠️ No data found. Please check the ticker and dates.")
        else:
            st.subheader("📊 Historical Stock Prices")
            st.line_chart(df["Close"])

            # Feature Engineering
            df["Returns"] = df["Close"].pct_change()
            df["Lag1"] = df["Close"].shift(1)
            df = df.dropna()

            # Prediction Model
            X = df[["Lag1"]]
            y = df["Close"]

            model = RandomForestRegressor(n_estimators=100)
            model.fit(X, y)
            df["Predicted_Close"] = model.predict(X)

            st.subheader("🔮 Predicted vs Actual")
            st.line_chart(df[["Close", "Predicted_Close"]])
    except Exception as e:
        st.error(f"Error: {e}")
