import pandas as pd
import numpy as np
import streamlit as st
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Load data
data = pd.read_csv('data.csv', parse_dates=['date'], index_col='date')

# Define the ARIMAX model
model = SARIMAX(data['expenses'], exog=data[['cpi', 'gdp_growth']], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
results = model.fit()

# Streamlit app
st.title('ARIMAX Model for Forecasting Expenses')
st.write('Model Summary:')
st.write(results.summary())

# Forecast
forecast_steps = st.slider('Forecast steps:', 1, 12, 3)
forecast = results.get_forecast(steps=forecast_steps, exog=data[['cpi', 'gdp_growth']].iloc[-forecast_steps:])
forecast_df = forecast.summary_frame()

st.write('Forecasted Expenses:')
st.write(forecast_df)

# Allow user to change estimates
new_cpi = st.number_input('New CPI:', value=data['cpi'].iloc[-1])
new_gdp_growth = st.number_input('New GDP Growth:', value=data['gdp_growth'].iloc[-1])
new_data = pd.DataFrame({'cpi': [new_cpi], 'gdp_growth': [new_gdp_growth]})

new_forecast = results.get_forecast(steps=1, exog=new_data)
new_forecast_df = new_forecast.summary_frame()

st.write('New Forecasted Expense:')
st.write(new_forecast_df)





