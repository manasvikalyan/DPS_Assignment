import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt

# Assuming df_alcohol['Value First Difference'] is your time series data
# and you have already determined the order and seasonal_order parameters

# Example values; replace them with your chosen parameters
df_alcohol = pd.read_csv('C:/github/DPS_Assignment/df_alcohol_processed.csv')
order = (1, 1, 1)
seasonal_order = (1, 1, 1, 12)

# Fit SARIMAX model
model = SARIMAX(df_alcohol['Value First Difference'], order=order, seasonal_order=seasonal_order)
results = model.fit()

# get value form value first difference prediction and return the value
def get_value_from_diff_prediction(df, value):
    return df['Value'].iloc[-1] + value

# Streamlit App
def main():
    st.title("ML Prediction App")

    # Input: Month
    month = st.selectbox("Select Month", [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ])

    # Input: Year
    year = st.number_input("Enter Year", min_value=2000, max_value=2100, value=2021)

    # Convert month to a numerical value
    month_mapping = {
        "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
        "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12
    }
    month_numeric = month_mapping[month]

    # Make prediction
    if st.button("Make Prediction"):
        # Make sure to provide the appropriate exogenous data based on your model
        # In this case, we use a constant as an example; replace it with your actual exogenous data
        exogenous_data = pd.DataFrame({'constant': [1]})
        
        # Forecast future values
        forecast_steps = 12  # You may adjust this based on your needs
        forecast = results.get_forecast(steps=forecast_steps, exog=exogenous_data)
        
        # get the values from the first difference prediction
        forecast_values = forecast.predicted_mean.cumsum()
        
        # get the values from the prediction
        forecast_values = forecast_values.apply(lambda x: get_value_from_diff_prediction(df_alcohol, x))
        
        # convert the values to int
        forecast_values = forecast_values.astype(int)

        # Display the predicted result for the selected month and year
        prediction_index = pd.date_range(start=f'{year}-{month_numeric}-01', periods=1, freq='M')
        prediction_value = forecast_values.loc[prediction_index].values[0]

        st.success(f"The predicted result for {month} {year} is: {prediction_value}")

if __name__ == "__main__":
    main()
