import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt

# Assuming df_alcohol['Value First Difference'] is your time series data
# and you have already determined the order and seasonal_order parameters

# Example values; replace them with your chosen parameters

# clean prediction
def get_values_from_diff_prediction(df, diff_prediction):
    return df['Value'].iloc[-1] + diff_prediction

# Streamlit App
def main():
    st.title("ML Prediction App")

    # Input: Month
    month = st.selectbox("Select Month", [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ])

    # Input: Year
    year = 2021

    # Convert month to a numerical value
    month_mapping = {
        "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
        "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12
    }
    
    month_numeric = month_mapping[month]
    
    df_alcohol = pd.read_csv('df_alcohol_processed.csv')
    # set Month column as index
    df_alcohol.index = pd.to_datetime(df_alcohol['Month'])

    order = (1, 1, 1)
    seasonal_order = (1, 1, 1, 12)

    # Fit SARIMAX model
    model = SARIMAX(df_alcohol['Value First Difference'], order=order, seasonal_order=seasonal_order)
    results = model.fit()

    # Make prediction
    if st.button("Make Prediction"):
        # forecast_steps = month_numeric - df_alcohol.index[-1].month + (year - df_alcohol.index[-1].year) * 12
        forecast_steps = month_numeric
        print(forecast_steps)
        forecast = results.get_forecast(steps=forecast_steps)
        predicted_values = forecast.predicted_mean
        predicted_values = predicted_values.cumsum()
        predicted_values = predicted_values.apply(lambda x: get_values_from_diff_prediction(df_alcohol, x))
        predicted_values = predicted_values.astype(int)
        print(predicted_values)

        # Display the predicted result for the selected month and year
        last_date = pd.to_datetime(df_alcohol.index[-1]) + pd.DateOffset(months=0)
        
        # Ensure the frequency of the index matches the frequency of your time series data
        predicted_values.index = pd.date_range(start=last_date, periods=forecast_steps, freq='M')

        # Set up the prediction_index using the selected month and year
        prediction_index = pd.date_range(start=f'{year}-{month_numeric}-01', periods=1, freq='M')
        
        try:
            prediction_value = predicted_values.loc[prediction_index].values[0]
            st.success(f"The predicted result for {month} {year} is: {prediction_value}")
        except KeyError:
            st.error("Error: The specified date is not in the predicted data.")


if __name__ == "__main__":
    main()