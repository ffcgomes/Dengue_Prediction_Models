#!/usr/bin/python
# -*- coding: utf-8 -*-
# Prophet example (simple)

import pandas as pd
from fbprophet import Prophet

# Import the data
df = pd.read_csv('dados/example.csv')
# Rename columns
df.columns = ['ds', 'y']

# Define interval dates
last_obs_day = '2019-05-31'
last_prev_day = '2019-06-05'

# Instant a new Prophet object
m = Prophet()
# Fit model to data
m.fit(df)
# Define number of days to forecast
future = m.make_future_dataframe(periods=365)
# Calculate forecast
forecast = m.predict(future)

# Save all calculated values into CSV file
forecast.to_csv('forecast.csv', index=False)
# Print first 5 forecasted values (today + 4 days)
forecast_select = forecast[(forecast['ds'] > last_obs_day) & (forecast['ds'] <= last_prev_day)]
print(forecast_select[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])

# Plot graphics
m.plot(forecast).savefig('model_forecast.png')
m.plot_components(forecast).savefig('components_forecast.png')
