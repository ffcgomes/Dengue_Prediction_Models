#!/usr/bin/python
# -*- coding: utf-8 -*-
# Prophet example (with external regressors)

import pandas as pd
from fbprophet import Prophet

# Import the data
df = pd.read_csv('dados/example.csv')
df_reg = pd.read_csv('dados/regressors.csv')
# Split observed and forecasted values
last_obs_day = '2019-05-31'
last_prev_day = '2019-06-05'
reg_past = df_reg[df_reg['data'] <= last_obs_day].reset_index(drop=True).copy()
reg_fut = df_reg[df_reg['data'] > last_obs_day].reset_index(drop=True).copy()
# Join main variable and regressor and Rename columns
df_past = pd.merge(df, reg_past, on='data')
df_past.columns = ['ds', 'y', 'exo']

# Instant a new Prophet object
m = Prophet()
# Add regressor by column name
m.add_regressor('exo')
# Fit model to data
m.fit(df_past)
# Define number of days to forecast
future = m.make_future_dataframe(periods=len(reg_fut))
# Insert column with forecasted vlues from regressor
future['exo'] = df_reg['exo']
# Calculate forecast
forecast = m.predict(future)

# Save all calculated values into CSV file
forecast.to_csv('forecast.csv', index=False)
# Print first forecasted values (today + 4 days)
forecast_select = forecast[(forecast['ds'] > last_obs_day) & (forecast['ds'] <= last_prev_day)]
print(forecast_select[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])

# Plot graphics
m.plot(forecast).savefig('model_forecast_regressor.png')
m.plot_components(forecast).savefig('components_forecast_regressor.png')
