import pandas as pd
from prophet import Prophet
import os

# Load and check file
file_path = r'C:\Users\user\PycharmProjects\PythonProject2\sales_data.csv'
print("File exists?", os.path.exists(file_path))

df = pd.read_csv(file_path)
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df_grouped = df.groupby(['Date', 'Item']).sum().reset_index()

# Forecast for each item
forecast_results = []
items = df_grouped['Item'].unique()

for item in items:
    item_df = df_grouped[df_grouped['Item'] == item][['Date', 'Quantity Sold']]
    item_df = item_df.rename(columns={'Date': 'ds', 'Quantity Sold': 'y'})

    if len(item_df) < 10:
        continue  # skip if not enough data

    model = Prophet()
    model.fit(item_df)

    future = model.make_future_dataframe(periods=14)
    forecast = model.predict(future)

    forecast['Item'] = item
    forecast_results.append(forecast[['ds', 'yhat', 'Item']])

# Combine forecasts
forecast_df = pd.concat(forecast_results)
forecast_df.columns = ['Date', 'Predicted Quantity', 'Item']

# Make sure 'Date' is datetime type (should already be, but just in case)
#forecast_df['Date'] = pd.to_datetime(forecast_df['Date'])
#forecast_df['Date'] = forecast_df['Date'].dt.date


# Sort by Date
forecast_df = forecast_df.sort_values(by='Date')
forecast_df['Predicted Quantity'] = forecast_df['Predicted Quantity'].round().astype(int)

forecast_df['Date'] = forecast_df['Date'].dt.strftime('%d/%m/%Y')

# Save to Excel
output_path = r'C:\Users\user\PycharmProjects\PythonProject2\pub_forecast.xlsx'
forecast_df.to_excel(output_path, index=False)

print(f"✅ Forecast saved to {output_path}")

