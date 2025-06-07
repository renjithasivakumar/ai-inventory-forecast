import pandas as pd
from prophet import Prophet
import os

def forecast_items(df):
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df_grouped = df.groupby(['Date', 'Item']).sum().reset_index()

    forecast_results = []
    items = df_grouped['Item'].unique()

    for item in items:
        item_df = df_grouped[df_grouped['Item'] == item][['Date', 'Quantity Sold']]
        item_df = item_df.rename(columns={'Date': 'ds', 'Quantity Sold': 'y'})


        if len(item_df) < 10:
            continue

        model = Prophet()
        model.fit(item_df)

        future = model.make_future_dataframe(periods=14)
        forecast = model.predict(future)

        forecast['Item'] = item
        forecast_results.append(forecast[['ds', 'yhat', 'Item']])

    forecast_df = pd.concat(forecast_results)
    forecast_df.columns = ['Date', 'Predicted Quantity', 'Item']
    forecast_df['Date'] = forecast_df['Date'].dt.strftime('%d/%m/%Y')
    forecast_df['Predicted Quantity'] = forecast_df['Predicted Quantity'].round().astype(int)

    return forecast_df

# ==== ⬇️ Original file-based forecast logic still works ====
if __name__ == "__main__":
    file_path = r'C:\Users\user\PycharmProjects\PythonProject2\sales_data.csv'
    print("File exists?", os.path.exists(file_path))

    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        forecast_df = forecast_items(df)

        output_path = r'C:\Users\user\PycharmProjects\PythonProject2\pub_forecast.xlsx'
        forecast_df.to_excel(output_path, index=False)
        print(f"✅ Forecast saved to {output_path}")

# ==== ⬇️ Streamlit UI below ====
try:
    import streamlit as st

    st.title("Pub Sales Forecast Dashboard")

    uploaded_file = st.file_uploader("Upload your pub sales CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        forecast_df = forecast_items(df)

        st.success("✅ Forecast generated!")
        st.dataframe(forecast_df)

        csv = forecast_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Forecast CSV", csv, "pub_forecast.csv", "text/csv")

except ModuleNotFoundError:
    print("Streamlit not installed — skipping UI. Install with `pip install streamlit` if needed.")
