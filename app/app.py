import os
import pandas as pd
import joblib
from flask import Flask, request, render_template

# Initialize the Flask app
app = Flask(__name__)

# Paths to model, scaler, and data
MODEL_PATH = os.path.join(os.getcwd(), 'models/vehicle_traffic_prediction_model.pkl')
SCALER_PATH = os.path.join(os.getcwd(), 'models/vehicle_traffic_scaler_total.pkl')
HISTORICAL_DATA_PATH = os.path.join(os.getcwd(), 'data/filtered_date_traffic_activity_data.parquet')
FUTURE_DATA_PATH = os.path.join(os.getcwd(), 'data/future_traffic_forecast.parquet')

# Load the model and scaler
rf_model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Load historical and future prediction data
historical_data = pd.read_parquet(HISTORICAL_DATA_PATH)
future_forecast_data = pd.read_parquet(FUTURE_DATA_PATH)

# Prepare the data
historical_data['Date'] = pd.to_datetime(historical_data['Timestamp']).dt.date
future_forecast_data['Date'] = pd.to_datetime(future_forecast_data['ds']).dt.date

# Ensure all columns have time-of-day features for future data
future_forecast_data['Hour'] = pd.to_datetime(future_forecast_data['ds']).dt.hour
future_forecast_data['TimeOfDay_Morning'] = (future_forecast_data['Hour'] < 12).astype(int)
future_forecast_data['TimeOfDay_Afternoon'] = ((future_forecast_data['Hour'] >= 12) & (future_forecast_data['Hour'] < 18)).astype(int)
future_forecast_data['TimeOfDay_Evening'] = ((future_forecast_data['Hour'] >= 18) & (future_forecast_data['Hour'] < 24)).astype(int)
future_forecast_data['TimeOfDay_Night'] = (future_forecast_data['Hour'] < 6).astype(int)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        # Get user input
        site = request.form['site']
        date = request.form['date']  # Input date from the form
        time_of_day = request.form['time_of_day']
        
        # Convert the input date to datetime.date format
        date = pd.to_datetime(date).date()
        
        # Filter the data for the selected site and date
        filtered_data = historical_data[
            (historical_data['Site'].str.strip().str.lower() == site.strip().lower()) &
            (historical_data['Date'] == date)
        ]

        if filtered_data.empty:
            # Check in future forecast data
            filtered_data = future_forecast_data[
                (future_forecast_data['Site'].str.strip().str.lower() == site.strip().lower()) &
                (future_forecast_data['Date'] == date)
            ]

        if filtered_data.empty:
            prediction = f"No data or forecast available for {site} on {date}."
        else:
            # Aggregate directional traffic data
            northbound = filtered_data['Northbound'].sum() if 'Northbound' in filtered_data else 0
            southbound = filtered_data['Southbound'].sum() if 'Southbound' in filtered_data else 0
            eastbound = filtered_data['Eastbound'].sum() if 'Eastbound' in filtered_data else 0
            westbound = filtered_data['Westbound'].sum() if 'Westbound' in filtered_data else 0

            # Process time_of_day into feature vector
            time_mapping = {
                'Morning': [1, 0, 0, 0],
                'Afternoon': [0, 1, 0, 0],
                'Evening': [0, 0, 1, 0],
                'Night': [0, 0, 0, 1],
            }
            time_features = time_mapping[time_of_day]

            # Prepare the input vector for the model
            input_vector = {
                'Northbound': [northbound],
                'Southbound': [southbound],
                'Eastbound': [eastbound],
                'Westbound': [westbound],
                'TimeOfDay_Morning': [time_features[0]],
                'TimeOfDay_Afternoon': [time_features[1]],
                'TimeOfDay_Evening': [time_features[2]],
                'TimeOfDay_Night': [time_features[3]],
                **{f'Site_{s}': int(site.strip().lower() == s.strip().lower()) for s in historical_data['Site'].unique()},
            }

            # Align features with model input
            input_df = pd.DataFrame(input_vector)
            for col in rf_model.feature_names_in_:
                if col not in input_df:
                    input_df[col] = 0
            input_df = input_df[rf_model.feature_names_in_]

            # Predict and inverse scale
            scaled_prediction = rf_model.predict(input_df)[0]
            prediction = scaler.inverse_transform([[scaled_prediction]])[0][0]

    return render_template('index.html', sites=historical_data['Site'].unique(), prediction=prediction)

if __name__ == '__main__':
    # Use port from environment or default to 6000
    port = int(os.environ.get('PORT', 6000))
    app.run(host='0.0.0.0', port=port)
