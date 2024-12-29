import os
import pandas as pd
import joblib
from flask import Flask, request, render_template

# Initialize the Flask app
app = Flask(__name__)

# Load the model and scaler
MODEL_PATH = os.path.join(os.getcwd(), 'models/vehicle_traffic_prediction_model.pkl')
SCALER_PATH = os.path.join(os.getcwd(), 'models/vehicle_traffic_scaler_total.pkl')
rf_model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Load historical data (if needed)
DATA_PATH = os.path.join(os.getcwd(), 'data/filtered_date_traffic_activity_data.parquet')
historical_data = pd.read_parquet(DATA_PATH)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        # Get user input
        site = request.form['site']
        date = request.form['date']
        time_of_day = request.form['time_of_day']
        
        # Process user input into feature vector
        time_mapping = {
            'Morning': [1, 0, 0, 0],
            'Afternoon': [0, 1, 0, 0],
            'Evening': [0, 0, 1, 0],
            'Night': [0, 0, 0, 1],
        }
        time_features = time_mapping[time_of_day]

        # Get forecasted traffic directions for the selected date
        row = historical_data[historical_data['Date'] == date]
        if row.empty:
            prediction = f"No data available for {date}"
        else:
            northbound = row['Northbound'].values[0]
            southbound = row['Southbound'].values[0]
            eastbound = row['Eastbound'].values[0]
            westbound = row['Westbound'].values[0]

            # Ensure eastbound and westbound are set to 0 if they are null/zero
            eastbound = eastbound if eastbound > 0 else 0
            westbound = westbound if westbound > 0 else 0

            # Create input vector
            input_vector = {
                'Northbound': [northbound],
                'Southbound': [southbound],
                'Eastbound': [eastbound],
                'Westbound': [westbound],
                'TimeOfDay_Morning': [time_features[0]],
                'TimeOfDay_Afternoon': [time_features[1]],
                'TimeOfDay_Evening': [time_features[2]],
                'TimeOfDay_Night': [time_features[3]],
                **{f'Site_{s}': int(site == s) for s in historical_data['Site'].unique()},
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
