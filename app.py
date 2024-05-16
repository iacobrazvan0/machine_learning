from flask import Flask, render_template, request
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("trained_model.pkl")

# Load the motorcycle dataset (without the 'Rating' column)
motorcycles_df = pd.read_csv("all_bikez_curated_without_columns.csv")

# Initialize Flask application
app = Flask(__name__)

# Define route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define route for making recommendations
@app.route('/recommend', methods=['POST'])
def recommend():
    # Get input data from the form
    year = request.form.get('Year')
    displacement = request.form.get('Displacement')
    power = request.form.get('Power')
    torque = request.form.get('Torque')
    bore = request.form.get('Bore')
    stroke = request.form.get('Stroke')
    fuel_capacity = request.form.get('Fuel_capacity')
    dry_weight = request.form.get('Dry_weight')
    wheelbase = request.form.get('Wheelbase')
    seat_height = request.form.get('Seat_height')
    category = request.form.get('Category')

    # Create a dictionary to hold the filled-out fields
    features = {}
    if year:
        features['Year'] = float(year)
    if displacement:
        features['Displacement (ccm)'] = float(displacement)
    if power:
        features['Power (hp)'] = float(power)
    if torque:
        features['Torque (Nm)'] = float(torque)
    if bore:
        features['Bore (mm)'] = float(bore)
    if stroke:
        features['Stroke (mm)'] = float(stroke)
    if fuel_capacity:
        features['Fuel capacity (lts)'] = float(fuel_capacity)
    if dry_weight:
        features['Dry weight (kg)'] = float(dry_weight)
    if wheelbase:
        features['Wheelbase (mm)'] = float(wheelbase)
    if seat_height:
        features['Seat height (mm)'] = float(seat_height)
    
    # Convert features to DataFrame
    df = pd.DataFrame([features])
    
    # Filter motorcycles based on provided specifications
    filtered_motorcycles = motorcycles_df.copy()
    for feature, value in features.items():
        filtered_motorcycles = filtered_motorcycles[filtered_motorcycles[feature] == value]

    # If 'Category' is provided, check if any category column has a value of 1
    if category:
        category_columns = [col for col in filtered_motorcycles.columns if col.startswith('Category_')]
        category_filter = filtered_motorcycles[category_columns].sum(axis=1) > 0
        filtered_motorcycles = filtered_motorcycles[category_filter]

    
    # Get names of recommended motorcycles
    # Concatenate 'Brand' and 'Model' columns
    filtered_motorcycles['Brand_Model'] = filtered_motorcycles['Brand'] + ' ' + filtered_motorcycles['Model']

    # Get names of recommended motorcycles
    recommended_motorcycles = filtered_motorcycles['Brand_Model'].tolist()
    
    return render_template('result.html', motorcycles=recommended_motorcycles)


if __name__ == '__main__':
    app.run(debug=True)
