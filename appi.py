from flask import Flask, render_template, request
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("trained_model.pkl")

# Load the second motorcycle dataset (the one to find similar ratings)
all_motorcycles_df = pd.read_csv("all_bikez_curated_imputed.csv")

# Initialize Flask application
app = Flask(__name__)

# Define route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define route for making recommendations
@app.route('/recommend', methods=['POST'])
def recommend():
    try:
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
        features = {
            'Year': float(year) if year else 0,
            'Displacement (ccm)': float(displacement) if displacement else 0,
            'Power (hp)': float(power) if power else 0,
            'Torque (Nm)': float(torque) if torque else 0,
            'Bore (mm)': float(bore) if bore else 0,
            'Stroke (mm)': float(stroke) if stroke else 0,
            'Fuel capacity (lts)': float(fuel_capacity) if fuel_capacity else 0,
            'Dry weight (kg)': float(dry_weight) if dry_weight else 0,
            'Wheelbase (mm)': float(wheelbase) if wheelbase else 0,
            'Seat height (mm)': float(seat_height) if seat_height else 0,
            'Category_ATV': 0,
            'Category_Allround': 0,
            'Category_Classic': 0,
            'Category_Cross / motocross': 0,
            'Category_Custom / cruiser': 0,
            'Category_Enduro / offroad': 0,
            'Category_Minibike, cross': 0,
            'Category_Minibike, sport': 0,
            'Category_Naked bike': 0,
            'Category_Prototype / concept model': 0,
            'Category_Scooter': 0,
            'Category_Speedway': 0,
            'Category_Sport': 0,
            'Category_Sport touring': 0,
            'Category_Super motard': 0,
            'Category_Touring': 0,
            'Category_Trial': 0,
            'Category_Unspecified category': 0
        }

        # Update the relevant category feature
        category_col = f"Category_{category}"
        if category_col in features:
            features[category_col] = 1

        # Convert features to DataFrame
        df_features = pd.DataFrame([features])

        # Debugging: Print features
        print("Features DataFrame:\n", df_features)

        # Make prediction using the loaded model
        predicted_rating = model.predict(df_features)[0]

        # Debugging: Print predicted rating
        print("Predicted Rating:", predicted_rating)

        # Ensure 'Rating' column exists in the second dataset and is numeric
        all_motorcycles_df['Rating'] = pd.to_numeric(all_motorcycles_df['Rating'], errors='coerce')

        # Filter motorcycles with similar ratings (within 0.10 difference)
        similar_motorcycles = all_motorcycles_df[abs(all_motorcycles_df['Rating'] - predicted_rating) <= 0.10]

        # Sort similar motorcycles by the absolute difference in rating
        similar_motorcycles_sorted = similar_motorcycles.iloc[abs(similar_motorcycles['Rating'] - predicted_rating).argsort()]

        # Get the top 10 recommended motorcycles with similar ratings
        top_motorcycles = similar_motorcycles_sorted.head(10)[['Brand', 'Model', 'Rating']].values.tolist()

        # Debugging: Print top motorcycles and ratings
        print("Top Motorcycles:", top_motorcycles)

        return render_template('result.html', predicted_rating=predicted_rating, motorcycles=top_motorcycles)
    except Exception as e:
        # Debugging: Print exception details
        print("Exception:", str(e))
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)
