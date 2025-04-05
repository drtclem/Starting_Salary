from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model
model = joblib.load('best_model.pkl')
#Load the scaler
scaler = joblib.load('scaler.pkl') 

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form inputs
        input_data = [
            float(request.form['feature1']),
            float(request.form['feature2']),
            float(request.form['feature3']),
            float(request.form['feature4'])
        ]
        #feed input data to scaler
        input_data = scaler.transform([input_data])
        #Put the comlumn names back in
        input_data = pd.DataFrame(input_data, columns=['University_Ranking', 'Internships_Completed', 'Certifications','Job_Offers'])
        # Convert to 2D array for model
        prediction = model.predict(input_data)[0]
        
        return render_template('index.html', prediction_text=f"Predicted Starting Salary: ${prediction:,.2f}")
    
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {e}")

if __name__ == "__main__":
    app.run(debug=True)