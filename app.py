from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('salary_model.pkl')
transformer = joblib.load('poly_transformer.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        try:
            level = float(request.form['level'])
            level_transformed = transformer.transform([[level]])
            prediction = round(model.predict(level_transformed)[0], 2)
        except Exception as e:
            prediction = f"Error: {str(e)}"
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)