from flask import Flask, render_template, request, redirect, url_for, session, flash, get_flashed_messages
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import numpy as np

app = Flask(__name__)
app.secret_key = 'secret_key'

model = joblib.load('salary_model.pkl')
transformer = joblib.load('poly_transformer.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    error = None

    if request.method == 'POST':
        try:
            level = request.form['level']
            if not level:
                raise ValueError("Input is required")

            level = float(level)
            if level < 1:
                raise ValueError("Level must not be lower than 1")

            level_transformed = transformer.transform([[level]])
            prediction = round(model.predict(level_transformed)[0], 2)
            session['prediction'] = prediction

            x_plot = np.linspace(1, 10, 100).reshape(-1, 1)
            x_plot_poly = transformer.transform(x_plot)
            y_plot = model.predict(x_plot_poly)

            plt.figure()
            plt.plot(x_plot, y_plot, label="Polynomial Regression Curve")
            plt.scatter([level], [prediction], color='red', label='Position Level')
            plt.title("Position Level vs Predicted Salary")
            plt.xlabel("Level")
            plt.ylabel("Salary")
            plt.legend()

            plot_path = os.path.join('static', 'prediction.png')
            plt.savefig(plot_path)
            plt.close()

        except ValueError as ve:
            flash(str(ve), "error")
        except Exception as e:
            print(f"Unexpected error: {e}")
            flash("Something went wrong. Please enter a valid number.", "error")

        return redirect(url_for("index"))

    prediction = session.pop('prediction', None)
    error_messages = get_flashed_messages(category_filter=["error"])
    error = error_messages[0] if error_messages else None

    return render_template('index.html', prediction=prediction, error=error)

if __name__ == '__main__':
    app.run(debug=True)