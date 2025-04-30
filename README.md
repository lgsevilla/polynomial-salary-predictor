# Polynomial Salary Predictor

This project is for a machine learning web app that predicts salaries based on position level using **Polynomial Regression**. It is built entirely in Python using Scikit-learn for modeling and Flask for the web interface

---

## 📁 Project Structure

```
.
├── Position_Salaries.csv         # Dataset
├── polynomial_regression.ipynb   # Model training and export
├── salary_model.pkl              # Trained regression model
├── poly_transformer.pkl          # Polynomial feature transformer
├── app.py                        # Flask web application
├── templates/
│   └── index.html                # HTML form for user input
└── static/                       # (optional) for styling
```

---

## 🚀 Getting Started

### 1. Clone the Repo

```bash
git clone https://github.com/YOUR_USERNAME/polynomial-salary-predictor.git
cd polynomial-salary-predictor
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the App

```bash
python app.py
```

Visit [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

---

## 📊 Model specs

- Polynomial Regression, degree = 5
- Trained on `Position_Salaries.csv`

---

## 🛠️ To Do

- [x] Add input validation
- [x] Add error handling
- [x] Add styling (CSS)
- [x] Add prediction plot
- [ ] Deploy to Heroku or Render

---

## 📄 License

MIT — feel free to use and modify.