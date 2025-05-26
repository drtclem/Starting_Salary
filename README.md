# Starting Salary Predictor

A machine learning project to explore the relationship between educational background, skills, and early career experiences on an individual's starting salary. The project includes data exploration, preprocessing, feature selection, model training, and deployment of a web application.

## Web Application

Explore the deployed web app: [https://starting-salary.onrender.com](https://starting-salary.onrender.com)

---

## Dataset Overview

The dataset contains information on 5,000 individuals with 19 features:

- **Academic:** High School GPA, SAT Score, University Ranking, University GPA  
- **Experience:** Internships, Projects, Certifications  
- **Personal Development:** Soft Skills Score, Networking Score  
- **Outcomes:** Job Offers, Career Satisfaction, Starting Salary  
- **Categorical Info:** Gender, Field of Study, Job Level, Entrepreneurship

Target Variable: **Starting_Salary**

---

## Data Processing

- Removed `Student_ID`
- Handled 0 missing values
- Separated into numeric and categorical features
- Used `StandardScaler` to scale numeric features
- Encoded categorical variables using `LabelEncoder`
- Combined into one scaled DataFrame

---

## Exploratory Data Analysis (EDA)

- Count plots for categorical features
- Histograms for numeric features
- Correlation matrix to understand feature interactions

---

## Feature Selection

Used `SelectKBest` with `f_regression` to identify the top 4 most predictive features:

- `University_Ranking`
- `Internships_Completed`
- `Certifications`
- `Job_Offers`

---

## Models Trained

### Basic Models
- **Linear Regression**
- **Ridge Regression**
- **Lasso Regression**

### With GridSearchCV
- **Ridge:** Alpha tuning
- **Lasso:** Alpha tuning
- **ElasticNet:** Alpha and L1 ratio tuning
- **Random Forest Regressor:** Tuning `n_estimators` and `max_depth`

### Evaluation Metrics
- **RMSE** (Root Mean Squared Error)
- **R² Score** (Goodness of Fit)

---

## Final Results

| Model            | Best Params                             | RMSE     | R²     |
|------------------|------------------------------------------|----------|--------|
| Ridge            | alpha=100                               | 14476.11 | 0.0024 |
| Lasso            | alpha=1                                  | 14476.11 | 0.0024 |
| ElasticNet       | alpha=1, l1_ratio=0.1                    | 14480.24 | 0.0018 |
| Linear Regression| --                                       | 14582.74 | 0.0009 |
| Random Forest    | max_depth=5, n_estimators=100            | 14643.55 | -0.0074|

---

## Deployment

- Best model and scaler saved using `joblib`:
  - `best_model.pkl`
  - `scaler.pkl`
- Deployed via Render at: [https://starting-salary.onrender.com](https://starting-salary.onrender.com)

---

## To Run Locally

```bash
# Clone the repo
$ git clone https://github.com/your-username/starting-salary
$ cd starting-salary

# Create virtual environment
$ python -m venv venv
$ source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
$ pip install -r requirements.txt

# Run the app
$ streamlit run app.py
