# 🏥 IBM Smart Healthcare Disease Prediction & Analytics Platform

> **An end-to-end healthcare machine learning system for disease-risk prediction, exploratory analytics, explainable AI, and interactive decision support.**

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Scikit--Learn-ML-orange?style=for-the-badge&logo=scikit-learn" alt="Scikit Learn">
  <img src="https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas" alt="Pandas">
  <img src="https://img.shields.io/badge/NumPy-Numerical-013243?style=for-the-badge&logo=numpy" alt="NumPy">
  <img src="https://img.shields.io/badge/SHAP-Explainable%20AI-red?style=for-the-badge" alt="SHAP">
  <img src="https://img.shields.io/badge/LIME-XAI-purple?style=for-the-badge" alt="LIME">
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit" alt="Streamlit">
  <img src="https://img.shields.io/badge/Git-GitHub-black?style=for-the-badge&logo=git" alt="Git">
</p>

---

## 🌟 Project Overview

The **IBM Smart Healthcare Disease Prediction & Analytics Platform** is a team-based major project designed to simulate a real-world healthcare analytics solution.

The platform analyzes structured patient information, performs data cleaning and exploratory analysis, engineers meaningful healthcare features, trains and compares multiple machine learning classification algorithms, evaluates their performance, explains model decisions using SHAP and LIME, and provides an interactive Streamlit dashboard.

The complete workflow follows an end-to-end machine learning lifecycle:

```text
Raw Healthcare Data
        │
        ▼
📥 Data Collection
        │
        ▼
🧹 Data Cleaning
        │
        ▼
📊 Exploratory Data Analysis
        │
        ▼
🧠 Feature Engineering
        │
        ▼
⚙️ Preprocessing
        │
        ▼
🤖 Model Training
        │
        ▼
📈 Model Evaluation
        │
        ▼
🪄 Explainable AI
        │
        ▼
🖥️ Streamlit Dashboard
        │
        ▼
💡 Healthcare Insights
```

---

## Project Report - https://drive.google.com/file/d/1_9ZuqCvcT5wRkNU7mUHtrmVAga33N2LZ/view?usp=sharing


## Documentation - https://docs.google.com/document/d/1lIEAJ52XN7Bz5DdSRCGBraA4Ymdy2tBP/edit?usp=sharing&ouid=118205314568634968757&rtpof=true&sd=true


## 🎯 Project Objectives

* 📥 Collect and prepare healthcare data
* 🧹 Clean and standardize patient records
* 📊 Perform comprehensive exploratory data analysis
* 🧠 Engineer meaningful healthcare-related features
* 🤖 Train multiple classification algorithms
* 🏆 Select the best-performing model
* 📈 Evaluate model performance using multiple metrics
* 🔍 Identify the most influential healthcare factors
* 🪄 Explain individual and global model predictions
* 🖥️ Build an interactive prediction dashboard
* 💡 Generate actionable business/healthcare insights
* 📦 Maintain a professional, modular ML project structure

---

## 🧠 Skills Demonstrated

### 📊 Data Science

* Data Cleaning
* Data Preprocessing
* Exploratory Data Analysis
* Statistical Analysis
* Feature Engineering
* Data Visualization
* Class Distribution Analysis
* Correlation Analysis
* Outlier Analysis

### 🤖 Machine Learning

* Binary Classification
* Logistic Regression
* Decision Trees
* Random Forest
* Extra Trees
* Gradient Boosting
* K-Nearest Neighbors
* XGBoost
* Cross-Validation
* Model Comparison
* Model Selection
* Feature Importance

### 🪄 Explainable AI

* SHAP
* LIME
* Global Feature Importance
* Local Patient-Level Explanations
* Model Interpretability

### 🖥️ Software Development

* Modular Python Architecture
* Reusable Components
* Streamlit Application Development
* Model Serialization
* Pipeline Serialization
* Git/GitHub
* Project Documentation

---

## 🛠️ Technology Stack

| Category                | Technology       |
| ----------------------- | ---------------- |
| 🐍 Programming Language | Python           |
| 📊 Data Processing      | Pandas, NumPy    |
| 📈 Visualization        | Matplotlib       |
| 🤖 Machine Learning     | Scikit-learn     |
| 🚀 Gradient Boosting    | XGBoost          |
| 🪄 Explainable AI       | SHAP, LIME       |
| 🖥️ Dashboard           | Streamlit        |
| 💾 Model Storage        | Joblib           |
| 📓 Development          | Jupyter Notebook |
| 🔧 Version Control      | Git              |
| ☁️ Repository           | GitHub           |

---

## 📌 Dataset Overview

The project uses a healthcare dataset containing:

### Dataset Size

```text
Rows    : 280,985
Columns : 39
```

### 🎯 Target Variable

The target variable is:

```text
label
```

with two classes:

```text
Abnormal
Normal
```

### 🔬 Major Dataset Features

#### 👤 Demographic Features

```text
age
gender
age_level
education_level
employment_status
```

#### ⚖️ Body / Health Features

```text
bmi
bmi_level
age_normalized
```

#### 🩸 Blood & Metabolic Features

```text
HbA1c_level
glucose
cholesterol
triglycerides
hdl
ldl
```

#### ❤️ Cardiovascular Features

```text
blood_pressure
systolic_bp
diastolic_bp
heart_rate
```

#### 🏃 Lifestyle Features

```text
smoking
physical_activity
sleep_hours
stress_level
alcohol_intake
salt_intake
```

#### 🧬 Risk Factors

```text
family_history
low_hdl_cholesterol
high_ldl_cholesterol
high_blood_pressure
```

---

## ⚠️ Data Leakage Audit

One of the most important parts of the project was identifying potential **target leakage**.

During analysis, several fields were found to be strongly associated with the target or directly derived from disease information.

The following columns were therefore excluded from the final predictive feature set:

```text
composite_key
source_dataset
sublabel
disease_flags
diabetes
hypertension
heart_disease
```

### Why?

For example, the analysis showed:

```text
diabetes = Yes
        ↓
Abnormal = 100%
```

This means allowing such a feature into the model could allow the model to effectively "see the answer."

Therefore, these fields were removed to make the ML pipeline more realistic and scientifically defensible.

---

## 📁 Project Architecture

```text
IBM-Healthcare-AI/
│
├── 📂 app/
│   ├── app.py / main.py
│   ├── components/
│   │   └── utils.py
│   ├── pages/
│   │   ├── Home.py
│   │   ├── Dataset_analytics.py
│   │   ├── EDA.py
│   │   ├── Prediction.py
│   │   ├── Model_Performance.py
│   │   ├── Explainability.py
│   │   ├── Business_Insights.py
│   │   └── About.py
│   └── styles/
│       └── style.css
│
├── 📂 data/
│   ├── raw/
│   ├── processed/
│   │   └── clean_data.csv
│   └── external/
│
├── 📂 models/
│   ├── best_model.pkl
│   ├── preprocessor.pkl
│   ├── label_encoder.pkl
│   └── feature_names.pkl
│
├── 📂 notebooks/
│
├── 📂 outputs/
│   ├── figures/
│   ├── metrics/
│   ├── reports/
│   └── explainability/
│
├── 📂 src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── evaluation/
│   ├── explainability/
│   └── utils/
│
├── 📂 tests/
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🧱 Module-by-Module Development

---

## 🟢 Module 1 — Project Setup

### Objective

Create a scalable architecture for the complete healthcare ML project.

### Work Completed

* Created project directory structure
* Separated data, source code, models and outputs
* Created reusable Python modules
* Established model and dataset paths
* Prepared Streamlit application structure

### Architecture

```text
Data
 ↓
Features
 ↓
Models
 ↓
Evaluation
 ↓
Explainability
 ↓
Dashboard
```

### Result

A modular ML architecture was established instead of putting the entire project inside a single notebook.

---

## 🟢 Module 2 — Data Collection & Ingestion

### Objective

Load and validate the healthcare dataset.

### Work Completed

* Loaded healthcare CSV dataset
* Checked dimensions
* Inspected columns
* Examined data types
* Checked missing values
* Checked duplicate records
* Verified target column

#### Dataset Result

```text
Dataset Shape
─────────────
280,985 rows
39 columns
```

### Output

```text
data/processed/clean_data.csv
```

---

## 🟢 Module 3 — Data Cleaning & Encoding

### Objective

Convert raw healthcare records into a consistent machine-learning-ready dataset.

### Cleaning Operations

#### 1️⃣ Missing Values

Missing values were identified and handled through preprocessing pipelines.

#### 2️⃣ Data Types

Numerical and categorical columns were automatically detected.

#### 3️⃣ Categorical Encoding

Categorical variables were processed using:

```python
OneHotEncoder(handle_unknown="ignore")
```

#### 4️⃣ Numerical Processing

Numerical values were handled using:

```python
SimpleImputer(strategy="median")
StandardScaler()
```

#### 5️⃣ Target Encoding

The target was encoded as:

```text
Abnormal → 0
Normal   → 1
```

### Result

A reusable preprocessing pipeline was created and saved:

```text
models/preprocessor.pkl
models/label_encoder.pkl
```

---

## 🟢 Module 4 — Exploratory Data Analysis

### Objective

Understand the underlying healthcare patterns before training machine learning models.

---

### 🎯 Target Distribution

The dataset contained:

```text
Abnormal : 174,101
Normal   : 106,884
```

#### Percentage

```text
Abnormal : 61.96%
Normal   : 38.04%
```

This showed a moderate class imbalance and made metrics such as Precision, Recall and F1-score important.

---

### 🩺 Hypertension Analysis

The analysis showed that hypertension-related values were strongly associated with abnormal cases.

However, because `hypertension` was identified as diagnosis-linked/derived information, it was removed from the refined model.

---

### 🩸 Diabetes Analysis

The strongest leakage signal was:

```text
Diabetes = No
    Abnormal → 40.74%
    Normal   → 59.26%

Diabetes = Yes
    Abnormal → 100%
    Normal   → 0%
```

This was a major reason for removing `diabetes` from the final predictive feature set.

---

### ❤️ Heart Disease Analysis

Heart-disease-related values also showed strong relationships with the target.

Because this feature can encode disease status directly, it was removed from the refined model.

---

### 🚬 Smoking Analysis

```text
Current
Abnormal → 76.99%
Normal   → 23.01%

Former
Abnormal → 69.98%
Normal   → 30.02%

Never
Abnormal → 48.34%
Normal   → 51.66%
```

This showed a meaningful relationship between smoking status and the target.

---

### 🧬 Family History

```text
No
Abnormal → 57.16%
Normal   → 42.84%

Yes
Abnormal → 68.06%
Normal   → 31.94%
```

Family history showed a noticeable association with abnormal cases.

---

### 🧪 Cholesterol Indicators

The project also analyzed:

```text
HDL
LDL
Triglycerides
Cholesterol
Low HDL
High LDL
```

These variables were investigated as potential risk indicators.

---

## Screenshot:
# -----------------
D:\programming\programming\python\age&genderdetection\yolov9\Resume History\Elytespark Major Project\IBM-Healthcare-AI\outputs\figures\age_boxplot.png




## 🟢 Module 5 — Feature Engineering & Preprocessing

### Objective

Create meaningful derived features while ensuring the pipeline is reusable and leakage-aware.

---

### 🧠 Engineered Features

#### Pulse Pressure

```text
pulse_pressure = systolic_bp - diastolic_bp
```

#### Mean Arterial Pressure

```text
MAP ≈ (diastolic_bp + 1/3 × pulse_pressure)
```

#### Cholesterol / HDL Ratio

```text
cholesterol_hdl_ratio = cholesterol / HDL
```

#### LDL / HDL Ratio

```text
ldl_hdl_ratio = LDL / HDL
```

#### BMI × Age

```text
bmi_age = BMI × Age
```

#### Age Normalization

Age was transformed into a normalized numerical representation.

---

### ⚙️ Preprocessing Pipeline

#### Numerical Pipeline

```text
Raw Numeric Data
      ↓
Median Imputation
      ↓
Standard Scaling
      ↓
ML-ready Features
```

#### Categorical Pipeline

```text
Raw Categories
      ↓
Most-Frequent Imputation
      ↓
One-Hot Encoding
      ↓
ML-ready Features
```

---

## 🟢 Module 6 — Machine Learning Model Development

### Objective

Train multiple classification algorithms and identify the strongest baseline model.

### Models Tested

```text
1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Extra Trees
5. Gradient Boosting
6. KNN
7. XGBoost
```

---

## 🏆 Initial Model Evaluation

> These results are from the **initial benchmark before the final leakage-aware refinement**.

| Rank | Model               |     Accuracy |    Precision |       Recall |           F1 |
| ---- | ------------------- | -----------: | -----------: | -----------: | -----------: |
| 🥇   | Random Forest       | **0.909906** | **0.920132** | **0.909906** | **0.907075** |
| 🥈   | Extra Trees         |     0.909648 |     0.919745 |     0.909648 |     0.906824 |
| 🥉   | XGBoost             |     0.909328 |     0.919592 |     0.909328 |     0.906466 |
| 4    | Gradient Boosting   |     0.907557 |     0.917376 |     0.907557 |     0.904683 |
| 5    | KNN                 |     0.885376 |     0.886137 |     0.885376 |     0.883770 |
| 6    | Decision Tree       |     0.868654 |     0.869032 |     0.868654 |     0.868818 |
| 7    | Logistic Regression |     0.868116 |     0.867872 |     0.868116 |     0.867976 |

### 🥇 Initial Best Model

```text
Random Forest
```

Saved as:

```text
models/best_model.pkl
```

---

## 🟢 Module 7 — Model Evaluation

### Objective

Evaluate the selected model beyond simple accuracy.

---

### 📊 Evaluation Metrics

#### Accuracy

Measures the percentage of correctly classified patients.

#### Precision

Measures how many predicted abnormal cases were actually abnormal.

#### Recall

Measures how many actual abnormal cases were detected.

#### F1-score

Balances Precision and Recall.

For healthcare screening, Recall and F1-score are especially important because missing a potentially abnormal case can be more serious than generating an additional false positive.

---

## 📈 Learning Curve Analysis

The learning curve was generated using the Random Forest model.

### Results

| Training Samples | Training F1 | Validation F1 |
| ---------------: | ----------: | ------------: |
|           14,985 |      1.0000 |        0.9045 |
|           44,957 |      1.0000 |        0.9065 |
|           74,929 |      1.0000 |        0.9070 |
|          104,900 |      1.0000 |        0.9069 |
|          149,858 |      1.0000 |        0.9070 |

### Interpretation

The model consistently achieved:

```text
Training F1     ≈ 1.00
Validation F1   ≈ 0.905–0.907
```

This indicates:

* strong learning capacity
* stable validation performance
* some overfitting behavior
* no major deterioration as training data increases

However, the leakage audit is important when interpreting these preliminary results.

---

## 🔍 Feature Importance

The initial Random Forest feature-importance analysis produced the following major features:

| Feature                 | Importance |
| ----------------------- | ---------: |
| `hypertension`          |   0.147696 |
| `diabetes_Yes`          |   0.096241 |
| `heart_disease`         |   0.090632 |
| `diabetes_No`           |   0.088103 |
| `HbA1c_level`           |   0.063297 |
| `triglycerides`         |   0.042045 |
| `sleep_hours`           |   0.034775 |
| `systolic_bp`           |   0.031147 |
| `hdl`                   |   0.027339 |
| `ldl_hdl_ratio`         |   0.026408 |
| `ldl`                   |   0.026218 |
| `salt_intake`           |   0.025247 |
| `alcohol_intake`        |   0.024907 |
| `pulse_pressure`        |   0.024370 |
| `cholesterol_hdl_ratio` |   0.023153 |

### Important Finding

The dominance of:

```text
hypertension
diabetes
heart_disease
```

was another indication that the model was benefiting from diagnosis-linked information.

Therefore, these variables were removed during the leakage-aware refinement.

This is an important part of the project because **a lower but more valid model is preferable to an artificially high score caused by leakage.**

---

## 🟢 Module 8 — Explainable AI

### Objective

Make machine learning predictions understandable.

A healthcare model should not simply return:

```text
Abnormal
```

It should also help answer:

> **Why did the model make this prediction?**

---

## 🪄 SHAP

SHAP was integrated for global and feature-level model interpretation.

### SHAP Provides

* Feature contribution
* Global feature importance
* Positive/negative contribution
* Model behavior visualization

### Generated Outputs

```text
outputs/explainability/
│
├── shap_summary.png
└── shap_bar.png
```

---

## 🔎 LIME

LIME was integrated to explain an individual patient prediction.

Example workflow:

```text
Patient Data
     ↓
ML Model
     ↓
Prediction
     ↓
LIME
     ↓
Important Factors
```

Generated output:

```text
lime_patient_explanation.html
```

---

## 🟢 Module 9 — Interactive Streamlit Dashboard

### Objective

Convert the complete ML pipeline into an interactive application.

---

### 🏠 Home

Provides:

* Project overview
* Dataset size
* Number of features
* Missing values
* Duplicate records
* Dataset preview

---

### 📊 Dataset Analytics

Displays:

* Dataset shape
* Column names
* Data types
* Missing values
* Target distribution
* Dataset preview

---

### 📈 EDA

Displays generated analysis charts including:

* Feature distributions
* Target distribution
* Risk-factor analysis
* Learning curve
* Feature importance

---

### 🤖 Prediction

The user can enter patient information and receive:

```text
Prediction
    ↓
Confidence
    ↓
Class probabilities
    ↓
Patient input summary
    ↓
Engineered feature summary
```

The dashboard uses the **same preprocessing pipeline used during training**, preventing inconsistencies between training and prediction.

---

### 🏆 Model Performance

Displays:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix
* ROC curve
* Precision-Recall curve
* Learning curve
* Feature importance

---

### 🧠 Explainability

Displays:

* SHAP summary
* SHAP feature importance
* LIME patient explanation
* Feature importance report

---

### 💡 Business Insights

The dashboard translates technical model results into practical recommendations.

Examples:

* Early screening of high-risk cases
* Monitoring metabolic indicators
* Lifestyle-risk monitoring
* Prioritization for manual review
* Continuous model monitoring

---

## 🟢 Module 10 — Final Packaging & Deployment

### Planned Finalization

The final stage includes:

* 🧪 Testing
* 📦 Dependency management
* 📄 Technical documentation
* 📝 README finalization
* 🐳 Optional Dockerization
* 🚀 Deployment
* 🔄 CI/CD integration
* 📊 Model monitoring

---

## 📦 Project Deliverables

```text
✅ Clean Dataset
✅ Jupyter Notebook
✅ Data Processing Pipeline
✅ Feature Engineering Pipeline
✅ Trained ML Model
✅ Saved Preprocessor
✅ Label Encoder
✅ Feature Names
✅ Model Evaluation Metrics
✅ Feature Importance
✅ Learning Curve
✅ SHAP Explanations
✅ LIME Explanation
✅ Streamlit Dashboard
✅ Business Insights
✅ Technical Documentation
✅ GitHub Repository
```

---

## 📊 Key Results at a Glance

```text
╔══════════════════════════════════════════╗
║          PROJECT PERFORMANCE             ║
╠══════════════════════════════════════════╣
║ Dataset Size        │ 280,985 × 39      ║
║ Target Classes      │ 2                 ║
║ Models Tested       │ 7                 ║
║ Initial Best Model  │ Random Forest     ║
║ Initial Accuracy    │ 90.99%            ║
║ Initial Precision   │ 92.01%            ║
║ Initial Recall      │ 90.99%            ║
║ Initial F1          │ 90.71%            ║
║ Validation F1       │ ~90.5–90.7%       ║
║ XAI                 │ SHAP + LIME       ║
║ Dashboard           │ Streamlit         ║
╚══════════════════════════════════════════╝
```

> ⚠️ **Evaluation note:** The headline model metrics above are from the initial benchmark. Because the leakage audit subsequently identified diagnosis-linked variables, the final scientifically valid report should use metrics from the **refined leakage-free model** once that model has been retrained and evaluated.

---

## 💡 Business Insights

The analysis suggests several areas that can be valuable for healthcare-oriented decision support.

### 🩸 Metabolic Monitoring

Features such as:

```text
glucose
HbA1c
cholesterol
triglycerides
```

can provide useful information for risk analysis.

### ❤️ Cardiovascular Monitoring

Variables such as:

```text
systolic BP
diastolic BP
heart rate
HDL
LDL
```

can contribute to risk assessment.

### 🏃 Lifestyle Analysis

The platform also considers:

```text
smoking
physical activity
sleep
stress
alcohol
salt intake
```

to provide a broader patient profile.

---

## 🔐 Responsible Healthcare AI

This project is an **academic/portfolio decision-support prototype**.

It should **not** be treated as a medical diagnosis system.

Machine learning predictions should be reviewed by qualified healthcare professionals and validated against appropriate clinical standards before any real-world medical deployment.

---

## 🔮 Future Scope

### 🚀 Technical Improvements

* REST API using FastAPI
* Docker containerization
* Cloud deployment
* CI/CD pipeline
* Model registry
* Automated retraining
* Model drift detection
* Monitoring dashboard

### 🧠 ML Improvements

* Disease-specific models
* Hyperparameter optimization
* Calibration
* Threshold optimization
* Advanced ensemble models
* Fairness analysis
* External validation dataset

### 🏥 Healthcare Improvements

* Patient history tracking
* Doctor authentication
* Role-based access
* Prediction history
* Clinical notes
* Patient-level explanation reports
* Alert system for high-risk cases

---

## ⚙️ Installation

### 1️⃣ Clone Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd IBM-Healthcare-AI
```

### 2️⃣ Create Environment

```bash
python -m venv .venv
```

#### Windows

```powershell
.venv\Scripts\activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run Model Pipeline

```bash
python -m src.models.train
```

### 5️⃣ Run Explainability

```bash
python -m src.explainability.explain
```

### 6️⃣ Launch Dashboard

```bash
streamlit run app/app.py
```

or:

```bash
streamlit run app/main.py
```

---

## 🧪 Recommended Execution Order

```text
01 → Data Collection
        ↓
02 → Data Cleaning
        ↓
03 → EDA
        ↓
04 → Feature Engineering
        ↓
05 → Preprocessing
        ↓
06 → Model Training
        ↓
07 → Evaluation
        ↓
08 → SHAP + LIME
        ↓
09 → Streamlit Dashboard
        ↓
10 → Testing + Deployment
```

---

## 👥 Team Project

This project is designed as a collaborative major project.

### Possible Team Responsibilities

| Area             | Responsibility                |
| ---------------- | ----------------------------- |
| 📥 Data          | Data ingestion & cleaning     |
| 📊 Analytics     | EDA & visualization           |
| 🧠 Features      | Feature engineering           |
| 🤖 ML            | Model training & optimization |
| 📈 Evaluation    | Metrics & validation          |
| 🪄 XAI           | SHAP & LIME                   |
| 🖥️ Frontend     | Streamlit dashboard           |
| 📄 Documentation | Reports & README              |
| 🚀 Deployment    | Packaging & deployment        |

---

## 🏆 Final Project Value

This project demonstrates an end-to-end workflow rather than a standalone machine learning notebook.

It combines:

```text
Data Science
     +
Machine Learning
     +
Explainable AI
     +
Software Engineering
     +
Dashboard Development
     +
Business Analytics
```

The strongest aspect of the project is its **leakage-aware modeling approach**. Rather than focusing only on achieving the highest possible accuracy, the project investigates whether the model is learning legitimate predictive patterns.

---

## 🌐 Project Links

### 📂 GitHub

`<YOUR_GITHUB_REPOSITORY_URL>`

### 📊 Dashboard

`<YOUR_DEPLOYED_STREAMLIT_URL>`

### 📄 Documentation

`<YOUR_DOCUMENTATION_URL>`

---

## 👨‍💻 Project Team

**IBM Smart Healthcare Disease Prediction & Analytics Platform**

> *Built with Python, Machine Learning, Explainable AI and Streamlit.*

---

<p align="center">

### 🏥 Turning Healthcare Data into Explainable Intelligence

**Data → Insights → Prediction → Explanation → Decision Support**

⭐ If you found this project useful, consider giving the repository a star!

</p>
