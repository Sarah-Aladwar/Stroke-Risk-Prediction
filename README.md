# 🧠 Stroke Risk Prediction Using Machine Learning

## 📌 Project Overview
Stroke is a leading cause of death and long-term disability worldwide. Early identification of individuals at high risk can support timely medical intervention and preventive care. This project aims to build a machine learning model that predicts whether an individual is at risk of stroke based on health-related symptoms and demographic information.

The project explores both regression and classification approaches, with a final focus on binary classification due to better generalization and reliability.

## 🎯 Objectives
- Analyze health and symptom data related to stroke risk
- Explore relationships between symptoms, age, and stroke likelihood
- Build and evaluate machine learning models for stroke risk prediction
- Compare regression vs classification approaches
- Prepare a deployable model for future integration

## 📊 Dataset
**Source:** Kaggle – Stroke Risk Prediction Dataset  
**Records:** ~7,000  
**Features:** 18  

### Feature Categories
- **Symptom-Based Binary Features (0/1):**  
  Chest pain, shortness of breath, irregular heartbeat, fatigue, dizziness, edema, high blood pressure, sleep apnea, anxiety, and others
- **Demographic Feature:**  
  Age (continuous numerical variable)
- **Targets:**  
  - At Risk (binary classification)  
  - Stroke Risk (%) (continuous value)

**🔗 Dataset link:**  
[https://www.kaggle.com/datasets/mahatiratusher/stroke-risk-prediction-dataset/data](https://www.kaggle.com/datasets/mahatiratusher/stroke-risk-prediction-dataset/data)

## 🛠️ Tools & Technologies
- **Programming:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
- **Modeling:** Logistic Regression, Random Forest, XGBoost, MLP
- **API:** Flask (basic deployment prototype)

## 🔄 Project Workflow
1. **Data Collection**  
   - Loaded and inspected dataset from Kaggle

2. **Data Cleaning & Preprocessing**  
   - Removed duplicates
   - Verified data types and consistency
   - Handled class imbalance using resampling techniques
   - Engineered additional features (symptom severity score, age groups)

3. **Exploratory Data Analysis (EDA)**  
   - Analyzed age distribution and stroke risk trends
   - Studied symptom frequency and correlations
   - Identified class imbalance and feature importance patterns

4. **Feature Selection**  
   - Used correlation analysis and model-based importance
   - Reduced noise and improved model performance

## 🤖 Machine Learning Models

### Regression Approach
- **Model:** Random Forest Regressor
- **Target:** Stroke Risk (%)
- **Result:** Showed signs of overfitting and poor generalization after tuning

### Classification Approach (Final)
- Converted stroke risk into binary classes:  
  `1` if risk > 50%, `0` otherwise
- **Models evaluated:**
  - Logistic Regression
  - Random Forest Classifier
  - XGBoost

## ✅ Best Model
**Random Forest Classifier**
- Accuracy: ~100%
- Precision, Recall, F1-score: ~1.00
- ROC AUC: ~1.00
- Handled class imbalance effectively

## 📈 Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- ROC AUC
- Confusion Matrix

The classification approach proved more reliable and suitable for this problem than regression.

## 🌐 API & Deployment (Prototype)
- Built a simple Flask API with a `/predict` endpoint
- Accepts JSON input and returns:
  - Binary prediction (At Risk / Not At Risk)
  - Class probabilities
- Included basic HTML interface using Flask and ngrok for testing

## 🚀 Future Improvements
- Add model explainability using SHAP or LIME
- Deploy using cloud platforms (AWS, Render)
- Support batch predictions via CSV uploads
- Improve model validation with real-world medical data
- Enhance UI and accessibility

## 📌 Conclusion
This project demonstrates an end-to-end machine learning workflow, from data preprocessing and EDA to model evaluation and deployment considerations. While regression approaches struggled to generalize, reframing the task as a classification problem led to robust and reliable results. The final model is suitable for further validation and potential integration into a clinical decision-support system.

---
👩‍💻 **Team**
- Sara Ali Mahmoud Ibrahim
- Mariam Gamal Askr
- Basmala Ahmed
- Engy Ahmed

