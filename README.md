# Flight Fare Prediction System

A **machine learning-based educational project** designed to predict flight ticket prices using historical data — built to understand real-world data preprocessing, model training, and deployment.

## 🚀 Project Overview

**What is it?**  
This project was developed as a **learning exercise** to explore how machine learning models can be applied to structured datasets. It focuses on analyzing flight fare data and predicting ticket prices based on features like date, route, airline, number of stops, and duration.

**Why build it?**  
- To gain hands-on experience in **data preprocessing**, **feature engineering**, and **model evaluation**.  
- To learn how to **deploy a trained model** using Streamlit for interactive predictions.  
- To understand the **end-to-end workflow** of a machine learning project — from data analysis to real-time user interaction.

## 📂 Repository Structure

```bash
├─ .gitignore
├─ README.md ← (this file)
├─ requirements.txt ← Python dependencies
├─ model.ipynb ← Jupyter notebook: exploration, modelling & evaluation
├─ model_preprocess.py ← Preprocessing script
├─ model.pkl ← Trained model artefact
├─ feature_columns.pkl ← Pickled list of feature-column names used by the model
└─ app.py ← Streamlit or web app front-end to allow interactive predictions
```

> ⚠️ Adjust file names/paths if yours differ.

## 🧠 Key Features & Approach

1. **Data Loading & Exploration**  
   - Study historical flight fare-dataset (from Kaggle)  
   - Visualise trends: fare vs date, stops, airlines, duration etc.

2. **Pre-processing**  
   - Handle missing values, categorical encoding (airline, source, destination, stops)  
   - Feature engineering (e.g., extracting day/month/year, duration in minutes)  
   - Save feature-columns list in `feature_columns.pkl`.

3. **Modelling**  
   - Split into train/test sets  
   - Try regression algorithms (e.g., XGBoost, RandomForest, etc)  
   - Evaluate using metrics like MAE, RMSE, etc  
   - Pick the best performing model and pickle it (`model.pkl`).

4. **Deployment / App**  
   - `app.py` provides a simple UI (Streamlit or similar) where a user enters inputs (date, source, destination, stops, etc) and the system returns a fare prediction in real-time.

## 📦 Installation & Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/Sayan-Mondal2022/flight-fare-prediction-system.git
   cd flight-fare-prediction-system
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # on mac/linux
   # or on Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the streamlit app:
   ```bash
   streamlit run app.py
   ```

5. Interact with the UI and input flight details. The model will output a predicted fare.

## 🛠 Tech Stack

- Language: Python
- Data analysis & modelling: pandas, numpy, scikit-learn, XGBoost
- Deployment: Streamlit app
- Storage: Python pickle for model and feature columns
- Environment management: Virtual env / pip

## 📁 Dataset & Source

- Original dataset: “Flight Fare Dataset” on Kaggle
 [`Flight-fare-dataset`](https://www.kaggle.com/datasets/yashdharme36/airfare-ml-predicting-flight-fares)
- Data columns include: Date_of_Journey, Departure_Time, Arrival_Time, Duration, Stops, Price
- Source code uses preprocessing script `model_preprocess.py` and training notebook `model.ipynb`.
