Student Name: Gabriel Silva
Student ID: 001321100

Dataset
The dataset exceeds Moodle’s 2GB limit. It is hosted on Google Drive.

Dataset Download Link: https://drive.google.com/drive/folders/1pUvPEczLoS6CZImIffSOzcyRAHBuSolL?usp=sharing

After downloading, place all files inside the data folder in the project root.

Setup

Open a terminal in the project root and run:

pip install -r requirements.txt

How to Run

Follow this order.

Step 1. Data Preparation

Run these scripts first:

python merge_dataset.py
python parse_snort.py
python anomaly_features.py

Step 2. Machine Learning

python anomaly_train_score.py

Step 3. Snort Evaluation

python snort_evaluate.py

Optional figures:

python core_figures.py
python extras.py
python comparisons.py

Step 4. Final Comparison

python metrics_compare.py

Pre-trained Models

These files are included:

isolation_forest.joblib
scaler.joblib
feature_meta.json

You should retrain the model.

Streamlit Dashboard

Run the app:

streamlit run app/app.py

Features:

Home page
Overview of the project

Live detector
Upload a CSV and run both detection methods

Head-to-head comparison
View metrics and charts

Hybrid fusion
Compare OR, AND, weighted, and tiered approaches
