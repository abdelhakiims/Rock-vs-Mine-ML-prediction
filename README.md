Rock vs Mine Prediction

A machine learning model that classifies sonar signals as either rocks or mines using logistic regression, trained on the UCI Sonar dataset.

Dataset
Download the dataset from Kaggle and place it in the /data folder as sonar.csv.

Project Structure

Rock vs Mine/

├── data/

│   └── sonar.csv

├── code.py


└── README.md

Requirements

Install dependencies with:
pip install numpy pandas scikit-learn

How to Run
python code.py

Results

AccuracyTraining~83%
Test~76%

How it Works
The model takes 60 sonar frequency readings as input and predicts whether the object is a rock (R) or a mine (M). Logistic regression is used as the classifier, with a 90/10 train/test split.
