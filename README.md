# 🏠 Housing Price Prediction

A machine learning project that predicts California housing prices using various regression models.

## 📌 Project Overview

This project uses the California Housing Dataset to predict median house values based on features like location, housing age, income, and population demographics.

## 🛠️ Technologies Used

- Python 3.x
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib

## 📊 Models Implemented

- Linear Regression
- Random Forest Regressor

## 🎯 Results

| Model | R² Score | Accuracy |
|-------|----------|----------|
| Linear Regression | 0.58 | 58% |
| **Random Forest** | **0.8051** | **80.51%** |

**Best Model:** Random Forest with **80.51% accuracy**

## 📈 Visualizations

The project generates the following plots:

- Price Distribution
- Correlation Heatmap
- Features vs Price
- Geographical Map
- Actual vs Predicted Values
- Feature Importance
- Residual Plot
- Model Comparison

## 🚀 How to Run

1. Clone the repository:
git clone https://github.com/abdulahad-ak/Housing-Price-Prediction.git

2. Install dependencies:
pip install -r requirements.txt

3. Run the script:
python housing_prediction.py

## 📁 Project Structure

Housing-Price-Prediction/
├── housing_prediction.py    # Main Python script
├── requirements.txt         # Dependencies
├── README.md               # Project documentation
└── outputs/
    └── plots/              # Generated visualizations

## 📝 Note

Model files (.pkl) are not included due to GitHub size limits. Running the script will automatically generate them.

## 👤 Author

Abdul Ahad Khan Kolachi

## 📄 License

This project is open source and available for educational purposes.
