# 🎓 Student Score Prediction App

A comprehensive machine learning application that predicts student exam scores based on various performance factors. Built with Streamlit for an interactive user experience.


## 📋 Project Overview

This application provides an interactive interface for:
- **Data Exploration**: Comprehensive EDA and visualizations
- **Model Training**: Linear and Polynomial regression with feature selection
- **Real-time Predictions**: Interactive prediction interface with grade estimation

## ✅ Implemented Requirements

### Core Requirements:
- ✅ Dataset: Student Performance Factors (Kaggle)
- ✅ Data cleaning and preprocessing
- ✅ Basic visualization and EDA
- ✅ Train/test split implementation
- ✅ Linear regression model training
- ✅ Prediction visualization
- ✅ Model performance evaluation

### Bonus Features:
- ✅ Polynomial regression implementation
- ✅ Feature combination experiments
- ✅ Advanced model evaluation metrics
- ✅ Interactive feature selection
- ✅ Real-time prediction interface

## 🚀 Quick Start

### Local Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/student-score-prediction.git
   cd student-score-prediction
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   cd Deplyment_App
   streamlit run app.py
   ```

4. **Open your browser**
   Navigate to: http://localhost:8501

## 📊 Application Features

### Tab 1: 📊 EDA & Visualizations
- Comprehensive data exploration
- Statistical summaries and correlation analysis
- Feature distribution plots
- Top 10 features by correlation with exam scores

### Tab 2: 📈 Model Evaluation & Insights
- **Feature Selection Options:**
  - Manual selection with "Select All" option
  - LassoCV automatic feature selection (cv=5)
- **Model Types:**
  - Linear Regression
  - Polynomial Regression (degree=2)
- **Evaluation Metrics:**
  - R² Score, Adjusted R²
  - MSE, RMSE, MAE
- **Visualizations:**
  - Actual vs Predicted plots
  - Residual plots
  - Feature importance analysis

### Tab 3: 🧮 Prediction
- Interactive input forms for all 19 features
- Real-time prediction generation
- Grade estimation (A+ to D)
- Performance level assessment
- Demo mode for users without trained models

## 🎯 Key Features

### Data Processing:
- Automatic missing value handling
- Categorical feature encoding (one-hot encoding)
- Outlier removal using IQR method
- Feature scaling with StandardScaler
- Consistent preprocessing pipeline

### Model Capabilities:
- Linear regression for simple relationships
- Polynomial regression for non-linear patterns
- LassoCV for automatic feature selection
- Comprehensive evaluation metrics
- Feature importance analysis

### User Interface:
- Modern gradient-based styling
- Responsive design with sidebar navigation
- Interactive widgets (sliders, selectboxes, radio buttons)
- Real-time feedback and status messages
- Professional visualizations

## 📈 Model Performance

The application provides comprehensive model evaluation including:

**Evaluation Metrics:**
- R² Score: Measures how well the model explains variance
- Adjusted R²: Accounts for number of features
- MSE: Mean squared error (lower is better)
- RMSE: Root mean squared error
- MAE: Mean absolute error

**Performance Indicators:**
- 🟢 Excellent (R² ≥ 0.8)
- 🟡 Good (R² ≥ 0.6)
- 🟠 Fair (R² ≥ 0.4)
- 🔴 Poor (R² < 0.4)

## 🔧 Technical Details

### Dataset Information:
- 6,607 student records
- 20 features (7 numerical, 13 categorical)
- Target variable: Exam_Score
- Features include: study hours, attendance, sleep, previous scores, tutoring sessions, family income, teacher quality, etc.

### Preprocessing Pipeline:
1. Missing value imputation
2. Categorical encoding (one-hot)
3. Outlier removal (IQR method)
4. Feature scaling (StandardScaler)
5. Feature selection (Manual/LassoCV)

### Model Architecture:
- Linear Regression: y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
- Polynomial Regression: Includes quadratic terms
- Consistent preprocessing for training and prediction

## 📁 Project Structure

```
Student Score Prediction/
├── Data/
│   └── StudentPerformanceFactors.csv    # Main dataset
├── Deplyment_App/
│   └── app.py                          # Main Streamlit application
├── src/
│   ├── data_cleaning.py                # Data preprocessing functions
│   ├── feature_selection.py            # Feature selection algorithms
│   ├── modeling.py                     # Model training and evaluation
│   ├── visualizations.py               # Plotting and visualization functions
│   └── utils.py                        # Utility functions
├── eda_model_dev/
│   └── Student_Score_Prediction.ipynb  # Jupyter notebook with analysis
├── requirements.txt                     # Python dependencies
├── README.md                           # This file
└── README.txt                          # Detailed documentation
```

## 🚀 Deployment Options

### Streamlit Cloud (Recommended)
1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select this repository
5. Deploy!

### Heroku
1. Create a `Procfile`:
   ```
   web: streamlit run Deplyment_App/app.py --server.port=$PORT --server.address=0.0.0.0
   ```
2. Deploy using Heroku CLI or GitHub integration

### Local Deployment
```bash
streamlit run Deplyment_App/app.py --server.port 8501
```

## 🔍 Troubleshooting

### Common Issues:
- **Predictions seem capped**: Check feature scaling consistency
- **Model training errors**: Verify feature selection
- **Visualizations don't appear**: Check matplotlib backend
- **Import errors**: Ensure all dependencies are installed

### Debug Information:
- The prediction tab includes debug information
- Shows input/output shapes and data ranges
- Displays model type and transformer information
- Provides raw prediction values

## 📚 Educational Value

This project demonstrates:
- Complete machine learning pipeline
- Data preprocessing best practices
- Model evaluation techniques
- Interactive application development
- Feature engineering concepts
- Regression analysis methods

### Covered Topics:
- Regression analysis
- Evaluation metrics
- Feature selection
- Data visualization
- Model comparison
- Real-world application development

## 🎓 Academic Requirements Fulfilled

✅ **Task 1: Student Score Prediction**
- ✅ Dataset: Student Performance Factors (Kaggle)
- ✅ Data cleaning and basic visualization
- ✅ Train/test split implementation
- ✅ Linear regression model training
- ✅ Prediction visualization and evaluation
- ✅ Tools: Python, Pandas, Scikit-learn, Matplotlib

### Bonus Features:
- ✅ Polynomial regression implementation
- ✅ Feature combination experiments
- ✅ Advanced evaluation metrics
- ✅ Interactive user interface

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset: [Student Performance Factors](https://www.kaggle.com/datasets/student-performance-factors)
- Streamlit for the amazing web framework
- Scikit-learn for machine learning tools
- The open-source community for inspiration and tools

---

⭐ **Star this repository if you find it helpful!**

🎯 **Project Status: COMPLETE & PRODUCTION-READY**

This application successfully implements all required features and exceeds the original requirements with additional professional-grade capabilities. 
