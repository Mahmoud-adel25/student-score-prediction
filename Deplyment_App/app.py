import os
import sys
import streamlit as st # type: ignore
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

# Setup path to use src folder
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_path)

from src.data_cleaning import clean_data
from src.feature_selection import get_features_lasso
from src.modeling import (
    train_and_evaluate, 
    train_and_save_model,
    get_model_metrics,
    get_enhanced_metrics,
    get_feature_importance,
    plot_actual_vs_predicted, 
    plot_residuals,
    predict_with_preprocessing
)
from src.visualizations import plot_eda, plot_feature_importance, plot_top_correlations
from src.utils import load_data

# --- Load and clean data ---
# --- Load and clean data ---
df = load_data("Data/StudentPerformanceFactors.csv")
df_clean = clean_data(df)

# Store in session_state for later use
st.session_state.df_clean = df_clean
st.session_state.df_original = df

df_clean_viz = clean_data(df, use_onehot=False, normalize=False, remove_outliers=True)
target_col = "Exam_Score"

# Cache default feature list
all_features = df_clean.columns.drop(target_col).tolist()

# App layout
st.set_page_config(
    page_title="Student Score Prediction", 
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🎓"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    
    .info-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    
    .success-box {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 16px;
        color: #262730;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #667eea;
        color: white;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    .stSelectbox > div > div {
        border-radius: 8px;
    }
    
    .stSlider > div > div > div > div {
        border-radius: 8px;
    }
    
    .stRadio > div > div > div > div {
        border-radius: 8px;
    }
    
    .stCheckbox > div > div > div > div {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar with app information
with st.sidebar:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 10px; color: white; margin-bottom: 1rem;">
        <h3 style="margin: 0; color: white;">🎓 App Info</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("**Dataset:** Student Performance Factors")
    st.markdown("**Features:** 19 student factors")
    st.markdown("**Target:** Exam Score")
    st.markdown("**Models:** Linear & Polynomial Regression")
    
    st.markdown("---")
    
    st.markdown("**📊 Navigation:**")
    st.markdown("1. **EDA Tab:** Explore data patterns")
    st.markdown("2. **Model Tab:** Train and evaluate models")
    st.markdown("3. **Prediction Tab:** Make predictions")
    
    st.markdown("---")
    
    st.markdown("**🔧 Features:**")
    st.markdown("• Feature selection (Manual/LassoCV)")
    st.markdown("• Model evaluation metrics")
    st.markdown("• Interactive visualizations")
    st.markdown("• Real-time predictions")

# Main header with gradient
st.markdown("""
<div class="main-header">
    <h1>🎓 Student Score Prediction App</h1>
    <p style="font-size: 1.2rem; margin-top: 0.5rem;">Predict exam scores based on student performance factors</p>
</div>
""", unsafe_allow_html=True)

# Tabs with enhanced styling
tab1, tab2, tab3 = st.tabs(["📊 EDA & Visualizations", "📈 Model Evaluation & Insights", "🧮 Prediction"])

# --- TAB 1: EDA ---
with tab1:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 10px; color: white; margin-bottom: 2rem;">
        <h2 style="margin: 0; color: white;">📊 Exploratory Data Analysis</h2>
        <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Explore the dataset and understand key patterns</p>
    </div>
    """, unsafe_allow_html=True)
    plot_eda(df_clean)
    st.subheader("Top 10 Features by Correlation with Exam Score")
    plot_top_correlations(df_clean_viz, target_col)
    st.markdown("---")
# --- TAB 2: Modeling & Insights ---
with tab2:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 1rem; border-radius: 10px; color: white; margin-bottom: 2rem;">
        <h2 style="margin: 0; color: white;">📈 Model Evaluation & Insights</h2>
        <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Select features, train models, and analyze performance</p>
    </div>
    """, unsafe_allow_html=True)

    # Feature selection with enhanced styling
    st.markdown("### 🎯 Feature Selection")
    st.markdown("Choose how to select features for your model:")
    
    method = st.radio("Feature Selection Method", ["Manual", "LassoCV (cv=5)"], horizontal=True)
    if method == "Manual":
        select_all = st.checkbox("Select All Features")
        if select_all:
            selected_features = all_features
        else:
            selected_features = st.multiselect("Choose features", all_features)
    else:
        selected_features = get_features_lasso(df_clean, target_col)
        st.success("LassoCV selected features:")
        #WRITE FEATURES IN BEAUTIFUL WAY
        st.write(", ".join(selected_features))

    # Model selection with enhanced styling
    st.markdown("### 🤖 Model Selection")
    st.markdown("Choose the type of regression model to train:")
    
    available_models = [
        "Linear Regression", 
        "Polynomial Regression (degree=2)"
    ]
    
    model_type = st.selectbox(
        "Select Regression Model", 
        available_models,
        help="Choose between Linear Regression for simple relationships or Polynomial Regression for non-linear patterns."
    )
    
    # Show model availability
    st.caption(f"Available models: {len(available_models)} models loaded")
    
    # Model description
    model_descriptions = {
        "Linear Regression": "Simple linear relationship between features and target",
        "Polynomial Regression (degree=2)": "Captures non-linear relationships with quadratic terms"
    }
    
    st.info(f"**{model_type}**: {model_descriptions[model_type]}")
    

    
    st.markdown("---")

    # Train and visualize
    if selected_features:
        # Model choice (Linear or Polynomial)
        model_name = model_type

        st.info(f"Training {model_name} with {len(selected_features)} features...")
        
        # Evaluation with selected model
        try:
            r2, mse, y_test, y_pred, model, transformer = train_and_evaluate(df_clean, selected_features, "Exam_Score", model_name)
            st.success(f"✅ {model_name} trained successfully!")
        except Exception as e:
            st.error(f"❌ Error training {model_name}: {str(e)}")
            st.stop()

        st.session_state.model = model
        st.session_state.transformer = transformer
        lin_model = model  # Store the model for predictions
        st.session_state.features = selected_features
        st.session_state.model_type = model_name
        
        # Show model type confirmation
        st.success(f"🎯 Model Type: {model_name}")
        if transformer:
            st.info(f"Transformer: {type(transformer).__name__}")
        else:
            st.info("Transformer: None")
        
        # Display enhanced model metrics
        st.markdown("### 📊 Model Evaluation Metrics")
        
        # Get enhanced metrics
        enhanced_metrics = get_enhanced_metrics(y_test, y_pred)
        
        st.caption("Comprehensive evaluation metrics to assess model performance. Higher R² and lower error values indicate better predictions.")
        
        # Show metrics in columns
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                label="📈 R² Score", 
                value=f"{enhanced_metrics['R²']:.3f}", 
                help="R² indicates how well the model explains the variability of the outcome."
            )
            st.metric(
                label="📊 Adjusted R²", 
                value=f"{enhanced_metrics['Adjusted R²']:.3f}", 
                help="Adjusted R² accounts for the number of features in the model."
            )
        with col2:
            st.metric(
                label="📉 RMSE", 
                value=f"{enhanced_metrics['RMSE']:.2f}", 
                help="Root Mean Squared Error — lower is better."
            )
            st.metric(
                label="📏 MAE", 
                value=f"{enhanced_metrics['MAE']:.2f}", 
                help="Mean Absolute Error — average absolute difference."
            )
        with col3:
            st.metric(
                label="📊 MSE", 
                value=f"{enhanced_metrics['MSE']:.2f}", 
                help="Mean Squared Error — average squared difference."
            )
            
            # Model performance indicator
            if enhanced_metrics['R²'] >= 0.8:
                performance = "🟢 Excellent"
            elif enhanced_metrics['R²'] >= 0.6:
                performance = "🟡 Good"
            elif enhanced_metrics['R²'] >= 0.4:
                performance = "🟠 Fair"
            else:
                performance = "🔴 Poor"
            
            st.metric(
                label="🎯 Performance", 
                value=performance,
                help="Overall model performance based on R² score."
            )

        st.markdown("### 📊 Model Visualizations")

        st.markdown("**📈 Actual vs Predicted Plot**")
        st.caption("This plot shows how well the model's predictions match actual exam scores. Points closer to the diagonal line = better predictions.")
        fig1 = plot_actual_vs_predicted(y_test, y_pred)
        st.pyplot(fig1)
        
        # Add interpretation for actual vs predicted
        st.markdown("**💡 Interpretation:**")
        st.markdown("""
        - **Diagonal line** = Perfect predictions (actual = predicted)
        - **Points above line** = Model under-predicts (actual > predicted)
        - **Points below line** = Model over-predicts (actual < predicted)
        - **Closer to diagonal** = Better model performance
        """)

        st.markdown("**📊 Residual Plot**")
        st.caption("This plot shows prediction errors. Random scatter around zero = good model. Patterns = potential issues.")
        fig2 = plot_residuals(y_test, y_pred)
        st.pyplot(fig2)
        
        # Add interpretation for residuals
        st.markdown("**💡 Interpretation:**")
        st.markdown("""
        - **Centered around 0** = Good model (errors are random)
        - **Bell-shaped curve** = Normal error distribution
        - **Wide spread** = Less accurate predictions
        - **Patterns** = Model may be missing important relationships
        """)

        # Feature importance analysis
        st.markdown("### 🎯 Feature Importance Analysis")
        st.caption("Shows the impact of each feature on the model's predictions.")
        
        # Get feature importance for supported models
        feature_importance = get_feature_importance(model, selected_features, model_name)
        
        if feature_importance:
            # Sort features by importance 
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse= True)
            st.write("**Top Features by Importance:**")
            #highest is above
            
            # Display top features with explanations
            st.markdown("**🎯 Top 10 Most Important Features:**")
            st.caption("These features have the strongest impact on exam score predictions. Higher values mean greater importance.")
            
            for i, (feature, importance) in enumerate(sorted_features[:10], 1):
                # Add explanations for key features
                explanations = {
                    'Hours_Studied': "Time spent studying directly impacts exam performance",
                    'Previous_Scores': "Past performance is a strong predictor of future success",
                    'Attendance': "Regular class attendance improves learning outcomes",
                    'Motivation_Level': "Student motivation drives engagement and retention",
                    'Teacher_Quality': "Better teachers provide clearer explanations and support",
                    'Sleep_Hours': "Adequate sleep improves cognitive function and memory",
                    'Parental_Involvement': "Parental support creates better learning environment",
                    'Access_to_Resources': "Better resources enable more effective studying",
                    'Tutoring_Sessions': "Additional personalized instruction improves performance",
                    'Family_Income': "Financial resources can provide better educational opportunities"
                }
                
                explanation = explanations.get(feature, "This feature contributes to exam score prediction")
                st.write(f"{i}. **{feature}**: {importance:.4f} - {explanation}")
            
            # Create feature importance plot (descending order)
            fig, ax = plt.subplots(figsize=(12, 8))
            features, importances = zip(*sorted_features[:10])
            
            # Create horizontal bar chart with descending order
            y_pos = range(len(features))
            bars = ax.barh(y_pos, importances, color='skyblue', edgecolor='navy')
            
            # Add value labels on bars
            for i, (bar, importance) in enumerate(zip(bars, importances)):
                ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                       f'{importance:.4f}', ha='left', va='center', fontweight='bold')
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(features, fontsize=10)
            ax.set_xlabel('Feature Importance Score', fontsize=12, fontweight='bold')
            ax.set_title('Top 10 Feature Importance (Most Important First)', fontsize=14, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            
            # Add interpretation
            st.markdown("### 📊 **Interpretation:**")
            st.markdown("""
            - **Higher bars** = More important features for predicting exam scores
            - **Top features** have the strongest influence on student performance
            - **Focus on improving** the top 3-5 features for maximum impact
            - **Model uses these weights** to make predictions in the next tab
            """)
        else:
            # Fallback to existing feature importance function
            plot_feature_importance(df_clean, selected_features, "Exam_Score")
        
        st.markdown("---")
        train_and_save_model(df_clean, selected_features, "Exam_Score", model_type="Polynomial")



    
    if st.button("🔁 Reset Selections"):
        st.session_state.clear()

# --- TAB 3: Prediction ---
with tab3:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); padding: 1rem; border-radius: 10px; color: white; margin-bottom: 2rem;">
        <h2 style="margin: 0; color: white;">🧮 Make a Prediction</h2>
        <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Input student data to predict exam scores</p>
    </div>
    """, unsafe_allow_html=True)
    
    if 'model' not in st.session_state:
        st.warning("⚠️ Please train a model first in the 'Model Evaluation & Insights' tab.")
        st.info("💡 Go to the previous tab, select features and model type, then come back here to make predictions.")
        
        # Demo mode with sample prediction
        st.markdown("---")
        st.markdown("### 🎮 Demo Mode")
        st.info("💡 This is a demo prediction based on typical student data. Train a real model for accurate predictions.")
        
        # Sample prediction for demo
        sample_inputs = {
            'Hours_Studied': 20,
            'Attendance': 85,
            'Parental_Involvement': 'Medium',
            'Access_to_Resources': 'Medium',
            'Extracurricular_Activities': 'Yes',
            'Sleep_Hours': 7,
            'Previous_Scores': 75,
            'Motivation_Level': 'Medium',
            'Internet_Access': 'Yes',
            'Tutoring_Sessions': 2,
            'Family_Income': 'Medium',
            'Teacher_Quality': 'Medium',
            'School_Type': 'Public',
            'Peer_Influence': 'Positive',
            'Physical_Activity': 3,
            'Learning_Disabilities': 'No',
            'Parental_Education_Level': 'College',
            'Distance_from_Home': 'Near',
            'Gender': 'Male'
        }
        
        # Demo prediction (simple formula based on key factors)
        demo_score = (
            sample_inputs['Hours_Studied'] * 0.3 +
            sample_inputs['Attendance'] * 0.2 +
            sample_inputs['Previous_Scores'] * 0.3 +
            sample_inputs['Sleep_Hours'] * 2 +
            (10 if sample_inputs['Motivation_Level'] == 'High' else 5 if sample_inputs['Motivation_Level'] == 'Medium' else 0)
        )
        demo_score = min(max(demo_score, 50), 95)  # Clamp between 50-95
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Demo Predicted Score", f"{demo_score:.1f}")
        with col2:
            st.metric("📝 Demo Grade", "B+" if demo_score >= 75 else "B" if demo_score >= 70 else "C+")
        with col3:
            st.metric("📈 Demo Performance", "Good")
        
        st.stop()
    
    # Load model and features from session
    model = st.session_state.model
    transformer = st.session_state.get("transformer", None)
    model_features = st.session_state.features
    model_type = st.session_state.model_type
    
    st.success(f"✅ Using {model_type} model ")
    
    # Get original data for input ranges
    original_df = st.session_state.df_original
    
    # Create input form with enhanced styling
    st.markdown("""
    <div style="background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); padding: 1rem; border-radius: 10px; margin: 1rem 0;">
        <h3 style="margin: 0; color: #333;">📥 Enter Student Information</h3>
        <p style="margin: 0.5rem 0 0 0; color: #666;">Fill in the student details to predict their exam score</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create two columns for better layout
    col1, col2 = st.columns(2)
    
    user_inputs = {}
    
    with col1:
        st.subheader("📊 Numerical Features")
        
        # Hours_Studied
        hours_studied = st.slider(
            "Hours Studied per Week", 
            min_value=int(original_df['Hours_Studied'].min()),
            max_value=int(original_df['Hours_Studied'].max()),
            value=int(original_df['Hours_Studied'].mean()),
            help="Number of hours spent studying per week"
        )
        user_inputs['Hours_Studied'] = hours_studied
        
        # Attendance
        attendance = st.slider(
            "Attendance Rate (%)", 
            min_value=0,
            max_value=int(original_df['Attendance'].max()),
            value=int(original_df['Attendance'].mean()),
            help="Percentage of classes attended"
        )
        user_inputs['Attendance'] = attendance
        
        # Sleep_Hours
        sleep_hours = st.slider(
            "Sleep Hours per Night", 
            min_value=int(original_df['Sleep_Hours'].min()),
            max_value=int(original_df['Sleep_Hours'].max()),
            value=int(original_df['Sleep_Hours'].mean()),
            help="Average hours of sleep per night"
        )
        user_inputs['Sleep_Hours'] = sleep_hours
        
        # Previous_Scores
        previous_scores = st.slider(
            "Previous Exam Scores", 
            min_value=int(original_df['Previous_Scores'].min()),
            max_value=int(original_df['Previous_Scores'].max()),
            value=int(original_df['Previous_Scores'].mean()),
            help="Average score from previous exams"
        )
        user_inputs['Previous_Scores'] = previous_scores
        
        # Tutoring_Sessions
        tutoring_sessions = st.slider(
            "Tutoring Sessions per Month", 
            min_value=int(original_df['Tutoring_Sessions'].min()),
            max_value=int(original_df['Tutoring_Sessions'].max()),
            value=int(original_df['Tutoring_Sessions'].mean()),
            help="Number of tutoring sessions attended per month"
        )
        user_inputs['Tutoring_Sessions'] = tutoring_sessions
        
        # Physical_Activity
        physical_activity = st.slider(
            "Physical Activity Hours per Week", 
            min_value=int(original_df['Physical_Activity'].min()),
            max_value=int(original_df['Physical_Activity'].max()),
            value=int(original_df['Physical_Activity'].mean()),
            help="Hours spent on physical activities per week"
        )
        user_inputs['Physical_Activity'] = physical_activity
        
        # Family_Income
        family_income = st.selectbox(
            "Family Income Level",
            options=original_df['Family_Income'].unique(),
            index=0,
            help="Family's income level"
        )
        user_inputs['Family_Income'] = family_income
    
    with col2:
        st.subheader("🔤 Categorical Features")
        
        # Parental_Involvement
        parental_involvement = st.selectbox(
            "Parental Involvement",
            options=original_df['Parental_Involvement'].unique(),
            index=0,
            help="Level of parental involvement in education"
        )
        user_inputs['Parental_Involvement'] = parental_involvement
        
        # Access_to_Resources
        access_to_resources = st.selectbox(
            "Access to Educational Resources",
            options=original_df['Access_to_Resources'].unique(),
            index=0,
            help="Level of access to educational resources"
        )
        user_inputs['Access_to_Resources'] = access_to_resources
        
        # Extracurricular_Activities
        extracurricular_activities = st.selectbox(
            "Extracurricular Activities",
            options=original_df['Extracurricular_Activities'].unique(),
            index=0,
            help="Participation in extracurricular activities"
        )
        user_inputs['Extracurricular_Activities'] = extracurricular_activities
        
        # Motivation_Level
        motivation_level = st.selectbox(
            "Motivation Level",
            options=original_df['Motivation_Level'].unique(),
            index=0,
            help="Student's motivation level"
        )
        user_inputs['Motivation_Level'] = motivation_level
        
        # Internet_Access
        internet_access = st.selectbox(
            "Internet Access",
            options=original_df['Internet_Access'].unique(),
            index=0,
            help="Access to internet at home"
        )
        user_inputs['Internet_Access'] = internet_access
        
        # Teacher_Quality
        teacher_quality = st.selectbox(
            "Teacher Quality",
            options=original_df['Teacher_Quality'].unique(),
            index=0,
            help="Quality of teaching"
        )
        user_inputs['Teacher_Quality'] = teacher_quality
        
        # School_Type
        school_type = st.selectbox(
            "School Type",
            options=original_df['School_Type'].unique(),
            index=0,
            help="Type of school (Public/Private)"
        )
        user_inputs['School_Type'] = school_type
        
        # Peer_Influence
        peer_influence = st.selectbox(
            "Peer Influence",
            options=original_df['Peer_Influence'].unique(),
            index=0,
            help="Influence of peers on academic performance"
        )
        user_inputs['Peer_Influence'] = peer_influence
        
        # Learning_Disabilities
        learning_disabilities = st.selectbox(
            "Learning Disabilities",
            options=original_df['Learning_Disabilities'].unique(),
            index=0,
            help="Presence of learning disabilities"
        )
        user_inputs['Learning_Disabilities'] = learning_disabilities
        
        # Parental_Education_Level
        parental_education = st.selectbox(
            "Parental Education Level",
            options=original_df['Parental_Education_Level'].unique(),
            index=0,
            help="Highest education level of parents"
        )
        user_inputs['Parental_Education_Level'] = parental_education
        
        # Distance_from_Home
        distance_from_home = st.selectbox(
            "Distance from Home",
            options=original_df['Distance_from_Home'].unique(),
            index=0,
            help="Distance from home to school"
        )
        user_inputs['Distance_from_Home'] = distance_from_home
        
        # Gender
        gender = st.selectbox(
            "Gender",
            options=original_df['Gender'].unique(),
            index=0,
            help="Student's gender"
        )
        user_inputs['Gender'] = gender
    
    # Prediction button
    if st.button("🎯 Predict Exam Score", type="primary"):
        try:
            # Use the new prediction function with proper preprocessing
            prediction, input_df_final, input_transformed = predict_with_preprocessing(
                model, transformer, model_features, user_inputs, original_df
            )
            
            # Display results with enhanced styling
            st.markdown("---")
            st.markdown("""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 10px; color: white; margin: 1rem 0;">
                <h3 style="margin: 0; color: white;">🎯 Prediction Results</h3>
                <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Your model's prediction for the student's exam score</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Create metrics display
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="📊 Predicted Exam Score",
                    value=f"{prediction:.1f}",
                    delta=f"{prediction - 70:.1f}" if prediction > 70 else f"{prediction - 70:.1f}",
                    delta_color="normal"
                )
            
            with col2:
                # Calculate grade based on score
                if prediction >= 90:
                    grade = "A+"
                elif prediction >= 85:
                    grade = "A"
                elif prediction >= 80:
                    grade = "A-"
                elif prediction >= 75:
                    grade = "B+"
                elif prediction >= 70:
                    grade = "B"
                elif prediction >= 65:
                    grade = "B-"
                elif prediction >= 60:
                    grade = "C+"
                elif prediction >= 55:
                    grade = "C"
                else:
                    grade = "D"
                
                st.metric(
                    label="📝 Estimated Grade",
                    value=grade
                )
            
            with col3:
                # Performance level
                if prediction >= 80:
                    performance = "Excellent"
                    color = "green"
                elif prediction >= 70:
                    performance = "Good"
                    color = "blue"
                elif prediction >= 60:
                    performance = "Average"
                    color = "orange"
                else:
                    performance = "Needs Improvement"
                    color = "red"
                
                st.metric(
                    label="📈 Performance Level",
                    value=performance
                )
            
            # Show input summary
            st.markdown("### 📋 Input Summary")
            input_summary = pd.DataFrame([user_inputs]).T
            input_summary.columns = ['Value']
            st.dataframe(input_summary, use_container_width=True)
            
            

            
        except Exception as e:
            st.error(f"❌ Error making prediction: {str(e)}")
            st.info("💡 Try training the model again in the previous tab.")
    
    # Add helpful information with enhanced styling
    st.markdown("---")
    st.markdown("""
    <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); padding: 1rem; border-radius: 10px; color: white; margin: 1rem 0;">
        <h3 style="margin: 0; color: white;">💡 Tips for Better Predictions</h3>
        <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Factors that influence exam performance</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **📚 Academic Factors:**
        - **Study Hours**: More study hours generally lead to better scores
        - **Attendance**: Higher attendance rates correlate with better performance
        - **Previous Scores**: Students with higher previous scores tend to perform better
        - **Tutoring**: Additional personalized instruction improves performance
        """)
    
    with col2:
        st.markdown("""
        **🌟 Environmental Factors:**
        - **Sleep**: Adequate sleep (7-9 hours) is important for academic performance
        - **Motivation**: Higher motivation levels lead to better outcomes
        - **Teacher Quality**: Better teachers significantly improve performance
        - **Parental Involvement**: More parental support creates better learning environment
        """)