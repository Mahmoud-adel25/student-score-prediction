# 🚀 Deployment Guide

## Quick Deployment Steps

### 1. GitHub Setup

```bash
# Initialize git repository (if not already done)
git init

# Add all files
git add .

# Commit changes
git commit -m "Initial commit: Student Score Prediction App"

# Add your GitHub repository as remote
git remote add origin https://github.com/yourusername/student-score-prediction.git

# Push to GitHub
git push -u origin main
```

### 2. Streamlit Cloud Deployment (Recommended)

1. **Go to Streamlit Cloud**
   - Visit: https://share.streamlit.io
   - Sign in with your GitHub account

2. **Deploy Your App**
   - Click "New app"
   - Select your repository: `yourusername/student-score-prediction`
   - Set the path to your app: `Deplyment_App/app.py`
   - Click "Deploy!"

3. **Your app will be live at:**
   - `https://your-app-name-yourusername.streamlit.app`

### 3. Alternative: Heroku Deployment

1. **Create a Procfile** (already included):
   ```
   web: streamlit run Deplyment_App/app.py --server.port=$PORT --server.address=0.0.0.0
   ```

2. **Deploy using Heroku CLI:**
   ```bash
   # Install Heroku CLI
   # Create new Heroku app
   heroku create your-app-name
   
   # Push to Heroku
   git push heroku main
   ```

### 4. Local Testing

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
cd Deplyment_App
streamlit run app.py
```

## 📋 Pre-deployment Checklist

- ✅ `requirements.txt` created
- ✅ `.gitignore` configured
- ✅ `README.md` updated
- ✅ All dependencies listed
- ✅ App runs locally without errors
- ✅ Data file included in repository

## 🔧 Troubleshooting

### Common Issues:

1. **Import Errors**
   - Ensure all dependencies are in `requirements.txt`
   - Check Python version compatibility

2. **Data File Not Found**
   - Verify `Data/StudentPerformanceFactors.csv` exists
   - Check file path in `app.py`

3. **Streamlit Cloud Issues**
   - Check app path is correct: `Deplyment_App/app.py`
   - Verify all imports work
   - Check Streamlit Cloud logs for errors

4. **Heroku Issues**
   - Ensure `Procfile` is in root directory
   - Check build logs for dependency issues

## 🎯 Success Indicators

Your deployment is successful when:
- ✅ App loads without errors
- ✅ All three tabs are accessible
- ✅ Data loads and displays correctly
- ✅ Model training works
- ✅ Predictions generate successfully

## 📞 Support

If you encounter issues:
1. Check the debug information in Tab 3
2. Review Streamlit Cloud logs
3. Test locally first
4. Check GitHub issues for similar problems

---

🎯 **Your app is ready for deployment!** 🚀 