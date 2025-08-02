#!/bin/bash

echo "🚀 Deploying Student Score Prediction App..."

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "❌ Git is not installed. Please install Git first."
    exit 1
fi

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "📁 Initializing git repository..."
    git init
fi

# Add all files
echo "📦 Adding files to git..."
git add .

# Commit changes
echo "💾 Committing changes..."
git commit -m "Initial commit: Student Score Prediction App"

# Check if remote exists
if ! git remote get-url origin &> /dev/null; then
    echo "🔗 Please add your GitHub repository as remote origin:"
    echo "git remote add origin https://github.com/yourusername/student-score-prediction.git"
    echo ""
    echo "Then run: git push -u origin main"
else
    echo "🚀 Pushing to GitHub..."
    git push -u origin main
fi

echo ""
echo "✅ Deployment preparation complete!"
echo ""
echo "📋 Next steps:"
echo "1. Push to GitHub: git push -u origin main"
echo "2. Deploy on Streamlit Cloud: https://share.streamlit.io"
echo "3. Or deploy locally: streamlit run Deplyment_App/app.py"
echo ""
echo "🎯 Your app is ready for deployment!" 