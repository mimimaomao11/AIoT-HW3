# Email/SMS Spam Classifier

A comprehensive spam classification system with a web interface, built using LogisticRegression and TF-IDF vectorization.

## Features

- 📊 Interactive web interface with Streamlit
- 🔍 Live inference for single messages
- 📈 Model performance visualization
- 🔄 Adjustable model parameters
- 📊 Token analysis and insights
- 📥 Downloadable analysis results

## Project Structure

```
.
├── app/                    # Streamlit web application
├── datasets/               # Raw and processed datasets
│   └── processed/         # Cleaned datasets for training
├── models/                # Saved model artifacts
├── scripts/               # Python scripts
└── reports/              
    └── visualizations/    # Generated plots
```

## Installation and Setup

1️⃣ **Install Dependencies**

First, create and activate a virtual environment:

```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# Linux/Mac
python -m venv .venv
source .venv/bin/activate
```

Then install the required packages:
```bash
pip install -r requirements.txt
```

2️⃣ **Preprocess Data**

Clean and prepare the dataset:
```bash
python scripts/preprocess_emails.py --input datasets/sample_sms_spam.csv --output datasets/processed/sample_clean.csv
```

3️⃣ **Train Model**

Train the classifier and save artifacts:
```bash
python scripts/train_spam_classifier.py --input datasets/processed/sample_clean.csv
```

4️⃣ **Launch Web Interface**

Start the Streamlit application:
```bash
streamlit run app/streamlit_app.py
```

The app will be available at http://localhost:8501

## Deployment Guide

### GitHub Setup

1. Create a new repository on GitHub
2. Initialize git and push your code:
```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

### Streamlit Cloud Deployment

1. Go to [Streamlit Cloud](https://streamlit.io/cloud)
2. Sign in with your GitHub account
3. Click "New app"
4. Select your repository and branch
5. Set the main file path to `app/streamlit_app.py`
6. Click "Deploy"

The app will be automatically deployed and accessible via a public URL.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Notes

- The current implementation uses a basic TF-IDF + LogisticRegression model
- Improve performance by:
  - Adding more preprocessing steps
  - Implementing feature engineering
  - Using more sophisticated models (e.g., BERT, XGBoost)
  - Adding cross-validation
  - Implementing model ensembles

