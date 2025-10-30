"""Streamlit app for Email/SMS Spam Classification.

Run with:
    streamlit run app/streamlit_app.py
"""

import sys
import argparse
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, roc_curve, precision_recall_curve,
    auc, average_precision_score, f1_score
)

# Add project root to path for importing scripts
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.train_spam_classifier import train  # noqa
from scripts.preprocess_emails import clean_text  # noqa

# Page config
st.set_page_config(
    page_title="Spam Classification Dashboard",
    page_icon="📧",
    layout="wide"
)

# Constants
SPAM_EXAMPLES = [
    """WINNER!! As a valued network customer you have been selected to receivea £900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only.""",
    """Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply.""",
    """SIX chances to win CASH! From 100 to 20,000 pounds txt> CSH11 and send to 87575. Cost 150p/day, 6days, 16+ TsandCs apply.""",
    """URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot! Txt the word: CLAIM to No: 81010"""
]

HAM_EXAMPLES = [
    """Hey, what time should we meet for coffee tomorrow? I was thinking about that new cafe downtown around 10am?""",
    """I'm gonna be home soon and i don't want to talk about this stuff anymore tonight, k? I've cried enough today.""",
    """Ok lar... Joking wif u oni...""",
    """Even my brother is not like to speak with me. They treat me like aids patent."""
]
CACHE_DIR = project_root / ".streamlit" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Initialize session state
for key in ['trained_model', 'vectorizer', 'train_data', 'test_data', 
           'X_test', 'y_test', 'y_pred', 'y_prob', 'model_params']:
    if key not in st.session_state:
        st.session_state[key] = None


def load_data(file_path: Path) -> pd.DataFrame:
    """Load dataset with caching."""
    return pd.read_csv(file_path)


@st.cache_data
def get_available_datasets() -> dict:
    """Get mapping of dataset names to file paths."""
    datasets_dir = project_root / "datasets"
    processed_dir = datasets_dir / "processed"
    
    datasets = {
        "Raw Dataset": datasets_dir / "sms_spam.csv",
        "Processed Dataset": processed_dir / "sms_spam_processed.csv"
    }
    return {k: v for k, v in datasets.items() if v.exists()}


def plot_class_distribution(df: pd.DataFrame, label_col: str) -> go.Figure:
    """Plot class distribution as a bar chart."""
    counts = df[label_col].value_counts()
    fig = px.bar(
        x=counts.index,
        y=counts.values,
        title="Class Distribution",
        labels={"x": "Class", "y": "Count"},
        color=counts.index,
        color_discrete_map={"spam": "#FF6B6B", "ham": "#4ECDC4"}
    )
    fig.update_layout(showlegend=False)
    return fig


def plot_top_tokens(df: pd.DataFrame, vectorizer, label_col: str, top_n: int) -> go.Figure:
    """Plot top tokens for each class."""
    feature_names = vectorizer.get_feature_names_out()
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Spam Tokens', 'Ham Tokens'))
    
    for i, label in enumerate(['spam', 'ham'], 1):
        # Get texts for this class
        texts = df[df[label_col] == label]['text']
        
        # Transform and get mean feature values
        X = vectorizer.transform(texts)
        mean_tfidf = X.mean(axis=0).A1
        
        # Get top tokens
        top_indices = mean_tfidf.argsort()[-top_n:][::-1]
        top_values = mean_tfidf[top_indices]
        top_terms = feature_names[top_indices]
        
        color = "#FF6B6B" if label == "spam" else "#4ECDC4"
        
        fig.add_trace(
            go.Bar(
                x=top_values,
                y=top_terms,
                orientation='h',
                name=label.upper(),
                marker_color=color
            ),
            row=1, col=i
        )
    
    fig.update_layout(
        height=400,
        title_text=f"Top {top_n} Tokens by Class",
        showlegend=False
    )
    return fig


def plot_confusion_matrix(y_true, y_pred, normalize: bool = False) -> go.Figure:
    """Plot confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Ham', 'Spam'],
        y=['Ham', 'Spam'],
        text=np.round(cm, 3),
        texttemplate="%{text}",
        textfont={"size": 16},
        hoverongaps=False,
        colorscale=[[0, '#4ECDC4'], [1, '#FF6B6B']]
    ))
    
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
        xaxis={'side': 'bottom'}
    )
    return fig


def plot_roc_pr_curves(y_true, y_prob) -> tuple[go.Figure, go.Figure]:
    """Plot ROC and PR curves."""
    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    fig_roc = px.line(
        x=fpr, y=tpr,
        title=f'ROC Curve (AUC = {roc_auc:.3f})',
        labels={'x': 'False Positive Rate', 'y': 'True Positive Rate'}
    )
    fig_roc.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )
    
    # PR curve
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    avg_precision = average_precision_score(y_true, y_prob)
    
    fig_pr = px.line(
        x=recall, y=precision,
        title=f'Precision-Recall Curve (AP = {avg_precision:.3f})',
        labels={'x': 'Recall', 'y': 'Precision'}
    )
    
    return fig_roc, fig_pr


def plot_threshold_sweep(y_true, y_prob) -> tuple[go.Figure, pd.DataFrame]:
    """Analyze model performance across different thresholds."""
    thresholds = np.linspace(0, 1, 100)
    results = []
    
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        precision = (y_pred & y_true).sum() / max(y_pred.sum(), 1)
        recall = (y_pred & y_true).sum() / max(y_true.sum(), 1)
        f1 = 2 * (precision * recall) / max((precision + recall), 1e-6)
        
        results.append({
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        })
    
    df_threshold = pd.DataFrame(results)
    
    fig = px.line(
        df_threshold,
        x='threshold',
        y=['precision', 'recall', 'f1_score'],
        title='Model Metrics vs. Classification Threshold'
    )
    
    return fig, df_threshold


def plot_prediction_probability(prob: float, threshold: float) -> go.Figure:
    """Create a visual probability gauge for spam prediction."""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = prob,
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = {
            'axis': {'range': [0, 1]},
            'bar': {'color': "#FF6B6B"},
            'steps': [
                {'range': [0, threshold], 'color': "#4ECDC4"},
                {'range': [threshold, 1], 'color': "#FF6B6B"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': threshold
            }
        },
        title = {'text': "Spam Probability"}
    ))
    
    fig.update_layout(height=300)
    return fig


def sidebar_settings():
    """Render sidebar settings."""
    st.sidebar.title("Settings")
    
    # Dataset selection
    datasets = get_available_datasets()
    dataset_name = st.sidebar.selectbox(
        "Select Dataset",
        options=list(datasets.keys()),
        key="dataset_name"
    )
    
    # Model parameters
    st.sidebar.subheader("Model Parameters")
    test_size = st.sidebar.slider(
        "Test Size",
        min_value=0.1,
        max_value=0.4,
        value=0.2,
        step=0.05,
        key="test_size"
    )
    
    random_seed = st.sidebar.number_input(
        "Random Seed",
        min_value=1,
        max_value=999999,
        value=42,
        key="random_seed"
    )
    
    threshold = st.sidebar.slider(
        "Decision Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
        key="threshold"
    )
    
    return datasets[dataset_name], test_size, random_seed, threshold


def overview_tab(df: pd.DataFrame):
    """Render Overview tab."""
    st.header("Dataset Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Data Preview")
        st.dataframe(df.head(), width="stretch")
        
        st.subheader("Dataset Info")
        st.text(f"Total samples: {len(df)}")
        st.text(f"Columns: {', '.join(df.columns)}")
    
    with col2:
        st.subheader("Class Distribution")
        fig = plot_class_distribution(df, 'label')
        st.plotly_chart(fig, width='stretch')


def tokens_tab(df: pd.DataFrame, vectorizer, top_n: int = 20):
    """Render Top Tokens tab."""
    st.header("Top Tokens Analysis")
    
    n_tokens = st.slider(
        "Number of top tokens to show",
        min_value=5,
        max_value=50,
        value=top_n
    )
    
    fig = plot_top_tokens(df, vectorizer, 'label', n_tokens)
    st.plotly_chart(fig, width='stretch')


def performance_tab(y_true, y_pred, y_prob):
    """Render Model Performance tab."""
    st.header("Model Performance")
    
    # Confusion Matrix
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Confusion Matrix")
        fig_cm = plot_confusion_matrix(y_true, y_pred)
        st.plotly_chart(fig_cm, width='stretch')
    
    with col2:
        st.subheader("Normalized Confusion Matrix")
        fig_cm_norm = plot_confusion_matrix(y_true, y_pred, normalize=True)
        st.plotly_chart(fig_cm_norm, width='stretch')
    
    # ROC and PR curves
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("ROC Curve")
        fig_roc, fig_pr = plot_roc_pr_curves(y_true, y_prob)
        st.plotly_chart(fig_roc, width='stretch')
    
    with col2:
        st.subheader("Precision-Recall Curve")
        st.plotly_chart(fig_pr, width='stretch')
    
    # Threshold analysis
    st.subheader("Threshold Analysis")
    fig_threshold, df_threshold = plot_threshold_sweep(y_true, y_prob)
    st.plotly_chart(fig_threshold, width='stretch')
    
    # Download threshold analysis
    st.download_button(
        "Download Threshold Analysis CSV",
        df_threshold.to_csv(index=False),
        "threshold_analysis.csv",
        "text/csv"
    )


def inference_tab(model, vectorizer, threshold: float):
    """Render Live Inference tab."""
    st.header("Live Inference")
    
    import random
    
    col1, col2 = st.columns(2)
    with col1:
        use_spam = st.button("Use Random Spam Example")
        if use_spam:
            st.session_state.input_text = random.choice(SPAM_EXAMPLES)
    
    with col2:
        use_ham = st.button("Use Random Ham Example")
        if use_ham:
            st.session_state.input_text = random.choice(HAM_EXAMPLES)
    
    input_text = st.text_area(
        "Enter text to classify:",
        value=st.session_state.get('input_text', ""),
        height=150
    )
    
    predict_button = st.button("Predict")
    
    if predict_button and input_text:
        st.session_state.input_text = input_text
        # Clean and transform text
        cleaned_text = clean_text(input_text)
        X = vectorizer.transform([cleaned_text])
        
        # Get prediction and probability
        pred = model.predict(X)[0]
        prob = model.predict_proba(X)[0, 1]
        
        # Show results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Cleaned Text")
            st.text(cleaned_text)
            
            st.subheader("Prediction")
            label = "SPAM" if prob >= threshold else "HAM"
            color = "#FF6B6B" if label == "SPAM" else "#4ECDC4"
            st.markdown(
                f'<h1 style="color: {color};">{label}</h1>',
                unsafe_allow_html=True
            )
        
        with col2:
            st.subheader("Spam Probability")
            fig = plot_prediction_probability(prob, threshold)
            st.plotly_chart(fig, width='stretch')


def main():
    """Main Streamlit app."""
    st.title("📧 Email/SMS Spam Classification Dashboard")
    
    # Sidebar settings
    dataset_path, test_size, random_seed, threshold = sidebar_settings()
    
    # Load and preprocess data
    try:
        df = load_data(dataset_path)
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return
    
    # Train model if needed
    model_needs_training = False
    if 'train_data' not in st.session_state:
        model_needs_training = True
    elif 'model_params' not in st.session_state:
        model_needs_training = True
    else:
        curr_params = {
            'test_size': test_size,
            'random_seed': random_seed,
            'dataset_path': str(dataset_path)
        }
        if st.session_state.model_params != curr_params:
            model_needs_training = True
    
    if model_needs_training:
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            df['text'], 
            (df['label'] == 'spam').astype(int),
            test_size=test_size,
            random_state=random_seed,
            stratify=df['label']
        )
        
        # Train model
        model_path = CACHE_DIR / "model.joblib"
        vector_path = CACHE_DIR / "vectorizer.joblib"
        
        try:
            # Create args namespace
            args = argparse.Namespace(
                input=str(dataset_path),
                label_col='label',
                test_size=test_size,
                random_state=random_seed,
                class_weight='balanced',
                max_features=10000,
                model_path=str(model_path),
                vector_path=str(vector_path)
            )
            train(args)
            
            st.session_state.trained_model = joblib.load(model_path)
            st.session_state.vectorizer = joblib.load(vector_path)
            st.session_state.train_data = X_train
            st.session_state.test_data = X_test
            st.session_state.model_params = {
                'test_size': test_size,
                'random_seed': random_seed,
                'dataset_path': str(dataset_path)
            }
            
            # Get predictions
            if st.session_state.vectorizer is not None and st.session_state.trained_model is not None:
                X_test_tfidf = st.session_state.vectorizer.transform(st.session_state.test_data)
                st.session_state.y_test = y_test
                st.session_state.y_pred = st.session_state.trained_model.predict(X_test_tfidf)
                st.session_state.y_prob = st.session_state.trained_model.predict_proba(X_test_tfidf)[:, 1]
            
        except Exception as e:
            st.error(f"Error training model: {str(e)}")
            return
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "Overview", "Top Tokens", "Model Performance", "Live Inference"
    ])
    
    with tab1:
        overview_tab(df)
    
    with tab2:
        tokens_tab(df, st.session_state.vectorizer)
    
    with tab3:
        performance_tab(
            st.session_state.y_test,
            st.session_state.y_pred,
            st.session_state.y_prob
        )
    
    with tab4:
        inference_tab(
            st.session_state.trained_model,
            st.session_state.vectorizer,
            threshold
        )


if __name__ == "__main__":
    main()