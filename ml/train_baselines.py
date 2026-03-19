import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn
import warnings
import os

warnings.filterwarnings('ignore')

def get_sentiment(rating):
    if pd.isna(rating):
        return None
    try:
        rating = float(rating)
        if rating <= 2:
            return 'Negative'
        elif rating >= 4:
            return 'Positive'
        else:
            return None # Drop Neutral
    except (ValueError, TypeError):
        return None

def main():
    # Make sure we are in the correct directory context
    data_path = 'data/clean_reviews.csv'
    if not os.path.exists(data_path) and os.path.exists('ml/data/clean_reviews.csv'):
        # Just in case this script is run from root rather than ml folder
        os.chdir('ml')
        
    print(f"Loading cleaned dataset from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Drop initial missing values
    df = df.dropna(subset=['clean_text', 'star_rating'])
    
    # Map sentiments
    df['sentiment'] = df['star_rating'].apply(get_sentiment)
    df = df.dropna(subset=['sentiment'])
    
    X = df['clean_text']
    y = df['sentiment']
    
    print(f"Dataset size: {len(df)}")
    print(f"Class distribution:\n{y.value_counts()}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    models = {
        'Naive_Bayes': MultinomialNB(),
        'Logistic_Regression': LogisticRegression(max_iter=1000, random_state=42),
        'LinearSVC': LinearSVC(random_state=42)
    }
    
    mlflow.set_experiment('Sentix_Tinder_Baselines')
    
    for model_name, clf in models.items():
        print(f"\nTraining {model_name}...")
        with mlflow.start_run(run_name=model_name):
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=50000)),
                ('clf', clf)
            ])
            
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            
            print(f"--- {model_name} Evaluation ---")
            report = classification_report(y_test, y_pred, output_dict=True)
            print(classification_report(y_test, y_pred))
            
            # ROC AUC computation
            y_prob = None
            if hasattr(pipeline['clf'], "predict_proba"):
                y_prob = pipeline.predict_proba(X_test)
            elif hasattr(pipeline['clf'], "decision_function"):
                y_prob = pipeline.decision_function(X_test)
                # LinearSVC decision function needs softmax-like transformation for strict ROC AUC if labels are multiple,
                # but sklearn handles decision_function for ovo/ovr if formatted properly.
                
            if y_prob is not None:
                try:
                    # Binary ROC AUC: use positive-class scores
                    if hasattr(y_prob, 'ndim') and y_prob.ndim == 2:
                        auc = roc_auc_score(y_test, y_prob[:, 1])
                    else:
                        auc = roc_auc_score(y_test, y_prob)
                    mlflow.log_metric('roc_auc_macro', auc)
                    print(f"ROC AUC: {auc:.4f}")
                except Exception as e:
                    print(f"Could not compute ROC AUC: {e}")
            
            # Log metrics to MLflow
            mlflow.log_param('max_features', 50000)
            mlflow.log_param('model_type', type(clf).__name__)
            
            # Log macro F1
            mlflow.log_metric('macro_f1', report['macro avg']['f1-score'])
            mlflow.log_metric('accuracy', report['accuracy'])
            
            mlflow.sklearn.log_model(pipeline, artifact_path="model")
            
            print(f"{model_name} logged to MLflow successfully.")

if __name__ == '__main__':
    main()
