import os
import re
import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import mlflow
import warnings

warnings.filterwarnings('ignore')

def get_sentiment(rating):
    if pd.isna(rating):
        return None
    try:
        rating = float(rating)
        if rating <= 2:
            return 0 # Negative
        elif rating >= 4:
            return 1 # Positive
        else:
            return None # Drop Neutral
    except (ValueError, TypeError):
        return None

def minimal_clean(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    try:
        import emoji
        text = emoji.demojize(text, delimiters=(" ", " "))
    except ImportError:
        text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"[^\w\s!?.,']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='macro')
    return {'accuracy': acc, 'f1_macro': f1}

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if device == "cpu":
        print("WARNING: Training on CPU. This will take a long time! We will run fewer epochs for speed on CPU if necessary.")
        epochs = 1
    else:
        epochs = 3

    print("Loading data from raw_reviews.csv...")
    data_path = os.path.join(script_dir, 'data', 'raw_reviews.csv')
    
    df = pd.read_csv(data_path)
    
    df['label'] = df['star_rating'].apply(get_sentiment)
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)
    
    print("Applying minimal cleaning to text...")
    df['text'] = df['review_text'].apply(minimal_clean)
    df = df[df["text"].str.split().str.len() >= 4].reset_index(drop=True)
    
    df = df[['text', 'label']]
    
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    
    train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
    test_dataset = Dataset.from_pandas(test_df.reset_index(drop=True))
    
    print(f"Train: {len(train_dataset)}  Test: {len(test_dataset)}")
    
    print("Initializing Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
        
    print("Tokenizing data...")
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_test = test_dataset.map(tokenize_function, batched=True)
    
    print("Initializing Model...")
    model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        report_to="none"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        compute_metrics=compute_metrics,
    )
    
    mlflow.set_experiment('Sentix_Tinder_BERT')
    with mlflow.start_run(run_name="BERT_minimal_prep_finetune"):
        print("Starting training...")
        trainer.train()
        
        print("Evaluating final model...")
        eval_results = trainer.evaluate()
        print(f"Eval Results: {eval_results}")
        
        mlflow.log_metric("best_f1_macro", eval_results['eval_f1_macro'])
        mlflow.log_metric("best_accuracy", eval_results['eval_accuracy'])
        
        model_save_path = "./models/bert_sentiment"
        os.makedirs(model_save_path, exist_ok=True)
        trainer.save_model(model_save_path)
        tokenizer.save_pretrained(model_save_path)
        print(f"PyTorch model saved to {model_save_path}")
        
        print("Exporting model to ONNX...")
        try:
            model.eval()
            model.to("cpu")
            dummy_text = "This app is awesome!"
            inputs = tokenizer(dummy_text, return_tensors="pt")
            
            onnx_path = os.path.join(model_save_path, "model.onnx")
            
            input_names = ['input_ids', 'attention_mask'] + (['token_type_ids'] if 'token_type_ids' in inputs else [])
            dummy_inputs = tuple(inputs[k] for k in input_names)
            
            torch.onnx.export(
                model, 
                args=dummy_inputs, 
                f=onnx_path, 
                input_names=input_names, 
                output_names=["logits"], 
                dynamic_axes={
                    k: {0: "batch_size", 1: "sequence_length"} for k in input_names
                } | {"logits": {0: "batch_size"}},
                opset_version=14,
                do_constant_folding=True
            )
            print(f"ONNX model successfully exported to {onnx_path}")
        except Exception as e:
            print(f"ONNX Export failed: {e}")

if __name__ == '__main__':
    main()
