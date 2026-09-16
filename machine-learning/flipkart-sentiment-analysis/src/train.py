#import necessary modules
import json
from pathlib import Path
import joblib 
import pandas as pd
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    classification_report,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix
)

from preprocessing import preprocess_data

MODEL_VERSION = "1.0.0"

TARGET_NAMES = [
    'negative',
    'neutral',
    'positive'
]

#define model and data path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = PROJECT_ROOT / "model"
DATA_PATH = PROJECT_ROOT / "data" / "dataset.csv"
MODEL_PATH = PROJECT_ROOT / "model" / "sentiment_pipeline.joblib"
REPORT_PATH = PROJECT_ROOT / "model" / "classification_report.json"
METRICS_PATH = PROJECT_ROOT / "model" / "metrics.json"

def train_multi_classification_model() -> None:
    """
    Train Multiclass classification model and store the trained model pipeline and metrics
    in model dir
    """

    #create model dir if not existing
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    #load the dataset
    df = pd.read_csv(DATA_PATH)

    #check the data set size before preprocessing
    print(f"Shape of data set before preprocessing : {df.shape}")

    #preprocess the data set
    (
        final_model_data,
        high_confidence_feedback,
        low_confidence_feedback,
        preprocessing_report,
    ) = preprocess_data(df)

    #check data set size after preprocessing
    print(f"Shape of data set after preprocessing: {final_model_data.shape}")

    #assign feature and target
    X = final_model_data["customer_review"]
    y = final_model_data["target_label"]

    #split train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    #check no.of train and test samples
    print(f"No. of train samples : {X_train.shape}")
    print(f"No. of test samples : {X_test.shape}")

    #pipeline for vectorization and training
    pipeline = Pipeline(
        steps=[
            (
                'tfidf',
                TfidfVectorizer(
                    max_features=30000,
                    ngram_range=(1,2),
                    min_df=3,
                    max_df=0.95,
                    sublinear_tf=True, # sublinear_tf=True often gives a small but consistent boost.
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    random_state=42,
                    #class_weight="balanced",
                ),
            ),
        ]
    )

    print("Customer Sentiment Distributions ...")
    print(final_model_data["target_sentiment"].value_counts())
    print(final_model_data["target_sentiment"].value_counts(normalize=True))

    print("\nTraining distribution")
    print(y_train.value_counts(normalize=True))
    
    print("\nTest distribution")
    print(y_test.value_counts(normalize=True))

    print("Training multiclass classification Model ...")
    #train the model
    pipeline.fit(
        X_train,
        y_train,
    )

    #evaluate the trained model
    y_pred = pipeline.predict(X_test)

    #calculate the accuracy metrics - overall correctness
    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    #evaluate - how trustworthy the +ve predictions are
    precision_macro = precision_score(
        y_test,
        y_pred,
        average='macro', # macro average to verify model performance on all classes
        zero_division=0, # prevents zero devision error
    )

    # how many actual samples are detected?
    recall_macro = recall_score(
        y_test,
        y_pred,
        average='macro',
        zero_division=0,
    )

     # balance between precision and recall
    f1_macro = f1_score(
        y_test,
        y_pred,
        average='macro',
        zero_division=0,
    )

    #weighted prcision
    precision_weighted = precision_score(
        y_test,
        y_pred,
        average="weighted", # weighted average to confirm that model is not biased
        zero_division=0,
    )

    #weighted recall
    recall_weighted = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    # Weighted F1
    f1_weighted = f1_score(
        y_test,
        y_pred,
        average='weighted',
        zero_division=0,
    )

    #create classification report
    report = classification_report(
        y_test,
        y_pred,
        target_names=TARGET_NAMES,
        output_dict=True,
        zero_division=0,
    )

    #calculate the confusion matrix
    confusion = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1, 2],
    )

    print(f"Accuracy: {accuracy:.4f}")
    print(report)
    print(f"Confusion Matrix :")
    print(confusion)

    #save the model
    joblib.dump(
        pipeline,
        MODEL_PATH
    )

    #save metrics to evaluate model quality
    metrics = {
        "dataset_rows": int(len(final_model_data)),
        "confusion_matrix": confusion.tolist(),
        "accuracy" : float(accuracy),
        "precision_macro" : float(precision_macro),
        "recall_macro" : float(recall_macro),
        "f1_macro" : float(f1_macro),
        "precision_weighted" : float(precision_weighted),
        "recall_weighted" : float(recall_weighted),
        "f1_weighted" : float(f1_weighted),
        "train_samples" : int(len(X_train)),
        "test_samples" : int(len(X_test)),
        "model_type" : "TF-IDF + Logistic Regression",
        "target_classes" : TARGET_NAMES,
    }

    metrics["model_version"] = MODEL_VERSION
    metrics["preprocessing_report"] = (
        preprocessing_report
    )
    #metrics["trained_at"] = datetime.now(datetime.UTC)

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

if __name__ == "__main__":
    train_multi_classification_model()