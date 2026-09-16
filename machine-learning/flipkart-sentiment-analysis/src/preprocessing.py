from ftplib import error_perm
import re
import pandas as pd
import json
import logging
import argparse
from pathlib import Path

#Label the rating with sentiment text
RATE_TO_SENTIMENT = {
    1: "negative",
    2: "negative",
    3: "neutral",
    4: "positive",
    5: "positive",
}

# taget label for the sentiment
LABEL_MAPPING = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}

REQUIRED_COLUMNS = {
    "product_name",
    "product_price",
    "Rate",
    "Review",
    "Summary",
    "Sentiment"
}

FINAL_COLUMNS = [
    "customer_review",
    "target_sentiment",
    "target_label",
]

#-------------------------------------------------------
# Text Cleaning
#-------------------------------------------------------
def clean_text(text: str) -> str:
    """
    Clean the feedback text before vectorization
    # convert all feedback text to lowercase
    # replace the noisy characters with space
    #replace with white space except chars,',', '!', '?'
    #replace whitespace sequence with single space
    """

    # convert all feedback text to lowercase
    text = str(text).lower()

    # replace the noisy characters with space
    text = re.sub(r"[#@*$]+", " ", text)

    #replace with white space except chars,',', '!', '?'
    text = re.sub(r"[^a-zA-Z0-9\s.,!?'-]", " ", text)

    #replace whitespace sequence with one space
    text = re.sub(r"\s+", " ", text)

    #return the cleaned string
    return text.strip()

#-------------------------------------------------------
# Input validation
#-------------------------------------------------------
def validate_schema(df: pd.DataFrame) -> None:
    """
    Validate whether the df contains all required columns and not empty
    """

    #check the df type
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"Expected : DataFrame. Received: {type(df).__name__}"
        )

    if df.empty:
        raise ValueError("The input dataframe is empty")

    missing_columns = sorted(REQUIRED_COLUMNS.difference(df.columns))

    if missing_columns:
        raise ValueError(
            f"Missing following required columns: {missing_columns}"
        )
    
#-------------------------------------------------------
# Numeric conversion of Rate and product_price columns
#-------------------------------------------------------
def convert_numeric_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    convert Rate and product_price in to numeric values
    Remove rows with missing and non-numeric values
    remove rows with out of range ratings
    """
    df = df.copy()

    initial_rows = len(df)

    df["Rate"] = pd.to_numeric(
        df["Rate"],
        errors="coerce"
    )

    df["product_price"] = pd.to_numeric(
        df["product_price"],
        errors="coerce"
    )

    invalid_rate_mask = (
        df["Rate"].isna()
        | ~df["Rate"].between(1,5)
    )

    invalid_rate_rows = int(invalid_rate_mask.sum())
    invalid_price_rows = int(df["product_price"].isna().sum())

    df = df.loc[~invalid_rate_mask].copy()
    df["Rate"] = df["Rate"].astype(int)

    report = {
        "initial_rows" : initial_rows,
        "invalid_rating_rows_removed" : invalid_rate_rows,
        "invalid_product_price_rows_removed": invalid_price_rows,
        "final_rows_after_numerical_cleaning": len(df)
    }

    return df, report

#-------------------------------------------------------
# create cleaned review
#-------------------------------------------------------
def create_review_and_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    create customer_review column by combining review and summary column
    clean the review from noisy chars
    """
    df = df.copy()

    # combine both Review and Summary
    df["customer_review"] = (
        df["Review"].fillna("").astype(str).str.strip()
        + " "
        + df["Summary"].fillna("").astype(str).str.strip()
    ).str.strip()

    #clean the text string in the customer_review column
    df["customer_review"] = (
        df["customer_review"].apply(clean_text)
    )

    # remove empty rows after combining and cleaning
    df = df.loc[df["customer_review"].ne("")].copy()

    # drive sentiment from rating
    df["rating_sentiment"] = (
        df["Rate"].map(RATE_TO_SENTIMENT)
    )

    # normalize original sentiment labels
    df["original_sentiment"] = (
        df["Sentiment"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # compare labels of rating_sentiment and original_sentiment
    df["labels_agree"] = (
        df["original_sentiment"] == df["rating_sentiment"]
    )

    #rating derived sentiment is used as target
    df["target_sentiment"] = df["rating_sentiment"]

    # derive target label from target sentiment
    df["target_label"] = (
        df["target_sentiment"]
        .map(LABEL_MAPPING)
    )

    #check for valid target labels
    if df["target_label"].isna().any():
        raise ValueError(
            f"Unknown sentiments in the sentiment column"
        )

    # convert the target label as numeric column
    df["target_label"] = df["target_label"].astype(int)

    return df

#-------------------------------------------------------
# Find conflicting sentiments for the same review
#-------------------------------------------------------
def find_conflicting_sentiments(df: pd.DataFrame) -> pd.Index:
    """
    Find same reviews assigned with conflicting target sentiments
    """
    label_counts = (
        df.groupby("customer_review")["target_sentiment"].nunique()
    )

    return label_counts[label_counts > 1].index

#-------------------------------------------------------
# Create confidence datasets by comparing sentiment and rating
#-------------------------------------------------------
def create_confidence_datasets(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    split the dataset according to label-source agreement
    high_confidence_dataset :  rating derived sentiment agrees with original sentiment
    low_confidence_dataset : disagrees
    """
    high_confidence_dataset = (
        df.loc[df["labels_agree"]].copy()
    )

    low_confidence_dataset = (
        df.loc[~df["labels_agree"]].copy()
    )

    return high_confidence_dataset, low_confidence_dataset

#-------------------------------------------------------
# create final model dataset from high confidence dataset
#-------------------------------------------------------
def create_final_model_data(high_confidence_dataset: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Create final model dataset from the high confidence dataset
    Steps: 
    - Remove duplicate records
    - remove same user reviews with conflicting labels
    - remove repeated reviews
    - retain only model input and target columns
    """
    df = high_confidence_dataset.copy()

    no_of_datasets = len(df)

    #remove duplicate records
    df = df.drop_duplicates().copy()

    no_of_duplicates = (no_of_datasets - len(df))

    # find any remaining reviews has conflicting labels
    conflicting_texts = find_conflicting_sentiments(df)

    no_of_conflicting_reviews = int(
        df["customer_review"].isin(conflicting_texts).sum()
    )

    #remove conflicting reviews
    df = df.loc[~df["customer_review"].isin(conflicting_texts)].copy()

    #drop duplicates from customer_review
    rows_b4_dup_removed = len(df)

    df = df.drop_duplicates(
        subset=["customer_review"],
        keep="first"
    ).copy()

    no_of_dup_reviews_removed = (rows_b4_dup_removed - len(df))

    final_model_data = (
        df[FINAL_COLUMNS]
        .reset_index(drop=True)
    )

    if final_model_data.empty:
        raise ValueError(
            "No record found, records are completely deleted during preprocessing"
        )

    #check for final remaining conflict
    remaining_conflicts = (
        final_model_data.groupby("customer_review")["target_sentiment"]
        .nunique()
        .gt(1)
        .sum()
    )

    if remaining_conflicts:
        raise ValueError(
            f"{remaining_conflicts} conflicting reviews are still in the data set"
        )

    report = {
        "high_confidence_rows_before_cleaning" : len(high_confidence_dataset),
        "exact_duplicates_removed" : no_of_duplicates,
        "distinct_conflicting_reviews_removed" : len(conflicting_texts),
        "rows_affected_by_confliects" : no_of_conflicting_reviews,
        "duplicate_review_texts_removed" : no_of_dup_reviews_removed,
        "final_model_rows" : len(final_model_data),
        "final_cloumns" : final_model_data.columns.to_list(),
    }

    return final_model_data, report

#-------------------------------------------------------
# Workflow for Data Preprocessing
#-------------------------------------------------------
def preprocess_data(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    Complete Preprocessing workflow

    Returns: 
        final_model_data -> df
        high_confidence_feedback -> df
        low_confidence_feedback -> df
        preprocessing_report -> dict
    """
    validate_schema(raw_df)

    df, numeric_report = convert_numeric_columns(raw_df)

    df = create_review_and_labels(df)

    rows_after_text_validation = len(df)

    high_confidence_dataset, low_confidence_dataset = (
        create_confidence_datasets(df)
    )

    final_model_data, model_report = (
        create_final_model_data(high_confidence_dataset)
    )

    preprocessing_report = {
        **numeric_report,
        "rows_after_text_validatation" : rows_after_text_validation,
        "high_confidence_rows": len(high_confidence_dataset),
        "low_confidence_rows": len(low_confidence_dataset),
        **model_report,
        "label_mapping": LABEL_MAPPING,
        "rating_to_sentiment": RATE_TO_SENTIMENT
    }

    return (
        final_model_data,
        high_confidence_dataset,
        low_confidence_dataset,
        preprocessing_report
    )

#-------------------------------------------------------
# Save processed datasets
#-------------------------------------------------------
def save_datasets(
        final_model_data: pd.DataFrame,
        high_confidence_dataset: pd.DataFrame,
        low_confidence_dataset: pd.DataFrame,
        preprocessing_report: dict,
        output_dir: Path
) -> None:
    """
    Save the processed Dataframe 
    save also the report as json 
    """
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    model_data_path = (
        output_dir / "cleaned_dataset.csv"
    )

    high_confidence_path = (
        output_dir / "high_confidence_feedback.csv"
    )

    low_confidence_path = (
        output_dir / "low_confidence_feedback.csv"
    )

    report_path = (
        output_dir / "preprocessing_report.json"
    )

    final_model_data.to_csv(
        model_data_path,
        index=False
    )

    high_confidence_dataset.to_csv(
        high_confidence_path,
        index=False,
    )

    low_confidence_dataset.to_csv(
        low_confidence_path,
        index=False,
    )

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(preprocessing_report, file, indent=2)

#-------------------------------------------------------
# parse input arguments
#-------------------------------------------------------
def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to raw Flipkart dataset"
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory for processed datasets and reports."
    )

    args = parser.parse_args()

    return args

def main():
    args = parse_arguments()

    #check the input data is existing in the given path
    if not args.input.exists():
        raise FileNotFoundError(
            f"{args.input}Input data not Found"
        )

    raw_df = pd.read_csv(args.input)

    (
        final_model_data,
        high_confidence_dataset,
        low_confidence_dataset,
        preprocessing_report
    ) = preprocess_data(raw_df)

    save_datasets(
        final_model_data=final_model_data,
        high_confidence_dataset=high_confidence_dataset,
        low_confidence_dataset=low_confidence_dataset,
        preprocessing_report=preprocessing_report,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main()