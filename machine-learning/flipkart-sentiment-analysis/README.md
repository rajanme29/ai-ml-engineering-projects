# Flipkart Customer Review Sentiment Analysis

## Overview

Customer reviews provide valuable insights into product quality, customer satisfaction, and user experience. Analyzing large volumes of textual feedback manually is time-consuming and difficult to scale.

This project implements a sentiment analysis pipeline for Flipkart customer reviews. The objective is to automatically classify customer feedback into one of three sentiment categories or unknown:

- Positive
- Neutral
- Negative
- Unknown

The project covers the complete machine learning workflow, including:

- Exploratory Data Analysis (EDA)
- Data cleaning and preprocessing
- Feature engineering
- Sentiment classification model development
- Model evaluation
- Production-ready deployment

The final solution transforms unstructured customer feedback into actionable insights that can support product quality monitoring and customer experience analysis.


## Dataset

The dataset contains customer reviews collected from Flipkart products and includes product information, ratings, review titles, review descriptions, and sentiment labels.


**Source:**

- Kaggle Dataset: [Flipkart Product reviews with sentiment Dataset](https://www.kaggle.com/datasets/niraliivaghani/flipkart-product-customer-reviews-dataset):

### Original Dataset Columns

| Column | Description |
|----------|----------|
| product_name | Product title and description |
| product_price | Product price |
| Rate | Customer rating (1-5) |
| Review | Short review title |
| Summary | Detailed customer review |
| Sentiment | Sentiment label assigned to the review |

### Example Record

```text
Product Name:
Candes 12 L Room/Personal Air Cooler

Rate:
5

Review:
super!

Summary:
great cooler excellent air flow and for this price its so amazing and unbelievable just love it

Sentiment:
positive
```

# EDA Findings
## Exploratory Data Analysis

The exploratory analysis focused on data quality, rating consistency,
duplicate records, class distribution, and the characteristics of the
customer-review text.

### Missing Values

Missing values were examined for all columns, particularly 'Review' and
'Summary'. Missing text values were replaced with empty strings during
the creation of the combined review feature. Records with no usable text
in either field were excluded.

### Review and Summary Combination

The short review title and detailed review summary were combined into a
single text feature named 'customer_review'.

### Target Label Quality

Exploratory analysis showed that neither the numerical rating nor the supplied
'Sentiment' column could be treated as completely reliable ground truth.

Several records contained obvious inconsistencies. For example, clearly
negative reviews such as 'absolute rubbish' and 'worst product' were sometimes
assigned a positive sentiment label. In other cases, positive review text was
associated with a one-star rating.

To reduce label noise, the rating was converted into a rating-based sentiment
using the following rule:

- Ratings 1–2: negative
- Rating 3: neutral
- Ratings 4–5: positive

Only records where the rating-derived sentiment and the normalized original
sentiment agreed were included in the high-confidence training pool.
Disagreement records were quarantined for separate analysis rather than
silently overwritten.

The remaining data was normalized and checked for identical review texts with
conflicting labels. Ambiguous review texts were excluded, and repeated reviews
were deduplicated before the train-test split to reduce information leakage.

### Label Consistency Analysis

The original dataset contained inconsistencies between customer ratings and sentiment labels. To reduce label noise, a high-confidence subset was created by retaining only records where the rating-derived sentiment agreed with the provided sentiment label.

This process reduced label conflicts from 442 unique review texts to 39 unique review texts. The remaining conflicting review texts represented ambiguous customer feedback with multiple sentiment assignments and were therefore removed from the final modeling dataset.

After resolving label conflicts, duplicate review texts were removed to prevent information leakage between training and test data.

# Data Preprocessing

The raw Flipkart dataset required several data-quality and text-preprocessing steps before it could be used for model training.

## Numeric Data Validation

The 'Rate' and 'product_price' columns were initially loaded as text and converted to numeric values using controlled coercion.

Rows containing missing, non-numeric, or out-of-range ratings were excluded because a valid rating between 1 and 5 was required for sentiment validation.

Invalid product prices were not used for sentiment prediction because the baseline model was intentionally designed to classify sentiment from customer-review text alone.

## Text Preparation

The short 'Review' title and the detailed 'Summary' were combined into a single feature named 'customer_review'.

# Model Selection
## Selected Model

The baseline sentiment classification model was implemented using:

TF-IDF Vectorizer
+
Logistic Regression

# Performance Metrics
## Model Performance

The baseline TF-IDF and Logistic Regression pipeline achieved the following
performance on the stratified test set:

| Metric | Score |
|---|---:|
| Accuracy | 98.62% |
| Macro Precision | 96.39% |
| Macro Recall | 87.23% |
| Macro F1 | 90.99% |
| Weighted F1 | 98.53% |

The model performed strongly for positive and negative sentiment. Neutral
sentiment remained the most challenging class:

| Sentiment | Precision | Recall | F1 |
|---|---:|---:|---:|
| Negative | 98.64% | 97.77% | 98.20% |
| Neutral | 91.78% | 64.14% | 75.51% |
| Positive | 98.74% | 99.77% | 99.26% |

Although overall accuracy is high, the final dataset is strongly imbalanced:
approximately 80.06% positive, 17.69% negative, and 2.26% neutral. Therefore,
macro F1 and neutral-class recall were used alongside accuracy to evaluate the
model more appropriately.

The main error pattern was the classification of neutral reviews as positive.
Future experiments will evaluate class weighting, character-level TF-IDF
features, and lightweight transformer models to improve minority-class recall.

# FastAPI Deployment
## Overview

To make the sentiment classification model accessible through a simple and scalable interface, the trained model was deployed as a REST API using **FastAPI**.

FastAPI was selected because it provides:

- High-performance API serving
- Automatic request validation using Pydantic
- Interactive Swagger documentation
- Easy integration with machine learning pipelines
- Container-friendly deployment

The API loads the trained sentiment classification pipeline and exposes prediction endpoints that allow users to classify customer reviews in real time.

---

## API Architecture
```text
Customer Review
      ↓
FastAPI Endpoint
      ↓
CustomerSentiment Class
      ↓
TF-IDF Vectorizer
      ↓
Logistic Regression Model
      ↓
Sentiment Prediction
```

# Docker Deployment
To ensure reproducible deployment and simplify environment management, the sentiment analysis application is packaged as a Docker container.

The container includes:

- FastAPI application
- Sentiment inference service
- Trained TF-IDF + Logistic Regression model
- Required Python dependencies

# CI/CD
## Overview

To ensure code quality, reproducibility, and deployment readiness, a Continuous Integration (CI) pipeline was implemented using **GitHub Actions**.

The pipeline automatically validates every code change by executing unit tests, building the Docker image, and verifying that the API can start successfully inside a clean environment.

This reduces the risk of regressions and helps ensure that the application remains deployable throughout the development lifecycle.

---

## CI Pipeline Workflow

The GitHub Actions workflow is defined in:

```text
.github/workflows/ci.yml
```

The pipeline executes the following stages:

```text
Checkout Repository
        ↓
Setup Python Environment
        ↓
Install Dependencies
        ↓
Build Docker Image
        ↓
Start Docker Container
        ↓
Execute Health Check
        ↓
Stop and Remove Container
```

# Future Improvements
