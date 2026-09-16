import joblib
from pathlib import Path
from preprocessing import clean_text

LABEL_TO_SENTIMENT = {
    0: "negative",
    1: "neutral",
    2: "positive",
}

class SentimentPredictor:
    """
    Wrapper class to load the prediction model and predict the customer sentiment
    """

    def __init__(self, model_path: str | None=None):
        #dynamically identify the project root folder
        proj_path = Path(__file__).resolve().parents[1]

        if model_path is None:
            model_path = proj_path / "model" / "sentiment_pipeline.joblib"
        self.model_path = Path(model_path)

        #load the model 
        self.model = self._load_model()

    def _load_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")

        return joblib.load(self.model_path)

    def predict_sentiment(self, user_comment:str) -> dict:
        #check the user comment contains a strig to be evaluated
        if user_comment is None or not str(user_comment).strip():
            raise ValueError('User comment must not be empty')

        #clean the noisy character from user comment
        cleaned_comment = clean_text(user_comment)

        #predict the sentiment
        label = self.model.predict([cleaned_comment])[0]
        sentiment = (
            LABEL_TO_SENTIMENT[label]
        )
        print(sentiment)

        #predict the probability
        all_probabilities = self.model.predict_proba([cleaned_comment])[0]
       
        probs = {
            cls: round(float(prob), 4)
            for cls, prob in zip(
                self.model.classes_,
                all_probabilities
            )
        }

        sentiment_probability = probs[label]

        #check whether the probability fulfill the threshold
        if sentiment_probability < 0.60:
            sentiment = "unknown"

        #return the result as dict
        return {
            'sentiment' : sentiment,
            'confidence' : sentiment_probability
        }

if __name__ == "__main__":
    predictor = SentimentPredictor()

    # Test sentiment predicition
    print('=' * 80)
    print("Testing Sentiment Predictions")
    print('=' * 80)

    sentiment_1 = "The claim process was very slow and communication was poor."
    sentiment_2 = "The service was fast, friendly and very professional."
    sentiment_3 = "The policy document was updated yesterday."
    sentiment_4 = "Fair quality"
    sentiment_5 = "must buy! awsm"


    print(f"\nSentiment_1 : {sentiment_1} ==> predicted as '{predictor.predict_sentiment(sentiment_1)}'")
    print(f"\nSentiment_2 : {sentiment_2} ==> predicted as '{predictor.predict_sentiment(sentiment_2)}'")
    print(f"\nSentiment_3 : {sentiment_3} ==> predicted as '{predictor.predict_sentiment(sentiment_3)}'")
    print(f"\nSentiment_4 : {sentiment_4} ==> predicted as '{predictor.predict_sentiment(sentiment_4)}'\n")
    print(f"\nSentiment_5 : {sentiment_5} ==> predicted as '{predictor.predict_sentiment(sentiment_5)}'\n")

    print('=' * 80)