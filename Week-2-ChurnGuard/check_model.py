
import joblib
import xgboost as xgb
from pathlib import Path

model_dir = Path("models")

preprocessor_path = model_dir / "final_preprocessor.joblib"
classifier_path = model_dir / "final_classifier.json"

preprocessor = joblib.load(preprocessor_path)

classifier = xgb.XGBClassifier()
classifier.load_model(classifier_path)

print("Both model components loaded successfully!")
print("Preprocessor:", type(preprocessor).__name__)
print("Classifier:", type(classifier).__name__)
print("Pipeline features:", len(preprocessor.get_feature_names_out()))
