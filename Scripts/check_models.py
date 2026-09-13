import joblib

model = joblib.load(r"D:\AI-Fraud-Detection\models\xgb_behavior.pkl")
threshold = joblib.load(r"D:\AI-Fraud-Detection\models\threshold.pkl")
preprocessor = joblib.load(r"D:\AI-Fraud-Detection\models\preprocessor_b.pkl")
feature_names = joblib.load(r"D:\AI-Fraud-Detection\models\feature_names_b.pkl")

print("MODEL:")
print(type(model))

print("\nTHRESHOLD:")
print(threshold)

print("\nPREPROCESSOR TRANSFORMERS:")
print(preprocessor.transformers_)

print("\nFEATURE NAMES:")
print(type(feature_names))
print(feature_names)