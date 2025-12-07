import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv("cleaned.csv")  


X = df[[
    "age", "gender", "height", "weight",
    "ap_hi", "ap_lo", "cholesterol", "gluc",
    "smoke", "alco", "active", "bmi"
]]
y = df["cardio"]  

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


pipe_lr = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])

pipe_rf = Pipeline([
    ("model", RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ))
])

pipe_lr.fit(X_train, y_train)
pipe_rf.fit(X_train, y_train)

joblib.dump(pipe_lr, "cardio_model_lr.pkl")
joblib.dump(pipe_rf, "cardio_model_rf.pkl")

print("Saved fitted models.")
