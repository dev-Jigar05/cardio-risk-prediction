import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
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


# pipe_lr = Pipeline([
#     ("scaler", StandardScaler()),
#     ("model", LogisticRegression(max_iter=1000))
# ])

# pipe_rf = Pipeline([

#     ("model", RandomForestClassifier(
#         n_estimators=200,
#         random_state=42
#     ))
# ])

# pipe_dt = Pipeline([
#     ("scaler",StandardScaler()),
#     ("clf",DecisionTreeClassifier(random_state=42))
# ])

# pipe_xgb = Pipeline([
#     ("scaler",StandardScaler()),
#     ("clf",XGBClassifier(
#         eval_metric = "logloss",
#         random_state = 42,
#         use_label_encoder = False
#     ))
# ])

# pipe_gb = Pipeline([
#     ("scaler",StandardScaler()),
#     ("clf",GradientBoostingClassifier())
# ])

pipe_knn = Pipeline([
    ("scaler",StandardScaler()),
    ("clf",KNeighborsClassifier())
])

# pipe_lr.fit(X_train, y_train)
# pipe_rf.fit(X_train, y_train)
# pipe_dt.fit(X_train,y_train)
# pipe_xgb.fit(X_train,y_train)
# pipe_gb.fit(X_train,y_train)

pipe_knn.fit(X_train,y_train)

y_pred = pipe_knn.predict(X_test)

print("Accuracy of Decision Tree Classifier: ",accuracy_score(y_test,y_pred))

# joblib.dump(pipe_lr, "cardio_model_lr.pkl")
# joblib.dump(pipe_rf, "cardio_model_rf.pkl")
joblib.dump(pipe_knn,"cardio_model_knn.pkl")

print("Saved fitted models.")

