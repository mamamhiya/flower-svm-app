import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# 1) Load data
df = pd.read_csv("synthetic_flower_classification.csv")

# 2) Separate features and target
X = df[["sepal_length_cm", "sepal_width_cm",
        "petal_length_cm", "petal_width_cm"]]
y = df["species"]

# 3) Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 4) Pipeline prevents data leakage during CV
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf"))
])

# 5) Tune RBF hyperparameters
param_grid = {
    "svm__C": [0.1, 1, 10, 100],
    "svm__gamma": ["scale", 0.01, 0.1, 1]
}

grid = GridSearchCV(
    pipeline, param_grid, cv=5, scoring="accuracy", n_jobs=-1
)
grid.fit(X_train, y_train)

# 6) Evaluate
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)
print("Best parameters:", grid.best_params_)
print("Test accuracy:", accuracy_score(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 7) Save the COMPLETE pipeline (Scaler + SVM)
joblib.dump(best_model, "svm_flower_model.joblib")
print("Saved: svm_flower_model.joblib")
