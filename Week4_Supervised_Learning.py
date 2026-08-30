
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    roc_curve
)

# Load dataset
bc = load_breast_cancer(as_frame=True)
df = bc.frame.copy()

# Separate predictors and target
X = df.drop(columns=["target"])
y = df["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Scaling + Logistic Regression pipeline
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        max_iter=5000,
        random_state=42
    ))
])

# Train model
pipe.fit(X_train, y_train)

# Predictions
y_pred = pipe.predict(X_test)
y_prob = pipe.predict_proba(X_test)[:, 1]

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-score:", f1_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))

# Confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Five-fold stratified cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_accuracy = cross_val_score(
    pipe, X, y, cv=cv, scoring="accuracy"
)

print("CV Accuracy:", cv_accuracy)
print("Mean CV Accuracy:", cv_accuracy.mean())

# ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

# Feature coefficients
coefficients = pipe.named_steps["model"].coef_[0]

for feature, coefficient in sorted(
    zip(X.columns, coefficients),
    key=lambda x: abs(x[1]),
    reverse=True
):
    print(feature, coefficient)
