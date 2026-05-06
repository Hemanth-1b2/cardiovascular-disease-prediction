# =========================================
# CARDIOVASCULAR DISEASE PREDICTION PROJECT
# =========================================

# 1. IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


# 2. LOAD DATASET
df = pd.read_csv('cardio_train (1).csv', sep=';')

print("Dataset Shape:", df.shape)
print(df.head())


# 3. DATA PREPROCESSING
# Drop ID column
df = df.drop('id', axis=1)

# Convert age from days to years
df['age'] = df['age'] / 365

# Check missing values
print("\nMissing Values:\n", df.isnull().sum())

# Remove unrealistic outliers (important step)
df = df[(df['ap_hi'] < 200) & (df['ap_lo'] < 150)]

print("After Cleaning Shape:", df.shape)


# 4. DATA VISUALIZATION

# Histograms
df.hist(figsize=(12,10))
plt.suptitle("Feature Distributions")
plt.show()

# Countplot (Target variable)
sns.countplot(x='cardio', data=df)
plt.title("Heart Disease Distribution")
plt.show()

# Boxplot
plt.figure(figsize=(12,6))
sns.boxplot(data=df)
plt.xticks(rotation=90)
plt.title("Boxplot for Outlier Detection")
plt.show()


# 5. CORRELATION MATRIX
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()


# 6. SPLIT DATA
X = df.drop('cardio', axis=1)
y = df['cardio']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)


# 7. TRAIN MODELS
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC()
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    results[name] = acc

    print("\n==============================")
    print(name)
    print("Accuracy:", acc)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))


# 8. ACCURACY COMPARISON
plt.figure(figsize=(8,5))
plt.bar(results.keys(), results.values())
plt.xticks(rotation=45)
plt.title("Model Accuracy Comparison")
plt.show()

print("\nFinal Accuracy Results:\n", results)


# 9. BEST MODEL
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]

print("\nBest Model:", best_model_name)


# 10. FINAL PREDICTION FUNCTION
def predict_heart_disease(input_data):
    input_data = np.array(input_data).reshape(1, -1)
    input_data = scaler.transform(input_data)
    prediction = best_model.predict(input_data)

    if prediction[0] == 1:
        return "Heart Disease Detected"
    else:
        return "No Heart Disease"


# Example prediction
sample = X.iloc[0]
print("\nSample Prediction:", predict_heart_disease(sample))