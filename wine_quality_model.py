#  Required Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

#  Load Dataset
red = pd.read_csv('winequality-red.csv')
white = pd.read_csv('winequality-white.csv')
red['type'] = 'red'
white['type'] = 'white'
df = pd.concat([red, white], ignore_index=True)

#  Define features and target
X = df.drop(['quality', 'type'], axis=1)
y_reg = df['quality']  # For regression

#  Create binary label for classification
df['quality_label'] = df['quality'].apply(lambda q: 'good' if q > 5 else 'bad')
y_cls = df['quality_label']

#  Train-Test Split
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_reg, test_size=0.2, random_state=42)
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_cls, test_size=0.2, random_state=42)

#  Scaling
scaler = StandardScaler()
X_train_r_scaled = scaler.fit_transform(X_train_r)
X_test_r_scaled = scaler.transform(X_test_r)
X_train_c_scaled = scaler.fit_transform(X_train_c)
X_test_c_scaled = scaler.transform(X_test_c)

#  Linear Regression
reg = LinearRegression()
reg.fit(X_train_r_scaled, y_train_r)
y_pred_r = reg.predict(X_test_r_scaled)
print("Regression RMSE:", np.sqrt(mean_squared_error(y_test_r, y_pred_r)))

#  Logistic Regression for Classification
log_cls = LogisticRegression(max_iter=1000)
log_cls.fit(X_train_c_scaled, y_train_c)
y_pred_c = log_cls.predict(X_test_c_scaled)
print("Classification Accuracy:", accuracy_score(y_test_c, y_pred_c))
print("\nConfusion Matrix:\n", confusion_matrix(y_test_c, y_pred_c))
print("\nClassification Report:\n", classification_report(y_test_c, y_pred_c))

#  PCA and Re-Train
pca = PCA(n_components=5)
X_train_pca = pca.fit_transform(X_train_c_scaled)
X_test_pca = pca.transform(X_test_c_scaled)

log_pca = LogisticRegression(max_iter=1000)
log_pca.fit(X_train_pca, y_train_c)
y_pca_pred = log_pca.predict(X_test_pca)
print("\n[With PCA] Classification Accuracy:", accuracy_score(y_test_c, y_pca_pred))

#  Plot PCA Variance Explained
plt.figure(figsize=(8,4))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA - Variance Explained')
plt.grid(True)
plt.tight_layout()
plt.show()


