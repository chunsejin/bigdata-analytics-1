# 11주차 실습과제: 고급 머신러닝 기법

## 과제 목표
- 신경망 기초 이해
- 차원 축소 기법 습득
- 비지도 학습 (클러스터링) 이해
- 이상치 탐지 기법

## 1. 신경망 기초 (20점)

### 1.1 다층 퍼셉트론 (MLP)

```python
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 분류용 신경망
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# 데이터 정규화
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 신경망 모델
mlp = MLPClassifier(
    hidden_layer_sizes=(100, 50),  # 2개 은닉층, 각각 100, 50 뉴런
    activation='relu',
    solver='adam',
    max_iter=1000,
    random_state=42
)

mlp.fit(X_train_scaled, y_train)
y_pred = mlp.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"신경망 정확도: {accuracy:.4f}")
print(f"손실: {mlp.loss_:.4f}")
```

### 1.2 Keras/TensorFlow 기초

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 1. 순차 모델 생성
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

# 2. 모델 컴파일
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 3. 모델 요약
model.summary()

# 4. 모델 학습 (MNIST 예시)
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
X_train = X_train.reshape(-1, 784) / 255.0
X_test = X_test.reshape(-1, 784) / 255.0

history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# 5. 모델 평가
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"테스트 정확도: {test_acc:.4f}")

# 6. 학습 곡선 시각화
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='훈련 정확도')
plt.plot(history.history['val_accuracy'], label='검증 정확도')
plt.legend()
plt.show()
```

---

## 2. 차원 축소 (20점)

### 2.1 PCA (Principal Component Analysis)

```python
from sklearn.decomposition import PCA
import numpy as np

# 1. PCA 적용
pca = PCA(n_components=2)
X_pca = pca.fit_transform(iris.data)

print(f"원본 차원: {iris.data.shape}")
print(f"축소된 차원: {X_pca.shape}")
print(f"설명 분산: {pca.explained_variance_ratio_}")
print(f"누적 설명 분산: {np.cumsum(pca.explained_variance_ratio_)}")

# 2. 차원 개수 결정 (95% 분산)
pca_95 = PCA(n_components=0.95)
X_pca_95 = pca_95.fit_transform(iris.data)
print(f"95% 분산 설명에 필요한 차원: {X_pca_95.shape[1]}")

# 3. 시각화
import matplotlib.pyplot as plt
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=iris.target)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()
```

### 2.2 t-SNE

```python
from sklearn.manifold import TSNE

# t-SNE 적용
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(iris.data)

plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=iris.target)
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.show()
```

### 2.3 자동 인코더

```python
from tensorflow.keras import layers, Model

# 인코더
input_img = layers.Input(shape=(784,))
encoded = layers.Dense(128, activation='relu')(input_img)
encoded = layers.Dense(64, activation='relu')(encoded)
encoded = layers.Dense(10, activation='relu')(encoded)

# 디코더
decoded = layers.Dense(64, activation='relu')(encoded)
decoded = layers.Dense(128, activation='relu')(decoded)
decoded = layers.Dense(784, activation='sigmoid')(decoded)

# 오토인코더
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# 학습
autoencoder.fit(X_train, X_train, epochs=10, batch_size=32, verbose=0)

# 인코더 추출
encoder = Model(input_img, encoded)
X_encoded = encoder.predict(X_test)
print(f"인코딩된 데이터 형태: {X_encoded.shape}")
```

---

## 3. 클러스터링 (20점)

### 3.1 K-Means

```python
from sklearn.cluster import KMeans

# K-Means 클러스터링
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(iris.data)
labels = kmeans.labels_

# 결과 시각화
plt.scatter(iris.data[:, 0], iris.data[:, 1], c=labels)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            marker='X', s=200, c='red')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# 엘보우 방법 (최적 클러스터 개수)
inertias = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(iris.data)
    inertias.append(kmeans.inertia_)

plt.plot(range(1, 11), inertias)
plt.xlabel('클러스터 개수')
plt.ylabel('관성')
plt.show()
```

### 3.2 계층 클러스터링

```python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# 계층 클러스터링
hierarchical = AgglomerativeClustering(n_clusters=3, linkage='ward')
h_labels = hierarchical.fit_predict(iris.data)

# 덴드로그램
linkage_matrix = linkage(iris.data, method='ward')
plt.figure(figsize=(12, 8))
dendrogram(linkage_matrix)
plt.show()
```

### 3.3 DBSCAN

```python
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# 데이터 정규화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(iris.data)

# DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)

print(f"클러스터 개수: {len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)}")
print(f"이상점 개수: {list(dbscan_labels).count(-1)}")

plt.scatter(iris.data[:, 0], iris.data[:, 1], c=dbscan_labels)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
```

---

## 4. 이상치 탐지 (20점)

### 4.1 Isolation Forest

```python
from sklearn.ensemble import IsolationForest

# Isolation Forest
iso_forest = IsolationForest(contamination=0.1, random_state=42)
outlier_labels = iso_forest.fit_predict(iris.data)

outlier_mask = outlier_labels == -1
print(f"이상점 개수: {outlier_mask.sum()}")

# 시각화
plt.scatter(iris.data[~outlier_mask, 0], iris.data[~outlier_mask, 1], 
           label='정상')
plt.scatter(iris.data[outlier_mask, 0], iris.data[outlier_mask, 1], 
           label='이상점', c='red')
plt.legend()
plt.show()
```

### 4.2 Local Outlier Factor

```python
from sklearn.neighbors import LocalOutlierFactor

# LOF
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
lof_labels = lof.fit_predict(iris.data)

lof_outlier_mask = lof_labels == -1
print(f"LOF 이상점 개수: {lof_outlier_mask.sum()}")
```

### 4.3 One-Class SVM

```python
from sklearn.svm import OneClassSVM

# One-Class SVM
oc_svm = OneClassSVM(gamma='auto', nu=0.05)
oc_labels = oc_svm.fit_predict(iris.data)

oc_outlier_mask = oc_labels == -1
print(f"One-Class SVM 이상점 개수: {oc_outlier_mask.sum()}")
```

---

## 5. 종합 프로젝트 (20점)

### 목표
다양한 머신러닝 기법을 실제 데이터에 적용

```python
# 손글씨 데이터셋
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 1. 신경망
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1000, random_state=42)
mlp.fit(X_train_scaled, y_train)
mlp_acc = mlp.score(X_test_scaled, y_test)
print(f"신경망 정확도: {mlp_acc:.4f}")

# 2. PCA + 분류
from sklearn.decomposition import PCA
pca = PCA(n_components=50)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

mlp_pca = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000)
mlp_pca.fit(X_train_pca, y_train)
mlp_pca_acc = mlp_pca.score(X_test_pca, y_test)
print(f"PCA + 신경망 정확도: {mlp_pca_acc:.4f}")

# 3. 클러스터링 분석
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=10, random_state=42)
kmeans.fit(X_train_scaled)
print(f"K-Means 관성: {kmeans.inertia_:.4f}")

# 4. 이상치 탐지
from sklearn.ensemble import IsolationForest
iso = IsolationForest(contamination=0.05, random_state=42)
iso.fit(X_train_scaled)
outliers = iso.predict(X_test_scaled)
print(f"이상점 개수: {(outliers == -1).sum()}")
```

---

## 6. 보너스 과제 (+10점)

### 앙상블 + 신경망 하이브리드

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# 다양한 모델 학습
rf = RandomForestClassifier(n_estimators=100, random_state=42)
gb = GradientBoostingClassifier(random_state=42)
mlp = MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42)

rf.fit(X_train_scaled, y_train)
gb.fit(X_train_scaled, y_train)
mlp.fit(X_train_scaled, y_train)

# 앙상블 예측
rf_pred = rf.predict_proba(X_test_scaled)
gb_pred = gb.predict_proba(X_test_scaled)
mlp_pred = mlp.predict_proba(X_test_scaled)

# 평균 앙상블
ensemble_pred = (rf_pred + gb_pred + mlp_pred) / 3
ensemble_labels = np.argmax(ensemble_pred, axis=1)

from sklearn.metrics import accuracy_score
ensemble_acc = accuracy_score(y_test, ensemble_labels)
print(f"하이브리드 앙상블 정확도: {ensemble_acc:.4f}")
```

---

## 제출 방법

1. **제출 파일:**
   - week11_neural_networks.py
   - week11_dimensionality_reduction.py
   - week11_clustering.py
   - week11_anomaly_detection.py
   - week11_project.py
   - advanced_ml_report.pdf

2. **제출 기한:** 다음 주 월요일 23:59
3. **제출 방식:** GitHub 레포지토리

---

## 평가 기준

| 항목 | 배점 |
|------|------|
| 신경망 기초 | 20점 |
| 차원 축소 | 20점 |
| 클러스터링 | 20점 |
| 이상치 탐지 | 20점 |
| 종합 프로젝트 | 20점 |
| **소계** | **100점** |
| 보너스 | +10점 |
