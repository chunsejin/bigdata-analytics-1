# 10주차 실습과제: 머신러닝 기초 (회귀 및 분류)

## 과제 목표
- 머신러닝 기본 개념 이해
- Scikit-learn 기초 사용
- 회귀 모델 구축 및 평가
- 분류 모델 기초

## 1. 머신러닝 기초 개념 (15점)

### 1.1 문제 정의

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. 데이터 준비
print("=== 머신러닝 워크플로우 ===")
print("1. 문제 정의 (Regression vs Classification)")
print("2. 데이터 수집 및 전처리")
print("3. 특성 엔지니어링")
print("4. 모델 선택")
print("5. 모델 학습")
print("6. 평가 및 튜닝")
print("7. 배포")

# 2. 데이터 로드
data = pd.DataFrame({
    'experience': [1, 3, 5, 7, 9, 11],
    'education': [12, 14, 16, 18, 20, 22],
    'salary': [25000, 35000, 50000, 65000, 80000, 95000]
})

X = data[['experience', 'education']]
y = data['salary']

# 3. 훈련/테스트 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n훈련 세트 크기: {X_train.shape}")
print(f"테스트 세트 크기: {X_test.shape}")

# 4. 데이터 정규화
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

---

## 2. 선형 회귀 (25점)

### 2.1 단순 선형 회귀

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 1. 모델 생성
model = LinearRegression()

# 2. 모델 학습
model.fit(X_train, y_train)

# 3. 예측
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# 4. 계수 확인
print(f"계수: {model.coef_}")
print(f"절편: {model.intercept_}")

# 5. 성능 평가
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)
train_mse = mean_squared_error(y_train, y_pred_train)
test_mse = mean_squared_error(y_test, y_pred_test)

print(f"훈련 R²: {train_r2:.4f}, 테스트 R²: {test_r2:.4f}")
print(f"훈련 MSE: {train_mse:.4f}, 테스트 MSE: {test_mse:.4f}")

# 6. 시각화
plt.scatter(X_test['experience'], y_test, label='실제값')
plt.plot(X_test['experience'], y_pred_test, 'r-', label='예측값')
plt.legend()
plt.show()
```

### 2.2 다중 선형 회귀

```python
# 이미 위에서 구현됨
# X에 여러 특성이 포함되는 경우

# 규제화된 회귀 (Ridge)
from sklearn.linear_model import Ridge

ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)
ridge_pred = ridge_model.predict(X_test)
ridge_r2 = r2_score(y_test, ridge_pred)

print(f"Ridge R²: {ridge_r2:.4f}")
```

### 2.3 비선형 회귀

```python
from sklearn.preprocessing import PolynomialFeatures

# 다항 회귀 (2차)
poly = PolynomialFeatures(degree=2)
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)

poly_model = LinearRegression()
poly_model.fit(X_poly_train, y_train)
poly_pred = poly_model.predict(X_poly_test)
poly_r2 = r2_score(y_test, poly_pred)

print(f"다항 회귀 R²: {poly_r2:.4f}")
```

---

## 3. 분류 기초 (25점)

### 3.1 로지스틱 회귀

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 이진 분류 데이터
iris_data = pd.read_csv('iris.csv')
X = iris_data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = (iris_data['species'] == 'setosa').astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 로지스틱 회귀 모델
log_model = LogisticRegression()
log_model.fit(X_train, y_train)

# 예측
y_pred = log_model.predict(X_test)
y_pred_proba = log_model.predict_proba(X_test)

# 성능 평가
accuracy = accuracy_score(y_test, y_pred)
print(f"정확도: {accuracy:.4f}")

# 혼동 행렬
cm = confusion_matrix(y_test, y_pred)
print(f"혼동 행렬:\n{cm}")

# 분류 보고서
print(classification_report(y_test, y_pred))
```

### 3.2 결정 트리

```python
from sklearn.tree import DecisionTreeClassifier

# 결정 트리 모델
tree_model = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_model.fit(X_train, y_train)

# 예측 및 성능 평가
y_pred_tree = tree_model.predict(X_test)
tree_accuracy = accuracy_score(y_test, y_pred_tree)
print(f"결정 트리 정확도: {tree_accuracy:.4f}")

# 특성 중요도
print(f"특성 중요도: {tree_model.feature_importances_}")

# 트리 시각화
from sklearn import tree
plt.figure(figsize=(20, 10))
tree.plot_tree(tree_model, feature_names=X.columns, filled=True)
plt.show()
```

### 3.3 랜덤 포레스트

```python
from sklearn.ensemble import RandomForestClassifier

# 랜덤 포레스트
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_pred_rf)
print(f"랜덤 포레스트 정확도: {rf_accuracy:.4f}")

# 특성 중요도
print(f"특성 중요도:\n{pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)}")
```

---

## 4. 모델 평가 및 선택 (20점)

### 4.1 교차 검증

```python
from sklearn.model_selection import cross_val_score, cross_validate

# K-fold 교차 검증
cv_scores = cross_val_score(LogisticRegression(), X, y, cv=5, scoring='accuracy')
print(f"교차 검증 점수: {cv_scores}")
print(f"평균: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# 상세 교차 검증
scoring = {'accuracy': 'accuracy', 'precision': 'precision', 'recall': 'recall'}
cv_results = cross_validate(LogisticRegression(), X, y, cv=5, scoring=scoring)
print(f"다중 지표 교차 검증: {cv_results}")
```

### 4.2 하이퍼파라미터 튜닝

```python
from sklearn.model_selection import GridSearchCV

# 그리드 서치
param_grid = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid,
    cv=5,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
print(f"최적 파라미터: {grid_search.best_params_}")
print(f"최적 점수: {grid_search.best_score_:.4f}")

# 최적 모델로 예측
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)
best_accuracy = accuracy_score(y_test, y_pred_best)
print(f"최적 모델 정확도: {best_accuracy:.4f}")
```

### 4.3 성능 곡선

```python
from sklearn.model_selection import learning_curve

# 학습 곡선
train_sizes, train_scores, val_scores = learning_curve(
    RandomForestClassifier(random_state=42),
    X, y,
    cv=5,
    train_sizes=np.linspace(0.1, 1.0, 10),
    n_jobs=-1
)

train_mean = train_scores.mean(axis=1)
val_mean = val_scores.mean(axis=1)

plt.plot(train_sizes, train_mean, 'o-', label='훈련 점수')
plt.plot(train_sizes, val_mean, 'o-', label='검증 점수')
plt.legend()
plt.xlabel('훈련 세트 크기')
plt.ylabel('정확도')
plt.show()
```

---

## 5. 종합 프로젝트 (15점)

### 목표
붓꽃(Iris) 데이터셋으로 다중 분류 수행

```python
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. 데이터 로드
iris = datasets.load_iris()
X = iris.data
y = iris.target

# 2. 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. 정규화
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. 모델 생성 및 학습
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 5. 평가
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"정확도: {accuracy:.4f}")
print("\n분류 보고서:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
```

---

## 6. 보너스 과제 (+10점)

### 앙상블 모델

```python
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC

# 다양한 모델 조합
ensemble = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression()),
        ('dt', DecisionTreeClassifier()),
        ('rf', RandomForestClassifier(n_estimators=100))
    ],
    voting='soft'
)

ensemble.fit(X_train, y_train)
ensemble_pred = ensemble.predict(X_test)
ensemble_accuracy = accuracy_score(y_test, ensemble_pred)

print(f"앙상블 정확도: {ensemble_accuracy:.4f}")
```

---

## 제출 방법

1. **제출 파일:**
   - week10_ml_basics.py
   - week10_regression.py
   - week10_classification.py
   - week10_model_evaluation.py
   - week10_project.py
   - ml_analysis_report.pdf

2. **제출 기한:** 다음 주 월요일 23:59
3. **제출 방식:** GitHub 레포지토리

---

## 평가 기준

| 항목 | 배점 |
|------|------|
| ML 기초 개념 | 15점 |
| 회귀 모델 | 25점 |
| 분류 모델 | 25점 |
| 평가 및 선택 | 20점 |
| 종합 프로젝트 | 15점 |
| **소계** | **100점** |
| 보너스 | +10점 |
