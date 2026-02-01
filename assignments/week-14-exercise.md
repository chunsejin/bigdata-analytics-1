# 14주차 실습과제: 특수 주제 및 심화 분석

## 과제 목표
- 시계열 분석 기초
- 자연어 처리 기초
- 특수 머신러닝 기법
- 실무 사례 분석

## 1. 시계열 분석 (25점)

### 1.1 시계열 데이터 처리

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# 시계열 데이터 생성
dates = pd.date_range('2024-01-01', periods=365)
values = np.cumsum(np.random.randn(365)) + 100

df_ts = pd.DataFrame({
    'date': dates,
    'value': values
})
df_ts.set_index('date', inplace=True)

# 1. 시계열 특성 추출
df_ts['year'] = df_ts.index.year
df_ts['month'] = df_ts.index.month
df_ts['quarter'] = df_ts.index.quarter
df_ts['dayofweek'] = df_ts.index.dayofweek
df_ts['is_weekend'] = df_ts['dayofweek'].isin([5, 6]).astype(int)

# 2. 이동 평균 (Moving Average)
df_ts['ma_7'] = df_ts['value'].rolling(window=7).mean()
df_ts['ma_30'] = df_ts['value'].rolling(window=30).mean()

# 3. 지수 평활 (Exponential Smoothing)
df_ts['ema_12'] = df_ts['value'].ewm(span=12).mean()

# 4. 변화율 (Rate of Change)
df_ts['returns'] = df_ts['value'].pct_change()
df_ts['log_returns'] = np.log(df_ts['value'] / df_ts['value'].shift(1))

# 시각화
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# 원본 데이터 + 이동평균
axes[0].plot(df_ts.index, df_ts['value'], label='Original', alpha=0.7)
axes[0].plot(df_ts.index, df_ts['ma_7'], label='MA(7)', linewidth=2)
axes[0].plot(df_ts.index, df_ts['ma_30'], label='MA(30)', linewidth=2)
axes[0].legend()
axes[0].set_title('Moving Average')

# 수익률
axes[1].plot(df_ts.index, df_ts['returns'], label='Returns', alpha=0.7)
axes[1].set_title('Daily Returns')
axes[1].legend()

# 누적 수익률
axes[2].plot(df_ts.index, (1 + df_ts['returns']).cumprod(), label='Cumulative Returns')
axes[2].set_title('Cumulative Returns')
axes[2].legend()

plt.tight_layout()
plt.show()

# 통계
print(f"평균: {df_ts['value'].mean():.2f}")
print(f"표준편차: {df_ts['value'].std():.2f}")
print(f"자기상관: {df_ts['value'].autocorr():.4f}")
```

### 1.2 ARIMA 모델

```python
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# 1. 정상성 확인
from statsmodels.tsa.stattools import adfuller

result = adfuller(df_ts['value'].dropna())
print(f"ADF 검정통계량: {result[0]:.4f}")
print(f"P-값: {result[1]:.4f}")

# 2. ACF/PACF 플롯
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(df_ts['value'].dropna(), ax=axes[0], lags=40)
plot_pacf(df_ts['value'].dropna(), ax=axes[1], lags=40)
plt.show()

# 3. ARIMA 모델 적합
model = ARIMA(df_ts['value'], order=(1, 1, 1))
fitted_model = model.fit()
print(fitted_model.summary())

# 4. 예측
forecast = fitted_model.get_forecast(steps=30)
forecast_df = forecast.conf_int()

# 예측 시각화
plt.figure(figsize=(12, 6))
plt.plot(df_ts.index, df_ts['value'], label='Original')
plt.plot(forecast_df.index, forecast_df['mean'], label='Forecast')
plt.fill_between(forecast_df.index,
                  forecast_df.iloc[:, 0],
                  forecast_df.iloc[:, 1],
                  alpha=0.2)
plt.legend()
plt.show()
```

### 1.3 변수 선택 및 모델 적합

```python
# auto_arima로 자동 매개변수 선택
from pmdarima import auto_arima

auto_model = auto_arima(
    df_ts['value'],
    start_p=0, start_q=0, max_p=5, max_q=5,
    seasonal=False,
    stepwise=True,
    trace=True
)

print(auto_model.summary())
```

---

## 2. 자연어 처리 (NLP) 기초 (25점)

### 2.1 텍스트 전처리

```python
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re

# 필수 NLTK 데이터 다운로드
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# 샘플 텍스트
text = """Natural language processing (NLP) is a subfield of linguistics, 
computer science, and artificial intelligence concerned with the 
interactions between computers and human language."""

# 1. 소문자 변환
text_lower = text.lower()

# 2. 특수문자 제거
text_clean = re.sub(r'[^a-zA-Z0-9\s]', '', text_lower)

# 3. 토큰화
tokens = word_tokenize(text_clean)
print(f"토큰: {tokens[:10]}")

# 4. 불용어 제거
stop_words = set(stopwords.words('english'))
tokens_filtered = [w for w in tokens if w not in stop_words]
print(f"필터링된 토큰: {tokens_filtered[:10]}")

# 5. 어간 추출 (Stemming)
stemmer = PorterStemmer()
tokens_stemmed = [stemmer.stem(w) for w in tokens_filtered]
print(f"어간 추출: {tokens_stemmed[:10]}")

# 6. 표제어 추출 (Lemmatization)
lemmatizer = WordNetLemmatizer()
tokens_lemma = [lemmatizer.lemmatize(w) for w in tokens_filtered]
print(f"표제어 추출: {tokens_lemma[:10]}")
```

### 2.2 텍스트 벡터화

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

texts = [
    "machine learning is powerful",
    "deep learning uses neural networks",
    "natural language processing is part of AI"
]

# 1. CountVectorizer
cv = CountVectorizer(stop_words='english', max_features=10)
count_matrix = cv.fit_transform(texts)
print("CountVectorizer:")
print(count_matrix.toarray())
print(cv.get_feature_names_out())

# 2. TF-IDF
tfidf = TfidfVectorizer(stop_words='english', max_features=10)
tfidf_matrix = tfidf.fit_transform(texts)
print("\nTF-IDF:")
print(tfidf_matrix.toarray())

# 3. Word2Vec
from gensim.models import Word2Vec

sentences = [text.split() for text in texts]
model = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=1)

print("\n단어 벡터:")
print(f"machine: {model.wv['machine'][:5]}")

# 유사 단어 찾기
similar = model.wv.most_similar('machine', topn=3)
print(f"'machine'과 유사한 단어: {similar}")
```

### 2.3 감정 분석

```python
from textblob import TextBlob

texts = [
    "I love this product! It's amazing.",
    "This is terrible. I hate it.",
    "It's okay, nothing special."
]

for text in texts:
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    
    print(f"Text: {text}")
    print(f"Polarity: {polarity:.2f}, Subjectivity: {subjectivity:.2f}\n")
```

---

## 3. 앙상블 고급 기법 (15점)

### 3.1 Stacking

```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# 기본 학습자
base_learners = [
    ('rf', RandomForestClassifier(n_estimators=100)),
    ('dt', DecisionTreeClassifier()),
    ('svm', SVC(probability=True))
]

# 메타 학습자
meta_learner = LogisticRegression()

# Stacking
stacking = StackingClassifier(
    estimators=base_learners,
    final_estimator=meta_learner,
    cv=5
)

stacking.fit(X_train, y_train)
stacking_pred = stacking.predict(X_test)
stacking_acc = accuracy_score(y_test, stacking_pred)

print(f"Stacking 정확도: {stacking_acc:.4f}")
```

### 3.2 Blending

```python
# 데이터 분할
X_train_split, X_blend, y_train_split, y_blend = train_test_split(
    X_train, y_train, test_size=0.3, random_state=42
)

# 모델 학습
models = [
    RandomForestClassifier(n_estimators=100),
    GradientBoostingClassifier(),
    SVC(probability=True)
]

meta_features_train = np.zeros((X_blend.shape[0], len(models)))
meta_features_test = np.zeros((X_test.shape[0], len(models)))

for i, model in enumerate(models):
    model.fit(X_train_split, y_train_split)
    meta_features_train[:, i] = model.predict_proba(X_blend)[:, 1]
    meta_features_test[:, i] = model.predict_proba(X_test)[:, 1]

# 메타 모델
meta_model = LogisticRegression()
meta_model.fit(meta_features_train, y_blend)
blending_pred = meta_model.predict(meta_features_test)
blending_acc = accuracy_score(y_test, blending_pred)

print(f"Blending 정확도: {blending_acc:.4f}")
```

---

## 4. 실무 사례 분석 (20점)

### 4.1 신용 위험 분석

```python
# 신용 데이터 분석 예시
credit_data = pd.DataFrame({
    'age': np.random.randint(20, 80, 1000),
    'income': np.random.randint(20000, 200000, 1000),
    'credit_score': np.random.randint(300, 850, 1000),
    'default': np.random.binomial(1, 0.1, 1000)
})

# EDA
print(credit_data.describe())
print(credit_data.groupby('default').mean())

# 모델 구축
X = credit_data[['age', 'income', 'credit_score']]
y = credit_data['default']

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# ROC 곡선
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)

plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
```

### 4.2 추천 시스템 기초

```python
from sklearn.metrics.pairwise import cosine_similarity

# 사용자-상품 평점 행렬
ratings = np.array([
    [5, 0, 3, 0],
    [4, 0, 0, 2],
    [0, 2, 0, 4],
    [1, 0, 4, 0]
])

# 코사인 유사도 계산
user_similarity = cosine_similarity(ratings)
item_similarity = cosine_similarity(ratings.T)

# 사용자 기반 협업 필터링
def recommend_user_based(user_id, n_users=2):
    similar_users = np.argsort(user_similarity[user_id])[::-1][1:n_users+1]
    recommendations = np.mean(ratings[similar_users], axis=0)
    return np.argsort(recommendations)[::-1]

# 아이템 기반 협업 필터링
def recommend_item_based(user_id, n_items=2):
    rated_items = np.nonzero(ratings[user_id])[0]
    similar_items = np.argsort(item_similarity[rated_items[0]])[::-1][:n_items]
    return similar_items

print("사용자 기반 추천:", recommend_user_based(0))
print("아이템 기반 추천:", recommend_item_based(0))
```

---

## 5. 종합 분석 프로젝트 (15점)

### 주제
실제 데이터셋 선택 후 시계열 분석, NLP, 또는 특수 기법 적용

요구사항:
- 데이터 수집 및 전처리
- 탐색적 분석
- 적절한 기법 적용
- 결과 해석 및 시각화

---

## 제출 방법

1. **제출 파일:**
   - week14_timeseries_analysis.py
   - week14_nlp_basics.py
   - week14_ensemble_advanced.py
   - week14_domain_applications.py
   - week14_project.py
   - advanced_topics_report.pdf

2. **제출 기한:** 다음 주 월요일 23:59
3. **제출 방식:** GitHub 레포지토리

---

## 평가 기준

| 항목 | 배점 |
|------|------|
| 시계열 분석 | 25점 |
| NLP 기초 | 25점 |
| 앙상블 고급 기법 | 15점 |
| 실무 사례 분석 | 20점 |
| 종합 프로젝트 | 15점 |
| **소계** | **100점** |
