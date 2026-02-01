# 3주차 실습과제: 데이터 전처리

## 과제 목표
- 결측치 처리 기법 이해 및 적용
- 이상치 탐지 및 처리
- 데이터 정규화 및 표준화
- 특성 엔지니어링 기초

## 1. 결측치 처리 (25점)

### 1.1 결측치 탐지

```python
import pandas as pd
import numpy as np

# 결측치를 포함한 샘플 데이터
data = {
    'id': [1, 2, 3, 4, 5, 6, 7, 8],
    'age': [25, 30, np.nan, 35, 40, np.nan, 28, 32],
    'salary': [50000, 60000, 55000, np.nan, 75000, 65000, np.nan, 58000],
    'department': ['Sales', 'IT', 'HR', np.nan, 'IT', 'Sales', 'HR', np.nan]
}
df = pd.DataFrame(data)

# 1) 결측치 확인
print("Missing values per column:")
print(df.isnull().sum())

print("\nTotal missing values:", df.isnull().sum().sum())
print("\nMissing percentage:")
print((df.isnull().sum() / len(df)) * 100)

# 2) 결측치가 있는 행 확인
print("\nRows with missing values:")
print(df[df.isnull().any(axis=1)])

# 3) 결측치 위치 시각화
print("\nMissing data pattern:")
print(df.isnull())
```

### 1.2 결측치 처리 방법

```python
import pandas as pd
import numpy as np

data = {
    'id': [1, 2, 3, 4, 5, 6, 7, 8],
    'age': [25, 30, np.nan, 35, 40, np.nan, 28, 32],
    'salary': [50000, 60000, 55000, np.nan, 75000, 65000, np.nan, 58000],
}
df = pd.DataFrame(data)

# 방법 1: 결측치 행 제거
df_dropped = df.dropna()
print("After dropna():", df_dropped.shape)

# 방법 2: 특정 열의 결측치만 제거
df_dropped_col = df.dropna(subset=['age'])
print("After dropna(subset=['age']):", df_dropped_col.shape)

# 방법 3: 평균값으로 채우기
df_filled_mean = df.copy()
df_filled_mean['age'] = df_filled_mean['age'].fillna(df_filled_mean['age'].mean())
print("After fillna with mean:", df_filled_mean['age'].tolist())

# 방법 4: 중앙값으로 채우기
df_filled_median = df.copy()
df_filled_median['salary'] = df_filled_median['salary'].fillna(df_filled_median['salary'].median())
print("After fillna with median:", df_filled_median['salary'].tolist())

# 방법 5: 전방/후방 이월 (시계열 데이터)
df_ffill = df.copy()
df_ffill['age'] = df_ffill['age'].fillna(method='ffill')  # 전방 이월
print("After forward fill:", df_ffill['age'].tolist())

# 방법 6: 선형 보간
df_interpolate = df.copy()
df_interpolate['age'] = df_interpolate['age'].interpolate()
print("After interpolation:", df_interpolate['age'].tolist())

# 방법 7: 상수값으로 채우기
df_filled_const = df.copy()
df_filled_const['age'] = df_filled_const['age'].fillna(-1)  # -1로 표시
print("After fillna with constant:", df_filled_const['age'].tolist())
```

**실습 과제:**
```python
# 여러 결측치 처리 방법 비교
# 1. dropna() 사용
# 2. 평균값 사용
# 3. 중앙값 사용
# 각 방법 후 데이터 확인 및 장단점 정리
```

---

## 2. 이상치 탐지 (25점)

### 2.1 이상치 탐지 방법

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# 이상치를 포함한 데이터
data = {
    'id': range(1, 11),
    'score': [85, 88, 92, 86, 1000, 89, 91, 87, 90, 999]  # 1000, 999 이상치
}
df = pd.DataFrame(data)

# 방법 1: IQR (Interquartile Range) 사용
Q1 = df['score'].quantile(0.25)
Q3 = df['score'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"Q1: {Q1}, Q3: {Q3}, IQR: {IQR}")
print(f"Lower bound: {lower_bound}, Upper bound: {upper_bound}")

outliers_iqr = df[(df['score'] < lower_bound) | (df['score'] > upper_bound)]
print("\nOutliers (IQR method):")
print(outliers_iqr)

# 방법 2: Z-score 사용
z_scores = np.abs(stats.zscore(df['score']))
outliers_zscore = df[z_scores > 3]
print("\nOutliers (Z-score > 3):")
print(outliers_zscore)

# 방법 3: 표준편차 사용
mean = df['score'].mean()
std = df['score'].std()
lower_std = mean - 3 * std
upper_std = mean + 3 * std

outliers_std = df[(df['score'] < lower_std) | (df['score'] > upper_std)]
print("\nOutliers (3-sigma rule):")
print(outliers_std)

# 방법 4: 상자수염그림 시각화
plt.boxplot(df['score'])
plt.title('Box Plot of Score')
plt.show()
```

### 2.2 이상치 처리

```python
# 처리 방법 1: 이상치 행 제거
df_removed = df[(df['score'] >= lower_bound) & (df['score'] <= upper_bound)]
print("After removing outliers:", df_removed.shape)

# 처리 방법 2: 경계값으로 제한 (capping)
df_capped = df.copy()
df_capped['score'] = df_capped['score'].clip(lower=lower_bound, upper=upper_bound)
print("After capping:", df_capped['score'].tolist())

# 처리 방법 3: 중앙값으로 대체
df_replaced = df.copy()
df_replaced.loc[df_replaced['score'] > upper_bound, 'score'] = df['score'].median()
print("After replacing with median:", df_replaced['score'].tolist())

# 처리 방법 4: 변환 (로그 변환)
df_log = df.copy()
df_log['score'] = np.log1p(df_log['score'])
print("After log transformation:", df_log['score'].tolist())
```

**실습 과제:**
```python
# 임의의 이상치를 포함한 데이터 생성
# 1. IQR 방법으로 이상치 탐지
# 2. Z-score 방법으로 이상치 탐지
# 3. 두 방법의 결과 비교
# 4. 이상치 처리 (제거 또는 대체)
```

---

## 3. 데이터 정규화 및 표준화 (25점)

### 3.1 정규화 (Normalization)

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 다양한 범위의 데이터
data = {
    'age': [25, 30, 35, 40, 45],
    'salary': [50000, 60000, 70000, 80000, 90000],
    'score': [85, 88, 92, 95, 98]
}
df = pd.DataFrame(data)

# 방법 1: Min-Max 정규화 (0-1 범위)
scaler_minmax = MinMaxScaler()
df_normalized = pd.DataFrame(
    scaler_minmax.fit_transform(df),
    columns=df.columns
)
print("Min-Max Normalized (0-1):")
print(df_normalized)

# 수동 계산
def min_max_normalize(x):
    return (x - x.min()) / (x.max() - x.min())

df_normalized_manual = df.apply(min_max_normalize)
print("\nManually calculated Min-Max Normalized:")
print(df_normalized_manual)

# 방법 2: Min-Max 정규화 (다른 범위, 예: -1 to 1)
df_normalized_custom = df.copy()
for col in df.columns:
    min_val = df[col].min()
    max_val = df[col].max()
    df_normalized_custom[col] = 2 * (df[col] - min_val) / (max_val - min_val) - 1
print("\nMin-Max Normalized (-1 to 1):")
print(df_normalized_custom)
```

### 3.2 표준화 (Standardization)

```python
# 방법 1: Z-score 표준화
scaler_standard = StandardScaler()
df_standardized = pd.DataFrame(
    scaler_standard.fit_transform(df),
    columns=df.columns
)
print("Standardized (Z-score):")
print(df_standardized)

# 수동 계산
def z_score_standardize(x):
    return (x - x.mean()) / x.std()

df_standardized_manual = df.apply(z_score_standardize)
print("\nManually calculated Z-score Standardized:")
print(df_standardized_manual)

# 표준화 후 통계
print("\nMean of standardized data:")
print(df_standardized.mean())
print("\nStd of standardized data:")
print(df_standardized.std())
```

### 3.3 다양한 변환

```python
# 방법 1: 로그 변환
df_log = df.copy()
df_log['salary'] = np.log1p(df_log['salary'])  # 우편향 분포 처리
print("Log transformed salary:", df_log['salary'].tolist())

# 방법 2: 제곱근 변환
df_sqrt = df.copy()
df_sqrt['score'] = np.sqrt(df_sqrt['score'])
print("Square root transformed score:", df_sqrt['score'].tolist())

# 방법 3: Box-Cox 변환 (양수 데이터만)
from scipy.stats import boxcox
salary_transformed, lambda_param = boxcox(df['salary'])
print(f"Box-Cox lambda: {lambda_param}")
print("Box-Cox transformed salary:", salary_transformed)
```

**실습 과제:**
```python
# 정규화 vs 표준화 비교
# 1. Min-Max 정규화 수행
# 2. Z-score 표준화 수행
# 3. 변환 전후 통계 비교 (min, max, mean, std)
# 4. 시각화 (히스토그램 또는 박스플롯)
```

---

## 4. 특성 엔지니어링 (25점)

### 4.1 특성 생성

```python
import pandas as pd
import numpy as np
from datetime import datetime

# 고객 데이터
customers = {
    'customer_id': [1, 2, 3, 4, 5],
    'signup_date': ['2023-01-15', '2023-03-20', '2023-06-10', '2023-08-05', '2023-11-30'],
    'total_purchases': [5, 12, 3, 8, 15],
    'total_spent': [500, 1800, 450, 900, 2250],
    'age': [25, 35, 28, 42, 31],
    'visit_frequency': [2, 15, 1, 5, 20]
}
df = pd.DataFrame(customers)

# 특성 1: 회원 기간 (일 수)
df['signup_date'] = pd.to_datetime(df['signup_date'])
df['member_days'] = (datetime.now() - df['signup_date']).dt.days
print("Member days:", df['member_days'].tolist())

# 특성 2: 평균 구매액
df['avg_purchase'] = df['total_spent'] / df['total_purchases']
print("Average purchase:", df['avg_purchase'].tolist())

# 특성 3: 방문 빈도 범주화
df['visit_category'] = pd.cut(df['visit_frequency'], 
                               bins=[0, 5, 10, 100],
                               labels=['Low', 'Medium', 'High'])
print("Visit category:", df['visit_category'].tolist())

# 특성 4: 나이 범주화
df['age_group'] = pd.cut(df['age'],
                          bins=[0, 30, 40, 100],
                          labels=['Young', 'Middle', 'Senior'])
print("Age group:", df['age_group'].tolist())

# 특성 5: 상호작용 특성
df['engagement_score'] = (df['total_purchases'] * 0.5 + 
                          df['visit_frequency'] * 0.3 +
                          df['member_days'] * 0.2)
print("Engagement score:", df['engagement_score'].tolist())
```

### 4.2 특성 인코딩

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

data = {
    'id': [1, 2, 3, 4, 5],
    'color': ['Red', 'Blue', 'Red', 'Green', 'Blue'],
    'size': ['S', 'M', 'L', 'M', 'S']
}
df = pd.DataFrame(data)

# 방법 1: Label Encoding (순서가 있는 경우)
le = LabelEncoder()
df['size_encoded'] = le.fit_transform(df['size'])
print("Label Encoded (size):")
print(df[['size', 'size_encoded']])

# 방법 2: One-Hot Encoding (순서가 없는 경우)
df_onehot = pd.get_dummies(df, columns=['color'], prefix='color')
print("\nOne-Hot Encoded (color):")
print(df_onehot)

# 방법 3: 수동 매핑
color_mapping = {'Red': 0, 'Blue': 1, 'Green': 2}
df['color_mapped'] = df['color'].map(color_mapping)
print("\nManual Mapping (color):")
print(df[['color', 'color_mapped']])
```

### 4.3 특성 선택

```python
import pandas as pd
import numpy as np

# 다중 특성 데이터
np.random.seed(42)
X = pd.DataFrame({
    'feature1': np.random.randn(100),
    'feature2': np.random.randn(100),
    'feature3': np.random.randn(100),
    'feature4': np.random.randn(100),
    'feature5': np.random.randn(100)
})
y = (X['feature1'] + 2*X['feature2'] + np.random.randn(100))

# 방법 1: 상관계수 기반 선택
correlations = X.corrwith(y).abs().sort_values(ascending=False)
print("Feature Correlations with target:")
print(correlations)

# 방법 2: 분산 기반 선택 (낮은 분산 특성 제거)
low_variance = X.var()[X.var() < 0.1]
print("\nLow Variance Features:")
print(low_variance)

# 방법 3: 통계적 테스트 (f_classif, chi2 등)
from sklearn.feature_selection import f_regression
scores, p_values = f_regression(X, y)
feature_scores = pd.DataFrame({
    'feature': X.columns,
    'score': scores,
    'p_value': p_values
}).sort_values('score', ascending=False)
print("\nFeature Importance (f_regression):")
print(feature_scores)
```

**실습 과제:**
```python
# 특성 엔지니어링 종합 실습
# 1. 샘플 데이터로 새로운 특성 5개 이상 생성
# 2. 범주형 특성 인코딩
# 3. 특성 정규화/표준화
# 4. 특성 중요도 분석
```

---

## 5. 종합 실습 프로젝트 (20점)

### 목표
실제 데이터셋을 이용한 전체 전처리 파이프라인 구축

### 샘플 데이터 (전자상거래 고객 데이터)

```python
import pandas as pd
import numpy as np

data = {
    'customer_id': range(1, 51),
    'age': np.random.randint(18, 80, 50),
    'gender': np.random.choice(['M', 'F'], 50),
    'signup_date': pd.date_range('2022-01-01', periods=50, freq='D'),
    'purchase_count': np.random.randint(0, 20, 50),
    'total_spent': np.random.randint(0, 5000, 50),
    'visit_frequency': np.random.randint(0, 100, 50),
    'satisfaction_score': np.concatenate([
        np.random.randint(1, 6, 45),
        [np.nan] * 5  # 일부 결측치
    ]),
    'region': np.random.choice(['Seoul', 'Busan', 'Daegu', 'Incheon'], 50)
}
df = pd.DataFrame(data)
df.to_csv('customer_data.csv', index=False)
```

### 요구사항

**1단계: 데이터 탐색**
- 기본 정보 확인 (shape, info, describe)
- 결측치 확인
- 이상치 초기 확인

**2단계: 데이터 정제**
- 결측치 처리
- 이상치 처리
- 중복 제거

**3단계: 특성 엔지니어링**
- 날짜 특성 추출 (member_days 등)
- 범주형 특성 인코딩
- 새로운 특성 생성 (engagement score 등)

**4단계: 데이터 정규화**
- 수치형 특성 정규화/표준화
- 시각화

**5단계: 최종 데이터 저장**
- 전처리된 데이터 저장
- 전처리 과정 문서화

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# 데이터 로드
df = pd.read_csv('customer_data.csv')

# 1. 데이터 탐색
print("Data shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())
print("\nBasic statistics:\n", df.describe())

# 2. 데이터 정제
# - 결측치 처리
df['satisfaction_score'].fillna(df['satisfaction_score'].mean(), inplace=True)

# - 이상치 처리 (visit_frequency)
Q1 = df['visit_frequency'].quantile(0.25)
Q3 = df['visit_frequency'].quantile(0.75)
IQR = Q3 - Q1
upper_bound = Q3 + 1.5 * IQR
df['visit_frequency'] = df['visit_frequency'].clip(upper=upper_bound)

# 3. 특성 엔지니어링
df['signup_date'] = pd.to_datetime(df['signup_date'])
df['member_days'] = (pd.Timestamp.now() - df['signup_date']).dt.days
df['avg_purchase'] = df['total_spent'] / (df['purchase_count'] + 1)
df['gender_encoded'] = df['gender'].map({'M': 0, 'F': 1})

# One-hot encoding for region
df_encoded = pd.get_dummies(df, columns=['region'], drop_first=True)

# 4. 정규화
numeric_features = ['age', 'purchase_count', 'total_spent', 'visit_frequency']
scaler = StandardScaler()
df_encoded[numeric_features] = scaler.fit_transform(df_encoded[numeric_features])

# 5. 최종 데이터 저장
df_encoded.to_csv('customer_data_processed.csv', index=False)
print("\nProcessed data saved!")
```

**제출 내용:**
- week03_preprocessing.py (전체 전처리 파이프라인)
- customer_data.csv (원본 데이터)
- customer_data_processed.csv (처리된 데이터)
- 전처리 보고서 (결측치, 이상치, 특성 엔지니어링 내용)

---

## 6. 보너스 과제 (+10점)

### 자동 데이터 프로파일 생성

```python
import pandas as pd

def generate_data_profile(df):
    """데이터셋의 자동 프로필 생성"""
    profile = {
        'rows': len(df),
        'columns': len(df.columns),
        'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,  # MB
        'missing_percentage': (df.isnull().sum() / len(df) * 100),
        'duplicates': df.duplicated().sum(),
        'numeric_features': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_features': df.select_dtypes(include=['object']).columns.tolist()
    }
    return profile

# 프로필 생성 및 출력
profile = generate_data_profile(df)
for key, value in profile.items():
    print(f"{key}: {value}")
```

---

## 제출 방법

1. **제출 파일:**
   - week03_missing_values.py
   - week03_outliers.py
   - week03_normalization.py
   - week03_feature_engineering.py
   - week03_preprocessing.py (종합)
   - 처리된 데이터 파일들
   - 전처리 보고서

2. **제출 기한:** 다음 주 월요일 23:59
3. **제출 방식:** GitHub 레포지토리

---

## 평가 기준

| 항목 | 배점 | 평가 기준 |
|------|------|---------|
| 결측치 처리 | 25점 | 다양한 방법 구현 및 비교 |
| 이상치 탐지 | 25점 | 여러 방법 사용 및 시각화 |
| 정규화/표준화 | 25점 | 정규화 vs 표준화 차이 이해 |
| 특성 엔지니어링 | 25점 | 의미있는 특성 생성 및 검증 |
| 보너스 | +10점 | 자동화된 프로파일 생성 |
| **총점** | **100점** | |
