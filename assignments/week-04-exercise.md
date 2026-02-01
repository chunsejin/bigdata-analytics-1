# 4주차 실습과제: 탐색적 데이터 분석 (EDA)

## 과제 목표
- 기술 통계를 이용한 데이터 이해
- 데이터 시각화 기법 습득
- 데이터의 분포, 상관관계 분석
- 인사이트 도출 및 해석

## 1. 기술 통계 분석 (25점)

### 1.1 기본 통계량

```python
import pandas as pd
import numpy as np

# 샘플 데이터
sales_data = pd.DataFrame({
    'date': pd.date_range('2024-01-01', periods=100),
    'sales': np.random.randint(1000, 10000, 100),
    'units': np.random.randint(10, 100, 100),
    'region': np.random.choice(['Seoul', 'Busan', 'Daegu'], 100),
    'product': np.random.choice(['A', 'B', 'C'], 100)
})

# 1) 기본 통계
print("Basic Statistics:")
print(sales_data.describe())

# 2) 세부 통계
print("\nDetailed Statistics:")
print(f"Mean: {sales_data['sales'].mean():.2f}")
print(f"Median: {sales_data['sales'].median():.2f}")
print(f"Mode: {sales_data['sales'].mode()[0]:.2f}")
print(f"Std: {sales_data['sales'].std():.2f}")
print(f"Variance: {sales_data['sales'].var():.2f}")
print(f"Skewness: {sales_data['sales'].skew():.2f}")
print(f"Kurtosis: {sales_data['sales'].kurtosis():.2f}")

# 3) 백분위수
print("\nPercentiles:")
print(f"25th percentile: {sales_data['sales'].quantile(0.25):.2f}")
print(f"50th percentile: {sales_data['sales'].quantile(0.50):.2f}")
print(f"75th percentile: {sales_data['sales'].quantile(0.75):.2f}")

# 4) 범주형 변수 분석
print("\nCategorical Analysis:")
print("Region counts:")
print(sales_data['region'].value_counts())
print("\nProduct counts:")
print(sales_data['product'].value_counts())
```

### 1.2 그룹별 통계

```python
# 지역별 판매액 통계
print("Sales by Region:")
print(sales_data.groupby('region')['sales'].agg([
    'count', 'mean', 'median', 'min', 'max', 'std'
]).round(2))

# 상품별 판매량 통계
print("\nUnits by Product:")
print(sales_data.groupby('product')['units'].agg([
    'sum', 'mean', 'std'
]).round(2))

# 다중 그룹핑
print("\nSales by Region and Product:")
print(sales_data.groupby(['region', 'product'])['sales'].mean().unstack().round(2))
```

---

## 2. 데이터 시각화 (30점)

### 2.1 히스토그램과 분포

```python
import matplotlib.pyplot as plt
import seaborn as sns

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1) 히스토그램
axes[0, 0].hist(sales_data['sales'], bins=20, edgecolor='black', alpha=0.7)
axes[0, 0].set_title('Distribution of Sales')
axes[0, 0].set_xlabel('Sales Amount')
axes[0, 0].set_ylabel('Frequency')

# 2) KDE 플롯
sales_data['sales'].plot(kind='density', ax=axes[0, 1])
axes[0, 1].set_title('KDE of Sales')

# 3) 박스 플롯
sales_data.boxplot(column='sales', by='region', ax=axes[1, 0])
axes[1, 0].set_title('Sales by Region')

# 4) 바이올린 플롯
sns.violinplot(data=sales_data, x='region', y='sales', ax=axes[1, 1])
axes[1, 1].set_title('Sales Distribution by Region')

plt.tight_layout()
plt.show()
```

### 2.2 관계 시각화

```python
# 1) 산점도
plt.figure(figsize=(10, 6))
plt.scatter(sales_data['units'], sales_data['sales'], alpha=0.6)
plt.xlabel('Units')
plt.ylabel('Sales')
plt.title('Relationship: Units vs Sales')
plt.grid(True)
plt.show()

# 2) 회귀선을 포함한 산점도
sns.regplot(data=sales_data, x='units', y='sales')
plt.title('Units vs Sales (with Regression Line)')
plt.show()

# 3) 상관계수 확인
correlation = sales_data[['sales', 'units']].corr()
print("Correlation Matrix:")
print(correlation)
```

### 2.3 상관계수 히트맵

```python
# 상관계수 행렬
corr_matrix = sales_data[['sales', 'units']].corr()

# 히트맵 그리기
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
            vmin=-1, vmax=1, square=True)
plt.title('Correlation Heatmap')
plt.show()
```

### 2.4 범주형 데이터 시각화

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 1) 막대 그래프
sales_data['region'].value_counts().plot(kind='bar', ax=axes[0], color='skyblue')
axes[0].set_title('Count by Region')
axes[0].set_ylabel('Count')

# 2) 파이 차트
sales_data['product'].value_counts().plot(kind='pie', ax=axes[1], autopct='%1.1f%%')
axes[1].set_title('Product Distribution')

plt.tight_layout()
plt.show()
```

### 2.5 시계열 시각화

```python
# 시간별 판매액 트렌드
sales_by_date = sales_data.groupby('date')['sales'].sum()

plt.figure(figsize=(12, 6))
plt.plot(sales_by_date.index, sales_by_date.values, linewidth=2)
plt.fill_between(sales_by_date.index, sales_by_date.values, alpha=0.3)
plt.title('Sales Trend Over Time')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 3. 상관관계 분석 (20점)

### 3.1 피어슨 상관계수

```python
import scipy.stats as stats

# 1) 상관계수 계산
correlation, p_value = stats.pearsonr(sales_data['units'], sales_data['sales'])
print(f"Pearson Correlation: {correlation:.4f}")
print(f"P-value: {p_value:.6f}")

if p_value < 0.05:
    print("Statistically significant correlation")
else:
    print("No statistically significant correlation")

# 2) 다중 변수 상관계수
numeric_cols = ['sales', 'units']
corr_matrix = sales_data[numeric_cols].corr()
print("\nCorrelation Matrix:")
print(corr_matrix)
```

### 3.2 스피어만 상관계수

```python
# 순위 기반 상관계수 (비선형 관계 감지)
spearman_corr, sp_p_value = stats.spearmanr(sales_data['units'], sales_data['sales'])
print(f"Spearman Correlation: {spearman_corr:.4f}")
print(f"P-value: {sp_p_value:.6f}")
```

### 3.3 상관계수 해석

```python
# 범주형 변수와 수치형 변수의 관계
print("\nAverage Sales by Region:")
print(sales_data.groupby('region')['sales'].mean())

print("\nAverage Units by Product:")
print(sales_data.groupby('product')['units'].mean())
```

---

## 4. 분포 분석 (15점)

### 4.1 정규분포 검정

```python
from scipy.stats import normaltest, shapiro, kstest

# 1) 정규성 검정 (Shapiro-Wilk)
stat, p_value = shapiro(sales_data['sales'])
print(f"Shapiro-Wilk Test:")
print(f"Statistic: {stat:.4f}, P-value: {p_value:.6f}")
if p_value > 0.05:
    print("Data is normally distributed")
else:
    print("Data is not normally distributed")

# 2) D'Agostino-Pearson 검정
k2, p = normaltest(sales_data['sales'])
print(f"\nD'Agostino-Pearson Test: p-value = {p:.6f}")

# 3) Q-Q 플롯
from scipy import stats
stats.probplot(sales_data['sales'], dist="norm", plot=plt)
plt.title('Q-Q Plot')
plt.show()
```

### 4.2 분포 형태 분석

```python
# 왜도와 첨도
skewness = sales_data['sales'].skew()
kurt = sales_data['sales'].kurtosis()

print(f"Skewness: {skewness:.4f}")
print(f"Kurtosis: {kurt:.4f}")

# 해석
if abs(skewness) < 0.5:
    print("Distribution is relatively symmetric")
elif skewness > 0:
    print("Distribution is right-skewed")
else:
    print("Distribution is left-skewed")
```

---

## 5. 이상 패턴 발견 (10점)

### 5.1 이상치 식별

```python
# 1) 통계적 이상치
Q1 = sales_data['sales'].quantile(0.25)
Q3 = sales_data['sales'].quantile(0.75)
IQR = Q3 - Q1
outliers = sales_data[
    (sales_data['sales'] < Q1 - 1.5*IQR) | 
    (sales_data['sales'] > Q3 + 1.5*IQR)
]
print(f"Number of outliers: {len(outliers)}")
print(outliers)

# 2) Z-score 기반 이상치
from scipy import stats
z_scores = np.abs(stats.zscore(sales_data['sales']))
outliers_zscore = sales_data[z_scores > 3]
print(f"\nOutliers (Z-score > 3): {len(outliers_zscore)}")
```

### 5.2 패턴 분석

```python
# 시간대별 패턴
sales_data['hour'] = sales_data['date'].dt.hour
sales_data['day_of_week'] = sales_data['date'].dt.dayofweek

print("Sales by Hour:")
print(sales_data.groupby('hour')['sales'].mean().round(2))

print("\nSales by Day of Week:")
print(sales_data.groupby('day_of_week')['sales'].mean().round(2))
```

---

## 6. 종합 실습 프로젝트 (30점)

### 목표
실제 데이터셋에 대한 완전한 EDA 수행

### 샘플 데이터 생성

```python
import pandas as pd
import numpy as np

# 고객 구매 데이터
np.random.seed(42)
customer_data = pd.DataFrame({
    'customer_id': range(1, 301),
    'age': np.random.randint(18, 75, 300),
    'income': np.random.randint(20000, 150000, 300),
    'purchase_count': np.random.randint(1, 50, 300),
    'total_spent': np.random.randint(100, 10000, 300),
    'satisfaction': np.random.randint(1, 6, 300),
    'region': np.random.choice(['North', 'South', 'East', 'West'], 300)
})

customer_data.to_csv('customer_data.csv', index=False)
```

### 요구 분석 사항

**1단계: 데이터 개요**
```python
import pandas as pd

df = pd.read_csv('customer_data.csv')

# 기본 정보
print("Data Shape:", df.shape)
print("\nData Info:")
print(df.info())
print("\nFirst 5 rows:")
print(df.head())
```

**2단계: 기술 통계**
```python
# 수치형 변수 통계
print("Descriptive Statistics:")
print(df.describe())

# 범주형 변수 분포
print("\nRegion Distribution:")
print(df['region'].value_counts())
```

**3단계: 시각화**
```python
# 다양한 시각화 생성
# - 히스토그램 (연령, 소득, 구매액)
# - 박스 플롯 (지역별 만족도)
# - 산점도 (소득 vs 구매액)
# - 상관계수 히트맵
```

**4단계: 상관관계 분석**
```python
# 상관계수 계산 및 시각화
# 유의미한 관계 식별
```

**5단계: 인사이트 도출**
```python
# 주요 발견사항 정리:
# - 가장 높은 구매율을 보이는 그룹
# - 만족도와 구매액의 관계
# - 지역별 특성 차이
```

### 보고서 작성

1. **데이터 개요** (5점)
   - 데이터셋 크기, 변수 타입, 결측치 등

2. **기술 통계 분석** (5점)
   - 주요 통계량, 분포 형태

3. **시각화** (10점)
   - 5개 이상의 의미 있는 그래프

4. **상관관계 분석** (5점)
   - 변수 간 관계 분석

5. **결론 및 인사이트** (5점)
   - 주요 발견사항, 비즈니스 의미

---

## 7. 보너스 과제 (+10점)

### 대시보드 스타일 시각화

여러 플롯을 하나의 대시보드로 구성

```python
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 다양한 서브플롯 추가
ax1 = fig.add_subplot(gs[0, :2])
ax1.hist(df['age'], bins=20, edgecolor='black')
ax1.set_title('Age Distribution')

# 추가 플롯들...

plt.savefig('eda_dashboard.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 제출 방법

1. **제출 파일:**
   - week04_descriptive_stats.py
   - week04_visualization.py
   - week04_correlation_analysis.py
   - week04_eda_report.py (종합)
   - eda_report.pdf (분석 보고서)
   - 생성된 시각화 이미지 파일들

2. **제출 기한:** 다음 주 월요일 23:59
3. **제출 방식:** GitHub 레포지토리

---

## 평가 기준

| 항목 | 배점 |
|------|------|
| 기술 통계 분석 | 25점 |
| 데이터 시각화 | 30점 |
| 상관관계 분석 | 20점 |
| 분포 분석 | 15점 |
| 이상 패턴 발견 | 10점 |
| **소계** | **100점** |
| 보너스 | +10점 |
