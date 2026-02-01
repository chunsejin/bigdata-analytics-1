# 15주차 실습과제: 종합 프로젝트 및 졸업 프로젝트

## 과제 목표
- 지난 14주간 학습한 모든 내용을 통합하는 완전한 빅데이터 분석 프로젝트
- 실무 수준의 데이터 분석 역량 입증
- 프레젠테이션 및 보고서 작성

## 1. 프로젝트 개요 (40점)

### 1.1 프로젝트 주제 선택

다음 중 하나를 선택하거나 자유 주제로 제안:

**옵션 1: 전자상거래 고객 분석**
- 고객 세분화
- 구매 행동 분석
- 이탈 예측
- 추천 시스템

**옵션 2: 금융 데이터 분석**
- 주식 가격 예측
- 신용 위험 평가
- 포트폴리오 분석

**옵션 3: 건강 및 의료 데이터**
- 질병 예측
- 환자 분류
- 치료 결과 분석

**옵션 4: IoT/센서 데이터**
- 실시간 이상 탐지
- 예지 정비
- 에너지 소비 최적화

**옵션 5: 소셜 미디어 분석**
- 감정 분석
- 주제 모델링
- 트렌드 예측

### 1.2 프로젝트 계획

```python
# project_plan.md
## 프로젝트: [제목]

### 1. 비즈니스 목표
- 주요 질문
- 성공 지표 (KPI)
- 예상 가치

### 2. 데이터 소개
- 데이터 소스
- 데이터 크기
- 주요 특성

### 3. 방법론
- 데이터 수집 방법
- 전처리 전략
- 분석 기법
- 모델링 접근방식

### 4. 예상 일정
- 주간별 마일스톤
- 주요 산출물
```

---

## 2. 데이터 수집 및 전처리 (30점)

### 2.1 데이터 수집

```python
# 1. 공개 데이터셋 이용
from sklearn import datasets
import pandas as pd

# 또는 온라인 소스에서
df = pd.read_csv('https://data.source.com/data.csv')

# 2. 웹 스크래핑
from bs4 import BeautifulSoup
import requests

url = 'https://example.com'
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')
data = soup.find_all('table')

# 3. API 호출
import requests

api_url = 'https://api.example.com/data'
response = requests.get(api_url)
data = response.json()

# 데이터 저장
df = pd.DataFrame(data)
df.to_csv('raw_data.csv', index=False)
```

### 2.2 데이터 탐색 및 정제

```python
import pandas as pd
import numpy as np

# 데이터 로드
df = pd.read_csv('raw_data.csv')

# 1. 기본 정보 확인
print(f"데이터 형태: {df.shape}")
print("\n데이터 타입:")
print(df.dtypes)
print("\n결측치:")
print(df.isnull().sum())
print("\n기본 통계:")
print(df.describe())

# 2. 결측치 처리
df = df.dropna(subset=['important_column'])
df['column_with_missing'].fillna(df['column_with_missing'].mean(), inplace=True)

# 3. 이상치 탐지 및 제거
Q1 = df['numeric_column'].quantile(0.25)
Q3 = df['numeric_column'].quantile(0.75)
IQR = Q3 - Q1

df = df[
    (df['numeric_column'] >= Q1 - 1.5*IQR) &
    (df['numeric_column'] <= Q3 + 1.5*IQR)
]

# 4. 데이터 타입 변환
df['date_column'] = pd.to_datetime(df['date_column'])
df['category_column'] = df['category_column'].astype('category')

# 전처리된 데이터 저장
df.to_csv('clean_data.csv', index=False)
```

---

## 3. 탐색적 분석 (EDA) (20점)

### 3.1 다차원 분석

```python
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('clean_data.csv')

# 1. 개별 변수 분석
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].hist(df['numeric_col1'], bins=30, edgecolor='black')
axes[0, 0].set_title('Numeric Column 1 Distribution')

axes[0, 1].hist(df['numeric_col2'], bins=30, edgecolor='black')
axes[0, 1].set_title('Numeric Column 2 Distribution')

axes[1, 0].value_counts = df['category_col'].value_counts().plot(kind='bar', ax=axes[1, 0])
axes[1, 0].set_title('Category Distribution')

axes[1, 1].text(0.5, 0.5, df.describe().to_string(), ha='center', va='center')
axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig('eda_individual.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. 변수 간 관계 분석
correlation_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. 그룹별 분석
grouped_analysis = df.groupby('category_col')['numeric_col'].agg([
    'count', 'mean', 'std', 'min', 'max'
])
print("그룹별 분석:")
print(grouped_analysis)

# 4. 시각화
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.boxplot(data=df, x='category_col', y='numeric_col1', ax=axes[0])
axes[0].set_title('Numeric Column 1 by Category')

sns.violinplot(data=df, x='category_col', y='numeric_col2', ax=axes[1])
axes[1].set_title('Numeric Column 2 by Category')

plt.tight_layout()
plt.savefig('eda_groups.png', dpi=300, bbox_inches='tight')
plt.show()
```

### 3.2 인사이트 도출

```python
# 주요 발견사항 정리
insights = """
## 주요 발견사항

1. 특성 A는 특성 B와 강한 양의 상관관계(r=0.85)
2. 카테고리 X는 다른 카테고리보다 평균적으로 30% 높은 값
3. 시계열 데이터에서 계절성 패턴 관찰
4. 이상치 5개 제거 후 분포가 더 정상적
5. ...
"""

with open('insights.txt', 'w') as f:
    f.write(insights)
```

---

## 4. 모델링 및 분석 (40점)

### 4.1 특성 엔지니어링

```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. 특성 생성
df['feature_interaction'] = df['col1'] * df['col2']
df['feature_ratio'] = df['col1'] / (df['col2'] + 1)
df['feature_log'] = np.log1p(df['numeric_col'])

# 2. 범주형 인코딩
df_encoded = pd.get_dummies(df, columns=['category_col'], drop_first=True)

# 3. 특성 스케일링
scaler = StandardScaler()
numeric_features = df_encoded.select_dtypes(include=[np.number]).columns
df_scaled = df_encoded.copy()
df_scaled[numeric_features] = scaler.fit_transform(df_encoded[numeric_features])

# 4. 차원 축소
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(df_scaled[numeric_features])

print(f"원본 특성 개수: {len(numeric_features)}")
print(f"축소된 특성 개수: {X_pca.shape[1]}")
print(f"설명된 분산: {pca.explained_variance_ratio_.sum():.2%}")
```

### 4.2 모델 구축

```python
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# 데이터 준비
X = df_scaled[numeric_features]
y = df['target_variable']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 모델 비교
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    # 교차 검증
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    
    # 모델 학습
    model.fit(X_train, y_train)
    
    # 예측
    y_pred = model.predict(X_test)
    
    # 평가
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {
        'model': model,
        'CV Mean': cv_scores.mean(),
        'CV Std': cv_scores.std(),
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }
    
    print(f"\n{name}:")
    print(f"  CV Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R²: {r2:.4f}")

# 최적 모델 선택
best_model_name = max(results, key=lambda x: results[x]['R2'])
best_model = results[best_model_name]['model']
print(f"\n최적 모델: {best_model_name}")
```

### 4.3 하이퍼파라미터 튜닝

```python
from sklearn.model_selection import GridSearchCV

# 그리드 서치
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    cv=5,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print(f"최적 파라미터: {grid_search.best_params_}")
print(f"최적 CV 점수: {grid_search.best_score_:.4f}")

# 최적 모델 평가
best_tuned_model = grid_search.best_estimator_
y_pred_tuned = best_tuned_model.predict(X_test)
tuned_r2 = r2_score(y_test, y_pred_tuned)
print(f"테스트 R²: {tuned_r2:.4f}")
```

---

## 5. 결과 시각화 및 해석 (20점)

### 5.1 결과 시각화

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 실제 vs 예측
axes[0, 0].scatter(y_test, y_pred_tuned, alpha=0.6)
axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
axes[0, 0].set_xlabel('Actual')
axes[0, 0].set_ylabel('Predicted')
axes[0, 0].set_title('Actual vs Predicted')

# 잔차
residuals = y_test - y_pred_tuned
axes[0, 1].scatter(y_pred_tuned, residuals, alpha=0.6)
axes[0, 1].axhline(y=0, color='r', linestyle='--')
axes[0, 1].set_xlabel('Predicted')
axes[0, 1].set_ylabel('Residuals')
axes[0, 1].set_title('Residual Plot')

# 특성 중요도
if hasattr(best_tuned_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': numeric_features,
        'importance': best_tuned_model.feature_importances_
    }).sort_values('importance', ascending=False).head(10)
    
    axes[1, 0].barh(feature_importance['feature'], feature_importance['importance'])
    axes[1, 0].set_xlabel('Importance')
    axes[1, 0].set_title('Top 10 Important Features')

# 잔차 분포
axes[1, 1].hist(residuals, bins=30, edgecolor='black')
axes[1, 1].set_xlabel('Residuals')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Residual Distribution')

plt.tight_layout()
plt.savefig('model_results.png', dpi=300, bbox_inches='tight')
plt.show()
```

### 5.2 보고서 작성

```markdown
# 최종 분석 보고서

## 1. 소개
- 프로젝트 개요
- 목표 및 질문

## 2. 방법론
- 데이터 수집 방법
- 전처리 과정
- 분석 기법

## 3. 결과
- 주요 발견사항
- 모델 성능
- 특성 중요도

## 4. 결론
- 주요 인사이트
- 비즈니스 임의
- 향후 개선 방안

## 5. 부록
- 코드
- 데이터 샘플
- 상세 통계
```

---

## 6. 프레젠테이션 (20점)

### 6.1 프레젠테이션 구성

```
1. 제목 슬라이드 (1)
   - 프로젝트 제목
   - 학생명
   - 날짜

2. 소개 (1-2)
   - 배경
   - 목표

3. 데이터 (2-3)
   - 데이터 소스
   - 크기 및 특성
   - EDA 결과

4. 방법론 (2-3)
   - 기법 설명
   - 모델 아키텍처

5. 결과 (3-4)
   - 주요 발견사항
   - 모델 성능
   - 시각화

6. 결론 (1-2)
   - 요약
   - 임의
   - Q&A
```

---

## 제출 방법

1. **제출 파일:**
   - `project_proposal.md` (프로젝트 제안)
   - `data_collection_preprocessing.py`
   - `eda_analysis.py`
   - `modeling.py`
   - `final_report.pdf` (30-50페이지)
   - `presentation.pptx`
   - `final_model.joblib` (훈련된 모델)
   - `README.md` (프로젝트 실행 방법)

2. **디렉토리 구조:**
   ```
   project/
   ├── data/
   │   ├── raw_data.csv
   │   └── clean_data.csv
   ├── notebooks/
   │   ├── 01_eda.ipynb
   │   └── 02_modeling.ipynb
   ├── src/
   │   ├── data_processing.py
   │   ├── feature_engineering.py
   │   └── models.py
   ├── results/
   │   ├── visualizations/
   │   └── models/
   ├── README.md
   └── requirements.txt
   ```

3. **제출 기한:** 15주차 마지막 날
4. **프레젠테이션:** 최종 발표 (10-15분)
5. **제출 방식:** GitHub 레포지토리 + 오프라인 발표

---

## 평가 기준

| 항목 | 배점 |
|------|------|
| 프로젝트 제안 및 계획 | 10점 |
| 데이터 수집 및 전처리 | 15점 |
| EDA | 15점 |
| 모델링 및 분석 | 30점 |
| 결과 해석 | 15점 |
| 보고서 및 문서화 | 10점 |
| 프레젠테이션 | 5점 |
| **총점** | **100점** |

---

## 성공을 위한 팁

1. **데이터 품질**: 좋은 데이터가 좋은 결과의 기반
2. **명확한 목표**: 분석의 방향을 명확히
3. **반복적 개선**: 첫 시도가 최선이 아님
4. **스토리텔링**: 기술적 정확성 + 전달력
5. **재현성**: 코드와 문서로 재현 가능하도록
6. **실무 관점**: 비즈니스 가치 고려
7. **피드백**: 동료 검토 및 피드백 수용

---

## 부록: 유용한 리소스

### Python 라이브러리
- Pandas: 데이터 조작
- NumPy: 수치 계산
- Scikit-learn: 머신러닝
- TensorFlow/PyTorch: 딥러닝
- Matplotlib/Seaborn: 시각화
- Plotly: 인터랙티브 시각화

### 온라인 데이터셋
- Kaggle (kaggle.com)
- UCI Machine Learning Repository
- Google Dataset Search
- 공공데이터포털 (data.go.kr)

### 학습 자료
- Scikit-learn 공식 문서
- TensorFlow 튜토리얼
- Papers with Code
- GitHub 저장소

---

**축하합니다! 빅데이터 분석 과정을 완료하셨습니다.**

성공적인 프로젝트를 위해 최선을 다하시길 바랍니다!
