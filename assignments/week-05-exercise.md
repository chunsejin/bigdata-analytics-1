# 5주차 실습과제: 통계 분석

## 과제 목표
- 가설 검정 개념 이해 및 적용
- 다양한 통계 검정 수행
- 통계적 의사결정 역량 배양
- 신뢰도 구간 및 p-값 해석

## 1. 가설 검정 기초 (20점)

### 1.1 가설 검정의 개념

```python
import numpy as np
from scipy import stats

# 기본 개념
print("=== 가설 검정의 단계 ===")
print("1. 귀무가설(H0) vs 대립가설(H1) 설정")
print("2. 유의수준(α) 결정: 0.05 (5%)")
print("3. 검정통계량 계산")
print("4. p-값 계산")
print("5. 결론: p-값 < α면 귀무가설 기각")

# 예시: 동전 던지기
coin_tosses = np.random.binomial(n=1, p=0.5, size=100)
heads = coin_tosses.sum()
print(f"\n100번 던져서 {heads}번 앞면")

# 이항 검정
p_value = stats.binom_test(heads, n=100, p=0.5, alternative='two-sided')
print(f"P-value: {p_value:.4f}")
print("결론: 동전이 공정하다/하지 않다")
```

### 1.2 p-값과 유의수준

```python
# p-값의 의미
print("=== p-값의 해석 ===")
print("p-값: 귀무가설이 참일 때 관찰된 데이터 이상 극단적인 결과가 나올 확률")
print("- p < 0.05: 귀무가설 기각 (통계적으로 유의함)")
print("- p >= 0.05: 귀무가설 채택 (통계적으로 유의하지 않음)")

# 신뢰도 구간
print("\n=== 신뢰도 구간 ===")
data = [23.5, 24.1, 23.8, 24.3, 23.9, 24.2]
mean = np.mean(data)
se = stats.sem(data)  # 표준오차
ci = stats.t.interval(0.95, len(data)-1, loc=mean, scale=se)
print(f"표본평균: {mean:.2f}")
print(f"95% 신뢰도 구간: ({ci[0]:.2f}, {ci[1]:.2f})")
```

---

## 2. 모수 검정 (30점)

### 2.1 단일 표본 t-검정

```python
from scipy import stats

# 약물의 효과 검정
print("=== 단일 표본 t-검정 ===")
# 약물 복용 후 혈압 (기준값: 120)
blood_pressure = [118, 122, 119, 121, 120, 123, 119, 121, 122, 120]

t_stat, p_value = stats.ttest_1samp(blood_pressure, 120)
print(f"표본평균: {np.mean(blood_pressure):.2f}")
print(f"기준값: 120")
print(f"t-통계량: {t_stat:.4f}")
print(f"p-값: {p_value:.4f}")

if p_value < 0.05:
    print("결론: 약물이 효과 있음 (p < 0.05)")
else:
    print("결론: 약물이 효과 없음 (p >= 0.05)")
```

### 2.2 독립 표본 t-검정

```python
print("\n=== 독립 표본 t-검정 ===")
# 두 그룹의 성적 비교
group_A = [85, 88, 92, 87, 89, 86, 90, 88]
group_B = [78, 82, 80, 75, 81, 79, 83, 77]

t_stat, p_value = stats.ttest_ind(group_A, group_B)
print(f"그룹 A 평균: {np.mean(group_A):.2f}")
print(f"그룹 B 평균: {np.mean(group_B):.2f}")
print(f"t-통계량: {t_stat:.4f}")
print(f"p-값: {p_value:.4f}")

if p_value < 0.05:
    print("결론: 두 그룹의 성적에 유의한 차이 있음")
else:
    print("결론: 두 그룹의 성적에 유의한 차이 없음")

# 등분산성 검정 (Levene test)
levene_stat, levene_p = stats.levene(group_A, group_B)
print(f"\nLevene 검정 p-값: {levene_p:.4f}")
if levene_p > 0.05:
    print("등분산 가정 만족")
else:
    print("등분산 가정 불만족 (Welch's t-test 사용)")
```

### 2.3 쌍표본 t-검정

```python
print("\n=== 쌍표본 t-검정 ===")
# 약물 복용 전후 체중 변화
before = [75, 82, 78, 85, 80, 88, 76, 84]
after = [73, 80, 76, 83, 78, 85, 74, 82]

t_stat, p_value = stats.ttest_rel(before, after)
print(f"복용 전 평균: {np.mean(before):.2f}")
print(f"복용 후 평균: {np.mean(after):.2f}")
print(f"평균 변화: {np.mean(after) - np.mean(before):.2f}")
print(f"t-통계량: {t_stat:.4f}")
print(f"p-값: {p_value:.4f}")

if p_value < 0.05:
    print("결론: 약물이 체중 감소에 효과 있음")
else:
    print("결론: 약물이 체중 감소에 효과 없음")
```

### 2.4 분산 분석 (ANOVA)

```python
print("\n=== 일원분산분석 (One-way ANOVA) ===")
# 세 지역의 평균 소비 비교
north = [85, 88, 92, 87, 89]
south = [78, 82, 80, 75, 81]
east = [92, 95, 90, 94, 91]

f_stat, p_value = stats.f_oneway(north, south, east)
print(f"North 평균: {np.mean(north):.2f}")
print(f"South 평균: {np.mean(south):.2f}")
print(f"East 평균: {np.mean(east):.2f}")
print(f"F-통계량: {f_stat:.4f}")
print(f"p-값: {p_value:.4f}")

if p_value < 0.05:
    print("결론: 지역별 평균 소비에 유의한 차이 있음")
else:
    print("결론: 지역별 평균 소비에 유의한 차이 없음")
```

---

## 3. 비모수 검정 (20점)

### 3.1 만-휘트니 U 검정

```python
print("\n=== 만-휘트니 U 검정 (비모수) ===")
# 두 제품의 품질 평가
product_A = [7, 8, 6, 9, 7, 8]
product_B = [5, 6, 4, 5, 6]

u_stat, p_value = stats.mannwhitneyu(product_A, product_B, alternative='two-sided')
print(f"제품 A 중앙값: {np.median(product_A):.2f}")
print(f"제품 B 중앙값: {np.median(product_B):.2f}")
print(f"U-통계량: {u_stat:.4f}")
print(f"p-값: {p_value:.4f}")

if p_value < 0.05:
    print("결론: 두 제품의 품질 평가에 유의한 차이 있음")
else:
    print("결론: 두 제품의 품질 평가에 유의한 차이 없음")
```

### 3.2 윌콕슨 부호 순위 검정

```python
print("\n=== 윌콕슨 부호 순위 검정 ===")
before = [75, 82, 78, 85, 80]
after = [73, 80, 76, 83, 78]

w_stat, p_value = stats.wilcoxon(before, after)
print(f"W-통계량: {w_stat:.4f}")
print(f"p-값: {p_value:.4f}")

if p_value < 0.05:
    print("결론: 전후에 유의한 차이 있음")
```

### 3.3 크루스칼-월리스 검정

```python
print("\n=== 크루스칼-월리스 검정 ===")
# 세 방법의 효과 비교
method_A = [8, 7, 9, 8, 7]
method_B = [6, 5, 7, 6, 5]
method_C = [9, 8, 10, 9, 8]

h_stat, p_value = stats.kruskal(method_A, method_B, method_C)
print(f"H-통계량: {h_stat:.4f}")
print(f"p-값: {p_value:.4f}")

if p_value < 0.05:
    print("결론: 세 방법 간에 유의한 차이 있음")
```

---

## 4. 카이제곱 검정 (15점)

### 4.1 독립성 검정

```python
print("\n=== 카이제곱 독립성 검정 ===")
# 성별과 구매 의도의 관계
import pandas as pd

contingency_table = pd.DataFrame({
    '구매': [45, 30],
    '미구매': [20, 35]
}, index=['남성', '여성'])

print("분할표:")
print(contingency_table)

chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)
print(f"\nχ²-통계량: {chi2_stat:.4f}")
print(f"p-값: {p_value:.4f}")
print(f"자유도: {dof}")

if p_value < 0.05:
    print("결론: 성별과 구매 의도는 관련이 있음")
else:
    print("결론: 성별과 구매 의도는 무관함")
```

### 4.2 적합도 검정

```python
print("\n=== 카이제곱 적합도 검정 ===")
# 주사위의 공정성 검정
observed = [15, 18, 16, 20, 17, 14]  # 주사위 600번 던진 결과
expected = [100, 100, 100, 100, 100, 100]  # 각각 100회씩 기대

chi2_stat, p_value = stats.chisquare(observed, expected)
print(f"χ²-통계량: {chi2_stat:.4f}")
print(f"p-값: {p_value:.4f}")

if p_value < 0.05:
    print("결론: 주사위는 공정하지 않음")
else:
    print("결론: 주사위는 공정함")
```

---

## 5. 상관관계 검정 (15점)

### 5.1 피어슨 상관계수 검정

```python
print("\n=== 피어슨 상관계수 검정 ===")
x = [1, 2, 3, 4, 5, 6, 7, 8]
y = [2, 4, 5, 4, 5, 7, 8, 8]

corr, p_value = stats.pearsonr(x, y)
print(f"상관계수: {corr:.4f}")
print(f"p-값: {p_value:.4f}")

if p_value < 0.05:
    print("결론: 두 변수 간에 유의한 상관관계 있음")
else:
    print("결론: 두 변수 간에 유의한 상관관계 없음")

# 신뢰도 구간
r = corr
n = len(x)
t = r * np.sqrt(n - 2) / np.sqrt(1 - r**2)
ci = stats.t.interval(0.95, n-2, loc=r, scale=np.sqrt((1-r**2)/(n-2)))
print(f"95% 신뢰도 구간: ({ci[0]:.4f}, {ci[1]:.4f})")
```

### 5.2 스피어만 순위 상관계수 검정

```python
print("\n=== 스피어만 상관계수 검정 ===")
x_rank = [1, 2, 3, 4, 5]
y_rank = [2, 1, 4, 3, 5]

corr, p_value = stats.spearmanr(x_rank, y_rank)
print(f"스피어만 상관계수: {corr:.4f}")
print(f"p-값: {p_value:.4f}")

if p_value < 0.05:
    print("결론: 두 순위 간에 유의한 상관관계 있음")
```

---

## 6. 효과 크기 (Effect Size) (10점)

### 6.1 Cohen's d

```python
print("\n=== 효과 크기 (Effect Size) ===")

def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std

group_A = [85, 88, 92, 87, 89]
group_B = [78, 82, 80, 75, 81]

d = cohens_d(group_A, group_B)
print(f"Cohen's d: {d:.4f}")

# 해석
if abs(d) < 0.2:
    print("효과 크기: 작음 (small)")
elif abs(d) < 0.5:
    print("효과 크기: 중간 (medium)")
elif abs(d) < 0.8:
    print("효과 크기: 중간~큼 (medium-large)")
else:
    print("효과 크기: 큼 (large)")
```

---

## 7. 종합 실습 프로젝트 (25점)

### 목표
실제 데이터를 이용한 통계 분석 프로젝트

### 샘플 데이터

```python
import pandas as pd
import numpy as np

np.random.seed(42)
# A/B 테스트 데이터
ab_test_data = pd.DataFrame({
    'group': ['A']*100 + ['B']*100,
    'conversion': list(np.random.binomial(1, 0.15, 100)) + 
                 list(np.random.binomial(1, 0.20, 100)),
    'spend': np.random.uniform(10, 100, 200),
    'age': np.random.randint(18, 65, 200)
})

ab_test_data.to_csv('ab_test_data.csv', index=False)
```

### 분석 요구사항

**1. 기술 통계**
- 그룹별 전환율, 평균 지출액 비교

**2. 가설 검정**
- 두 그룹 전환율 차이 검정 (카이제곱)
- 두 그룹 지출액 차이 검정 (t-검정)

**3. 신뢰도 구간**
- 각 그룹 전환율의 95% 신뢰도 구간

**4. 효과 크기**
- 두 그룹 간 효과 크기 계산

**5. 결론**
- 통계 검정 결과 해석 및 비즈니스 의사결정

---

## 8. 보너스 과제 (+10점)

### 베이지안 분석 기초

```python
# 사전 분포, 우도, 사후 분포 계산
from scipy.stats import binom

n_trials = 100
n_successes = 25

# 베타-이항 모형
prior = stats.beta(1, 1)  # 균등 사전
likelihood_func = lambda p: stats.binom.pmf(n_successes, n_trials, p)
posterior = stats.beta(n_successes + 1, n_trials - n_successes + 1)

print(f"사전 분포 평균: {prior.mean():.4f}")
print(f"사후 분포 평균: {posterior.mean():.4f}")
```

---

## 제출 방법

1. **제출 파일:**
   - week05_hypothesis_testing.py
   - week05_parametric_tests.py
   - week05_nonparametric_tests.py
   - week05_chi_square_tests.py
   - week05_statistical_analysis_project.py
   - statistical_analysis_report.pdf

2. **제출 기한:** 다음 주 월요일 23:59
3. **제출 방식:** GitHub 레포지토리

---

## 평가 기준

| 항목 | 배점 |
|------|------|
| 가설 검정 기초 | 20점 |
| 모수 검정 | 30점 |
| 비모수 검정 | 20점 |
| 카이제곱 검정 | 15점 |
| 상관관계 검정 | 15점 |
| **소계** | **100점** |
| 보너스 | +10점 |
