# 1주차 실습과제: 빅데이터 기초 및 환경 설정

## 과제 목표
- 빅데이터 분석 환경 구축
- Python 기본 문법 복습
- Pandas를 이용한 기초 데이터 처리

## 1. 환경 설정 (20점)

### 1.1 필수 라이브러리 설치
다음 명령어를 이용하여 필수 라이브러리를 설치하세요.

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### 1.2 설치 확인
Python에서 다음을 실행하여 모든 라이브러리가 정상 설치되었는지 확인하세요.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets

print(f"Pandas version: {pd.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"Scikit-learn version: {sklearn.__version__}")
print("All libraries installed successfully!")
```

**제출 내용**: 실행 결과 스크린샷 또는 버전 정보

---

## 2. Python 기본 문법 (30점)

### 2.1 데이터 타입 및 변수
다음을 구현하세요:

```python
# 1) 다양한 데이터 타입 변수 선언
# - 정수형 변수: num_records = 1000
# - 실수형 변수: avg_salary = 45000.50
# - 문자열 변수: company_name = "TechCorp"
# - 불린 변수: is_active = True

# 2) 리스트 생성 및 조작
data_list = [10, 20, 30, 40, 50]
# - 리스트 길이 출력
# - 첫 번째 원소 출력
# - 슬라이싱으로 [20, 30, 40] 출력
# - append() 메서드로 60 추가

# 3) 딕셔너리 생성 및 조작
person = {"name": "John", "age": 30, "city": "Seoul"}
# - 모든 키와 값 출력
# - 특정 값 접근하여 출력
```

### 2.2 조건문 및 반복문
다음을 구현하세요:

```python
# 1) 조건문 사용
# score 변수에 85를 할당
# - score >= 90: "A"
# - score >= 80: "B"
# - score >= 70: "C"
# - score < 70: "F"
# 위의 로직에 따라 학점 결정하고 출력

# 2) for 루프로 1부터 10까지의 합 계산
total = 0
# for 루프를 사용하여 합계 계산 후 출력

# 3) while 루프로 구구단 출력 (2단)
# while 루프를 사용하여 2*1, 2*2, ... 2*9 출력
```

### 2.3 함수 정의
다음 함수를 구현하세요:

```python
# 1) 두 수를 더하는 함수
def add(a, b):
    # 구현
    pass

# 2) 리스트 평균을 구하는 함수
def calculate_average(numbers):
    # 구현
    pass

# 3) 숫자가 짝수인지 판단하는 함수
def is_even(num):
    # 구현
    pass

# 함수 호출하여 결과 출력
```

**제출 내용**: week01_python_basics.py 파일

---

## 3. Pandas 기초 (50점)

### 3.1 데이터프레임 생성 및 기본 조작

```python
import pandas as pd
import numpy as np

# 1) 딕셔너리로 데이터프레임 생성
data = {
    'employee_id': [101, 102, 103, 104, 105],
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'department': ['Sales', 'IT', 'HR', 'Sales', 'IT'],
    'salary': [50000, 60000, 55000, 52000, 65000]
}
df = pd.DataFrame(data)

# 2) 기본 정보 확인
# - df.info() 실행 및 결과 해석
# - df.describe() 실행 및 통계 정보 확인
# - df.head() / df.tail() 사용

# 3) 데이터프레임 크기 정보
# - 행 개수: len(df)
# - 열 개수: df.shape[1]
# - 전체 원소 개수: df.size

print(f"행: {len(df)}, 열: {df.shape[1]}, 전체 원소: {df.size}")
```

### 3.2 선택 및 필터링

```python
# 1) 특정 열 선택
salaries = df['salary']
# 또는
department = df[['name', 'department']]

# 2) 조건으로 필터링
# - IT 부서 직원 필터링
# - 급여가 55000 이상인 직원 필터링
# - 이름이 'A'로 시작하는 직원 필터링

# 3) loc와 iloc 사용
# - 첫 번째 행 선택: df.iloc[0]
# - 특정 조건 행 선택: df.loc[df['salary'] > 55000]
```

### 3.3 데이터 조작

```python
# 1) 새로운 열 추가
df['bonus'] = df['salary'] * 0.1

# 2) 열 이름 변경
df = df.rename(columns={'employee_id': 'id'})

# 3) 데이터 정렬
# - salary 기준 오름차순 정렬
# - salary 기준 내림차순 정렬

# 4) 결측치 처리
# 결측치가 있는지 확인: df.isnull().sum()
```

### 3.4 실습 과제

다음 CSV 파일을 다운로드하여 분석하세요:
```
student_id,name,korean,english,math
1,Alice,90,85,92
2,Bob,78,82,88
3,Charlie,92,88,95
4,David,85,90,80
5,Eve,88,86,89
```

**요구사항:**
1. CSV 파일을 pandas로 읽기
2. 각 과목 평균 점수 계산
3. 각 학생의 평균 점수 계산
4. 평균 점수가 85 이상인 학생 필터링
5. 학생 이름, 평균 점수를 포함한 새로운 데이터프레임 생성
6. 평균 점수 기준 내림차순 정렬

```python
import pandas as pd

# CSV 파일 읽기
df = pd.read_csv('students.csv')

# 분석 코드 작성
# ...

# 결과 출력
```

**제출 내용**: 
- week01_pandas_exercise.py (분석 코드)
- students.csv (샘플 데이터)
- 실행 결과 (평균, 필터링된 데이터 등)

---

## 4. 보너스 과제 (+10점)

### 추천도서 요약
"빅데이터란 무엇인가"라는 주제로 200-300자 요약 작성

**핵심 포함 내용:**
- 빅데이터의 정의
- 빅데이터의 특징 (3V: Volume, Velocity, Variety)
- 빅데이터 활용 분야 1-2가지 예시

---

## 제출 방법

1. **제출 파일:**
   - week01_python_basics.py
   - week01_pandas_exercise.py
   - students.csv
   - 환경 설정 스크린샷
   - 보너스 요약 파일 (선택사항)

2. **제출 기한:** 다음 주 월요일 23:59
3. **제출 방식:** 학사시스템 또는 GitHub 레포지토리

---

## 평가 기준

| 항목 | 배점 | 평가 기준 |
|------|------|---------|
| 환경 설정 | 20점 | 모든 라이브러리 정상 설치 및 동작 확인 |
| Python 기초 | 30점 | 모든 코드 정상 작동 및 결과 정확성 |
| Pandas 실습 | 50점 | 요구사항 완성도 및 코드 품질 |
| 보너스 | +10점 | 요약 내용의 정확성 및 이해도 |
| **총점** | **100점** | |

---

## 추가 학습 자료

- [Pandas 공식 문서](https://pandas.pydata.org/docs/)
- [Python 공식 튜토리얼](https://docs.python.org/3/tutorial/)
- [Jupyter Notebook 가이드](https://jupyter.org/documentation)
