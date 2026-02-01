# 2주차 실습과제: 데이터 수집 및 저장

## 과제 목표
- 다양한 데이터 소스에서 데이터 수집
- 데이터 저장 형식 이해 및 활용
- 웹 스크래핑 기초 습득

## 1. 데이터 수집 방법 (30점)

### 1.1 파일 기반 데이터 수집

```python
import pandas as pd
import numpy as np

# 1) CSV 파일 읽기
df_csv = pd.read_csv('data.csv')

# 2) Excel 파일 읽기
df_excel = pd.read_excel('data.xlsx', sheet_name=0)

# 3) JSON 파일 읽기
df_json = pd.read_json('data.json')

# 4) 각 형식에 대해 성공/실패 여부 출력
print("CSV read successfully:", df_csv.shape)
print("Excel read successfully:", df_excel.shape)
print("JSON read successfully:", df_json.shape)
```

**실습:**
- 3가지 형식의 샘플 파일 생성 (content는 동일)
- 각각 읽어서 결과 확인

### 1.2 API를 이용한 데이터 수집

```python
import requests
import pandas as pd
import json

# 1) REST API 호출 (공공 API 예시)
# OpenWeather API를 이용한 날씨 데이터 수집
api_key = "YOUR_API_KEY"
city = "Seoul"
url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}"

response = requests.get(url)
weather_data = response.json()

# 2) 응답 상태 코드 확인
print(f"Status Code: {response.status_code}")

# 3) JSON 응답을 pandas DataFrame으로 변환
weather_df = pd.DataFrame([weather_data])

# 4) 주요 정보 추출
print(f"Temperature: {weather_data['main']['temp']}")
print(f"Humidity: {weather_data['main']['humidity']}")
print(f"Weather: {weather_data['weather'][0]['description']}")
```

**실습:**
- requests 라이브러리 설치: `pip install requests`
- 공공 API 중 하나 선택하여 데이터 수집
- 수집한 데이터를 DataFrame으로 변환

### 1.3 웹 스크래핑

```python
from bs4 import BeautifulSoup
import requests
import pandas as pd

# 1) 웹페이지 HTML 가져오기
url = "https://example.com"
response = requests.get(url)
html_content = response.text

# 2) BeautifulSoup로 파싱
soup = BeautifulSoup(html_content, 'html.parser')

# 3) 특정 요소 추출
# 예: 모든 링크 추출
links = soup.find_all('a')
for link in links:
    print(link.get('href'), link.get_text())

# 4) 테이블 데이터 추출
tables = pd.read_html(url)
print(f"Found {len(tables)} tables")
```

**실습:**
- BeautifulSoup 설치: `pip install beautifulsoup4`
- 간단한 웹사이트에서 테이블 데이터 추출
- 추출한 데이터를 CSV로 저장

**주의:** 웹 스크래핑 시 robots.txt 확인 및 서버 부하 고려

---

## 2. 데이터 저장 형식 (40점)

### 2.1 다양한 파일 형식으로 저장

```python
import pandas as pd
import numpy as np

# 샘플 데이터 생성
data = {
    'customer_id': range(1, 6),
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'age': [25, 30, 35, 28, 32],
    'purchase_amount': [1000, 1500, 2000, 1200, 1800]
}
df = pd.DataFrame(data)

# 1) CSV 형식으로 저장
df.to_csv('customers.csv', index=False)
print("Saved as CSV")

# 2) Excel 형식으로 저장
df.to_excel('customers.xlsx', index=False)
print("Saved as Excel")

# 3) JSON 형식으로 저장
df.to_json('customers.json', orient='records')
print("Saved as JSON")

# 4) 각 형식 파일 읽어서 확인
df_csv = pd.read_csv('customers.csv')
df_excel = pd.read_excel('customers.xlsx')
df_json = pd.read_json('customers.json')

print("CSV shape:", df_csv.shape)
print("Excel shape:", df_excel.shape)
print("JSON shape:", df_json.shape)
```

### 2.2 데이터베이스 저장

```python
import pandas as pd
import sqlite3

# 1) SQLite 데이터베이스 연결
conn = sqlite3.connect('business.db')

# 2) DataFrame을 테이블로 저장
df.to_sql('customers', conn, if_exists='replace', index=False)
print("Data saved to SQLite database")

# 3) SQL 쿼리로 데이터 조회
query = "SELECT * FROM customers WHERE age > 28"
result = pd.read_sql_query(query, conn)
print(result)

# 4) 연결 종료
conn.close()
```

### 2.3 HDF5 형식 (대용량 데이터용)

```python
import pandas as pd
import numpy as np

# 1) 대용량 데이터 생성
large_data = pd.DataFrame({
    'id': range(100000),
    'value': np.random.randn(100000),
    'category': np.random.choice(['A', 'B', 'C'], 100000)
})

# 2) HDF5로 저장 (압축)
store = pd.HDFStore('large_data.h5')
store['data'] = large_data
store.close()

# 3) HDF5에서 읽기
store = pd.HDFStore('large_data.h5')
loaded_data = store['data']
store.close()

print("HDF5 format suitable for large datasets")
```

### 2.4 형식별 비교

| 형식 | 장점 | 단점 | 용도 |
|------|------|------|------|
| CSV | 텍스트, 호환성 좋음 | 타입 정보 없음 | 일반적인 데이터 교환 |
| Excel | 가시성, 그래프 | 파일 크기 큼 | 비즈니스 보고서 |
| JSON | 계층 구조 지원 | 복잡한 데이터 | 웹 API, 문서 DB |
| SQLite | 쿼리 가능, 경량 | 관계형만 | 로컬 데이터베이스 |
| HDF5 | 압축, 빠른 I/O | 특화 라이브러리 필요 | 대용량 과학 데이터 |
| Parquet | 효율적 압축 | 전문가 필요 | 빅데이터 처리 (Spark) |

**실습:**
각 형식으로 저장 후 로딩 시간 측정

```python
import time

data = pd.DataFrame({
    'id': range(100000),
    'value': np.random.randn(100000)
})

# CSV 저장/로드 시간
start = time.time()
data.to_csv('test.csv', index=False)
csv_load_time = time.time() - start

start = time.time()
df_csv = pd.read_csv('test.csv')
csv_read_time = time.time() - start

print(f"CSV - Write: {csv_load_time:.4f}s, Read: {csv_read_time:.4f}s")
```

---

## 3. 실전 프로젝트: 다중 소스 데이터 통합 (30점)

### 목표
다양한 소스에서 데이터를 수집하여 하나의 통합 데이터셋 만들기

### 요구사항

**1단계: 데이터 수집**
- 소스 1: CSV 파일 (고객 정보)
- 소스 2: Excel 파일 (구매 기록)
- 소스 3: JSON 파일 (리뷰 데이터) 또는 웹 API

**2단계: 데이터 탐색**
```python
import pandas as pd

# 각 데이터 로드
customers = pd.read_csv('customers.csv')
purchases = pd.read_excel('purchases.xlsx')
reviews = pd.read_json('reviews.json')

# 각 데이터셋 정보 확인
print("Customers shape:", customers.shape)
print("Purchases shape:", purchases.shape)
print("Reviews shape:", reviews.shape)

# 컬럼명 확인
print("\nCustomers columns:", customers.columns.tolist())
print("Purchases columns:", purchases.columns.tolist())
print("Reviews columns:", reviews.columns.tolist())
```

**3단계: 데이터 통합**
```python
# 공통 키를 기준으로 merge
merged = customers.merge(purchases, on='customer_id', how='inner')
final_data = merged.merge(reviews, on='customer_id', how='left')

print("Merged data shape:", final_data.shape)
```

**4단계: 데이터 검증 및 저장**
```python
# 결측치 확인
print("Missing values:\n", final_data.isnull().sum())

# 중복 확인
print("Duplicates:", final_data.duplicated().sum())

# 통합 데이터 저장
final_data.to_csv('integrated_data.csv', index=False)
final_data.to_excel('integrated_data.xlsx', index=False)
final_data.to_json('integrated_data.json', orient='records')
```

### 샘플 데이터 생성 코드

```python
import pandas as pd
import numpy as np

# 고객 정보
customers = pd.DataFrame({
    'customer_id': [1, 2, 3, 4, 5],
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'email': ['alice@email.com', 'bob@email.com', 'charlie@email.com', 'david@email.com', 'eve@email.com']
})
customers.to_csv('customers.csv', index=False)

# 구매 기록
purchases = pd.DataFrame({
    'customer_id': [1, 1, 2, 3, 4, 5, 5],
    'purchase_date': ['2024-01-15', '2024-02-20', '2024-01-10', '2024-02-05', '2024-01-30', '2024-02-10', '2024-02-25'],
    'amount': [100, 150, 200, 75, 300, 125, 180]
})
purchases.to_excel('purchases.xlsx', index=False)

# 리뷰 데이터
reviews = pd.DataFrame({
    'customer_id': [1, 2, 3, 5],
    'review': ['Good product', 'Excellent service', 'Average', 'Highly recommended'],
    'rating': [4, 5, 3, 5]
})
reviews.to_json('reviews.json', orient='records')
```

**제출 내용:**
- 각 데이터 소스 파일
- 통합 분석 Python 코드 (week02_data_integration.py)
- 최종 통합 데이터 (CSV, Excel, JSON)
- 분석 결과 정리

---

## 4. 보너스 과제 (+10점)

### 웹 스크래핑 심화
특정 웹사이트 (예: 뉴스, 주식 정보 등)에서 데이터 스크래핑하여 분석

**요구사항:**
- BeautifulSoup 사용
- 최소 50개 이상의 데이터 수집
- CSV 형식으로 저장
- 간단한 통계 분석 (평균, 최대, 최소 등)

---

## 제출 방법

1. **제출 파일:**
   - week02_data_collection.py (다양한 형식 저장 코드)
   - week02_data_integration.py (통합 프로젝트 코드)
   - 생성된 모든 데이터 파일 (CSV, Excel, JSON, DB)
   - 분석 결과 문서

2. **제출 기한:** 다음 주 월요일 23:59
3. **제출 방식:** GitHub 레포지토리

---

## 평가 기준

| 항목 | 배점 | 평가 기준 |
|------|------|---------|
| 데이터 수집 방법 | 30점 | 파일, API, 스크래핑 구현 |
| 데이터 저장 형식 | 40점 | 다양한 형식 저장/로드 성공 |
| 통합 프로젝트 | 30점 | 데이터 통합 및 검증 완성도 |
| 보너스 | +10점 | 웹 스크래핑 심화 과제 |
| **총점** | **100점** | |

---

## 추가 학습 자료

- [Pandas I/O tools](https://pandas.pydata.org/docs/reference/io.html)
- [Requests 문서](https://docs.python-requests.org/)
- [BeautifulSoup 튜토리얼](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)
- [공공 API 포털](https://www.data.go.kr/)
