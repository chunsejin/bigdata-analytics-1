# 8주차 실습과제: Apache Spark 기초 및 설치

## 과제 목표
- Apache Spark 환경 구축
- Spark의 기본 개념 이해 (RDD, DataFrame)
- 기본 변환 및 액션 연산 습득
- 성능 최적화 기초

## 1. Spark 설치 및 환경 설정 (15점)

### 1.1 설치

```bash
# Mac에서 Homebrew를 이용한 설치
brew install apache-spark

# 또는 Docker 사용
docker run -it -p 8888:8888 jupyter/pyspark-notebook

# 설치 확인
spark-shell --version
pyspark --version
```

### 1.2 환경 설정

```python
# PySpark 초기화
from pyspark.sql import SparkSession

# Spark 세션 생성
spark = SparkSession.builder \
    .appName("MyFirstSparkApp") \
    .master("local[*]") \
    .getOrCreate()

# Spark Context 확인
sc = spark.sparkContext
print(f"Spark version: {spark.version}")
print(f"Python version: {sc.pythonVer}")
print(f"App name: {sc.appName}")
print(f"Master: {sc.master}")
```

### 1.3 로그 설정

```python
# 로그 레벨 조정
spark.sparkContext.setLogLevel("WARN")

# 또는 Python 로깅
import logging
logging.getLogger('pyspark').setLevel(logging.WARN)
```

---

## 2. RDD (Resilient Distributed Dataset) (20점)

### 2.1 RDD 생성

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("RDDExample").master("local[*]").getOrCreate()
sc = spark.sparkContext

# 1. 컬렉션으로부터 생성
data = [1, 2, 3, 4, 5]
rdd = sc.parallelize(data)

# 2. 외부 저장소로부터 생성
text_rdd = sc.textFile("path/to/file.txt")

# 3. 다른 RDD로부터 생성
rdd2 = rdd.map(lambda x: x * 2)

# RDD 확인
print(f"RDD partitions: {rdd.getNumPartitions()}")
print(f"RDD collect: {rdd.collect()}")
```

### 2.2 RDD 변환 (Transformation)

```python
rdd = sc.parallelize([1, 2, 3, 4, 5, 6])

# 1. map: 각 요소에 함수 적용
mapped_rdd = rdd.map(lambda x: x * 2)
print(mapped_rdd.collect())  # [2, 4, 6, 8, 10, 12]

# 2. filter: 조건에 맞는 요소 선택
filtered_rdd = rdd.filter(lambda x: x > 3)
print(filtered_rdd.collect())  # [4, 5, 6]

# 3. flatMap: map 후 결과 평탄화
flatmap_rdd = rdd.flatMap(lambda x: [x, x*2])
print(flatmap_rdd.collect())  # [1, 2, 2, 4, 3, 6, ...]

# 4. distinct: 중복 제거
dup_rdd = sc.parallelize([1, 1, 2, 2, 3, 3])
distinct_rdd = dup_rdd.distinct()
print(distinct_rdd.collect())  # [1, 2, 3]

# 5. union: 두 RDD 합집합
rdd1 = sc.parallelize([1, 2, 3])
rdd2 = sc.parallelize([3, 4, 5])
union_rdd = rdd1.union(rdd2)
print(union_rdd.collect())  # [1, 2, 3, 3, 4, 5]

# 6. intersection: 교집합
inter_rdd = rdd1.intersection(rdd2)
print(inter_rdd.collect())  # [3]

# 7. subtract: 차집합
sub_rdd = rdd1.subtract(rdd2)
print(sub_rdd.collect())  # [1, 2]

# 8. groupByKey: 키별 그룹화
pair_rdd = sc.parallelize([("a", 1), ("b", 1), ("a", 1)])
grouped = pair_rdd.groupByKey()
print(grouped.collect())  # [('a', [1, 1]), ('b', [1])]
```

### 2.3 RDD 액션 (Action)

```python
rdd = sc.parallelize([1, 2, 3, 4, 5])

# 1. collect: 모든 요소 반환
result = rdd.collect()
print(result)  # [1, 2, 3, 4, 5]

# 2. count: 요소 개수
count = rdd.count()
print(count)  # 5

# 3. first: 첫 번째 요소
first = rdd.first()
print(first)  # 1

# 4. take: 처음 n개 요소
taken = rdd.take(3)
print(taken)  # [1, 2, 3]

# 5. reduce: 누적 연산
sum_result = rdd.reduce(lambda x, y: x + y)
print(sum_result)  # 15

# 6. saveAsTextFile: 파일로 저장
rdd.saveAsTextFile("output/path")

# 7. foreach: 각 요소에 함수 적용 (부작용)
rdd.foreach(lambda x: print(f"Element: {x}"))

# 8. countByValue: 값별 개수
value_counts = rdd.countByValue()
print(value_counts)  # {1: 1, 2: 1, 3: 1, ...}
```

### 2.4 RDD 페어 연산

```python
pair_rdd = sc.parallelize([
    ("apple", 1), ("apple", 2), ("banana", 1), ("banana", 3)
])

# 1. reduceByKey: 키별 축약
result = pair_rdd.reduceByKey(lambda x, y: x + y)
print(result.collect())  # [('apple', 3), ('banana', 4)]

# 2. sortByKey: 키로 정렬
sorted_rdd = pair_rdd.sortByKey()
print(sorted_rdd.collect())

# 3. join: 키 기반 조인
rdd1 = sc.parallelize([("a", 1), ("b", 2)])
rdd2 = sc.parallelize([("a", 3), ("b", 4)])
joined = rdd1.join(rdd2)
print(joined.collect())  # [('a', (1, 3)), ('b', (2, 4))]

# 4. leftOuterJoin: 왼쪽 외부 조인
left_joined = rdd1.leftOuterJoin(rdd2)

# 5. rightOuterJoin: 오른쪽 외부 조인
right_joined = rdd1.rightOuterJoin(rdd2)
```

---

## 3. DataFrame (30점)

### 3.1 DataFrame 생성

```python
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
import pandas as pd

spark = SparkSession.builder.appName("DataFrameExample").master("local[*]").getOrCreate()

# 1. 리스트로부터 생성
data = [("Alice", 25), ("Bob", 30), ("Charlie", 35)]
df = spark.createDataFrame(data, ["name", "age"])

# 2. 스키마 정의 후 생성
schema = StructType([
    StructField("name", StringType(), True),
    StructField("age", IntegerType(), True),
    StructField("salary", IntegerType(), True)
])

data = [
    ("Alice", 25, 50000),
    ("Bob", 30, 60000),
    ("Charlie", 35, 70000)
]
df = spark.createDataFrame(data, schema)

# 3. Pandas DataFrame으로부터 생성
pdf = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35]
})
df = spark.createDataFrame(pdf)

# 4. CSV 파일로부터 생성
df = spark.read.csv("data.csv", header=True, inferSchema=True)

# 5. JSON 파일로부터 생성
df = spark.read.json("data.json")
```

### 3.2 DataFrame 탐색

```python
# 1. 스키마 확인
df.printSchema()

# 2. 기본 정보
print(f"Rows: {df.count()}")
print(f"Columns: {df.columns}")

# 3. 첫 5개 행 확인
df.show(5)

# 4. 특정 열 조회
df.select("name", "age").show()

# 5. 필터링
df.filter(df.age > 25).show()

# 6. 정렬
df.orderBy("age").show()
df.orderBy(df.age.desc()).show()

# 7. 그룹화 및 집계
df.groupBy("department").count().show()
df.groupBy("department").agg({"salary": "avg"}).show()

# 8. 기술 통계
df.describe().show()
df.select("age", "salary").describe().show()
```

### 3.3 DataFrame 변환

```python
# 1. 새 열 추가
df_with_bonus = df.withColumn("bonus", df.salary * 0.1)

# 2. 열 이름 변경
df_renamed = df.withColumnRenamed("age", "years_old")

# 3. 열 삭제
df_dropped = df.drop("bonus")

# 4. 조건부 값 할당
from pyspark.sql.functions import when
df_classified = df.withColumn(
    "salary_level",
    when(df.salary >= 60000, "High").otherwise("Low")
)

# 5. SQL 쿼리
df.createOrReplaceTempView("employees")
result = spark.sql("SELECT name, salary FROM employees WHERE salary > 55000")
result.show()

# 6. 조인
df1 = spark.createDataFrame([("Alice", 1), ("Bob", 2)], ["name", "id"])
df2 = spark.createDataFrame([(1, "Sales"), (2, "IT")], ["id", "dept"])
joined = df1.join(df2, on="id")
joined.show()
```

### 3.4 DataFrame에서 RDD로, RDD에서 DataFrame으로

```python
# DataFrame → RDD
rdd = df.rdd

# RDD → DataFrame
new_df = rdd.toDF()

# RDD → DataFrame (스키마 지정)
new_df = spark.createDataFrame(rdd, schema)
```

---

## 4. Spark SQL (20점)

### 4.1 SQL 쿼리

```python
# 임시 테이블 생성
df.createOrReplaceTempView("employees")

# 기본 SELECT
spark.sql("SELECT * FROM employees").show()

# WHERE 조건
spark.sql("SELECT name, salary FROM employees WHERE salary > 50000").show()

# GROUP BY
spark.sql("""
    SELECT department, COUNT(*) as count, AVG(salary) as avg_salary
    FROM employees
    GROUP BY department
""").show()

# ORDER BY
spark.sql("SELECT * FROM employees ORDER BY salary DESC").show()

# JOIN
spark.sql("""
    SELECT e.name, d.dept_name
    FROM employees e
    JOIN departments d ON e.dept_id = d.id
""").show()

# 전역 임시 테이블 (여러 세션에서 접근 가능)
df.createGlobalTempView("global_employees")
spark.sql("SELECT * FROM global_temp.global_employees").show()
```

### 4.2 내장 함수

```python
from pyspark.sql.functions import *

# 1. 문자열 함수
df.select(upper("name")).show()
df.select(lower("name")).show()
df.select(length("name")).show()
df.select(concat_ws(" ", "name", "age")).show()

# 2. 수치 함수
df.select(round("salary", -3)).show()  # 천의 자리로 반올림
df.select(abs("salary")).show()

# 3. 날짜 함수
df.select(current_date()).show()
df.select(year("hire_date")).show()
df.select(month("hire_date")).show()

# 4. 집계 함수
df.agg(sum("salary"), avg("salary"), max("salary"), min("salary")).show()

# 5. 윈도우 함수
from pyspark.sql.window import Window
window = Window.partitionBy("department").orderBy("salary")
df.withColumn("rank", rank().over(window)).show()
```

---

## 5. 성능 최적화 (15점)

### 5.1 캐싱

```python
# 1. 메모리 캐싱
df.cache()
df.show()  # 캐시에 저장됨

# 2. 디스크 캐싱
df.persist()

# 3. 캐시 제거
df.unpersist()

# 4. 캐시 상태 확인
print(df.is_cached)
```

### 5.2 파티셔닝

```python
# 1. 파티션 개수 확인
print(df.rdd.getNumPartitions())

# 2. 파티션 재설정
df_repartitioned = df.repartition(10)

# 3. 특정 컬럼으로 파티셔닝
df_partitioned = df.repartition("department")

# 4. 파티션 개수 줄이기
df_coalesced = df.coalesce(4)
```

### 5.3 실행 계획 분석

```python
# 1. 실행 계획 확인
df.explain()

# 2. 상세 계획
df.explain(extended=True)

# 3. 쿼리 계획 비교
df1.select("name").explain()
df2.select("name").explain()
```

---

## 6. 종합 실습 프로젝트 (25점)

### 목표
대규모 판매 데이터 분석

### 샘플 데이터

```python
import pandas as pd
from datetime import datetime, timedelta

# 판매 데이터 생성
dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(365)]
regions = ['Seoul', 'Busan', 'Daegu', 'Incheon', 'Daejeon']
products = ['ProductA', 'ProductB', 'ProductC', 'ProductD', 'ProductE']

data = []
for _ in range(10000):
    data.append((
        random.choice(dates),
        random.choice(regions),
        random.choice(products),
        random.randint(100, 1000)
    ))

pdf = pd.DataFrame(data, columns=['date', 'region', 'product', 'sales'])
df = spark.createDataFrame(pdf)

df.write.csv("sales_data.csv", header=True)
```

### 분석 요구사항

**1. 기본 분석**
- 전체 판매액, 평균 판매액, 최대/최소 판매액
- 지역별 판매액
- 상품별 판매액

**2. 시계열 분석**
- 월별 판매액 추이
- 계절성 분석

**3. 심화 분석**
- 지역-상품별 판매액 크로스탭
- 상위 5개 판매 지역
- 상위 3개 판매 상품

```python
spark = SparkSession.builder.appName("SalesAnalysis").master("local[*]").getOrCreate()

# 데이터 로드
df = spark.read.csv("sales_data.csv", header=True, inferSchema=True)

# 1. 기본 통계
print("Total sales:", df.select(sum("sales")).collect()[0][0])
df.describe("sales").show()

# 2. 지역별 분석
regional = df.groupBy("region").agg(sum("sales").alias("total_sales"))
regional.show()

# 3. SQL 분석
df.createOrReplaceTempView("sales")
result = spark.sql("""
    SELECT region, product, SUM(sales) as total
    FROM sales
    GROUP BY region, product
    ORDER BY total DESC
""")
result.show()

# 4. 시각화용 데이터 추출
top_regions = spark.sql("""
    SELECT region, SUM(sales) as total_sales
    FROM sales
    GROUP BY region
    ORDER BY total_sales DESC
    LIMIT 5
""")
```

---

## 7. 보너스 과제 (+10점)

### Spark Streaming 기초

```python
from pyspark.sql.streaming import StreamingQueryListener

spark = SparkSession.builder \
    .appName("SparkStreamingExample") \
    .master("local[*]") \
    .getOrCreate()

# 1. 스트리밍 소켓 데이터 읽기
lines = spark.readStream \
    .format("socket") \
    .option("host", "localhost") \
    .option("port", 9999) \
    .load()

# 2. 데이터 처리
words = lines.select(explode(split(lines.value, " ")).alias("word"))
word_counts = words.groupBy("word").count()

# 3. 결과 출력
query = word_counts.writeStream \
    .outputMode("complete") \
    .format("console") \
    .start()

query.awaitTermination()
```

---

## 제출 방법

1. **제출 파일:**
   - week08_spark_basics.py
   - week08_rdd_operations.py
   - week08_dataframe_operations.py
   - week08_spark_sql.py
   - week08_project.py (종합)
   - Spark_analysis_report.pdf

2. **제출 기한:** 다음 주 월요일 23:59
3. **제출 방식:** GitHub 레포지토리

---

## 평가 기준

| 항목 | 배점 |
|------|------|
| 설치 및 환경 설정 | 15점 |
| RDD 연산 | 20점 |
| DataFrame | 30점 |
| Spark SQL | 20점 |
| 성능 최적화 | 15점 |
| **소계** | **100점** |
| 보너스 | +10점 |
