# 9주차 실습과제: Spark SQL과 DataFrame 심화

## 과제 목표
- Spark SQL 고급 기능 습득
- 복잡한 DataFrame 연산
- 성능 튜닝 및 최적화
- 실시간 데이터 처리

## 1. DataFrame 고급 연산 (25점)

### 1.1 복잡한 집계

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window

spark = SparkSession.builder.appName("AdvancedDF").master("local[*]").getOrCreate()

# 데이터 로드
df = spark.read.csv("sales_data.csv", header=True, inferSchema=True)

# 1. 여러 조건 집계
agg_df = df.groupBy("region").agg(
    sum("sales").alias("total_sales"),
    avg("sales").alias("avg_sales"),
    max("sales").alias("max_sales"),
    min("sales").alias("min_sales"),
    count("*").alias("transaction_count"),
    stddev("sales").alias("std_sales")
).orderBy(desc("total_sales"))

agg_df.show()

# 2. HAVING 필터링
high_volume = df.groupBy("product") \
    .agg(count("*").alias("count")) \
    .filter(col("count") > 100)

high_volume.show()

# 3. 피벗 테이블
pivot_df = df.groupBy("region") \
    .pivot("product") \
    .agg(sum("sales"))

pivot_df.show()
```

### 1.2 윈도우 함수

```python
# 윈도우 파티션 정의
window = Window.partitionBy("region").orderBy(desc("sales"))

# 1. 순위 함수
ranked_df = df.withColumn("rank", rank().over(window)) \
              .withColumn("dense_rank", dense_rank().over(window)) \
              .withColumn("row_num", row_number().over(window))

ranked_df.show()

# 2. 누적 함수
cumulative_window = Window.partitionBy("region").orderBy("date")
cumsum_df = df.withColumn(
    "cumulative_sales",
    sum("sales").over(cumulative_window)
)

cumsum_df.show()

# 3. LAG/LEAD
lag_lead_window = Window.partitionBy("region").orderBy("date")
lag_lead_df = df.withColumn(
    "prev_sales",
    lag("sales", 1).over(lag_lead_window)
).withColumn(
    "next_sales",
    lead("sales", 1).over(lag_lead_window)
)

lag_lead_df.show()
```

### 1.3 UDF (User Defined Function)

```python
from pyspark.sql.types import DoubleType, StringType

# 1. Python UDF 정의
@udf(returnType=DoubleType())
def custom_discount(sales):
    if sales > 1000:
        return sales * 0.9
    else:
        return sales

# 2. UDF 등록 및 사용
df_with_discount = df.withColumn("discounted_sales", custom_discount(col("sales")))
df_with_discount.show()

# 3. 문자열 UDF
@udf(returnType=StringType())
def sales_category(sales):
    if sales < 500:
        return "Low"
    elif sales < 1000:
        return "Medium"
    else:
        return "High"

df_categorized = df.withColumn("category", sales_category(col("sales")))
df_categorized.show()

# 4. 벡터화 UDF (Pandas UDF - 성능 향상)
import pandas as pd

@pandas_udf(DoubleType())
def vectorized_discount(s):
    return s.apply(lambda x: x * 0.9 if x > 1000 else x)

df_fast = df.withColumn("fast_discount", vectorized_discount(col("sales")))
```

---

## 2. Spark SQL 고급 쿼리 (25점)

### 2.1 CTE (Common Table Expression)

```python
spark.sql("""
    WITH regional_sales AS (
        SELECT region, SUM(sales) as total
        FROM sales
        GROUP BY region
    ),
    avg_sales AS (
        SELECT AVG(total) as avg_total FROM regional_sales
    )
    SELECT r.region, r.total
    FROM regional_sales r
    CROSS JOIN avg_sales a
    WHERE r.total > a.avg_total
""").show()
```

### 2.2 복합 조인

```python
spark.sql("""
    SELECT 
        e.name,
        d.dept_name,
        e.salary,
        s.avg_dept_salary,
        CASE 
            WHEN e.salary > s.avg_dept_salary THEN 'Above Average'
            ELSE 'Below Average'
        END as salary_level
    FROM employees e
    INNER JOIN departments d ON e.dept_id = d.id
    LEFT JOIN (
        SELECT dept_id, AVG(salary) as avg_dept_salary
        FROM employees
        GROUP BY dept_id
    ) s ON e.dept_id = s.dept_id
    ORDER BY d.dept_name, e.salary DESC
""").show()
```

### 2.3 분석 함수

```python
spark.sql("""
    SELECT 
        region,
        product,
        sales,
        SUM(sales) OVER (PARTITION BY region) as region_total,
        RANK() OVER (PARTITION BY region ORDER BY sales DESC) as rank,
        PERCENT_RANK() OVER (PARTITION BY region ORDER BY sales) as percent_rank,
        NTILE(4) OVER (ORDER BY sales) as quartile
    FROM sales
    ORDER BY region, rank
""").show()
```

---

## 3. 성능 최적화 (25점)

### 3.1 캐싱 전략

```python
# 1. DataFrame 캐싱
df.cache()
df.count()  # 캐시에 저장

# 2. 부분 캐싱
filtered_df = df.filter(col("region") == "Seoul")
filtered_df.cache()

# 3. 캐시 모니터링
spark.sql("CACHE TABLE sales_table AS SELECT * FROM sales")
spark.sql("SELECT COUNT(*) FROM CACHE")

# 4. 캐시 제거
df.unpersist()
spark.sql("UNCACHE TABLE sales_table")
```

### 3.2 파티셔닝 최적화

```python
# 1. 적절한 파티션 개수 설정
df_partitioned = df.repartition(100)

# 2. 컬럼 기반 파티셔닝
df_bucketed = df.repartition("region")

# 3. 정렬된 저장
df.sortWithinPartitions("sales").write \
    .parquet("output/path")

# 4. 파티션 통계
print(f"Partitions: {df.rdd.getNumPartitions()}")
print(f"Partition sizes: {df.rdd.mapPartitions(lambda x: [sum(1 for _ in x)]).collect()}")
```

### 3.3 쿼리 플랜 분석

```python
# 1. 기본 플랜
df.select("*").explain()

# 2. 상세 플랜
df.groupBy("region").agg(sum("sales")).explain(extended=True)

# 3. 쿼리 플랜 비교
query1 = spark.sql("SELECT * FROM sales WHERE region='Seoul'")
query2 = spark.sql("SELECT * FROM sales").filter(col("region")=="Seoul")

print("Query 1:")
query1.explain()
print("\nQuery 2:")
query2.explain()
```

---

## 4. 데이터 포맷과 I/O (15점)

### 4.1 다양한 포맷

```python
# 1. CSV
df.write.mode("overwrite").csv("output/data.csv", header=True)
df_csv = spark.read.csv("output/data.csv", header=True, inferSchema=True)

# 2. Parquet (권장)
df.write.mode("overwrite").parquet("output/data.parquet")
df_parquet = spark.read.parquet("output/data.parquet")

# 3. JSON
df.write.mode("overwrite").json("output/data.json")
df_json = spark.read.json("output/data.json")

# 4. ORC
df.write.mode("overwrite").orc("output/data.orc")
df_orc = spark.read.orc("output/data.orc")

# 5. SQL 데이터베이스
df.write.format("jdbc") \
    .option("url", "jdbc:mysql://localhost:3306/mydb") \
    .option("dbtable", "sales") \
    .option("user", "root") \
    .option("password", "password") \
    .mode("overwrite") \
    .save()
```

### 4.2 포맷 성능 비교

```python
import time

# CSV 쓰기/읽기 시간 측정
start = time.time()
df.write.csv("test_csv")
csv_write = time.time() - start

start = time.time()
spark.read.csv("test_csv", header=True).count()
csv_read = time.time() - start

# Parquet 쓰기/읽기 시간 측정
start = time.time()
df.write.parquet("test_parquet")
parquet_write = time.time() - start

start = time.time()
spark.read.parquet("test_parquet").count()
parquet_read = time.time() - start

print(f"CSV - Write: {csv_write:.2f}s, Read: {csv_read:.2f}s")
print(f"Parquet - Write: {parquet_write:.2f}s, Read: {parquet_read:.2f}s")
```

---

## 5. 종합 실습 프로젝트 (30점)

### 목표
대규모 e-commerce 데이터 분석 및 최적화

### 분석 요구사항

```python
# 1. 고객별 RFM 분석
rfm_analysis = spark.sql("""
    SELECT 
        customer_id,
        MAX(order_date) as last_purchase,
        DATEDIFF(current_date(), MAX(order_date)) as recency,
        COUNT(DISTINCT order_id) as frequency,
        SUM(amount) as monetary
    FROM orders
    GROUP BY customer_id
""")

rfm_analysis.show()

# 2. 상품 추천 (협업 필터링 기초)
recommendations = spark.sql("""
    WITH user_purchases AS (
        SELECT customer_id, product_id, COUNT(*) as purchase_count
        FROM orders
        GROUP BY customer_id, product_id
    ),
    similar_purchases AS (
        SELECT 
            u1.customer_id as cust1,
            u2.customer_id as cust2,
            COUNT(*) as common_products
        FROM user_purchases u1
        JOIN user_purchases u2 ON u1.product_id = u2.product_id
        WHERE u1.customer_id != u2.customer_id
        GROUP BY u1.customer_id, u2.customer_id
    )
    SELECT * FROM similar_purchases
    WHERE common_products > 3
""")

recommendations.show()

# 3. 시계열 분석
time_series = spark.sql("""
    SELECT 
        DATE_TRUNC('day', order_date) as order_date,
        SUM(amount) as daily_revenue,
        COUNT(DISTINCT customer_id) as unique_customers
    FROM orders
    GROUP BY DATE_TRUNC('day', order_date)
    ORDER BY order_date
""")

time_series.show()
```

---

## 6. 보너스 과제 (+10점)

### Catalyst 옵티마이저 분석

```python
# 쿼리 플랜을 통한 최적화 학습
df1 = spark.read.csv("customers.csv", header=True, inferSchema=True)
df2 = spark.read.csv("orders.csv", header=True, inferSchema=True)

# 최적화 전
unoptimized = df1.join(df2, df1.id == df2.customer_id) \
    .filter(col("amount") > 100) \
    .select("name", "amount")

unoptimized.explain()

# 최적화됨 (Catalyst에 의해 자동)
unoptimized.collect()
```

---

## 제출 방법

1. **제출 파일:**
   - week09_advanced_dataframe.py
   - week09_advanced_sql.py
   - week09_optimization.py
   - week09_project.py
   - performance_analysis_report.pdf

2. **제출 기한:** 다음 주 월요일 23:59
3. **제출 방식:** GitHub 레포지토리

---

## 평가 기준

| 항목 | 배점 |
|------|------|
| DataFrame 고급 연산 | 25점 |
| SQL 고급 쿼리 | 25점 |
| 성능 최적화 | 25점 |
| 데이터 포맷과 I/O | 15점 |
| 종합 프로젝트 | 10점 |
| **소계** | **100점** |
| 보너스 | +10점 |
