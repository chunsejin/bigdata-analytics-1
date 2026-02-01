# 6주차 실습과제: 관계형 데이터베이스 (SQL)

## 과제 목표
- SQL 기본 문법 습득
- 데이터베이스 설계 및 구축
- 복잡한 쿼리 작성
- 데이터베이스와 Python 연동

## 1. SQL 기초 (20점)

### 1.1 기본 쿼리

```sql
-- 1. 테이블 생성
CREATE TABLE employees (
    emp_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    department TEXT,
    salary INTEGER,
    hire_date DATE
);

-- 2. 데이터 삽입
INSERT INTO employees (emp_id, name, department, salary, hire_date)
VALUES (1, 'Alice', 'Sales', 50000, '2020-01-15');

INSERT INTO employees VALUES
(2, 'Bob', 'IT', 60000, '2019-03-20'),
(3, 'Charlie', 'HR', 55000, '2021-06-10'),
(4, 'David', 'Sales', 52000, '2020-08-05'),
(5, 'Eve', 'IT', 65000, '2018-11-30');

-- 3. 조회 (SELECT)
SELECT * FROM employees;

SELECT name, salary FROM employees;

SELECT name, salary FROM employees WHERE department = 'IT';

-- 4. 정렬
SELECT * FROM employees ORDER BY salary DESC;

-- 5. 조건문
SELECT name, salary FROM employees 
WHERE salary > 55000 AND department = 'IT';
```

### 1.2 집계 함수

```sql
-- 1. COUNT: 행 개수
SELECT COUNT(*) FROM employees;

-- 2. SUM: 합계
SELECT SUM(salary) FROM employees;

-- 3. AVG: 평균
SELECT AVG(salary) FROM employees;

-- 4. MIN, MAX
SELECT MIN(salary), MAX(salary) FROM employees;

-- 5. 그룹별 집계
SELECT department, COUNT(*) as emp_count, AVG(salary) as avg_salary
FROM employees
GROUP BY department;

-- 6. HAVING: 그룹 필터링
SELECT department, AVG(salary) as avg_salary
FROM employees
GROUP BY department
HAVING AVG(salary) > 55000;
```

### 1.3 데이터 수정 및 삭제

```sql
-- UPDATE: 데이터 수정
UPDATE employees
SET salary = 70000
WHERE name = 'Alice';

-- DELETE: 데이터 삭제
DELETE FROM employees
WHERE emp_id = 5;

-- ALTER: 테이블 구조 변경
ALTER TABLE employees ADD COLUMN bonus DECIMAL(10,2);

ALTER TABLE employees DROP COLUMN bonus;
```

---

## 2. JOIN 연산 (25점)

### 2.1 JOIN 기본

```sql
-- 테이블 생성
CREATE TABLE departments (
    dept_id INTEGER PRIMARY KEY,
    dept_name TEXT,
    location TEXT
);

CREATE TABLE employees (
    emp_id INTEGER PRIMARY KEY,
    name TEXT,
    dept_id INTEGER,
    salary INTEGER,
    FOREIGN KEY(dept_id) REFERENCES departments(dept_id)
);

-- 데이터 삽입
INSERT INTO departments VALUES
(1, 'Sales', 'Seoul'),
(2, 'IT', 'Busan'),
(3, 'HR', 'Seoul');

INSERT INTO employees VALUES
(1, 'Alice', 1, 50000),
(2, 'Bob', 2, 60000),
(3, 'Charlie', 1, 52000),
(4, 'David', 2, 65000);

-- 1. INNER JOIN
SELECT e.name, e.salary, d.dept_name, d.location
FROM employees e
INNER JOIN departments d ON e.dept_id = d.dept_id;

-- 2. LEFT JOIN
SELECT e.name, e.salary, d.dept_name
FROM employees e
LEFT JOIN departments d ON e.dept_id = d.dept_id;

-- 3. RIGHT JOIN
SELECT e.name, d.dept_name
FROM employees e
RIGHT JOIN departments d ON e.dept_id = d.dept_id;

-- 4. FULL OUTER JOIN
SELECT e.name, d.dept_name
FROM employees e
FULL OUTER JOIN departments d ON e.dept_id = d.dept_id;

-- 5. CROSS JOIN
SELECT e.name, d.dept_name
FROM employees e
CROSS JOIN departments d;
```

### 2.2 자체 JOIN

```sql
-- 직원과 상사 관계
CREATE TABLE staff (
    emp_id INTEGER PRIMARY KEY,
    name TEXT,
    manager_id INTEGER,
    FOREIGN KEY(manager_id) REFERENCES staff(emp_id)
);

INSERT INTO staff VALUES
(1, 'CEO', NULL),
(2, 'Manager1', 1),
(3, 'Manager2', 1),
(4, 'Employee1', 2),
(5, 'Employee2', 2);

-- 직원과 상사 정보 함께 조회
SELECT e.name as employee, m.name as manager
FROM staff e
LEFT JOIN staff m ON e.manager_id = m.emp_id;
```

---

## 3. 부분쿼리 및 고급 쿼리 (20점)

### 3.1 부분쿼리 (Subquery)

```sql
-- 1. WHERE 절 부분쿼리
SELECT name, salary
FROM employees
WHERE salary > (SELECT AVG(salary) FROM employees);

-- 2. FROM 절 부분쿼리
SELECT *
FROM (
    SELECT department, AVG(salary) as avg_salary
    FROM employees
    GROUP BY department
) dept_avg
WHERE avg_salary > 55000;

-- 3. IN 연산자와 부분쿼리
SELECT name
FROM employees
WHERE dept_id IN (
    SELECT dept_id FROM departments 
    WHERE location = 'Seoul'
);

-- 4. EXISTS 연산자
SELECT name FROM employees e
WHERE EXISTS (
    SELECT 1 FROM departments d
    WHERE e.dept_id = d.dept_id AND d.location = 'Seoul'
);
```

### 3.2 UNION 연산

```sql
-- 1. UNION (중복 제거)
SELECT name, 'Employee' as type FROM employees
UNION
SELECT dept_name, 'Department' FROM departments;

-- 2. UNION ALL (중복 포함)
SELECT salary FROM employees
UNION ALL
SELECT salary FROM employees
WHERE salary > 60000;
```

### 3.3 윈도우 함수

```sql
-- 1. ROW_NUMBER: 행 번호
SELECT name, salary,
    ROW_NUMBER() OVER (ORDER BY salary DESC) as rank
FROM employees;

-- 2. RANK: 순위 (동일값 시 같은 순위)
SELECT name, salary,
    RANK() OVER (ORDER BY salary DESC) as rank
FROM employees;

-- 3. DENSE_RANK: 조밀한 순위
SELECT name, salary,
    DENSE_RANK() OVER (ORDER BY salary DESC) as rank
FROM employees;

-- 4. LAG, LEAD: 이전/다음 행 값
SELECT name, salary,
    LAG(salary) OVER (ORDER BY salary) as prev_salary,
    LEAD(salary) OVER (ORDER BY salary) as next_salary
FROM employees;

-- 5. 누적 합계
SELECT name, salary,
    SUM(salary) OVER (ORDER BY salary) as cumsum
FROM employees;
```

---

## 4. 인덱스 및 성능 최적화 (15점)

### 4.1 인덱스 생성

```sql
-- 1. 단일 인덱스
CREATE INDEX idx_name ON employees(name);

-- 2. 복합 인덱스
CREATE INDEX idx_dept_salary ON employees(department, salary);

-- 3. UNIQUE 인덱스
CREATE UNIQUE INDEX idx_email ON employees(email);

-- 4. 인덱스 조회
SELECT * FROM sqlite_master WHERE type='index';

-- 5. 인덱스 삭제
DROP INDEX idx_name;
```

### 4.2 성능 분석

```sql
-- 1. EXPLAIN QUERY PLAN: 실행 계획 보기
EXPLAIN QUERY PLAN
SELECT * FROM employees WHERE name = 'Alice';

-- 2. 쿼리 성능 측정
-- .timer ON (SQLite의 경우)
SELECT COUNT(*) FROM employees;
```

---

## 5. Python과 SQL 연동 (20점)

### 5.1 SQLite 연동

```python
import sqlite3
import pandas as pd

# 1. 데이터베이스 연결
conn = sqlite3.connect('company.db')
cursor = conn.cursor()

# 2. 테이블 생성
cursor.execute('''
    CREATE TABLE IF NOT EXISTS employees (
        emp_id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        department TEXT,
        salary INTEGER
    )
''')

# 3. 데이터 삽입
employees = [
    (1, 'Alice', 'Sales', 50000),
    (2, 'Bob', 'IT', 60000),
    (3, 'Charlie', 'HR', 55000)
]

cursor.executemany(
    'INSERT INTO employees VALUES (?, ?, ?, ?)',
    employees
)
conn.commit()

# 4. 데이터 조회
cursor.execute('SELECT * FROM employees')
results = cursor.fetchall()
for row in results:
    print(row)

# 5. Pandas와 통합
df = pd.read_sql_query('SELECT * FROM employees', conn)
print(df)

# 6. 연결 종료
conn.close()
```

### 5.2 데이터베이스 작업 함수

```python
import sqlite3
import pandas as pd

class DatabaseManager:
    def __init__(self, db_name):
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()
    
    def execute_query(self, query):
        """쿼리 실행"""
        self.cursor.execute(query)
        self.conn.commit()
    
    def fetch_all(self, query):
        """모든 결과 반환"""
        return pd.read_sql_query(query, self.conn)
    
    def insert_dataframe(self, df, table_name):
        """DataFrame을 테이블로 저장"""
        df.to_sql(table_name, self.conn, if_exists='append', index=False)
    
    def close(self):
        """연결 종료"""
        self.conn.close()

# 사용 예시
db = DatabaseManager('company.db')
df = db.fetch_all('SELECT * FROM employees WHERE salary > 50000')
print(df)
db.close()
```

---

## 6. 종합 실습 프로젝트 (25점)

### 목표
온라인 쇼핑 데이터베이스 설계 및 구축

### 데이터베이스 스키마

```sql
-- 고객 테이블
CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT UNIQUE,
    phone TEXT,
    join_date DATE,
    city TEXT
);

-- 상품 테이블
CREATE TABLE products (
    product_id INTEGER PRIMARY KEY,
    product_name TEXT NOT NULL,
    category TEXT,
    price DECIMAL(10, 2),
    stock_quantity INTEGER
);

-- 주문 테이블
CREATE TABLE orders (
    order_id INTEGER PRIMARY KEY,
    customer_id INTEGER,
    order_date DATE,
    total_amount DECIMAL(10, 2),
    status TEXT,
    FOREIGN KEY(customer_id) REFERENCES customers(customer_id)
);

-- 주문 항목 테이블
CREATE TABLE order_items (
    item_id INTEGER PRIMARY KEY,
    order_id INTEGER,
    product_id INTEGER,
    quantity INTEGER,
    price DECIMAL(10, 2),
    FOREIGN KEY(order_id) REFERENCES orders(order_id),
    FOREIGN KEY(product_id) REFERENCES products(product_id)
);

-- 리뷰 테이블
CREATE TABLE reviews (
    review_id INTEGER PRIMARY KEY,
    customer_id INTEGER,
    product_id INTEGER,
    rating INTEGER,
    comment TEXT,
    review_date DATE,
    FOREIGN KEY(customer_id) REFERENCES customers(customer_id),
    FOREIGN KEY(product_id) REFERENCES products(product_id)
);
```

### 요구 분석 사항

**1. 데이터 삽입**
```sql
-- 샘플 데이터 삽입
INSERT INTO customers VALUES
(1, 'Alice', 'alice@email.com', '010-1234-5678', '2023-01-15', 'Seoul'),
(2, 'Bob', 'bob@email.com', '010-2345-6789', '2023-02-20', 'Busan');

INSERT INTO products VALUES
(1, 'Laptop', 'Electronics', 1000000, 50),
(2, 'Mouse', 'Electronics', 30000, 200),
(3, 'Keyboard', 'Electronics', 80000, 150);
```

**2. 기본 조회 쿼리**
```sql
-- 고객별 주문 현황
SELECT c.name, COUNT(o.order_id) as order_count, SUM(o.total_amount) as total_spent
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.customer_id;

-- 상품별 판매량
SELECT p.product_name, SUM(oi.quantity) as total_quantity, SUM(oi.quantity * oi.price) as revenue
FROM products p
LEFT JOIN order_items oi ON p.product_id = oi.product_id
GROUP BY p.product_id
ORDER BY revenue DESC;
```

**3. 고급 분석 쿼리**
```sql
-- 고객별 평균 주문액
SELECT c.name, AVG(o.total_amount) as avg_order
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.customer_id;

-- 최근 주문 상위 5개
SELECT order_id, customer_id, order_date, total_amount
FROM orders
ORDER BY order_date DESC
LIMIT 5;

-- 카테고리별 평균 평점
SELECT p.category, AVG(r.rating) as avg_rating, COUNT(r.review_id) as review_count
FROM products p
LEFT JOIN reviews r ON p.product_id = r.product_id
GROUP BY p.category;
```

**4. Python에서 분석**
```python
import sqlite3
import pandas as pd

conn = sqlite3.connect('shopping.db')

# 1. 고객별 구매액
query = """
SELECT c.name, COUNT(o.order_id) as orders, SUM(o.total_amount) as total
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.customer_id
"""
df_customer = pd.read_sql_query(query, conn)
print(df_customer)

# 2. 상품별 판매 현황
query = """
SELECT p.product_name, SUM(oi.quantity) as qty, AVG(r.rating) as rating
FROM products p
LEFT JOIN order_items oi ON p.product_id = oi.product_id
LEFT JOIN reviews r ON p.product_id = r.product_id
GROUP BY p.product_id
"""
df_product = pd.read_sql_query(query, conn)
print(df_product)

conn.close()
```

---

## 7. 보너스 과제 (+10점)

### 트랜잭션 및 데이터 무결성

```python
import sqlite3

conn = sqlite3.connect('company.db')
cursor = conn.cursor()

try:
    # 트랜잭션 시작
    cursor.execute('BEGIN TRANSACTION')
    
    # 계정 A에서 1000원 출금
    cursor.execute('UPDATE accounts SET balance = balance - 1000 WHERE account_id = 1')
    
    # 계정 B에 1000원 입금
    cursor.execute('UPDATE accounts SET balance = balance + 1000 WHERE account_id = 2')
    
    # 트랜잭션 커밋
    conn.commit()
    print("Transaction successful")
    
except Exception as e:
    # 오류 시 롤백
    conn.rollback()
    print(f"Transaction failed: {e}")

finally:
    conn.close()
```

---

## 제출 방법

1. **제출 파일:**
   - week06_sql_basics.py
   - week06_join_operations.py
   - week06_advanced_queries.py
   - week06_optimization.py
   - week06_project_queries.sql
   - shopping_database.db (생성된 데이터베이스)
   - database_analysis_report.pdf

2. **제출 기한:** 다음 주 월요일 23:59
3. **제출 방식:** GitHub 레포지토리

---

## 평가 기준

| 항목 | 배점 |
|------|------|
| SQL 기초 | 20점 |
| JOIN 연산 | 25점 |
| 부분쿼리 및 고급 쿼리 | 20점 |
| 인덱스 및 성능 | 15점 |
| Python 연동 | 20점 |
| **소계** | **100점** |
| 보너스 | +10점 |
