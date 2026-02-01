# 7주차 실습과제: NoSQL 데이터베이스 (MongoDB)

## 과제 목표
- MongoDB 기본 개념 이해
- 문서 지향 데이터 모델링
- CRUD 연산 습득
- Python으로 MongoDB 조작

## 1. MongoDB 기초 (20점)

### 1.1 기본 개념 및 설치

```python
# MongoDB 설치 및 연결
# 설치: brew install mongodb-community (macOS)
# 또는 Docker: docker run -d -p 27017:27017 --name mongodb mongo

import pymongo

# MongoDB 연결
client = pymongo.MongoClient('mongodb://localhost:27017/')
db = client['company_db']  # 데이터베이스 선택
collection = db['employees']  # 컬렉션 선택

print("Connected to MongoDB successfully!")
```

### 1.2 데이터 삽입

```python
# 1. 단일 문서 삽입
employee = {
    "_id": 1,
    "name": "Alice",
    "department": "Sales",
    "salary": 50000,
    "skills": ["Excel", "PowerPoint"],
    "hire_date": "2020-01-15"
}
result = collection.insert_one(employee)
print(f"Inserted ID: {result.inserted_id}")

# 2. 다중 문서 삽입
employees = [
    {
        "_id": 2,
        "name": "Bob",
        "department": "IT",
        "salary": 60000,
        "skills": ["Python", "JavaScript", "SQL"]
    },
    {
        "_id": 3,
        "name": "Charlie",
        "department": "HR",
        "salary": 55000,
        "skills": ["Communication", "Leadership"]
    },
    {
        "_id": 4,
        "name": "David",
        "department": "Sales",
        "salary": 52000,
        "skills": ["Excel", "CRM"]
    }
]
result = collection.insert_many(employees)
print(f"Inserted IDs: {result.inserted_ids}")
```

---

## 2. CRUD 연산 (25점)

### 2.1 조회 (Read)

```python
# 1. 모든 문서 조회
all_employees = collection.find()
for emp in all_employees:
    print(emp)

# 2. 첫 번째 문서
first_emp = collection.find_one()
print(first_emp)

# 3. 특정 필드로 조회
it_employees = collection.find({"department": "IT"})
for emp in it_employees:
    print(emp)

# 4. 조건 쿼리
# 급여가 55000 이상
high_salary = collection.find({"salary": {"$gte": 55000}})
for emp in high_salary:
    print(emp)

# 5. OR 연산
or_query = collection.find({
    "$or": [
        {"department": "Sales"},
        {"department": "IT"}
    ]
})
for emp in or_query:
    print(emp)

# 6. AND 연산
and_query = collection.find({
    "department": "Sales",
    "salary": {"$gt": 50000}
})

# 7. IN 연산
in_query = collection.find({
    "department": {"$in": ["Sales", "HR"]}
})

# 8. NOT 연산
not_query = collection.find({
    "department": {"$ne": "IT"}
})
```

### 2.2 정렬 및 제한

```python
# 1. 정렬
# 급여 기준 오름차순
sorted_asc = collection.find().sort("salary", 1)

# 급여 기준 내림차순
sorted_desc = collection.find().sort("salary", -1)

# 2. 개수 제한
top_3 = collection.find().sort("salary", -1).limit(3)
for emp in top_3:
    print(f"{emp['name']}: {emp['salary']}")

# 3. 건너뛰기
skip_query = collection.find().skip(2).limit(2)

# 4. 프로젝션 (필드 선택)
projection = collection.find(
    {},
    {"name": 1, "salary": 1, "_id": 0}  # name, salary만 선택
)
for emp in projection:
    print(emp)
```

### 2.3 수정 (Update)

```python
# 1. 단일 문서 수정
collection.update_one(
    {"_id": 1},
    {"$set": {"salary": 55000}}
)

# 2. 다중 문서 수정
collection.update_many(
    {"department": "Sales"},
    {"$set": {"bonus": 5000}}
)

# 3. 필드 증가
collection.update_one(
    {"_id": 1},
    {"$inc": {"salary": 1000}}  # 1000 증가
)

# 4. 배열에 요소 추가
collection.update_one(
    {"_id": 1},
    {"$push": {"skills": "Tableau"}}
)

# 5. 배열에서 요소 제거
collection.update_one(
    {"_id": 1},
    {"$pull": {"skills": "Excel"}}
)

# 6. 문서 생성 또는 수정 (upsert)
collection.update_one(
    {"_id": 5},
    {
        "$set": {
            "name": "Eve",
            "department": "Finance",
            "salary": 58000
        }
    },
    upsert=True  # 없으면 생성
)
```

### 2.4 삭제 (Delete)

```python
# 1. 단일 문서 삭제
collection.delete_one({"_id": 5})

# 2. 다중 문서 삭제
collection.delete_many({"department": "Finance"})

# 3. 모든 문서 삭제
# collection.delete_many({})
```

---

## 3. 집계 연산 (25점)

### 3.1 집계 파이프라인

```python
# 1. 그룹 및 집계
pipeline = [
    {
        "$group": {
            "_id": "$department",
            "avg_salary": {"$avg": "$salary"},
            "count": {"$sum": 1},
            "max_salary": {"$max": "$salary"},
            "min_salary": {"$min": "$salary"}
        }
    }
]
result = collection.aggregate(pipeline)
for item in result:
    print(item)

# 2. 정렬 및 제한
pipeline = [
    {"$sort": {"salary": -1}},
    {"$limit": 3}
]
top_earners = collection.aggregate(pipeline)
for emp in top_earners:
    print(f"{emp['name']}: {emp['salary']}")

# 3. 매칭 (필터링)
pipeline = [
    {"$match": {"salary": {"$gt": 50000}}},
    {"$group": {
        "_id": "$department",
        "count": {"$sum": 1}
    }}
]

# 4. 프로젝션
pipeline = [
    {
        "$project": {
            "name": 1,
            "salary": 1,
            "senior_level": {
                "$cond": [
                    {"$gte": ["$salary", 55000]},
                    "Senior",
                    "Junior"
                ]
            }
        }
    }
]
result = collection.aggregate(pipeline)
for emp in result:
    print(emp)

# 5. 언래핑 (배열 펼치기)
# skills 배열의 각 요소를 별도 문서로
pipeline = [
    {"$unwind": "$skills"}
]
```

### 3.2 복잡한 집계

```python
# 부서별 급여 총액 및 직원 수
pipeline = [
    {
        "$group": {
            "_id": "$department",
            "total_salary": {"$sum": "$salary"},
            "emp_count": {"$sum": 1},
            "avg_salary": {"$avg": "$salary"}
        }
    },
    {
        "$sort": {"total_salary": -1}
    },
    {
        "$project": {
            "_id": 1,
            "total_salary": 1,
            "emp_count": 1,
            "avg_salary": {"$round": ["$avg_salary", 0]}
        }
    }
]

result = collection.aggregate(pipeline)
for item in result:
    print(item)
```

---

## 4. 인덱싱 및 최적화 (15점)

### 4.1 인덱스 생성

```python
# 1. 단일 필드 인덱스
collection.create_index("department")
collection.create_index("salary")

# 2. 복합 인덱스
collection.create_index([("department", 1), ("salary", -1)])

# 3. 고유 인덱스
collection.create_index("email", unique=True)

# 4. 인덱스 목록 조회
indexes = collection.list_indexes()
for index in indexes:
    print(index)

# 5. 인덱스 삭제
collection.drop_index("department_1")
```

### 4.2 쿼리 최적화

```python
# explain() 사용하여 쿼리 성능 분석
explain = collection.find({"department": "IT"}).explain()
print(f"Executed stages: {len(explain['executionStats']['executionStages'])}")
print(f"Documents examined: {explain['executionStats']['totalDocsExamined']}")
print(f"Documents returned: {explain['executionStats']['nReturned']}")
```

---

## 5. 트랜잭션 및 데이터 유효성 (15점)

### 5.1 트랜잭션

```python
from pymongo import client_session

# MongoDB 4.0+ 필요
client = pymongo.MongoClient('mongodb://localhost:27017/')

# 트랜잭션 세션 시작
session = client.start_session()

try:
    session.start_transaction()
    
    # 여러 작업 수행
    db = client['company_db']
    collection = db['employees']
    
    collection.insert_one(
        {"_id": 5, "name": "Eve", "department": "Finance"},
        session=session
    )
    
    collection.update_one(
        {"_id": 1},
        {"$inc": {"salary": 1000}},
        session=session
    )
    
    session.commit_transaction()
    print("Transaction completed successfully")
    
except Exception as e:
    session.abort_transaction()
    print(f"Transaction failed: {e}")

finally:
    session.end_session()
```

### 5.2 데이터 검증 스키마

```python
# 컬렉션 검증 규칙 정의
validator = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["name", "department", "salary"],
        "properties": {
            "_id": {"bsonType": "int"},
            "name": {
                "bsonType": "string",
                "description": "must be a string and is required"
            },
            "department": {
                "enum": ["Sales", "IT", "HR", "Finance"],
                "description": "must be one of the enum values"
            },
            "salary": {
                "bsonType": "int",
                "minimum": 0,
                "description": "must be a positive integer"
            },
            "skills": {
                "bsonType": "array",
                "items": {"bsonType": "string"}
            }
        }
    }
}

# 유효성 검사 적용
db.command("collMod", "employees", validator=validator)
```

---

## 6. 종합 실습 프로젝트 (25점)

### 목표
전자상거래 데이터를 MongoDB로 관리

### 문서 구조

```python
import pymongo
from datetime import datetime

client = pymongo.MongoClient('mongodb://localhost:27017/')
db = client['ecommerce_db']

# 고객 컬렉션
customers = db['customers']
customer_doc = {
    "_id": 1,
    "name": "Alice",
    "email": "alice@email.com",
    "phone": "010-1234-5678",
    "address": {
        "city": "Seoul",
        "district": "Gangnam",
        "zip": "12345"
    },
    "join_date": datetime.now(),
    "purchase_history": [1, 2, 3]  # 주문 ID
}

# 상품 컬렉션
products = db['products']
product_doc = {
    "_id": 1,
    "name": "Laptop",
    "category": "Electronics",
    "price": 1000000,
    "stock": 50,
    "rating": 4.5,
    "reviews_count": 120,
    "tags": ["computer", "portable"],
    "created_at": datetime.now()
}

# 주문 컬렉션
orders = db['orders']
order_doc = {
    "_id": 1,
    "customer_id": 1,
    "order_date": datetime.now(),
    "items": [
        {
            "product_id": 1,
            "quantity": 1,
            "price": 1000000
        },
        {
            "product_id": 2,
            "quantity": 2,
            "price": 30000
        }
    ],
    "total_amount": 1060000,
    "status": "Delivered",
    "shipping_address": {
        "city": "Seoul",
        "zip": "12345"
    }
}
```

### 요구 분석 사항

**1. 데이터 삽입**
```python
# 각 컬렉션에 샘플 데이터 추가
customers.insert_many([customer_doc, ...])
products.insert_many([product_doc, ...])
orders.insert_many([order_doc, ...])
```

**2. 기본 조회**
```python
# 고객별 주문 수
pipeline = [
    {
        "$group": {
            "_id": "$customer_id",
            "order_count": {"$sum": 1},
            "total_spent": {"$sum": "$total_amount"}
        }
    },
    {"$sort": {"total_spent": -1}}
]
```

**3. 고급 분석**
```python
# 상품별 판매량 및 수익
pipeline = [
    {"$unwind": "$items"},
    {
        "$group": {
            "_id": "$items.product_id",
            "total_qty": {"$sum": "$items.quantity"},
            "total_revenue": {"$sum": {
                "$multiply": ["$items.quantity", "$items.price"]
            }}
        }
    },
    {"$sort": {"total_revenue": -1}}
]

result = orders.aggregate(pipeline)
for item in result:
    print(item)
```

**4. 데이터 업데이트**
```python
# 고객 구매 이력 업데이트
customers.update_one(
    {"_id": 1},
    {"$push": {"purchase_history": 4}}
)

# 상품 재고 감소
products.update_one(
    {"_id": 1},
    {"$dec": {"stock": 1}}
)
```

---

## 7. 보너스 과제 (+10점)

### 실시간 데이터 모니터링

```python
# Change Streams를 이용한 실시간 모니터링
with collection.watch([]) as stream:
    for change in stream:
        print(f"Change detected: {change}")
        if change['operationType'] == 'insert':
            print(f"New document inserted: {change['fullDocument']}")
        elif change['operationType'] == 'update':
            print(f"Document updated: {change['updateDescription']}")
        elif change['operationType'] == 'delete':
            print(f"Document deleted: {change['documentKey']}")
```

---

## 제출 방법

1. **제출 파일:**
   - week07_mongodb_basics.py
   - week07_crud_operations.py
   - week07_aggregation.py
   - week07_indexing.py
   - week07_project.py (종합)
   - MongoDB_analysis_report.pdf

2. **제출 기한:** 다음 주 월요일 23:59
3. **제출 방식:** GitHub 레포지토리

---

## 평가 기준

| 항목 | 배점 |
|------|------|
| MongoDB 기초 | 20점 |
| CRUD 연산 | 25점 |
| 집계 연산 | 25점 |
| 인덱싱 및 최적화 | 15점 |
| 트랜잭션 및 검증 | 15점 |
| **소계** | **100점** |
| 보너스 | +10점 |
