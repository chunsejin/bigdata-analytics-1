# 13주차 실습과제: 고급 프로젝트 및 배포

## 과제 목표
- 엔드-투-엔드 머신러닝 프로젝트 구축
- 모델 저장 및 로드
- 웹 서비스 배포
- 모니터링 및 유지보수

## 1. 엔드-투-엔드 프로젝트 (30점)

### 1.1 주택 가격 예측 프로젝트

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle

# 1단계: 데이터 로드 및 탐색
df = pd.read_csv('house_prices.csv')
print("데이터 형태:", df.shape)
print("\n결측치:")
print(df.isnull().sum())
print("\n기본 통계:")
print(df.describe())

# 2단계: 데이터 전처리
# 결측치 처리
df['price'].fillna(df['price'].mean(), inplace=True)
df['sqft_living'].fillna(df['sqft_living'].mean(), inplace=True)

# 범주형 변수 인코딩
df['has_basement'] = df['has_basement'].astype(int)

# 특성 엔지니어링
df['price_per_sqft'] = df['price'] / df['sqft_living']
df['age'] = 2024 - df['year_built']

# 3단계: 특성 선택
features = ['sqft_living', 'bedrooms', 'bathrooms', 'age', 'price_per_sqft']
X = df[features]
y = df['price']

# 결측치 제거
X = X.dropna()
y = y[X.index]

# 4단계: 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5단계: 데이터 정규화
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6단계: 모델 학습
models = {
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {
        'model': model,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }
    
    print(f"\n{name}:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R²: {r2:.4f}")

# 7단계: 최적 모델 선택
best_model_name = max(results, key=lambda x: results[x]['R2'])
best_model = results[best_model_name]['model']
print(f"\n최적 모델: {best_model_name}")
```

### 1.2 모델 저장 및 로드

```python
import pickle
import joblib

# 1. Pickle 사용
with open('best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# 2. Joblib 사용 (더 효율적)
joblib.dump(best_model, 'best_model.joblib')
joblib.dump(scaler, 'scaler.joblib')

# 3. 모델 로드
loaded_model = joblib.load('best_model.joblib')
loaded_scaler = joblib.load('scaler.joblib')

# 4. 저장된 모델으로 예측
new_data = X_test_scaled[:5]
predictions = loaded_model.predict(new_data)
print(f"예측값: {predictions}")
```

---

## 2. Flask 웹 서비스 (25점)

### 2.1 기본 Flask 앱

```python
# app.py
from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# 모델 로드
model = joblib.load('best_model.joblib')
scaler = joblib.load('scaler.joblib')

@app.route('/')
def home():
    return jsonify({'message': '주택 가격 예측 API'})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 요청 데이터 받기
        data = request.json
        
        # 특성 추출
        features = [
            data['sqft_living'],
            data['bedrooms'],
            data['bathrooms'],
            data['age'],
            data['price_per_sqft']
        ]
        
        # 데이터 정규화
        features_scaled = scaler.transform([features])
        
        # 예측
        prediction = model.predict(features_scaled)[0]
        
        return jsonify({
            'predicted_price': round(prediction, 2),
            'input_features': data
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    try:
        data = request.json
        df = pd.DataFrame(data)
        
        # 정규화 및 예측
        df_scaled = scaler.transform(df)
        predictions = model.predict(df_scaled)
        
        return jsonify({
            'predictions': predictions.tolist()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/model_info', methods=['GET'])
def model_info():
    return jsonify({
        'model_type': str(type(model)),
        'features': ['sqft_living', 'bedrooms', 'bathrooms', 'age', 'price_per_sqft'],
        'version': '1.0'
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

### 2.2 클라이언트 테스트

```python
import requests

# API 테스트
url = 'http://localhost:5000/predict'

data = {
    'sqft_living': 2000,
    'bedrooms': 3,
    'bathrooms': 2,
    'age': 10,
    'price_per_sqft': 500
}

response = requests.post(url, json=data)
print(response.json())

# 배치 예측
batch_url = 'http://localhost:5000/batch_predict'
batch_data = [
    {'sqft_living': 2000, 'bedrooms': 3, 'bathrooms': 2, 'age': 10, 'price_per_sqft': 500},
    {'sqft_living': 1500, 'bedrooms': 2, 'bathrooms': 1, 'age': 5, 'price_per_sqft': 600}
]

batch_response = requests.post(batch_url, json=batch_data)
print(batch_response.json())
```

---

## 3. Docker 컨테이너화 (15점)

### 3.1 Dockerfile

```dockerfile
# Dockerfile
FROM python:3.9

WORKDIR /app

# 의존성 복사
COPY requirements.txt .
RUN pip install -r requirements.txt

# 애플리케이션 파일 복사
COPY app.py .
COPY best_model.joblib .
COPY scaler.joblib .

# 포트 노출
EXPOSE 5000

# 실행 명령어
CMD ["python", "app.py"]
```

### 3.2 docker-compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  web:
    build: .
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=production
    volumes:
      - ./data:/app/data
    
  db:
    image: postgres:13
    environment:
      - POSTGRES_USER=admin
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=predictions
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

### 3.3 빌드 및 실행

```bash
# 이미지 빌드
docker build -t house-price-predictor .

# 컨테이너 실행
docker run -p 5000:5000 house-price-predictor

# Docker Compose 실행
docker-compose up -d

# 로그 확인
docker logs <container_id>

# 컨테이너 중지
docker stop <container_id>
```

---

## 4. 모니터링 및 로깅 (15점)

### 4.1 로깅

```python
import logging
from logging.handlers import RotatingFileHandler

# 로깅 설정
logger = logging.getLogger('house_price_predictor')
logger.setLevel(logging.DEBUG)

# 파일 핸들러
file_handler = RotatingFileHandler(
    'app.log',
    maxBytes=10485760,  # 10MB
    backupCount=10
)
file_handler.setLevel(logging.INFO)

# 콘솔 핸들러
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# 포매터
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# 핸들러 추가
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# 로깅 사용
logger.info('애플리케이션 시작')
logger.debug('디버그 정보')
logger.error('오류 발생')
```

### 4.2 성능 모니터링

```python
import time
from datetime import datetime

class PredictionMonitor:
    def __init__(self):
        self.predictions = []
    
    def log_prediction(self, features, prediction, latency):
        self.predictions.append({
            'timestamp': datetime.now(),
            'features': features,
            'prediction': prediction,
            'latency': latency
        })
    
    def get_statistics(self):
        latencies = [p['latency'] for p in self.predictions]
        return {
            'total_predictions': len(self.predictions),
            'avg_latency': np.mean(latencies),
            'max_latency': np.max(latencies),
            'min_latency': np.min(latencies)
        }

# 사용 예시
monitor = PredictionMonitor()

for _ in range(100):
    start_time = time.time()
    prediction = model.predict(X_test_scaled[:1])[0]
    latency = time.time() - start_time
    monitor.log_prediction(X_test[0].values, prediction, latency)

print(monitor.get_statistics())
```

---

## 5. 종합 프로젝트 (30점)

### 요구사항

1. **데이터 파이프라인**
   - 데이터 수집
   - 전처리 및 정제
   - 특성 엔지니어링

2. **모델 개발**
   - 여러 모델 비교
   - 하이퍼파라미터 튜닝
   - 교차 검증

3. **배포**
   - Flask/FastAPI 웹 서비스
   - Docker 컨테이너화
   - 모니터링

4. **문서화**
   - README
   - API 문서
   - 성능 보고서

---

## 6. 보너스 과제 (+10점)

### CI/CD 파이프라인

```yaml
# .github/workflows/ci-cd.yml
name: CI/CD Pipeline

on: [push, pull_request]

jobs:
  build:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    
    - name: Run tests
      run: |
        pytest tests/
    
    - name: Build Docker image
      run: |
        docker build -t myapp:latest .
    
    - name: Push to Docker Hub
      run: |
        docker push myapp:latest
```

---

## 제출 방법

1. **제출 파일:**
   - week13_end_to_end_project.py
   - app.py (Flask)
   - Dockerfile
   - docker-compose.yml
   - requirements.txt
   - tests/ (테스트 코드)
   - README.md
   - deployment_guide.pdf

2. **제출 기한:** 다음 주 월요일 23:59
3. **제출 방식:** GitHub 레포지토리

---

## 평가 기준

| 항목 | 배점 |
|------|------|
| 엔드-투-엔드 프로젝트 | 30점 |
| Flask 웹 서비스 | 25점 |
| Docker 배포 | 15점 |
| 모니터링 및 로깅 | 15점 |
| 종합 프로젝트 | 15점 |
| **소계** | **100점** |
| 보너스 | +10점 |
