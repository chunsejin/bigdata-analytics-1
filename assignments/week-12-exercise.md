# 12주차 실습과제: 데이터 시각화 및 대시보드

## 과제 목표
- 다양한 시각화 기법 습득
- 인터랙티브 대시보드 구축
- 데이터 스토리텔링 이해
- BI 도구 활용

## 1. Matplotlib 심화 (20점)

### 1.1 고급 플롯

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 1. Subplots 활용
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Subplot 1: 선 그래프
x = np.linspace(0, 10, 100)
axes[0, 0].plot(x, np.sin(x), label='sin(x)')
axes[0, 0].plot(x, np.cos(x), label='cos(x)')
axes[0, 0].set_title('삼각함수')
axes[0, 0].legend()

# Subplot 2: 히스토그램
axes[0, 1].hist(np.random.randn(1000), bins=30, edgecolor='black')
axes[0, 1].set_title('정규분포')

# Subplot 3: 산점도
axes[1, 0].scatter(np.random.randn(100), np.random.randn(100), alpha=0.6)
axes[1, 0].set_title('산점도')

# Subplot 4: 막대 그래프
categories = ['A', 'B', 'C', 'D']
values = [23, 45, 56, 78]
axes[1, 1].bar(categories, values, color=['red', 'blue', 'green', 'yellow'])
axes[1, 1].set_title('카테고리별 값')

plt.tight_layout()
plt.show()

# 2. 그리드 지정 레이아웃
fig = plt.figure(figsize=(12, 8))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

ax1 = fig.add_subplot(gs[0, :])
ax2 = fig.add_subplot(gs[1, :-1])
ax3 = fig.add_subplot(gs[1:, -1])

ax1.plot([1, 2, 3, 4])
ax2.scatter(np.random.randn(50), np.random.randn(50))
ax3.imshow(np.random.randn(10, 10), cmap='viridis')

plt.show()
```

### 1.2 고급 스타일링

```python
# 스타일 설정
plt.style.use('seaborn-v0_8-darkgrid')

# 컬러맵
x = np.linspace(0, 10, 100)
fig, ax = plt.subplots()

# 그라디언트 색상
colors = plt.cm.viridis(np.linspace(0, 1, 5))
for i in range(5):
    ax.plot(x, np.sin(x + i), color=colors[i], label=f'sin(x+{i})')

ax.legend()
plt.show()

# 주석 추가
fig, ax = plt.subplots()
ax.plot([1, 2, 3, 4, 5], [1, 4, 2, 3, 5])
ax.annotate('Peak', xy=(2, 4), xytext=(2.5, 4.5),
            arrowprops=dict(arrowstyle='->', color='red'),
            fontsize=12)
plt.show()
```

---

## 2. Seaborn 고급 기법 (20점)

### 2.1 통계 시각화

```python
import seaborn as sns

# 데이터 로드
tips = sns.load_dataset('tips')

# 1. 관계 시각화
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Scatter with regression
sns.regplot(data=tips, x='total_bill', y='tip', ax=axes[0, 0])

# Hexbin plot
axes[0, 1].hexbin(tips['total_bill'], tips['tip'], gridsize=15, cmap='YlOrRd')

# Joint plot
sns.jointplot(data=tips, x='total_bill', y='tip', kind='scatter')

# Pair plot
sns.pairplot(tips, hue='sex')

plt.show()

# 2. 분포 시각화
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Violin plot
sns.violinplot(data=tips, x='day', y='total_bill', ax=axes[0])

# Box plot
sns.boxplot(data=tips, x='day', y='total_bill', hue='sex', ax=axes[1])

# Strip plot
sns.stripplot(data=tips, x='day', y='total_bill', hue='sex', ax=axes[2])

plt.show()

# 3. 범주형 플롯
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

sns.barplot(data=tips, x='day', y='total_bill', ax=axes[0])
sns.countplot(data=tips, x='day', hue='sex', ax=axes[1])

plt.show()

# 4. 행렬 시각화
pivot_table = tips.pivot_table(
    values='tip', index='day', columns='sex', aggfunc='mean'
)
sns.heatmap(pivot_table, annot=True, fmt='.2f', cmap='coolwarm')
plt.show()
```

---

## 3. Plotly 인터랙티브 시각화 (20점)

### 3.1 기본 인터랙티브 플롯

```python
import plotly.graph_objects as go
import plotly.express as px

# 1. Scatter plot
fig = px.scatter(
    tips,
    x='total_bill',
    y='tip',
    color='sex',
    size='total_bill',
    hover_data=['day', 'time'],
    title='Total Bill vs Tip'
)
fig.show()

# 2. Box plot
fig = px.box(
    tips,
    x='day',
    y='total_bill',
    color='sex',
    title='Distribution of Total Bill by Day'
)
fig.show()

# 3. Bar chart
fig = px.bar(
    tips.groupby('day')['total_bill'].sum().reset_index(),
    x='day',
    y='total_bill',
    title='Total Bill by Day'
)
fig.show()

# 4. Line chart
dates = pd.date_range('2024-01-01', periods=100)
values = np.cumsum(np.random.randn(100))
df_ts = pd.DataFrame({'date': dates, 'value': values})

fig = px.line(df_ts, x='date', y='value', title='Time Series')
fig.show()
```

### 3.2 다중 서브플롯

```python
from plotly.subplots import make_subplots

# 서브플롯 생성
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Scatter', 'Box', 'Bar', 'Histogram'),
    specs=[[{'type': 'scatter'}, {'type': 'box'}],
           [{'type': 'bar'}, {'type': 'histogram'}]]
)

# Scatter
fig.add_trace(
    go.Scatter(x=tips['total_bill'], y=tips['tip'], mode='markers'),
    row=1, col=1
)

# Box
fig.add_trace(
    go.Box(y=tips['total_bill']),
    row=1, col=2
)

# Bar
fig.add_trace(
    go.Bar(x=tips['day'].unique(), y=tips.groupby('day')['total_bill'].sum()),
    row=2, col=1
)

# Histogram
fig.add_trace(
    go.Histogram(x=tips['total_bill']),
    row=2, col=2
)

fig.update_layout(height=800, showlegend=False)
fig.show()
```

---

## 4. 대시보드 구축 (20점)

### 4.1 Streamlit 대시보드

```python
# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# 페이지 설정
st.set_page_config(page_title="판매 대시보드", layout="wide")

# 제목
st.title("📊 판매 데이터 대시보드")

# 사이드바
st.sidebar.header("필터")
selected_region = st.sidebar.multiselect(
    "지역 선택",
    options=['Seoul', 'Busan', 'Daegu', 'Incheon'],
    default=['Seoul']
)

# 샘플 데이터
@st.cache_data
def load_data():
    df = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=100),
        'region': np.random.choice(['Seoul', 'Busan', 'Daegu', 'Incheon'], 100),
        'sales': np.random.randint(1000, 10000, 100)
    })
    return df

df = load_data()
df_filtered = df[df['region'].isin(selected_region)]

# 주요 지표
col1, col2, col3, col4 = st.columns(4)
col1.metric("총 판매액", f"${df_filtered['sales'].sum():,.0f}")
col2.metric("평균 판매액", f"${df_filtered['sales'].mean():,.0f}")
col3.metric("최대 판매액", f"${df_filtered['sales'].max():,.0f}")
col4.metric("최소 판매액", f"${df_filtered['sales'].min():,.0f}")

# 그래프
col1, col2 = st.columns(2)

with col1:
    st.subheader("일별 판매액")
    daily = df_filtered.groupby('date')['sales'].sum()
    fig = px.line(x=daily.index, y=daily.values)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("지역별 판매액")
    regional = df_filtered.groupby('region')['sales'].sum()
    fig = px.bar(x=regional.index, y=regional.values)
    st.plotly_chart(fig, use_container_width=True)

# 테이블
st.subheader("데이터 테이블")
st.dataframe(df_filtered)
```

### 4.2 Dash 대시보드

```python
from dash import Dash, dcc, html, Input, Output
import plotly.express as px

app = Dash(__name__)

# 데이터
df = px.data.gapminder()

# 레이아웃
app.layout = html.Div([
    html.H1("📊 국가별 GDP 대시보드"),
    
    html.Div([
        html.Label("연도 선택:"),
        dcc.Slider(
            id='year-slider',
            min=df['year'].min(),
            max=df['year'].max(),
            value=df['year'].max(),
            marks={str(year): str(year) for year in range(1952, 2008, 4)},
            step=None
        )
    ]),
    
    dcc.Graph(id='scatter-plot'),
    dcc.Graph(id='bar-chart')
])

@app.callback(
    [Output('scatter-plot', 'figure'),
     Output('bar-chart', 'figure')],
    [Input('year-slider', 'value')]
)
def update_graphs(selected_year):
    filtered_df = df[df['year'] == selected_year]
    
    # 산점도
    scatter = px.scatter(
        filtered_df,
        x='gdpPercap',
        y='lifeExp',
        size='pop',
        color='continent',
        hover_name='country',
        title=f'{selected_year}년 GDP vs 기대수명'
    )
    
    # 막대 그래프
    top_10 = filtered_df.nlargest(10, 'gdpPercap')
    bar = px.bar(
        top_10,
        x='country',
        y='gdpPercap',
        title=f'{selected_year}년 상위 10 국가 GDP'
    )
    
    return scatter, bar

if __name__ == '__main__':
    app.run_server(debug=True)
```

---

## 5. 지리 데이터 시각화 (10점)

### 5.1 Folium

```python
import folium

# 기본 지도
m = folium.Map(location=[37.5665, 126.9780], zoom_start=13)

# 마커 추가
folium.Marker(
    location=[37.5665, 126.9780],
    popup="서울시청",
    tooltip="클릭하세요"
).add_to(m)

# 원 추가
folium.Circle(
    location=[37.5665, 126.9780],
    radius=1000,
    color='red',
    fill=True
).add_to(m)

m.save('map.html')

# Choropleth (등치 지도)
import geopandas as gpd

geo_data = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

fig = px.choropleth(
    geo_data,
    locations='iso_a3',
    color='pop_est',
    hover_name='name',
    title='인구 분포'
)
fig.show()
```

---

## 6. 종합 프로젝트 (30점)

### 목표
완전한 대시보드 구축

구성 요소:
- 메인 페이지 (핵심 지표)
- 판매 분석 페이지
- 고객 분석 페이지
- 예측 페이지
- 설정 페이지

---

## 제출 방법

1. **제출 파일:**
   - week12_matplotlib_advanced.py
   - week12_seaborn_advanced.py
   - week12_plotly.py
   - week12_dashboard.py (Streamlit/Dash)
   - week12_geo_visualization.py
   - visualization_report.pdf

2. **제출 기한:** 다음 주 월요일 23:59
3. **제출 방식:** GitHub 레포지토리

---

## 평가 기준

| 항목 | 배점 |
|------|------|
| Matplotlib 심화 | 20점 |
| Seaborn | 20점 |
| Plotly | 20점 |
| 대시보드 | 20점 |
| 지리 시각화 | 10점 |
| 종합 프로젝트 | 10점 |
| **소계** | **100점** |
