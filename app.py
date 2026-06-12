import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# --------------------
# 한글 폰트 설정
# --------------------
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# --------------------
# 페이지 설정
# --------------------
st.set_page_config(
    page_title="AI 중고차 가격 예측",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 AI 중고차 가격 예측 시스템")
st.write("Random Forest 기반 중고차 가격 예측")

# --------------------
# 파일 불러오기
# --------------------
with open("car_price_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

with open("feature_importance.pkl", "rb") as f:
    feature_importance = pickle.load(f)

with open("brand_list.pkl", "rb") as f:
    brand_list = pickle.load(f)

with open("fuel_list.pkl", "rb") as f:
    fuel_list = pickle.load(f)

with open("transmission_list.pkl", "rb") as f:
    transmission_list = pickle.load(f)

# --------------------
# 입력 UI
# --------------------
st.header("차량 정보 입력")

col1, col2 = st.columns(2)

with col1:
    brand = st.selectbox("브랜드", brand_list)
    fuel = st.selectbox("연료 종류", fuel_list)
    transmission = st.selectbox("변속기", transmission_list)

with col2:
    seats = st.number_input(
        "좌석 수",
        min_value=2,
        max_value=12,
        value=5
    )

# 차량 나이
st.subheader("차량 나이")

c1, c2 = st.columns(2)

with c1:
    age_slider = st.slider(
        "차량 나이 (년)",
        0,
        30,
        5
    )

with c2:
    car_age = st.number_input(
        "직접 입력",
        min_value=0,
        max_value=30,
        value=age_slider
    )

# 주행거리
st.subheader("주행거리")

c1, c2 = st.columns(2)

with c1:
    km_slider = st.slider(
        "주행거리 (km)",
        0,
        500000,
        50000
    )

with c2:
    km_driven = st.number_input(
        "직접 입력 (km)",
        min_value=0,
        max_value=500000,
        value=km_slider
    )

# 기타 정보
mileage = st.number_input(
    "연비 (km/L)",
    min_value=0.0,
    value=18.0
)

engine = st.number_input(
    "배기량 (cc)",
    min_value=500,
    value=1500
)

max_power = st.number_input(
    "최대 출력 (hp)",
    min_value=10.0,
    value=100.0
)

# --------------------
# 예측 버튼
# --------------------
if st.button("🚀 가격 예측하기"):

    input_df = pd.DataFrame(
        np.zeros((1, len(feature_columns))),
        columns=feature_columns
    )

    # 수치형 입력
    input_df["차량나이"] = car_age
    input_df["주행거리"] = km_driven
    input_df["연비"] = mileage
    input_df["배기량"] = engine
    input_df["최대출력"] = max_power
    input_df["좌석수"] = seats

    # 브랜드
    brand_col = f"브랜드_{brand}"
    if brand_col in input_df.columns:
        input_df[brand_col] = 1

    # 연료
    fuel_col = f"연료종류_{fuel}"
    if fuel_col in input_df.columns:
        input_df[fuel_col] = 1

    # 변속기
    trans_col = f"변속기_{transmission}"
    if trans_col in input_df.columns:
        input_df[trans_col] = 1

    # 예측
    predicted_price = model.predict(input_df)[0]

    # 환율
    INR_TO_KRW = 16
    INR_TO_USD = 0.012

    price_krw = predicted_price * INR_TO_KRW
    price_usd = predicted_price * INR_TO_USD

    st.header("💰 예상 중고차 가격")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "인도 루피 (INR)",
            f"₹ {predicted_price:,.0f}"
        )

    with c2:
        st.metric(
            "대한민국 원 (KRW)",
            f"₩ {price_krw:,.0f}"
        )

    with c3:
        st.metric(
            "미국 달러 (USD)",
            f"$ {price_usd:,.0f}"
        )

    # 가격 등급
    if predicted_price < 300000:
        grade = "🟢 저가 차량"
    elif predicted_price < 1000000:
        grade = "🟡 중가 차량"
    else:
        grade = "🔴 고가 차량"

    st.subheader(f"가격 등급 : {grade}")

    st.info(
        "※ 환율은 1 INR = 16 KRW, 0.012 USD 기준입니다."
    )

    # --------------------
    # 중요도 분석
    # --------------------
    st.header("📊 가격 예측 변수 중요도")

    importance_df = pd.DataFrame({
        "Feature": feature_columns,
        "Importance": feature_importance
    })

    brand_importance = importance_df[
        importance_df["Feature"].str.startswith("브랜드_")
    ]["Importance"].sum()

    fuel_importance = importance_df[
        importance_df["Feature"].str.startswith("연료종류_")
    ]["Importance"].sum()

    transmission_importance = importance_df[
        importance_df["Feature"].str.startswith("변속기_")
    ]["Importance"].sum()

    numeric_df = importance_df[
        ~importance_df["Feature"].str.startswith("브랜드_") &
        ~importance_df["Feature"].str.startswith("연료종류_") &
        ~importance_df["Feature"].str.startswith("변속기_")
    ]

    grouped_df = pd.concat([
        numeric_df,
        pd.DataFrame({
            "Feature": [
                "브랜드",
                "연료종류",
                "변속기"
            ],
            "Importance": [
                brand_importance,
                fuel_importance,
                transmission_importance
            ]
        })
    ])

    grouped_df["Importance (%)"] = (
        grouped_df["Importance"] /
        grouped_df["Importance"].sum()
    ) * 100

    grouped_df = grouped_df.sort_values(
        "Importance (%)",
        ascending=False
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.barh(
        grouped_df["Feature"],
        grouped_df["Importance (%)"]
    )

    ax.set_xlabel("중요도 (%)")
    ax.set_title("가격 예측 변수 중요도")
    ax.invert_yaxis()

    st.pyplot(fig)

    display_df = grouped_df.copy()
    display_df["Importance (%)"] = (
        display_df["Importance (%)"]
        .round(2)
    )

    st.dataframe(
        display_df[
            ["Feature", "Importance (%)"]
        ],
        use_container_width=True
    )
