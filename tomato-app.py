import streamlit as st
import pandas as pd
import joblib
import os

# -----------------------------
# 페이지 설정
# -----------------------------
st.set_page_config(
    page_title="방울토마토 착과율 예측",
    page_icon="🍅",
    layout="centered"
)

# -----------------------------
# 제목
# -----------------------------
st.title("🍅 방울토마토 착과율 예측 시스템")
st.markdown("환경 데이터를 이용하여 착과율을 예측합니다.")
st.markdown("---")

# -----------------------------
# 모델 불러오기
# -----------------------------
@st.cache_resource
def load_model():

    model_path = "tomato_model.pkl"

    if not os.path.exists(model_path):
        return None

    try:
        model = joblib.load(model_path)
        return model

    except Exception as e:
        st.error(f"❌ 모델 로드 오류: {e}")
        return None


model = load_model()

# -----------------------------
# 사이드바 입력
# -----------------------------
st.sidebar.header("🌱 환경 데이터 입력")

temp = st.sidebar.slider(
    "내부온도 (℃)",
    min_value=0.0,
    max_value=50.0,
    value=25.0,
    step=0.1
)

hum = st.sidebar.slider(
    "내부습도 (%)",
    min_value=0.0,
    max_value=100.0,
    value=65.0,
    step=0.1
)

soil_temp = st.sidebar.slider(
    "지온 (℃)",
    min_value=0.0,
    max_value=50.0,
    value=22.0,
    step=0.1
)

flower = st.sidebar.slider(
    "개화군",
    min_value=1,
    max_value=10,
    value=5
)

# -----------------------------
# 현재 입력값 표시
# -----------------------------
col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

col1.metric("🌡 내부온도", f"{temp} ℃")
col2.metric("💧 내부습도", f"{hum} %")
col3.metric("🌍 지온", f"{soil_temp} ℃")
col4.metric("🌸 개화군", f"{flower} 군")

st.markdown("---")

# -----------------------------
# 설명
# -----------------------------
st.info("버튼을 눌러 예상 착과율을 확인하세요.")

# -----------------------------
# 예측 버튼
# -----------------------------
if st.button("🚀 착과율 예측", use_container_width=True):

    # 입력 데이터
    input_df = pd.DataFrame(
        [[temp, hum, soil_temp, flower]],
        columns=["내부온도", "내부습도", "지온", "개화군"]
    )

    # -------------------------
    # 실제 모델 예측
    # -------------------------
    if model is not None:

        try:
            pred = model.predict(input_df)[0]

            # 값 범위 제한
            pred = max(0, min(100, pred))

            st.success(f"📊 예상 착과율: {pred:.1f}%")

            # 추가 메시지
            if pred >= 80:
                st.balloons()
                st.write("✅ 매우 좋은 환경입니다!")

            elif pred >= 60:
                st.write("🙂 양호한 환경입니다.")

            else:
                st.warning("⚠️ 환경 조건 개선이 필요합니다.")

        except Exception as e:
            st.error(f"❌ 예측 오류: {e}")

    # -------------------------
    # 데모 모드
    # -------------------------
    else:

        st.warning("⚠️ 모델 파일이 없어 데모 모드로 실행됩니다.")

        demo_pred = (
            65
            + (temp - 25) * 0.8
            - abs(hum - 70) * 0.3
            + (soil_temp - 21) * 0.5
        )

        demo_pred = max(0, min(100, demo_pred))

        st.success(f"📊 데모 착과율: {demo_pred:.1f}%")

# -----------------------------
# 하단 정보
# -----------------------------
st.markdown("---")

st.caption("📌 tomato_model.pkl 파일을 같은 폴더에 넣어주세요.")
