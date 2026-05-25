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
st.title("🍅 방울토마토 착과율 예측")
st.markdown("온실 환경 데이터를 입력하면 머신러닝 모델이 착과율을 예측합니다.")
st.markdown("---")

# -----------------------------
# 모델 로드
# -----------------------------
@st.cache_resource
def load_model():
    model_path = "tomato_model.pkl"
    if not os.path.exists(model_path):
        st.warning(f"⚠️ 모델 파일(`{model_path}`)을 찾을 수 없습니다. 데모 모드로 실행합니다.")
        return None
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"❌ 모델 로드 오류: {e}")
        return None

model = load_model()

# -----------------------------
# 사용자 입력 (슬라이더)
# -----------------------------
st.subheader("🌱 환경 데이터 입력")

temp = st.slider(
    "내부온도 (°C)",
    min_value=0.0, max_value=50.0, value=25.0, step=0.5,
    help="권장 범위: 20~30°C"
)

humidity = st.slider(
    "내부습도 (%)",
    min_value=0.0, max_value=100.0, value=65.0, step=1.0,
    help="권장 범위: 60~80%"
)

soil_temp = st.slider(
    "지온 (°C)",
    min_value=0.0, max_value=50.0, value=22.0, step=0.5,
    help="권장 범위: 18~25°C"
)

flower_group = st.slider(
    "개화군",
    min_value=1.0, max_value=10.0, value=5.0, step=1.0,
    help="1군(초기) ~ 10군(후기)"
)

st.markdown("---")

# -----------------------------
# 입력값 요약 표시
# -----------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("🌡️ 내부온도", f"{temp:.1f} °C")
col2.metric("💧 내부습도", f"{humidity:.0f} %")
col3.metric("🌍 지온", f"{soil_temp:.1f} °C")
col4.metric("🌸 개화군", f"{int(flower_group)} 군")

st.markdown("---")

# -----------------------------
# 예측 버튼 및 결과
# -----------------------------
if st.button("🚀 착과율 예측", use_container_width=True):
    # 입력 데이터를 DataFrame으로 변환 (컬럼명 중요!)
    input_data = pd.DataFrame(
        [[temp, humidity, soil_temp, flower_group]],
        columns=['내부온도', '내부습도', '지온', '개화군']
    )

    if model is not None:
        # 실제 모델 예측
        try:
            predicted = model.predict(input_data)[0]
            # 착과율은 0~100% 사이로 제한
            predicted = max(0, min(100, predicted))
            st.success(f"📊 **예측 착과율: {predicted:.1f}%**")

            # 추가 피드백
            if predicted >= 80:
                st.balloons()
                st.info("🎉 최적 환경에 가깝습니다! 현재 상태를 유지하세요.")
            elif predicted >= 60:
                st.info("🙂 양호한 환경입니다. 조금만 더 개선하면 좋습니다.")
            else:
                st.warning("⚠️ 환경 조건을 개선하면 착과율이 높아질 수 있습니다.")
        except Exception as e:
            st.error(f"예측 중 오류 발생: {e}")
    else:
        # 데모 모드 (모델 없을 때)
        st.warning("⚠️ 모델 파일이 없어 데모 예측값을 보여줍니다.")
        # 간단한 데모 공식 (실제 모델과 무관)
        demo_pred = 65 + (temp-25)*0.8 - abs(humidity-70)*0.3 + (soil_temp-21)*0.5
        demo_pred = max(0, min(100, demo_pred))
        st.success(f"📊 **데모 착과율: {demo_pred:.1f}%**")
        st.caption("📌 정확한 예측을 위해 `tomato_model.pkl` 파일을 같은 폴더에 넣어주세요.")

# -----------------------------
# 하단 정보
# -----------------------------
st.markdown("---")
st.caption("🤖 사용 모델: RandomForestRegressor | 입력 변수: 내부온도, 내부습도, 지온, 개화군")
