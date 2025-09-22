import streamlit as st
from PIL import Image
import cv2
import backend

st.set_page_config(page_title="AI Weed Sprayer", layout="wide")
st.title("🌱 AI-Powered Weed Sprayer — Demo")

@st.cache_resource
def get_model():
    return backend.load_model_once()

get_model()

col1, col2 = st.columns([1,1])

with col1:
    st.header("Upload image")
    uploaded = st.file_uploader("Upload an image (jpg/png)", type=["jpg","jpeg","png"])
    sample = st.checkbox("Use sample image (if available)")

    if uploaded or sample:
        if uploaded:
            img_pil = Image.open(uploaded).convert("RGB")
        else:
            try:
                img_pil = Image.open("dataset/samples/sample1.jpg").convert("RGB")
            except Exception:
                st.error("No sample image found in dataset/samples/")
                img_pil = None

        if img_pil:
            st.image(img_pil, caption="Original image", use_container_width=True)
            threshold = st.slider("Spray decision threshold (probability)", 0.0, 1.0, 0.5, 0.05)

            if st.button("Run detection"):
                with st.spinner("Running model..."):
                    cv2_img = backend.pil_to_cv2(img_pil)
                    decision, prob, raw_pred = backend.predict_and_decide_from_cv2(cv2_img, threshold=threshold)
                    if decision:
                        processed_img, boxes, mask = backend.simulate_spray_by_green_detection(cv2_img)
                        annotated = backend.annotate_image(processed_img, decision, prob)
                    else:
                        annotated = backend.annotate_image(cv2_img, decision, prob)

                    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    st.image(
                        annotated_rgb,
                        caption=f"Result — {'SPRAY' if decision else 'DON\'T SPRAY'} ({prob*100:.1f}%)",
                        use_container_width=True,
                    )

                    is_success, buffer = cv2.imencode(".jpg", annotated)
                    if is_success:
                        st.download_button("Download result", buffer.tobytes(), file_name="result.jpg", mime="image/jpeg")

with col2:
    st.header("About")
    st.markdown("""
    This demo uses an AI model trained to detect weeds in crop images.
    Adjust the threshold and click **Run Detection** to view recommended spray regions.
    """)
    st.subheader("Quick run command")
    st.code("streamlit run app.py", language="bash")
