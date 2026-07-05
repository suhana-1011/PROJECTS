import streamlit as st
from deep_translator import GoogleTranslator
from deep_translator.constants import GOOGLE_LANGUAGES_TO_CODES
from textblob import TextBlob
from PIL import Image
import pytesseract
import logging
import queue
import numpy as np
import speech_recognition as sr
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av

logging.disable(logging.WARNING)

# ── Language setup ──
LANG_LIST = {k.title(): v for k, v in GOOGLE_LANGUAGES_TO_CODES.items()}
LANG_NAMES = sorted(LANG_LIST.keys())

# ── RTC Config (for online deployment) ──
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

# ── Page Config ──
st.set_page_config(
    page_title="🌍 Global Translator Pro",
    page_icon="🌍",
    layout="wide"
)

# ── Session State ──
for key in ["input_text", "translated_text", "chat_history", "voice_queue"]:
    if key not in st.session_state:
        if key == "chat_history":
            st.session_state[key] = []
        elif key == "voice_queue":
            st.session_state[key] = queue.Queue()
        else:
            st.session_state[key] = ""

# ── Title ──
st.title("🌍 Global Translator Pro")
st.caption("Translate text, voice, images into any language — works on phone anywhere!")
st.divider()

# ── TABS ──
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "✏️ Translate",
    "🔄 Multi-Language",
    "😊 Emotion Detection",
    "🖼️ Image to Text",
    "💬 Chat Translator"
])

# ══════════════════════════════════════
# TAB 1 — Basic Translate + Voice
# ══════════════════════════════════════
with tab1:
    st.subheader("✏️ Text & Voice Translator")

    input_method = st.radio(
        "Input method:",
        ["✏️ Type Text", "🎤 Voice Input"],
        horizontal=True
    )

    if input_method == "✏️ Type Text":
        st.session_state.input_text = st.text_area(
            "Enter text to translate:",
            height=150,
            placeholder="Type anything here in any language..."
        )

    else:
        st.info("🎤 Click START below — speak into your mic — click STOP when done!")

        audio_buffer = queue.Queue()

        def audio_callback(frame: av.AudioFrame) -> av.AudioFrame:
            sound = frame.to_ndarray()
            audio_buffer.put(sound)
            return frame

        webrtc_ctx = webrtc_streamer(
            key="voice-translator",
            mode=WebRtcMode.SENDONLY,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"audio": True, "video": False},
            audio_frame_callback=audio_callback,
        )

        if st.button("🔍 Convert Speech to Text", use_container_width=True):
            audio_chunks = []
            while not audio_buffer.empty():
                audio_chunks.append(audio_buffer.get())

            if audio_chunks:
                with st.spinner("Converting speech to text..."):
                    try:
                        audio_data = np.concatenate(audio_chunks, axis=1).flatten()
                        audio_int16 = (audio_data * 32768).astype(np.int16)
                        audio_bytes = audio_int16.tobytes()

                        recognizer = sr.Recognizer()
                        audio_source = sr.AudioData(audio_bytes, 48000, 2)
                        text = recognizer.recognize_google(audio_source)
                        st.session_state.input_text = text
                        st.success(f"✅ Captured: **{text}**")
                    except Exception as e:
                        st.error(f"❌ Could not convert: {e}")
            else:
                st.warning("⚠️ No audio captured! Please speak and try again.")

        if st.session_state.input_text:
            st.text_area("Captured Text:", value=st.session_state.input_text,
                         height=100, disabled=True)

    st.divider()
    target_lang = st.selectbox("Translate to:", LANG_NAMES, key="tab1_lang")

    if st.button("🌐 Translate", use_container_width=True, type="primary", key="tab1_btn"):
        if not st.session_state.input_text.strip():
            st.warning("⚠️ Please enter or speak some text first!")
        else:
            with st.spinner("Translating..."):
                try:
                    result = GoogleTranslator(
                        source="auto",
                        target=LANG_LIST[target_lang]
                    ).translate(st.session_state.input_text)
                    st.session_state.translated_text = result
                    st.subheader("✅ Translation")
                    st.success(result)
                    st.text_area("Copy from here:", value=result, height=100)
                except Exception as e:
                    st.error(f"❌ Error: {e}")

# ══════════════════════════════════════
# TAB 2 — Multi Language at Once
# ══════════════════════════════════════
with tab2:
    st.subheader("🔄 Translate into Multiple Languages at Once!")
    st.caption("Type once — get translation in many languages simultaneously!")

    multi_text = st.text_area(
        "Enter text:", height=150,
        placeholder="Type anything here...",
        key="multi_input"
    )

    selected_langs = st.multiselect(
        "Choose languages:",
        LANG_NAMES,
        default=["Tamil", "Hindi", "French", "Spanish", "Arabic"]
    )

    if st.button("🔄 Translate All", use_container_width=True,
                 type="primary", key="multi_btn"):
        if not multi_text.strip():
            st.warning("⚠️ Please enter some text!")
        elif not selected_langs:
            st.warning("⚠️ Please select at least one language!")
        else:
            st.divider()
            st.subheader("🌍 All Translations")
            cols = st.columns(2)
            for i, lang in enumerate(selected_langs):
                try:
                    result = GoogleTranslator(
                        source="auto",
                        target=LANG_LIST[lang]
                    ).translate(multi_text)
                    with cols[i % 2]:
                        st.markdown(f"**🌐 {lang}**")
                        st.success(result)
                except Exception as e:
                    with cols[i % 2]:
                        st.error(f"❌ {lang}: {e}")

# ══════════════════════════════════════
# TAB 3 — Emotion & Sentiment Detection
# ══════════════════════════════════════
with tab3:
    st.subheader("😊 Emotion & Sentiment Detection")
    st.caption("Detects the mood of your text BEFORE translating!")

    emotion_text = st.text_area(
        "Enter text to analyze:",
        height=150,
        placeholder="Type something...",
        key="emotion_input"
    )

    target_lang_emotion = st.selectbox(
        "Also translate to:", LANG_NAMES, key="emotion_lang"
    )

    if st.button("😊 Detect & Translate", use_container_width=True,
                 type="primary", key="emotion_btn"):
        if not emotion_text.strip():
            st.warning("⚠️ Please enter some text!")
        else:
            with st.spinner("Analyzing..."):
                blob = TextBlob(emotion_text)
                polarity = blob.sentiment.polarity
                subjectivity = blob.sentiment.subjectivity

                if polarity > 0.5:
                    emotion = "😄 Very Positive / Happy"
                elif polarity > 0:
                    emotion = "🙂 Slightly Positive"
                elif polarity == 0:
                    emotion = "😐 Neutral"
                elif polarity > -0.5:
                    emotion = "😟 Slightly Negative"
                else:
                    emotion = "😢 Very Negative / Sad"

                if subjectivity > 0.6:
                    subj_label = "💭 Very Opinionated"
                elif subjectivity > 0.3:
                    subj_label = "🤔 Somewhat Opinionated"
                else:
                    subj_label = "📰 Factual / Objective"

                st.divider()
                st.subheader("📊 Emotion Result")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("😊 Emotion", emotion)
                with col2:
                    st.metric("📈 Polarity", f"{polarity:.2f}")
                with col3:
                    st.metric("💭 Subjectivity", subj_label)

                st.markdown("**Sentiment Scale:** Negative ◀️ ─── ▶️ Positive")
                st.progress((polarity + 1) / 2)

                st.divider()
                try:
                    translated = GoogleTranslator(
                        source="auto",
                        target=LANG_LIST[target_lang_emotion]
                    ).translate(emotion_text)
                    st.subheader(f"🌐 Translated to {target_lang_emotion}")
                    st.success(translated)
                except Exception as e:
                    st.error(f"❌ Translation Error: {e}")

# ══════════════════════════════════════
# TAB 4 — Image to Text & Translate
# ══════════════════════════════════════
with tab4:
    st.subheader("🖼️ Image to Text & Translate")
    st.caption("Upload an image with text — we extract and translate it!")

    uploaded_image = st.file_uploader(
        "Upload Image (JPG, PNG):",
        type=["jpg", "jpeg", "png"]
    )

    target_lang_img = st.selectbox(
        "Translate extracted text to:", LANG_NAMES, key="img_lang"
    )

    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("🔍 Extract & Translate", use_container_width=True,
                     type="primary", key="img_btn"):
            with st.spinner("Extracting text from image..."):
                try:
                    extracted = pytesseract.image_to_string(image).strip()
                    if not extracted:
                        st.warning("⚠️ No text found! Try a clearer image.")
                    else:
                        st.subheader("📝 Extracted Text")
                        st.info(extracted)

                        translated = GoogleTranslator(
                            source="auto",
                            target=LANG_LIST[target_lang_img]
                        ).translate(extracted)

                        st.subheader(f"🌐 Translated to {target_lang_img}")
                        st.success(translated)
                except Exception as e:
                    st.error(f"❌ Error: {e}")

# ══════════════════════════════════════
# TAB 5 — Real Time Chat Translator
# ══════════════════════════════════════
with tab5:
    st.subheader("💬 Real Time Chat Translator")
    st.caption("Two people chat in different languages and understand each other!")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 👤 Person 1")
        p1_lang = st.selectbox("Person 1 Language:", LANG_NAMES,
                                index=LANG_NAMES.index("English"), key="p1_lang")
        p1_msg = st.text_input("Message:", placeholder="Type here...", key="p1_msg")
        p1_send = st.button("📤 Send", key="p1_send", use_container_width=True)

    with col2:
        st.markdown("### 👥 Person 2")
        p2_lang = st.selectbox("Person 2 Language:", LANG_NAMES,
                                index=LANG_NAMES.index("Tamil"), key="p2_lang")
        p2_msg = st.text_input("Message:", placeholder="Type here...", key="p2_msg")
        p2_send = st.button("📤 Send", key="p2_send", use_container_width=True)

    if p1_send and p1_msg.strip():
        try:
            translated = GoogleTranslator(
                source="auto",
                target=LANG_LIST[p2_lang]
            ).translate(p1_msg)
            st.session_state.chat_history.append({
                "sender": f"👤 Person 1 ({p1_lang})",
                "original": p1_msg,
                "translated": translated,
                "target_lang": p2_lang
            })
        except Exception as e:
            st.error(f"❌ Error: {e}")

    if p2_send and p2_msg.strip():
        try:
            translated = GoogleTranslator(
                source="auto",
                target=LANG_LIST[p1_lang]
            ).translate(p2_msg)
            st.session_state.chat_history.append({
                "sender": f"👥 Person 2 ({p2_lang})",
                "original": p2_msg,
                "translated": translated,
                "target_lang": p1_lang
            })
        except Exception as e:
            st.error(f"❌ Error: {e}")

    if st.session_state.chat_history:
        st.divider()
        st.subheader("💬 Chat History")
        for chat in reversed(st.session_state.chat_history):
            st.markdown(f"**{chat['sender']}:** {chat['original']}")
            st.success(f"🌐 → {chat['target_lang']}: {chat['translated']}")
            st.markdown("---")

        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

# ── Footer ──
st.divider()
st.markdown(
    "<center>🌍 Global Translator Pro — Made with ❤️ using Python & Streamlit</center>",
    unsafe_allow_html=True
)

