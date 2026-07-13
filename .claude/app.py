import numpy as np
import streamlit as st

from video_processor import get_transcript
from rag_engine import YouTubeRAG
from utils import extract_video_id, format_timestamp, youtube_url_at
from i18n import INPUT_LANGUAGES, OUTPUT_LANGUAGES
from voice import VoiceIO, wav_bytes_to_numpy

# -------------------- Page setup --------------------
st.set_page_config(page_title="🎥 YouTube Professor", layout="wide")
st.title("🎥 YouTube Professor")
st.caption("Chat with any YouTube video — local, multilingual, with timestamps and voice.")

# -------------------- Voice backend (lazy) --------------------
@st.cache_resource
def get_voice():
    return VoiceIO()


# Lazy: only instantiated when the user actually triggers TTS or STT.
# This avoids pulling in piper / faster-whisper on app startup, which on
# Windows can crash the Streamlit process if a backend is broken.
voice = None

# -------------------- Session state --------------------
def _init_state():
    st.session_state.setdefault("rag_engine", None)
    st.session_state.setdefault("video_id", None)
    st.session_state.setdefault("video_title", None)
    st.session_state.setdefault("transcript", "")
    st.session_state.setdefault("video_processed", False)
    st.session_state.setdefault("messages", [])          # [{role, content, timestamps, sources}]
    st.session_state.setdefault("audio_buffers", {})     # message_idx -> wav bytes
    st.session_state.setdefault("input_lang", "English")
    st.session_state.setdefault("output_lang", "English")
    st.session_state.setdefault("use_memory", False)
    st.session_state.setdefault("voice_question", "")


_init_state()

# -------------------- Sidebar --------------------
with st.sidebar:
    st.header("⚙️ Settings")
    st.session_state.input_lang = st.selectbox(
        "🌐 Transcript language",
        INPUT_LANGUAGES,
        index=INPUT_LANGUAGES.index(
            next(p for p in INPUT_LANGUAGES if p[0] == st.session_state.input_lang)
        ) if st.session_state.input_lang in [n for n, _ in INPUT_LANGUAGES] else 0,
        format_func=lambda p: p[0],
    )[0]
    st.session_state.output_lang = st.selectbox(
        "🗣️ Answer in",
        OUTPUT_LANGUAGES,
        index=OUTPUT_LANGUAGES.index(st.session_state.output_lang)
        if st.session_state.output_lang in OUTPUT_LANGUAGES else 0,
    )
    st.session_state.use_memory = st.checkbox(
        "🧠 Use past Q&A as context (cross-video memory)",
        value=st.session_state.use_memory,
        help="When on, the assistant can reference Q&A from previously processed videos.",
    )

    st.divider()
    st.header("📥 Step 1: Load Video")
    url = st.text_input("Paste YouTube URL here:", placeholder="https://youtube.com/watch?v=...")

    if st.button("🚀 Process Video", type="primary"):
        if not url:
            st.error("Please enter a YouTube URL.")
        else:
            video_id = extract_video_id(url)
            if not video_id:
                st.error("Invalid YouTube URL. Please check and try again.")
            else:
                with st.spinner("Fetching transcript..."):
                    try:
                        lang_code = next(
                            code for name, code in INPUT_LANGUAGES
                            if name == st.session_state.input_lang
                        )
                        transcript = get_transcript(url, lang=lang_code)
                        st.session_state.transcript = transcript.text
                        st.session_state.video_id = video_id

                        rag = YouTubeRAG(video_id)
                        rag.index_transcript(transcript)
                        st.session_state.rag_engine = rag
                        st.session_state.video_processed = True
                        st.session_state.messages = []  # reset chat on new video
                        st.session_state.audio_buffers = {}
                        st.success(
                            f"✅ Video processed! "
                            f"({len(transcript.text)} chars, "
                            f"{len(transcript.segments)} segments)"
                        )
                    except Exception as e:
                        st.error(f"❌ Error: {e}")

    if st.session_state.video_processed:
        st.divider()
        st.subheader("📊 Status")
        st.success(f"Indexed: `{st.session_state.video_id}`")
        if st.button("🗑️ Clear chat & reload"):
            st.session_state.rag_engine = None
            st.session_state.video_processed = False
            st.session_state.messages = []
            st.session_state.audio_buffers = {}
            st.rerun()

# -------------------- Main layout --------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("💬 Conversation")

    if not st.session_state.video_processed:
        st.info("👈 Paste a YouTube URL and click 'Process Video' to start.")
    else:
        # ----- chat history -----
        for idx, msg in enumerate(st.session_state.messages):
            role = msg["role"]
            avatar = "🧑" if role == "user" else "🧠"
            with st.chat_message(role, avatar=avatar):
                st.markdown(msg["content"])

                if role == "assistant":
                    timestamps = msg.get("timestamps", [])
                    # Dedupe while preserving order — the retriever can return
                    # the same chunk multiple times when the same top-k is
                    # fetched from two calls in ask().
                    seen_ts: set = set()
                    unique_ts: list = []
                    for t in timestamps:
                        key = round(float(t), 1)
                        if key not in seen_ts:
                            seen_ts.add(key)
                            unique_ts.append(float(t))
                    if unique_ts:
                        ts_links = []
                        for t in unique_ts[:4]:
                            label = format_timestamp(t)
                            url = youtube_url_at(st.session_state.video_id, t)
                            ts_links.append(f"[{label}]({url})")
                        st.caption("Key moments: " + " · ".join(ts_links))

                    if msg.get("sources"):
                        with st.expander("📚 Source chunks"):
                            for i, doc in enumerate(msg["sources"]):
                                st.caption(
                                    f"Chunk {i+1} · {format_timestamp(doc.metadata.get('start', 0))}"
                                )
                                st.text(doc.page_content[:300] + "...")

                    # TTS button per assistant message
                    audio_key = f"audio_{idx}"
                    if st.button(f"🔊 Read aloud ({st.session_state.output_lang})", key=audio_key):
                        if voice is None:
                            voice = get_voice()
                        try:
                            wav = voice.speak(msg["content"], st.session_state.output_lang)
                            if wav:
                                st.session_state.audio_buffers[idx] = wav
                        except Exception as e:
                            st.error(f"TTS error: {e}")
                    if idx in st.session_state.audio_buffers:
                        st.audio(st.session_state.audio_buffers[idx], format="audio/wav")

        # ----- input row -----
        st.divider()
        st.markdown("##### Ask a question")

        # Voice input is opt-in and gated behind a checkbox so the mic
        # component only mounts when the user actually wants it.
        voice_enabled = st.checkbox(
            "🎙️ Enable voice input (mic + STT)",
            value=False,
            help="Turn on the mic button to record a question and transcribe it with Whisper.",
        )

        if voice_enabled:
            st.caption("Click 🎤 Record, ask your question, then click ⏹️ Stop")
            mic_col, txt_col, btn_col = st.columns([1, 3, 1])
        else:
            mic_col, txt_col, btn_col = st.columns([0.01, 4, 1])

        with mic_col:
            if voice_enabled:
                if voice is None:
                    voice = get_voice()
                try:
                    from streamlit_mic_recorder import mic_recorder

                    # Uses the browser's native MediaRecorder API — no
                    # aiortc/PyAV signaling server involved, so it doesn't
                    # have the Windows-crash problem streamlit-webrtc did.
                    # just_once=True means this returns audio bytes exactly
                    # once, right after you hit stop, then None afterwards —
                    # so we don't re-transcribe the same clip on every rerun.
                    audio = mic_recorder(
                        start_prompt="🎤 Record",
                        stop_prompt="⏹️ Stop",
                        just_once=True,
                        use_container_width=True,
                        format="wav",
                        key="mic_recorder",
                    )

                    if audio and audio.get("bytes"):
                        # DIAGNOSTIC: play back exactly what was captured,
                        # before it ever reaches Whisper. If this sounds
                        # silent/garbled, the problem is in mic capture, not
                        # in transcription — cheapest way to isolate it.
                        st.audio(audio["bytes"], format="audio/wav")

                        with st.spinner("Transcribing..."):
                            try:
                                audio_np = wav_bytes_to_numpy(audio["bytes"])
                                # DIAGNOSTIC: peak/RMS amplitude tells us if
                                # real signal reached this point at all.
                                peak = float(np.max(np.abs(audio_np))) if len(audio_np) else 0.0
                                rms = float(np.sqrt(np.mean(audio_np ** 2))) if len(audio_np) else 0.0
                                print(f"🔊 Parsed audio: {len(audio_np)} samples, peak={peak:.4f}, rms={rms:.4f}")
                                st.caption(f"debug — peak: {peak:.3f}, rms: {rms:.4f}")

                                text = voice.transcribe(audio_np)
                                if text:
                                    st.session_state.voice_question = text
                                else:
                                    st.warning(
                                        "Didn't catch any speech in that clip — "
                                        "try recording again and speak for a couple seconds."
                                    )
                            except Exception as e:
                                st.error(f"STT error: {e}")
                except ImportError:
                    st.caption(
                        "⚠️ `streamlit-mic-recorder` not installed. "
                        "Run `pip install streamlit-mic-recorder`."
                    )

        with txt_col:
            # When the user just transcribed audio, seed the keyed text_input
            # before rendering. `value=` is ignored once a `key` is registered,
            # so we have to write to the session-state slot directly.
            if st.session_state.voice_question:
                st.session_state.question_input = st.session_state.voice_question
                st.session_state.voice_question = ""
            question = st.text_input(
                "Question",
                placeholder="e.g., What are the 3 main points?",
                label_visibility="collapsed",
                key="question_input",
            )

        with btn_col:
            ask_clicked = st.button("🔍 Ask", type="primary", use_container_width=True)

        if ask_clicked:
            if not question.strip():
                st.warning("Please type or speak a question.")
            else:
                with st.spinner(f"Thinking with {st.session_state.output_lang}..."):
                    try:
                        result = st.session_state.rag_engine.ask(
                            question,
                            history=st.session_state.messages,
                            target_language=st.session_state.output_lang,
                            use_memory=st.session_state.use_memory,
                        )
                        # Append user + assistant turns
                        st.session_state.messages.append({"role": "user", "content": question})
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": result["answer"],
                            "timestamps": result["timestamps"],
                            "sources": result["source_documents"],
                        })
                        # Remember the Q&A in the cross-video collection
                        st.session_state.rag_engine.remember_qa(question, result["answer"])
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error generating answer: {e}")

        # ----- transcript preview -----
        with st.expander("📄 Show transcript preview"):
            st.text(st.session_state.transcript[:1000] + "...")

with col2:
    st.subheader("⚙️ Your Setup")
    st.metric("🧠 LLM", "qwen3.5:4b (local)")
    st.metric("📦 Embeddings", "nomic-embed-text (local)")
    st.metric("💾 Vector DB", "ChromaDB")
    st.metric("🎤 STT", "faster-whisper (small)")
    st.metric("🗣️ TTS", "piper-tts (local)")
    st.divider()
    st.caption(
        "💡 **Tips**\n"
        "- Switch answer language any time — the next answer translates in place.\n"
        "- Click a timestamp under an answer to jump straight to that moment on YouTube.\n"
        "- Toggle 'Use past Q&A as context' to let the assistant reference earlier videos."
    )
