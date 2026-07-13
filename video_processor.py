import os
import torch
import yt_dlp
import whisper
from youtube_transcript_api import YouTubeTranscriptApi
from utils import (
    Segment,
    Transcript,
    extract_video_id,
    clean_transcript,
)

# Ensure the downloads folder exists
DOWNLOADS_DIR = "downloads"
os.makedirs(DOWNLOADS_DIR, exist_ok=True)


def _segments_from_youtube(fetched) -> list:
    """Convert youtube-transcript-api FetchedTranscript snippets to Segment list."""
    segments = []
    for snip in fetched:
        # Newer API exposes .text, .start, .duration; older uses dict access
        try:
            text = snip.text
            start = float(snip.start)
            end = float(snip.start) + float(snip.duration)
        except AttributeError:
            text = snip["text"]
            start = float(snip["start"])
            end = start + float(snip.get("duration", 0.0))
        segments.append(Segment(text=text, start=start, end=end))
    return segments


def get_transcript(url: str, lang: str = "en") -> Transcript:
    """
    Tries to get YouTube captions in `lang`.
    Falls back to downloading audio and transcribing with Whisper on GPU.
    Returns a Transcript with both flat text and timestamped segments.
    """
    video_id = extract_video_id(url)
    if not video_id:
        raise ValueError("Invalid YouTube URL")

    # --- Attempt 1: Get existing captions ---
    try:
        print(f"⏳ Fetching captions for {video_id} (lang={lang})...")
        ytt_api = YouTubeTranscriptApi()
        # Try requested language, then fall back to any English variant
        try:
            fetched = ytt_api.fetch(video_id, languages=[lang])
        except Exception:
            fetched = ytt_api.fetch(video_id, languages=["en", "en-US", "en-GB"])
        segments = _segments_from_youtube(fetched)
        if not segments:
            raise ValueError("Empty caption list")
        full_text = " ".join(s.text for s in segments)
        print(f"✅ Captions fetched successfully! ({len(segments)} segments)")
        return Transcript(
            text=clean_transcript(full_text),
            segments=segments,
            lang=lang,
        )
    except Exception as e:
        print(f"⚠️ No captions found ({e}). Transcribing audio with Whisper...")

    # --- Attempt 2: Download Audio and Transcribe ---
    audio_path = os.path.join(DOWNLOADS_DIR, f"{video_id}.mp3")
    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
        "outtmpl": os.path.join(DOWNLOADS_DIR, f"{video_id}"),
        "quiet": False,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as dl_err:
        raise Exception(f"Failed to download audio: {dl_err}")

    if not os.path.exists(audio_path):
        for file in os.listdir(DOWNLOADS_DIR):
            if file.startswith(video_id) and file.endswith(".mp3"):
                audio_path = os.path.join(DOWNLOADS_DIR, file)
                break

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🎤 Transcribing audio on {device.upper()}...")
    if device == "cpu":
        print("⚠️ GPU not detected — transcription will be slow.")

    model = whisper.load_model("small", device=device)
    result = model.transcribe(audio_path, fp16=(device == "cuda"))

    # Build segments from whisper's per-segment output
    segments = []
    for seg in result.get("segments", []):
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        segments.append(Segment(
            text=text,
            start=float(seg.get("start", 0.0)),
            end=float(seg.get("end", 0.0)),
        ))

    if os.path.exists(audio_path):
        try:
            os.remove(audio_path)
        except OSError:
            pass

    return Transcript(
        text=clean_transcript(result.get("text", "")),
        segments=segments,
        lang=lang,
    )
