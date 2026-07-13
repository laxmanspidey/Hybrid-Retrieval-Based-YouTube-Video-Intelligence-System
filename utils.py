import re
import streamlit as st
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Segment:
    """A timestamped piece of transcript text."""
    text: str
    start: float  # seconds
    end: float    # seconds


@dataclass
class Transcript:
    """Full transcript with both flat text and timestamped segments."""
    text: str
    segments: List[Segment] = field(default_factory=list)
    lang: str = "en"


def extract_video_id(url):
    """Extracts the unique video ID from a YouTube URL."""
    patterns = [
        r'(?:youtube\.com\/watch\?v=)([\w-]+)',
        r'(?:youtu\.be\/)([\w-]+)',
        r'(?:youtube\.com\/embed\/)([\w-]+)',
        r'(?:youtube\.com\/shorts\/)([\w-]+)'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def clean_transcript(text):
    """Cleans up extra spaces and newlines from transcript text."""
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def format_timestamp(seconds: float) -> str:
    """Formats seconds as MM:SS or H:MM:SS for long videos."""
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def youtube_url_at(video_id: str, seconds: float) -> str:
    """Builds a YouTube deep-link that jumps to a given timestamp."""
    return f"https://youtu.be/{video_id}?t={int(seconds)}"


def chunk_transcript_with_timestamps(
    segments: List[Segment],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[dict]:
    """
    Walks segments and groups them into chunks of roughly `chunk_size` chars.
    Returns a list of dicts:
        {"text": str, "start": float, "end": float, "segments": [int, ...]}
    Overlap is achieved by re-including the trailing text of the previous chunk
    at the start of the next one (no re-use of segment indices to keep it simple).
    """
    if not segments:
        return []

    chunks: List[dict] = []
    current_text_parts: List[str] = []
    current_start: Optional[float] = None
    current_end: Optional[float] = None
    current_len = 0

    for seg in segments:
        seg_text = seg.text.strip()
        if not seg_text:
            continue

        if current_start is None:
            current_start = seg.start

        projected = current_len + len(seg_text) + (1 if current_text_parts else 0)
        if projected > chunk_size and current_text_parts:
            # Close current chunk
            chunks.append({
                "text": " ".join(current_text_parts).strip(),
                "start": current_start,
                "end": current_end or seg.start,
            })
            # Build overlap tail from the end of the just-closed chunk
            full = " ".join(current_text_parts)
            tail = full[-chunk_overlap:] if chunk_overlap > 0 else ""
            current_text_parts = [tail] if tail else []
            current_len = len(tail)
            current_start = seg.start  # overlap inherits the next segment's start

        current_text_parts.append(seg_text)
        current_end = seg.end
        current_len += len(seg_text) + 1

    # Flush remainder
    if current_text_parts:
        chunks.append({
            "text": " ".join(current_text_parts).strip(),
            "start": current_start or 0.0,
            "end": current_end or 0.0,
        })

    return [c for c in chunks if c["text"]]


@st.cache_data
def load_sample_text():
    """Returns a sample text for testing if no video is processed."""
    return "This is a sample transcript. The speaker discusses the basics of artificial intelligence and machine learning."
