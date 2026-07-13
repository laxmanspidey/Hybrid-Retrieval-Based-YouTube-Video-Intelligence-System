import os
import re
import hashlib
import json
from typing import List, Optional, Callable

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from utils import chunk_transcript_with_timestamps
from i18n import lang_instruction

# ---- CONFIGURATION ----
LLM_MODEL = "qwen3.5:4b"
EMBEDDING_MODEL = "nomic-embed-text"
PERSIST_DIR = "chroma_db"

# Global collection used for cross-video Q&A memory.
MEMORY_COLLECTION = "user_memory"
MEMORY_DIR = os.path.join(PERSIST_DIR, MEMORY_COLLECTION)

# How many prior turns to fold into the prompt as "conversation so far".
HISTORY_TURNS = 3
# How many cross-video memories to include when the user opts in.
MEMORY_K = 2
# How many source chunks the retriever returns.
RETRIEVER_K = 4

# Requests matching this get routed to full-video map-reduce summarization
# instead of top-k similarity retrieval (see _is_summary_query below).
_SUMMARY_INTENT_RE = re.compile(
    r"\b(summar(y|ize|ise|is)|overview|recap|gist|"
    r"(main|key)\s+points?|highlights?|tl;?dr)\b",
    re.IGNORECASE,
)

# Target chars per map-reduce group when summarizing the whole video. Kept
# modest since local Ollama models commonly run with a small default context
# window (Ollama defaults to num_ctx=2048 tokens unless explicitly raised),
# so each group + prompt overhead needs to comfortably fit regardless of
# how that's configured.
# Hybrid summarization:
# - short/medium transcripts: one LLM call over the full transcript
# - large transcripts: cached map-reduce with larger groups
SUMMARY_SINGLE_PASS_CHARS = 24000
SUMMARY_GROUP_CHARS = 10000
SUMMARY_CACHE_DIR = os.path.join(PERSIST_DIR, "summary_cache")
os.makedirs(SUMMARY_CACHE_DIR, exist_ok=True)


def _is_summary_query(question: str) -> bool:
    return bool(_SUMMARY_INTENT_RE.search(question))


def _strip_think(raw_text: str) -> Optional[str]:
    """
    qwen3.5:4b (and other reasoning models) prefix responses with a
    <think>...</think> block before the actual answer. Strip that out so
    only the real answer is stored/shown.

    Returns the cleaned answer text, or None if the response was truncated
    mid-reasoning (a <think> that never closes) — i.e. num_predict ran out
    before the model ever got to the real answer, so there's nothing usable
    to show. Callers should treat None as "retry with a bigger budget",
    not as an empty-but-valid answer.
    """
    text = raw_text.replace("<|im_start|>", "").replace("<|im_end|>", "")
    if "<think>" in text:
        if "</think>" not in text:
            return None
        text = text.split("</think>")[-1]
    text = text.strip()
    if text.startswith("assistant\n"):
        text = text[len("assistant\n"):]
    elif text.startswith("assistant"):
        text = text[len("assistant"):]
    return text.strip()


class YouTubeRAG:
    def __init__(self, video_id: str):
        self.video_id = video_id
        self.collection_name = f"video_{video_id}"
        self.persist_directory = os.path.join(PERSIST_DIR, self.collection_name)
        self.embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        # NOTE: qwen3.5:4b is a reasoning model — it spends part of its
        # num_predict budget on a <think>...</think> block before the real
        # answer. Too small a budget means it can burn the whole thing on
        # reasoning and never reach (or close) the actual answer, silently
        # producing nothing usable. Kept generous here; _run_llm below also
        # retries once with an even bigger budget if that still happens.
        self.llm = OllamaLLM(
            model=LLM_MODEL,
            temperature=0.2,
            num_ctx=16384,
            num_predict=2048,
        )
        self.llm_retry = OllamaLLM(
            model=LLM_MODEL,
            temperature=0.2,
            num_ctx=16384,
            num_predict=4096,
        )
        # Map calls should be short and fast; final answers use self.llm.
        self.map_llm = OllamaLLM(
            model=LLM_MODEL,
            temperature=0.1,
            num_ctx=8192,
            num_predict=500,
        )
        self.vectordb: Optional[Chroma] = None
        self.lang = "en"

    # ---------- transcript indexing ----------

    def index_transcript(self, transcript) -> None:
        """
        `transcript` is a utils.Transcript (with .text and .segments).
        We build timestamped chunks and store them as Documents with metadata.
        """
        segments = transcript.segments or []
        self.lang = getattr(transcript, "lang", "en") or "en"

        if segments:
            chunk_dicts = chunk_transcript_with_timestamps(
                segments, chunk_size=1000, chunk_overlap=200
            )
        else:
            # Fallback: text-only — synthesize one fake chunk
            chunk_dicts = [{"text": transcript.text, "start": 0.0, "end": 0.0}]

        if not chunk_dicts:
            raise ValueError("Transcript is empty or too short to process.")

        print(f"🧠 Indexing {len(chunk_dicts)} chunks into ChromaDB...")

        # Clear out any previously-indexed chunks for this video first. Without
        # this, re-processing the same video (e.g. picking a different
        # transcript language, or just hitting "process" again) appends a
        # fresh set of embeddings on top of the old ones in the same
        # collection, so retrieval starts mixing stale and current chunks.
        try:
            existing = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name=self.collection_name,
            )
            existing.delete_collection()
        except Exception as e:
            # No existing collection to clear (first time processing this
            # video) — nothing to do.
            print(f"ℹ️ No existing collection to clear ({e}).")

        docs = [
            Document(
                page_content=c["text"],
                metadata={
                    "start": c["start"],
                    "end": c["end"],
                    "video_id": self.video_id,
                    "lang": self.lang,
                },
            )
            for c in chunk_dicts
        ]

        self.vectordb = Chroma.from_documents(
            documents=docs,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
            collection_name=self.collection_name,
        )
        print("✅ Indexing complete!")

    def _load_vectordb_if_needed(self) -> Chroma:
        if self.vectordb is None:
            self.vectordb = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name=self.collection_name,
            )
        return self.vectordb

    # ---------- cross-video memory ----------

    def _memory_db(self) -> Chroma:
        """Lazy-loaded global Q&A memory collection."""
        os.makedirs(MEMORY_DIR, exist_ok=True)
        return Chroma(
            persist_directory=MEMORY_DIR,
            embedding_function=self.embeddings,
            collection_name=MEMORY_COLLECTION,
        )

    def remember_qa(self, question: str, answer: str) -> None:
        """Persist a Q&A pair to the cross-video memory collection."""
        if not question.strip() or not answer.strip():
            return
        try:
            db = self._memory_db()
            db.add_texts(
                texts=[f"Question: {question}\nAnswer: {answer}"],
                metadatas=[{
                    "video_id": self.video_id,
                    "ts": 0.0,
                }],
            )
        except Exception as e:
            print(f"⚠️ Failed to store memory: {e}")

    def fetch_memories(self, question: str) -> List[Document]:
        """Top-K most relevant past Q&As. Empty list on any error."""
        try:
            db = self._memory_db()
            retriever = db.as_retriever(search_kwargs={"k": MEMORY_K})
            return retriever.invoke(question)
        except Exception:
            return []

    # ---------- full-video summarization (map-reduce) ----------

    def _all_chunks_chronological(self) -> List[Document]:
        """Every indexed chunk for this video, sorted by start time — used
        for summarization instead of similarity search, since a summary
        request has no specific content to match against."""
        vectordb = self._load_vectordb_if_needed()
        raw = vectordb.get(include=["documents", "metadatas"])
        docs = [
            Document(page_content=text, metadata=meta or {})
            for text, meta in zip(raw["documents"], raw["metadatas"])
        ]
        docs.sort(key=lambda d: d.metadata.get("start", 0.0))
        return docs

    def _group_chunks(self, docs: List[Document], target_chars: int) -> List[Document]:
        """Merge consecutive chronological chunks into larger groups of
        roughly `target_chars` each, to cut down the number of map-step LLM
        calls needed to cover the whole video."""
        groups: List[Document] = []
        buf_text: List[str] = []
        buf_start: Optional[float] = None
        buf_len = 0

        for doc in docs:
            if buf_start is None:
                buf_start = doc.metadata.get("start", 0.0)
            buf_text.append(doc.page_content)
            buf_len += len(doc.page_content)
            if buf_len >= target_chars:
                groups.append(Document(page_content=" ".join(buf_text), metadata={"start": buf_start}))
                buf_text, buf_start, buf_len = [], None, 0

        if buf_text:
            groups.append(Document(page_content=" ".join(buf_text), metadata={"start": buf_start or 0.0}))

        return groups

    def _summary_cache_path(self, question: str, target_language: str, content_hash: str) -> str:
        # Including a hash of the transcript content itself (not just
        # video_id) means that re-indexing this video with different/new
        # content invalidates any previously cached summary automatically,
        # instead of silently serving a stale one that no longer matches
        # what's actually in the vector store.
        key = hashlib.sha256(
            f"{self.video_id}|{question.strip().lower()}|{target_language}|{content_hash}|v3".encode("utf-8")
        ).hexdigest()
        return os.path.join(SUMMARY_CACHE_DIR, f"{key}.json")

    def _summarize_full_video(
        self,
        question: str,
        target_language: str,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> dict:
        """
        Hybrid summarization:
        1. Use one LLM call when the whole transcript is small enough.
        2. Otherwise use cached map-reduce with large chronological groups.
        """
        all_docs = self._all_chunks_chronological()
        if not all_docs:
            return {
                "answer": "I don't have that information in the video.",
                "source_documents": [],
                "timestamps": [],
            }

        full_text = "\n\n".join(doc.page_content for doc in all_docs)
        lang_instr = lang_instruction(target_language)

        # Fast path: one call for short/medium transcripts.
        if len(full_text) <= SUMMARY_SINGLE_PASS_CHARS:
            if progress_callback:
                progress_callback(1, 1, "Summarizing the full video in one pass...")
            prompt = PromptTemplate(
                template=(
                    "You are given the complete transcript of a video in chronological order. "
                    "Use the ENTIRE transcript to fulfill the user's request. Be accurate, concise, "
                    "and do not add facts that are not present in the transcript.\n\n"
                    "Complete transcript:\n{text}\n\n"
                    "User's request: {question}\n\n"
                    f"{lang_instr}\n\nAnswer:"
                ),
                input_variables=["text", "question"],
            )
            answer = self._run_llm(prompt, {"text": full_text, "question": question})
            sample_docs = all_docs[::max(1, len(all_docs) // 4)][:4]
            return {
                "answer": answer,
                "source_documents": sample_docs,
                "timestamps": [d.metadata.get("start", 0.0) for d in sample_docs],
            }

        # Slow path: large videos only.
        groups = self._group_chunks(all_docs, SUMMARY_GROUP_CHARS)
        content_hash = hashlib.sha256(full_text.encode("utf-8")).hexdigest()[:16]
        cache_path = self._summary_cache_path(question, target_language, content_hash)

        section_summaries = []
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                section_summaries = cached.get("section_summaries", [])
                if len(section_summaries) != len(groups):
                    section_summaries = []
            except Exception:
                section_summaries = []

        map_prompt = PromptTemplate(
            template=(
                "Summarize this chronological video transcript section in 3-5 concise bullet points. "
                "Preserve important facts, arguments, examples, and conclusions. "
                "Use only the supplied text.\n\nSection:\n{text}\n\nSummary:"
            ),
            input_variables=["text"],
        )
        map_chain = map_prompt | self.map_llm | StrOutputParser()

        if not section_summaries:
            total = len(groups)
            for i, group in enumerate(groups, start=1):
                if progress_callback:
                    progress_callback(i, total + 1, f"Summarizing section {i} of {total}...")
                try:
                    raw_summary = map_chain.invoke({"text": group.page_content})
                    summary = _strip_think(raw_summary) or ""
                except Exception as e:
                    print(f"⚠️ Map step failed for section {i}: {e}")
                    summary = ""
                section_summaries.append(summary)

            try:
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {"section_summaries": section_summaries},
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )
            except Exception as e:
                print(f"⚠️ Could not save summary cache: {e}")

        if progress_callback:
            progress_callback(len(groups) + 1, len(groups) + 1, "Creating the final summary...")

        combined = "\n\n".join(
            f"[Section {i + 1}] {summary}"
            for i, summary in enumerate(section_summaries)
            if summary
        )

        reduce_prompt = PromptTemplate(
            template=(
                "The following chronological section summaries cover the entire video. "
                "Use ALL sections together to fulfill the user's request. Remove repetition, "
                "preserve the video's overall flow and important details, and do not invent facts.\n\n"
                "Section summaries:\n{combined}\n\n"
                "User's request: {question}\n\n"
                f"{lang_instr}\n\nAnswer:"
            ),
            input_variables=["combined", "question"],
        )
        answer = self._run_llm(reduce_prompt, {"combined": combined, "question": question})

        return {
            "answer": answer,
            "source_documents": groups,
            "timestamps": [g.metadata.get("start", 0.0) for g in groups],
        }

    def _run_llm(self, prompt: PromptTemplate, inputs: dict) -> str:
        """
        Runs `prompt` through self.llm, strips any <think> reasoning block,
        and retries once with self.llm_retry (bigger num_predict) if the
        first attempt got truncated mid-reasoning and produced nothing
        usable. This is what stands between "the model ran out of tokens
        while thinking" and the chat showing an empty answer bubble.
        """
        chain = prompt | self.llm | StrOutputParser()
        raw = chain.invoke(inputs)
        cleaned = _strip_think(raw)

        if not cleaned:
            print("⚠️ Answer was empty/truncated mid-reasoning — retrying with a larger token budget...")
            retry_chain = prompt | self.llm_retry | StrOutputParser()
            raw_retry = retry_chain.invoke(inputs)
            cleaned = _strip_think(raw_retry)

        if not cleaned:
            return (
                "I generated a response but it got cut off before finishing. "
                "Try asking again, or ask a more specific/shorter question."
            )
        return cleaned

    # ---------- Q&A ----------

    def ask(
        self,
        question: str,
        history: Optional[List[dict]] = None,
        target_language: str = "English",
        use_memory: bool = False,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> dict:
        """
        Returns:
            {
              "answer": str,
              "source_documents": [Document, ...],
              "timestamps": [float, ...]   # chunk start times in seconds
            }
        """
        if _is_summary_query(question):
            try:
                return self._summarize_full_video(question, target_language, progress_callback)
            except Exception as e:
                print(f"⚠️ Full-video summarization failed ({e}); falling back to standard retrieval.")
                # fall through to the normal retrieval-based path below

        vectordb = self._load_vectordb_if_needed()
        retriever = vectordb.as_retriever(search_kwargs={"k": RETRIEVER_K})

        # Build the contextual prompt
        memory_block = ""
        if use_memory:
            mems = self.fetch_memories(question)
            if mems:
                memory_block = "Relevant past Q&A from other videos:\n" + "\n\n".join(
                    d.page_content for d in mems
                ) + "\n\n"

        history_block = ""
        if history:
            recent = history[-HISTORY_TURNS * 2:]  # last N turns (user+assistant)
            lines = []
            for turn in recent:
                role = turn.get("role", "user")
                content = turn.get("content", "")
                prefix = "Q" if role == "user" else "A"
                lines.append(f"{prefix}: {content}")
            if lines:
                history_block = "Conversation so far:\n" + "\n".join(lines) + "\n\n"

        lang_instr = lang_instruction(target_language)

        prompt_template = (
            "You are an expert assistant. Use the provided context from the video "
            "transcript to answer the question. If the context does not contain the "
            "answer, say \"I don't have that information in the video.\"\n\n"
            "{memory_block}"
            "{history_block}"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            f"{lang_instr}\n\n"
            "Accurate Answer:"
        )
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"],
            partial_variables={
                "memory_block": memory_block,
                "history_block": history_block,
            },
        )

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # Retrieve once and reuse the same documents for both the answer
        # chain and the returned source_documents/timestamps, instead of
        # invoking the retriever a second time afterwards.
        source_docs = retriever.invoke(question)
        context = format_docs(source_docs)

        answer = self._run_llm(PROMPT, {"context": context, "question": question})

        timestamps = []
        for doc in source_docs:
            start = doc.metadata.get("start")
            if start is not None:
                timestamps.append(float(start))

        return {
            "answer": answer,
            "source_documents": source_docs,
            "timestamps": timestamps,
        }
