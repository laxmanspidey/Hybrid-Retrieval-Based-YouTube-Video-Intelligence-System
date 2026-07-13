import os
from typing import List, Optional

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

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


class YouTubeRAG:
    def __init__(self, video_id: str):
        self.video_id = video_id
        self.collection_name = f"video_{video_id}"
        self.persist_directory = os.path.join(PERSIST_DIR, self.collection_name)
        self.embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        self.llm = OllamaLLM(model=LLM_MODEL, temperature=0.2)
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

    # ---------- Q&A ----------

    def ask(
        self,
        question: str,
        history: Optional[List[dict]] = None,
        target_language: str = "English",
        use_memory: bool = False,
    ) -> dict:
        """
        Returns:
            {
              "answer": str,
              "source_documents": [Document, ...],
              "timestamps": [float, ...]   # chunk start times in seconds
            }
        """
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

        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | PROMPT
            | self.llm
            | StrOutputParser()
        )

        answer = chain.invoke(question)
        source_docs = retriever.invoke(question)

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
