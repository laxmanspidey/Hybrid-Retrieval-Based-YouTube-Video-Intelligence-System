from langchain_ollama import OllamaLLM

llm = OllamaLLM(model='qwen3.5:4b', temperature=0.2, num_ctx=16384, num_predict=768)

prompt = """You are an expert assistant. Use the provided context from the video transcript to answer the question. If the context does not contain the answer, say "I don't have that information in the video."

Context:
The VSS may not be the best weapon but it sure is fun to pick off unsuspecting victims while being purely stealth.

Question: which weapon is pure stealth

Answer in English.

Accurate Answer:"""

print(repr(llm.invoke(prompt)))
