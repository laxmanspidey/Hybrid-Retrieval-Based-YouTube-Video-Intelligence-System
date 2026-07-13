from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = OllamaLLM(model='qwen3.5:4b', temperature=0.2, num_ctx=16384, num_predict=768)

prompt_str = """You are an expert assistant. Use the provided context from the video transcript to answer the question. If the context does not contain the answer, say "I don't have that information in the video."

Conversation so far:
Q: which gun is talked about in this video
A: The gun talked about in this video is the VSS.

Context:
The VSS may not be the best weapon but it sure is fun to pick off unsuspecting victims while being purely stealth.

Question: can you tell me about it

Answer in English.

Accurate Answer:"""

prompt = PromptTemplate.from_template(prompt_str)
chain = prompt | llm | StrOutputParser()
print("Test OllamaLLM with history:")
print(repr(chain.invoke({})))
