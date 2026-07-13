from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOllama(model='qwen3.5:4b', temperature=0.2, num_ctx=16384, num_predict=768)

system_content = (
    "You are an expert assistant. Use the provided context from the video "
    "transcript to answer the question. If the context does not contain the "
    "answer, say \"I don't have that information in the video.\"\n\n"
    "Context:\n{context}"
)

messages = [
    ("system", system_content),
    ("human", "which gun is talked about in this video"),
    ("ai", "The gun talked about in this video is the VSS."),
    ("human", "{question}\n\nAnswer in English.")
]

prompt = ChatPromptTemplate.from_messages(messages)
chain = prompt | llm | StrOutputParser()

res = chain.invoke({
    "context": "The VSS may not be the best weapon but it sure is fun to pick off unsuspecting victims while being purely stealth.",
    "question": "can you tell me about it"
})
print("Test ChatPromptTemplate:")
print(repr(res))
