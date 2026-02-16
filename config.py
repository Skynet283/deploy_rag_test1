from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# llm=ChatGroq(
#     model_name="llama-3.1-8b-instant"
# )



llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    temperature=0.2,
    max_output_tokens=2048*2,  # increase when you want detail
)


#gemini-2.5-flash-lite



# system_prompt = """You are an assistant for question-answering tasks.
# Use only the following pieces of retrieved context to answer the question.
# If you don't know the answer, just say that you don't know.
# Use three sentences maximum and keep the answer concise
# you can explain in detail if the query says so.


# Context:
# {context}
# """

system_prompt = """You are an assistant for question-answering tasks.
Use only the retrieved context below to answer.

If the answer is not in the context, say: "I don't know based on the provided context."

Style rules:
- Default: give a concise answer (3–5 sentences max).
- If the user explicitly asks for detail (e.g., "explain", "in detail", "why", "how", "steps"),
  then give a detailed, structured answer using bullet points or numbered steps.

Context:
{context}
"""

llm_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{question}")
])
