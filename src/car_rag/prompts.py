"""Versioned prompt templates."""

from langchain_core.prompts import ChatPromptTemplate

RAG_SYSTEM_PROMPT = (
    "You are an intelligent in-car virtual assistant for an MG ZS vehicle. "
    "Explain car warning messages clearly, concisely, and calmly, as your response "
    "will be read aloud via text-to-speech software to the driver. "
    "Always base your response strictly on the provided context. If the warning "
    "is not in the context, politely inform the driver that you cannot find it.\n\n"
    "Context: {context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)


def build_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_template(RAG_SYSTEM_PROMPT)
