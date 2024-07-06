from openai import OpenAI
import streamlit as st
import os
from operator import itemgetter
from typing import List, Tuple
from langchain_pinecone import PineconeVectorStore
from langchain.callbacks import StreamlitCallbackHandler
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    format_document,
)
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables import RunnableConfig

st.title("🌌文化元宇宙前沿研究成果")

if prompt := st.chat_input(placeholder="请提问"):
    embedding = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
    PINECONE_INDEX_NAME = st.secrets["PINECONE_INDEX_NAME"]
    LLAMA_API_KEY=st.secrets["LLAMA_API_KEY"]
    GROQ_API_KEY=st.secrets["GROQ_API_KEY"]
    PINECONE_API_KEY=st.secrets["PINECONE_API_KEY"]
    PINECONE_ENVIRONMENT=st.secrets["PINECONE_ENVIRONMENT"]
    llama = ChatGroq(model="llama3-8b-8192",temperature=0)
    vectorstore = PineconeVectorStore.from_existing_index(
    PINECONE_INDEX_NAME, embedding
    )
    retriever = vectorstore.as_retriever()


    # Prompt Templates for Answer Synthesis
    # Ask: Condense a chat history and follow-up question into a standalone question
    _template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""  # noqa: E501
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

    # Answer: RAG answer synthesis prompt-context and format, chat history and user question
    template = """你是一位在中国元宇宙领域有深厚专业知识的教授。你的目标是与元宇宙爱好者、学者以及业界和学术界的专业人士分享你的知识。你应该耐心、正式地用中文交流，总是使用案例研究来解释你的观点。仅根据以下上下文用中文回答问题:
    <context>
    {context}
    </context>"""
    ANSWER_PROMPT = ChatPromptTemplate.from_messages(
        [
            ("system", template),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{question}"),
        ]
    )

    # Chain for Conversational Retrieval
    DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")
    # Combine the documents by using document_separator
    def _combine_documents(
        docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
    ):
        doc_strings = [format_document(doc, document_prompt) for doc in docs]
        return document_separator.join(doc_strings)

    # Human Message+ AI Message[HumanMessage(content=''), AIMessage(content=""), ...]
    def _format_chat_history(chat_history: List[Tuple[str, str]]) -> List:
        buffer = []
        for human, ai in chat_history:
            buffer.append(HumanMessage(content=human))
            buffer.append(AIMessage(content=ai))
        return buffer

    # Define chat history and user input
    class ChatHistory(BaseModel):
        chat_history: List[Tuple[str, str]] = Field(..., extra={"widget": {"type": "chat"}})
        question: str
        
    _search_query = RunnableBranch(
        # If input includes chat_history, we condense it with the follow-up question
        (
            RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
                run_name="HasChatHistoryCheck"
            ),  # Condense follow-up question and chat into a standalone_question
            RunnablePassthrough.assign(
                chat_history=lambda x: _format_chat_history(x["chat_history"])
            )
            | CONDENSE_QUESTION_PROMPT
            | llama
            | StrOutputParser(),
        ),
        # Else, we have no chat history, so just pass through the question
        RunnableLambda(itemgetter("question")),
    )

    _inputs = RunnableParallel(
        {
            "question": lambda x: x["question"],
            "chat_history": lambda x: _format_chat_history(x["chat_history"]),
            "context": _search_query | retriever | _combine_documents,
        }
    ).with_types(input_type=ChatHistory)
    chain = _inputs | ANSWER_PROMPT | llama | StrOutputParser()


    with st.chat_message("assistant"):
        st.write(prompt)
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        cfg = RunnableConfig()
        cfg["callbacks"] = [st_cb]
        try:
            response = chain.invoke({"question": prompt, "chat_history": chat_history}, cfg)
        except NameError:
            response = chain.invoke({"question": prompt, "chat_history": []}, cfg)
        st.write(response)
