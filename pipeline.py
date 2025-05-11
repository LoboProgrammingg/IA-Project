from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

documents = [
    Document(page_content="The capital of France is Paris."),
    Document(page_content="The Eiffel Tower is one of the most famous landmarks in Paris."),
    Document(page_content="France is located in Western Europe and is known for its wine and cuisine."),
]

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)

llm = ChatOpenAI(model="gpt-4", temperature=0.2)

retriever = vectorstore.as_retriever()

prompt_template = PromptTemplate(
    input_variables=["question", "context"],
    template="Question: {question}\nContext: {context}\nAnswer:"
)

qa_chain = RetrievalQA.from_chain_type(
    retriever=retriever,
    chain_type="stuff",
    llm=llm,
    chain_type_kwargs={"prompt": prompt_template},
)

def query_rag(question: str):
    """
    Faz uma consulta ao pipeline RAG.

    Args:
        question (str): Pergunta realizada pelo usuário.

    Returns:
        str: Resposta gerada pelo pipeline RAG.
    """
    return qa_chain.run(question)

if __name__ == "__main__":
    user_query = "What is the capital of France?"
    response = query_rag(user_query)
    print("Generated response:", response)