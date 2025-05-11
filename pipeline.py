from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import Document  # Importa a classe Document
from dotenv import load_dotenv, find_dotenv

# Carrega variáveis de ambiente
_ = load_dotenv(find_dotenv())

# Converte os documentos para objetos Document
documents = [
    Document(page_content="The capital of France is Paris."),
    Document(page_content="The Eiffel Tower is one of the most famous landmarks in Paris."),
    Document(page_content="France is located in Western Europe and is known for its wine and cuisine."),
]

# Cria embeddings para os documentos
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)

# Define o modelo de linguagem para geração de respostas
llm = ChatOpenAI(model="gpt-4", temperature=0.2)

# Cria o retriever baseado no FAISS
retriever = vectorstore.as_retriever()

# Define o template do prompt
prompt_template = PromptTemplate(
    input_variables=["question", "context"],
    template="Question: {question}\nContext: {context}\nAnswer:"
)

# Configura o pipeline RAG (Retrieval + Generation)
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

# Exemplo de uso
if __name__ == "__main__":
    user_query = "What is the capital of France?"
    response = query_rag(user_query)
    print("Generated response:", response)