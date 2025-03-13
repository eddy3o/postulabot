from flask import Flask, render_template, request
from langchain_community.llms import Ollama
from openai import OpenAI
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate


# CARGANDO OPENAI KEY

load_dotenv()
# openai.api_key = os.environ["OPENAI_API_KEY"]
# print(openai.api_key)
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


app = Flask(__name__)

# Llama 3 Model
cached_llm = Ollama(model="llama3")


folder_path = "db"
chat_history = []

embedding = FastEmbedEmbeddings()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024, chunk_overlap=80, length_function=len, is_separator_regex=False
)

raw_prompt = PromptTemplate.from_template("""
<s> [INST] Tu eres un asistente llamado PostulaBot que ayuda emprededores a postular sus empreendimientos a fondos como corfo y otros, ademas ayuda con info de como postuar tu eres un desarrollo de Hello Future, una empresa de capacitacion fundanda por amanda Zerbinatti [/INST] </s>
[INST] {input}
Context: {context}
Answer:
[/INST] 
""")


@app.route('/')
def inicio():
    return render_template('index.html')


@app.route('/nosotros')
def Nosotros():
    return render_template('nosotros.html')


@app.route('/proyecto')
def Proyecto():
    return render_template('proyecto.html')


@app.route('/postulabot', methods=['GET', 'POST'])
def postulabot():
    if request.method == 'GET':
        return render_template('postulabot.html')
    if request.form['prompt_user']:
        # Get the message input from the the user
        prompt = 'Empreendedor:' + request.form['prompt_user']

        # Get the message input from the the user
        # user_input = request.form["question"]

        # Use the OpenAI API to generate a responde
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Eres un BOT que ayuda empreendedores a postular fondos y un excelente asistente."},
                {"role": "user", "content": prompt}
            ]
        )

        chat_response = completion.choices[0].message

        print(completion.choices[0].message)

        # Add the user input and bot response to the chat history
        chat_history.append(prompt)
        chat_history.append(chat_response)
        return render_template('postulabot.html', chat=chat_history)
    else:
        return render_template('postulabot.html')

    # Render the Bot template with the response text
    # return user_input
    # return render_template('www/index1.html')
    # else:
    # return render_template('www/index1.html')


@app.route("/ai", methods=["POST"])
def aiPost():
    print("Post /ai called")
    json_content = request.json
    query = json_content.get("query")

    print(f"query: {query}")

    response = cached_llm.invoke(query)

    print(response)

    response_answer = {"answer": response}
    return response_answer


@app.route('/upload', methods=["POST"])
def pdfPost():
    file = request.files["file"]
    file_name = file.filename
    save_file = "Upload/pdf/" + file_name
    file.save(save_file)
    print(f"filename: {file_name}")

    loader = PDFPlumberLoader(save_file)
    docs = loader.load_and_split()
    print(f"docs len={len(docs)}")

    chunks = text_splitter.split_documents(docs)
    print(f"chunks len={len(chunks)}")

    vector_store = Chroma.from_documents(
        documents=chunks, embedding=embedding, persist_directory=folder_path)

    vector_store.persist()

    response_upload = {
        "statud": "Upload Exitoso",
        "filename": file_name,
        "doc_len": len(docs),
        "chunks": len(chunks),
    }
    return response_upload


@app.route('/ask_pdf', methods=["POST"])
def ragPdfPost():
    json_content = request.json
    query = json_content.get("query")
    print(f"query: {query}")

    print("Loading vector store")
    vector_store = Chroma(persist_directory=folder_path,
                          embedding_function=embedding)

    print("Creating Chain")
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 20,
            "score_threshold": 0.1,
        },
    )
    document_chain = create_stuff_documents_chain(cached_llm, raw_prompt)
    chain = create_retrieval_chain(retriever, document_chain)
    result = chain.invoke({"input": query})
    print(result)

    sources = []
    for doc in result["context"]:
        sources.append(
            {"source": doc.metadata["source"],
             "page_content": doc.page_content}
        )

    response_answer = {"answer": result["answer"], "sources": sources}
    return response_answer


def start_app():
    app.run(port=8000, debug=True)


if __name__ == "__main__":
    start_app()