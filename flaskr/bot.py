import os
from flask import (
    Blueprint, flash, g, redirect, render_template, request, url_for, session, jsonify
)
from werkzeug.exceptions import abort
from flaskr.auth import login_required
from flaskr.db import get_db
from langchain_community.llms import Ollama
from openai import OpenAI
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate

load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

cached_llm = Ollama(
    model="llama3",
    base_url="http://localhost:11434"  
)

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

bp = Blueprint('bot', __name__, url_prefix='/bot')

@bp.route('/bot', methods=['GET', 'POST'])
@login_required
def bot():
    db = get_db()
    user_id = g.user['id']
    user = db.execute('SELECT * FROM user WHERE id = ?', (user_id,)).fetchone()
        
    return render_template('bot/postulabot.html', user=user)


@bp.route('/chat', methods=['POST'])
@login_required
def chat():
    data = request.get_json()  
    if 'chat_history' not in session:
        session['chat_history'] = []

    prompt = data.get('inputMessage', '').strip() 
    
    if not prompt:
        return jsonify({'error': "Por favor, escribe un mensaje antes de enviar."}), 400

    session['chat_history'].append({'role': 'user', 'content': prompt})

    session['chat_history'] = session['chat_history'][-10:]

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Eres Postulabot, un BOT que ayuda a emprendedores a postular fondos y eres un excelente asistente."}
        ] + session['chat_history']
    )
    
    chat_response = completion.choices[0].message.content
    session['chat_history'].append({'role': 'assistant', 'content': chat_response})

    session.modified = True 

    return jsonify({'user_message': prompt, 'bot_response': chat_response})

@bp.route('/ask_pdf', methods=['POST'])
@login_required
def ragPdfPost():
    data = request.get_json()
    
    if 'chat_history' not in session:
        session['chat_history'] = []

    prompt = data.get('inputMessage', '').strip()
    
    if not prompt:
        return jsonify({'error': "Por favor, escribe un mensaje antes de enviar."}), 400

    session['chat_history'].append({'role': 'user', 'content': prompt})
    session['chat_history'] = session['chat_history'][-10:] 

    vector_store = Chroma(persist_directory=folder_path,
                          embedding_function=embedding)

    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 20,
            "score_threshold": 0.1,
        },
    )
    
    document_chain = create_stuff_documents_chain(cached_llm, raw_prompt)
    chain = create_retrieval_chain(retriever, document_chain)
    
    result = chain.invoke({
        "input": prompt,
        "chat_history": session['chat_history'][:-1]  
    })

    sources = []
    for doc in result["context"]:
        sources.append({
            "source": doc.metadata["source"],
            "page_content": doc.page_content
        })

    response_answer = {
        "answer": result["answer"],
        "sources": sources
    }
    
    session['chat_history'].append({'role': 'assistant', 'content': response_answer['answer']})
    session.modified = True

    return jsonify({
        'user_message': prompt,
        'bot_response': response_answer['answer'],
        'sources': response_answer['sources']
    })

@bp.route('/upload', methods=["POST"])
@login_required
def pdfPost():
    file = request.files["file"]
    file_name = file.filename
    save_file = "Upload/pdf/" + file_name
    file.save(save_file)

    loader = PDFPlumberLoader(save_file)
    docs = loader.load_and_split()

    chunks = text_splitter.split_documents(docs)

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