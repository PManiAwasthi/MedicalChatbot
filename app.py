from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
from pinecone import Pinecone as pinecone
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

load_dotenv()

api_key = os.getenv("PINECONE_API_KEY")
env_key = os.getenv("PINECONE_API_ENV")
index_pinecone = os.getenv("PINECONE_INDEX")

embeddings = download_hugging_face_embeddings()

#initializing pinecone
pc=pinecone(api_key=api_key)

#loading the index
docsearch_1 = Pinecone.from_existing_index(index_pinecone, embeddings)

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs = {"prompt":PROMPT}

model_path = "C:\\Users\\ASUS\\.cache\\huggingface\\hub\\models--TheBloke--Llama-2-13B-chat-GGML\\snapshots\\3140827b4dfcb6b562cd87ee3d7f07109b014dd0\\llama-2-13b-chat.ggmlv3.q5_1.bin"

llm = CTransformers(model=model_path, model_type="llama", config={'max_new_tokens':512, 'temperature':0.8})

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch_1.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=['GET', 'POST'])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result=qa({"query":input})
    print("Response : ", result['result'])
    return str(result["result"])

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, debug=True)