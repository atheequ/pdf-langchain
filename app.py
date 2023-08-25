import os
import json
from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    redirect,flash,url_for
    
)
from werkzeug.utils import secure_filename
#from werkzeug.exceptions import HTTPException
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter #,NLTKTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings #OpenAIEmbeddings,s
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI,HuggingFaceHub
from langchain.chains import RetrievalQA
import openai
#from dotenv import load_dotenv

#load_dotenv()


app = Flask(__name__)
app.secret_key = "Genpact Task"
#os.environ['REQUESTS_CA_BUNDLE'] = 'C:/Users/703099374/OneDrive - Genpact/python_code/new_test/exkey_access.crt'
#os.environ['CURL_CA_BUNDLE'] ='huggingface.crt'

def get_pdf_text(pdf_doc):
    text=""
    # for pdf in pdf_doc:
    pdf_reader=PdfReader(pdf_doc)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text
def get_pdf_chunks(raw_text):
    text_splitter=CharacterTextSplitter(
        separator=".",
        chunk_overlap=10,
        chunk_size=2000,
        length_function=len
    )
    chunks=text_splitter.split_text(raw_text)
    return chunks

def load_locally(indee):
    index=os.path.basename(indee).split(".")[0]
    if(os.path.exists(f"faiss/{index}")):
        #print(f"faiss/{index}")
        embeddings=HuggingFaceInstructEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2") 
        vectordb = FAISS.load_local(f"faiss/{index}", embeddings)
    else:
        vectordb=""
    return vectordb
@app.route("/get_similar/<query>",methods=["GET","POST"])
def similar_search(query):
    qq=query.split(" filename=")
    if len(qq)==2:
        vectordb=load_locally(qq[1])
        if vectordb !="":
            df=vectordb.similarity_search(qq[0].strip())
            dd=[]
            for i in df:
                #dd +=i.page_content
                dd.append(i.page_content)
            #print(dd)
            return get_vectorstore(dd)
        else:
            return "Embedding is not found"
    else:
        return "file name is not there"
def get_vectorstore(text_chunks):
    #embeddings=OpenAIEmbeddings()
    embeddings=HuggingFaceInstructEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")#hkunlp/instructor-large
    vectorstore=FAISS.from_texts(texts=text_chunks,embedding=embeddings)
    ##vectorstore=Pinecone.from_texts(text_chunks,embedding=embeddings,index_name="genpacta")
    return vectorstore #(model_name="sentence-transformers/all-MiniLM-L6-v2")#

def conn_open(names):
    doc=get_pdf_text(names)
    text_c=get_pdf_chunks(doc)
    vectordb=get_vectorstore(text_c)
    name=os.path.basename(names).split(".")[0]
    # if(os.path.exists("faiss_index")):
    #     db = FAISS.load_local("faiss_index", query_embedding)
    vectordb.save_local(f"faiss/{name}")
    return name
@app.get("/openai_chat/<question>")
def openai_chat(question):
    content=run_query(question)
    return jsonify(answer=content)

def run_query(question):
    
    vectordb=similar_search(question)
    #if(len(vectordb))
    #print(type(vectordb))
    if(type(vectordb)!=str):
        #print(vectordb)
        #return "Currently its disabled.."
        try:
            #embeddings=HuggingFaceInstructEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2") 
            #vectordb = FAISS.load_local("faiss_index", embeddings)
            #print(vectordb)
            llm=OpenAI(model_name='text-davinci-003')#google/flan-t5-large
            #llm=HuggingFaceHub(repo_id="google/flan-t5-large", model_kwargs={"temperature":0.6, "max_length":1024})
            chain=RetrievalQA.from_chain_type(
                #llm=AzureOpenAI(model_kwargs={'engine':"gpt-35-turbo-16k","api-version":"2022-12-01","":""}),#'engine':"gpt-35-turbo-16k",
                llm=llm,
                retriever=vectordb.as_retriever(),
                chain_type="stuff"
            )
            # print(question)
            return chain.run(question)
        except Exception as ee:
            return str(ee)
    else:
        return vectordb
     
    
@app.route("/azure",methods=['POST','GET'])
def az_view(): #Azure OpenAI
    question=request.args.get("user_input")#request.form["user_input"]
    #type=request.args.get("type")#request.form['type']
    try:
        if(len(question)>2):
            content=run_query(question)
            # content=question

        else:
            content="Wrong Question or Question is empty"
    except Exception as e:
        #print(str(e),"Openai Ai")
        content=str(e)
    return jsonify(content=content)


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files or request.files["file"].filename == "":
        flash("Please select the file", category="error")
        return redirect(url_for("home"))
    else:
        f = request.files["file"]
        if f.filename != "":
            f.save("upload/" + secure_filename(f.filename))
            names = "upload/" + secure_filename(f.filename)
            #session["raww"] = None
            conn_open(names)
            
            os.remove(names)
            # os.unlink(names)
    return {"message":"upload successfully"}

@app.route("/home")
def home():
    return render_template("file_upload.html")
@app.route("/")
@app.route("/chat")
def chat_with():
    return render_template("index.html")    


if(__name__=="__main__"):
    app.run(debug=True)
