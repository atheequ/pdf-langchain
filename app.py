from flask  import Flask,request,jsonify #,render_template
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

#from pdf2image import convert_from_path
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline

import os
import sys
import torch
from langchain import HuggingFacePipeline, PromptTemplate
from langchain.chains import RetrievalQA

app=Flask(__name__)

@app.route("/extract",methods=["GET","POST"])
def extract():
    if(request.method=="POST"):
       filename=request.form['filename']
       question=request.form['question']
    elif(request.method=="GET"):
       filename=request.form.get("filename")
       question=request.form.get(question)
    else:
       return jsonify(answer="Please check the request method")
    embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")#sentence-transformers/all-mpnet-base-v2 #all-MiniLM-L6-v2
    vectorstore=FAISS.load_local(f"faiss/{filename}",embeddings=embeddings)
    tokenizer=AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")#,use_auto_token=True)
    model=AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf",
                                           device_map="auto",
                                           torch_dtype=torch.float15,
                                           #use_auth_token=True,
                                           load_in_8bit=True)
    pipe=pipeline("text-generation",
              model=model,
              torch_dtype=torch.bfloat16,
              device_map="auto",
              max_new_tokens=512,
              do_sample=True,top_k=30,
              num_return_sequences=1,
              eos_token_id=tokenizer.eos_token_id
              )
    llm=HuggingFacePipeline(pipeline=pipe,model_kwargs={"temperature":0.1})

    System_prompt="""
    Use the following pieces of context to answer the question at the end.
    if you don't know the answer, just say that you don't know, don't try to make up an answer
    """

    B_INST,E_INST="[INST]","[/INST]"
    B_SYS, E_SYS="<<SYS>>\n","\n<</SYS>>\n\n"
    System_prompt=B_SYS+System_prompt+E_SYS

    Instruction="""
        {context}
    Question: {question}
    """
    template=B_INST +System_prompt+Instruction+E_INST

    prompt=PromptTemplate(template=template,input_variables=["content","question"])

    qa_chain=RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k":2}),
        chain_type_kwargs={"prompt":prompt}
    )

    res=qa_chain(question)
    return jsonify(answer=res['result'])


if(__name__=="__main__"):
   app.run()
