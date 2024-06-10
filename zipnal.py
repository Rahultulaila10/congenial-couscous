import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.llms import HuggingFaceHub
from langchain.llms import CTransformers
from langchain.llms import OpenAI 
from langchain.chains import RetrievalQA
import os
import time


def main():
    load_dotenv()

    st.set_page_config(page_title="Apollo Chat Bot",page_icon=":hospital:",layout="wide")
    st.snow()

    st.header("Chat with Medical Docs :page_facing_up:")
    

    col1,col2,col3=st.columns([0.2,0.6,0.2])

    if 'embeddings' not in st.session_state:
        print('loading embeddings...')              
        embeddings=HuggingFaceEmbeddings(model_name="BAAI/bge-large-en",model_kwargs={'device':'cpu'})
        st.session_state['embeddings'] = embeddings
        print('embeddings loaded')

    if 'LLM' not in st.session_state:
            localmodel="/home/bugada/.cache/huggingface/hub/models--TheBloke--Llama-2-7B-Chat-GGUF/blobs/08a5566d61d7cb6b420c3e4387a39e0078e1f2fe5f055f3a03887385304d4bfa"
            llm = CTransformers(
                        model=localmodel,
                        model_type="llama",
                        lib="avx2",
                        temperature=0.1
                    )
            st.session_state['LLM'] = llm
            print('LLM Initialized')

    with col3:
        st.subheader("Parameters")
        print("abc")
        Chunksize=st.number_input("Chunk size",key="1")
        Chunkoverlap=st.number_input("Chunk Overlap",key="2")
        temperature=st.number_input("Temperature",key="3")
        maxlength=st.number_input("Max length",key="4")
        embedding=st.text_input("Input a embedding model",key="5")
        llmmodel=st.text_input("select a llm model",key="6")
        llmmodel_version=st.text_input("select the version of the model to download")
        llm_model_type=st.text_input("select the model type pf the llms model")
        waytorun=st.radio("Select a way to run",["Download locally","Open AI","HuggingFace Hub"])



    with col1:
        st.subheader("Your Documents")
        pdf_docs=st.file_uploader("Upload your documents",accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                text=""
                for pdf in pdf_docs:
                    pdf_reader=PdfReader(pdf)
                    for page in pdf_reader.pages:
                        text += page.extract_text()

                if Chunksize:
                    chunk__size=Chunksize
                else:
                    chunk__size=300

                if Chunkoverlap:
                    chunk__overlap=Chunkoverlap
                else:
                    chunk__overlap=30    

                text_splitter=RecursiveCharacterTextSplitter(
                    separators="\n",
                    chunk_size=chunk__size,
                    chunk_overlap=chunk__overlap,
                    length_function=len
                    )
                chunks=text_splitter.split_text(text)
                
                if embedding:
                    embedding1=embedding
                else:
                    embedding1="BAAI/bge-large-en"
                
                st.session_state['embeddings']=HuggingFaceEmbeddings(model_name=embedding1,model_kwargs={'device':'cpu'})
                print('vector store creating')
                vectorstore=FAISS.from_texts(texts=chunks,embedding=st.session_state['embeddings'])
                print('vector store created!')

                if llmmodel:
                    localmodel=llmmodel
                else:
                    pass

                if llmmodel_version:
                    model_file=llmmodel_version
                
                if llm_model_type:
                    model_type=llm_model_type
                
                if temperature:
                    temp = temperature
                else:
                    temp=0.1

                if maxlength:
                    mal=maxlength
                else:
                    mal=512

                if waytorun == "Download locally":

                    st.session_state['LLM'] = CTransformers(
                        model=localmodel,
                        model_file=model_file,
                        model_type=model_type,
                        lib="avx2",
                        temperature=temp
                    )

                elif waytorun == "HuggingFace Hub":
                    os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")
                    st.session_state['LLM']=HuggingFaceHub(repo_id=localmodel,model_kwargs={"temperature":temp,"max_length":mal})
                    

                else:                    
                    os.environ["OPENAI_API_TOKEN"]=os.getenv("OPENAI_API_TOKEN")
                    st.session_state['LLM']=OpenAI(model_name=localmodel)

                qa=RetrievalQA.from_chain_type(
                    llm=st.session_state['LLM'],
                    chain_type="stuff",
                    retriever=vectorstore.as_retriever(),
                    return_source_documents=True
                )

                if 'QA' not in st.session_state:
                    print('loading QA...')
                    
                    st.session_state['QA'] = qa
                    print('QA loaded')

    with col2:
        user_question=st.text_input("Ask me anything",key="<uniquevlueofsomesort>")
        generate=st.button("Run RAG")

        st.subheader("Response")
        if generate and user_question:
            with st.spinner("Generating response..."):
                start=time.time_ns()
                qa = st.session_state['QA']
                response=qa(user_question)
                end=time.time_ns()
                if response:
                    st.write(response)
                    st.write(end-start,"ns")
                    st.write("Response generated!")
                else:
                    st.error("Failed to generate response.")


if __name__ == '__main__':
    main()