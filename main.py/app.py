
Search
Write
Sign up

Sign in



Building a multi–PDF GEN AI Chatbot Using Open AI, LangChain and Streamlit.
Binay Pradhan
Binay Pradhan

·
Follow

3 min read
·
Nov 2, 2023
25


1



In this tutorial, we will understand the process of creating a multi-PDF reader Generative AI Chatbot using Open AI, LangChain libraries and Streamlit.

But before jumping into the process and code, first the libraries’ basics

Open AI embeddings:
OpenAI’s text embeddings measure the relatedness of text strings. Embeddings are commonly used for:

Search (where results are ranked by relevance to a query string)
Clustering (where text strings are grouped by similarity)
Recommendations (where items with related text strings are recommended)
Anomaly detection (where outliers with little relatedness are identified)
Diversity measurement (where similarity distributions are analyzed)
Classification (where text strings are classified by their most similar label)
Open AI Models:
GPT-4, a large multimodal model that can solve difficult problems with greater accuracy than any of the Open AI previous models.

LangChain Chains libraries:
RetrievalQAWithSourcesChain- Question-answering with sources over an index. Create a new model by parsing and validating input data from keyword arguments. Raises Validation Error if the input data cannot be parsed to form a valid model.

PDF Libraries:
PyPDF2 is a free and open-source pure-python PDF library capable of splitting, merging, cropping, and transforming the pages of PDF files.

Web Application Library:
Streamlit: An open source app framework in Python language. It helps us create web apps for data science and machine learning in a short time.

Note : We will be using Github as our Code hosting repository.

So, Without further ado, lets jump into the code.

Create two files on GitHub:

I. main.py/app.py

II. requirements.txt

main.py/app.py
import streamlit as st
import langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain import OpenAI, VectorDBQA
from langchain.chains import RetrievalQAWithSourcesChain
import PyPDF2

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

#This function will go through pdf and extract and return list of page texts.
def read_and_textify(files):
    text_list = []
    sources_list = []
    for file in files:
        pdfReader = PyPDF2.PdfReader(file)
        #print("Page Number:", len(pdfReader.pages))
        for i in range(len(pdfReader.pages)):
          pageObj = pdfReader.pages[i]
          text = pageObj.extract_text()
          pageObj.clear()
          text_list.append(text)
          sources_list.append(file.name + "_page_"+str(i))
    return [text_list,sources_list]

st.set_page_config(layout="centered", page_title="GoldDigger")
st.header("GoldDigger")
st.write("---")
  
#file uploader
uploaded_files = st.file_uploader("Upload documents",accept_multiple_files=True, type=["txt","pdf"])
st.write("---")

if uploaded_files is None:
  st.info(f"""Upload files to analyse""")
elif uploaded_files:
  st.write(str(len(uploaded_files)) + " document(s) loaded..")
  
  textify_output = read_and_textify(uploaded_files)
  
  documents = textify_output[0]
  sources = textify_output[1]
  
  #extract embeddings
  embeddings = OpenAIEmbeddings(openai_api_key = st.secrets["openai_api_key"])
  #vstore with metadata. Here we will store page numbers.
  vStore = Chroma.from_texts(documents, embeddings, metadatas=[{"source": s} for s in sources])
  #deciding model
  model_name = "gpt-3.5-turbo"
  # model_name = "gpt-4"

  retriever = vStore.as_retriever()
  retriever.search_kwargs = {'k':2}

  #initiate model
  llm = OpenAI(model_name=model_name, openai_api_key = st.secrets["openai_api_key"], streaming=True)
  model = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
  
  st.header("Ask your data")
  user_q = st.text_area("Enter your questions here")
  
  if st.button("Get Response"):
    try:
      with st.spinner("Model is working on it..."):
        result = model({"question":user_q}, return_only_outputs=True)
        st.subheader('Your response:')
        st.write(result['answer'])
        st.subheader('Source pages:')
        st.write(result['sources'])
    except Exception as e:
      st.error(f"An error occurred: {e}")
      st.error('Oops, the GPT response resulted in an error :( Please try again with a different question.')
requirements.txt
streamlit
PyPDF2
openai
chromadb
tiktoken
langchain
typing-inspect==0.8.0
typing_extensions==4.5.0
pysqlite3-binary
Now, Lets Initialize Streamlit — Login to Streamlit via GitHub and create a new app with the Github repo.

Streamlit Login
Deploy an App on Streamlit
Wait till your app is ready. Once live, click on the settings, add your Open AI secret (this can be grabbed from you Open AI account) and Save.


Voila! Your Chatbot is ready, run the app, upload the pdfs, and ask questions. Below is a snapshot of my app running:


Cheers! 👏👏👏

Generative Ai Tools
ChatGPT
Python Programming
Chatbots
25


1


Binay Pradhan
Written by Binay Pradhan
22 Followers
·
22 Following
👨‍💻Solutions Consultant @Adobe ✍Technologist + MarTech + Digital👨‍🏫

Follow
Responses (1)
What are your thoughts?

Cancel
Respond
Respond

Also publish to my profile

Ragathebrown
Ragathebrown

10 months ago


Hi Binay,
I wanted to know if the solution can respond with multiple answers. Let say if the answer is present in multiple documents, can it display all these results along with the main result.
Reply

More from Binay Pradhan
How to Create Multi-Step Forms in Marketo
Adobe Tech Blog
In

Adobe Tech Blog

by

Binay Pradhan

How to Create Multi-Step Forms in Marketo
How to keep your site visitors’ attention spans, resulting in higher conversation rates and avoiding form abandonments.
Jun 2, 2022
20
Google Ads Enhanced Conversions via Adobe Experience Platform
Adobe Tech Blog
In

Adobe Tech Blog

by

Binay Pradhan

Google Ads Enhanced Conversions via Adobe Experience Platform
Set up enhanced conversions by leveraging Google Ads API via Adobe Experience Platform
May 22, 2023
11
1
firebase web push Permission Allow & Block. Notification post permission
Adobe Tech Blog
In

Adobe Tech Blog

by

Binay Pradhan

Firebase Web Push Notification Directly from Marketo
This article will help you implement firebase web push notification with a web app creation, all within Marketo.
Feb 14, 2023
6
See all from Binay Pradhan
Recommended from Medium
Building a RAG Chatbot Using Langchain and Streamlit:Engage with Your PDFs
AI Advances
In

AI Advances

by

Tarun Singh

Building a RAG Chatbot Using Langchain and Streamlit:Engage with Your PDFs
Interacting with extensive PDFs has never been more fascinating. Imagine having the ability to converse with your notes, books, and…

Jun 19
165
3
Building a Multi PDF RAG Chatbot: Langchain, Streamlit with code
GoPenAI
In

GoPenAI

by

Paras Madan

Building a Multi PDF RAG Chatbot: Langchain, Streamlit with code
Talking to big PDF’s is cool. You can chat with your notes, books and documents etc. This blog post will help you build a Multi RAG…
Jun 7
842
7
Lists



What is ChatGPT?
9 stories
·
472 saves
Image by vectorjuice on FreePik


The New Chatbots: ChatGPT, Bard, and Beyond
12 stories
·
512 saves

AI-generated image of a cute tiny robot in the backdrop of ChatGPT’s logo

ChatGPT
21 stories
·
889 saves



ChatGPT prompts
50 stories
·
2281 saves
How to Build One Chatbot for Multiple Databases
Samar Singh
Samar Singh

How to Build One Chatbot for Multiple Databases
Building a chatbot that seamlessly interacts with multiple databases like CSV files, PDFs, and images is a powerful way to enhance user…

Nov 17
125
1
Deploy a Streamlit LLM App on Azure Web App (GPT-4o Azure OpenAI and SSO auth)
Enric Domingo - AI Engineering
Enric Domingo - AI Engineering

Deploy a Streamlit LLM App on Azure Web App (GPT-4o Azure OpenAI and SSO auth)
In this tutorial, we will see how to deploy a Streamlit Python web app on the Microsoft Azure Cloud, using Azure App Service Plan, Azure…
Sep 29
10
Building a Multi-Modal RAG System for Visual Question Answering
Level Up Coding
In

Level Up Coding

by

Bhargob Deka

Building a Multi-Modal RAG System for Visual Question Answering
Build a multi-modal RAG chatbot using LangChain and GPT-4o to chat with a PDF document.

Aug 1
568
4
Streamlining Information Retrieval: Building a Chatbot with Django and OpenAI
Grant Palmer
Grant Palmer

Streamlining Information Retrieval: Building a Chatbot with Django and OpenAI
Build a professor-info chatbot using Django and OpenAI’s GPT-3.5, integrating a backend for professor data and a user-friendly frontend.
Aug 5
27
See more recommendations
Help

Status

About

Careers

Press

Blog

Privacy

Terms

Text to speech

Teams
