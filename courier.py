import time
import os
import streamlit as st
import openai
import logging
import sys
import llama_index
from qdrant_client import QdrantClient
from llama_index import VectorStoreIndex, ServiceContext
from llama_index.llms import OpenAI
from llama_index import SimpleDirectoryReader
from llama_index.storage.storage_context import StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index import set_global_service_context
from llama_index.embeddings import VoyageEmbedding
from qdrant_client.models import Distance, VectorParams

version = "1.0.2"
st.set_page_config(page_title=f"Gaia v{version}", page_icon="🌎", layout="centered", initial_sidebar_state="auto", menu_items=None)
st.title(f"Courier v{version}")

# Set up logging and tracing via Arize Phoenix
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# Use Voyager Lite Embeddings
model_name = "voyage-lite-01-instruct"

voyage_api_key = st_secrets["voyage_key"]

embed_model = VoyageEmbedding(
    model_name=model_name, voyage_api_key=voyage_api_key
)

if "messages" not in st.session_state.keys(): 
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello, my name is Courier. I'm an Generative AI Assistant designed to help Proctor Academy students. Ask me anything about the Proctor Handbook or any current Proctor Academy staff member as of 2023-2024."}
    ]

openai.api_key = st_secrets["openai_key"]

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text=f"Loading Courier v{version} ..."):
        docs = SimpleDirectoryReader(input_dir="./data", recursive=True).load_data()
        qdrant_client = QdrantClient(
            url="https://02aec354-4932-4062-9e00-422eacb506fc.us-east4-0.gcp.cloud.qdrant.io",
            api_key=st_secrets["qdrant_key"],
        )
        qdrant_client.create_collection(collection_name="courierv2",vectors_config=VectorParams(size=1024, distance=Distance.EUCLID),)
        service_context = ServiceContext.from_defaults(embed_model=embed_model,llm=OpenAI(model="gpt-4", max_tokens=1500, temperature=0.5, system_prompt="Keep your answers technical and based on facts and do not hallucinate in responses. In addition, make sure all responses look natural, no Answer: or Query: in the response. Courier can also respond in any language including but not limited to English,Spanish,German,Dutch,Chinese,Thai,Korean and Japanese.  Try to keep translation short to about 4-5 sentences. Always attempt to query database."))
        set_global_service_context(service_context)
        vector_store = QdrantVectorStore(client=qdrant_client, collection_name="courierv2")
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(
            docs, storage_context=storage_context, service_context=service_context,
        )
        return index

index = load_data()

if "chat_engine" not in st.session_state.keys():
    st.session_state.chat_engine = index.as_chat_engine(streaming=True,chat_mode="condense_question",max_tokens=1500,verbose=True)

if prompt := st.chat_input("Your question"):
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: 
        st.write(message["content"])

if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            res_box = st.empty()  
            with st.spinner("Thinking..."):
                response = st.session_state.chat_engine.stream_chat(prompt)
                full_response = ""
                for token in response.response_gen:
                    full_response += "".join(token)
                    res_box.write(full_response)
                message = {"role": "assistant", "content": response.response}
                st.session_state.messages.append(message)