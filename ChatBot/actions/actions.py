# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions

import datetime as dt
from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import ActionExecuted

import os
import pinecone
from sentence_transformers import SentenceTransformer
from langchain import HuggingFaceHub
from langchain import PromptTemplate, LLMChain

#database

PINECONE_API_TOKEN = "ENTER_YOUR_API_KEY"
pinecone.init(
    api_key= PINECONE_API_TOKEN,
    environment="asia-southeast1-gcp-free"
)

retriever = SentenceTransformer(
    "flax-sentence-embeddings/all_datasets_v3_mpnet-base"
)
retriever

index_name = "abstractive-question-answering"
index = pinecone.Index(index_name)

#Falcon model

HUGGINGFACE_API_TOKEN = "ENTER_YOUR_API_KEY"
os.environ["HUGGINGFACE_API_TOKEN"] = HUGGINGFACE_API_TOKEN

repo_id = "tiiuae/falcon-7b-instruct"
llm = HuggingFaceHub(huggingfacehub_api_token=HUGGINGFACE_API_TOKEN, 
                     repo_id=repo_id, 
                     model_kwargs={"temperature":0.1, "max_new_tokens":700})

template = """
Answer the question using the given context within 10 words.
{question}
Answer:
"""
prompt = PromptTemplate(template=template, input_variables=["question"])
llm_chain = LLMChain(prompt=prompt, llm=llm)


class ActionDefaultFallback(Action):
    def name(self) -> Text:
        return "action_default_fallback"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        dispatcher.utter_message(template="utter_default")
        return []
    
class ActionSayTime(Action):
    def name(self) -> Text:
        return "action_show_time"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        current_time = dt.datetime.now().strftime("%H:%M:%S")
        message = f"The current time is {current_time}"
        dispatcher.utter_message(text=message)
        return []
    
class ActionContextQuestionAnswering(Action):
    def name(self) -> Text:
        return "action_context_answering"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        def query_pinecone(query, top_k):
            xq = retriever.encode([query]).tolist()
            xc = index.query(xq, top_k=top_k, include_metadata=True)
            return xc
        
        query = tracker.latest_message['text']
        result = query_pinecone(query, top_k=1)
        
        def format_query(query,  context):
            context = [f" {m['metadata']['content']}" for m in context]
            context = " ".join(context)
            query = f"question: {query} context: {context}"
            return query
        query_for_model = format_query(query, result["matches"])

        ans = llm_chain.run(query_for_model)
        
        dispatcher.utter_message(text= 'Answer: "' + ans +'"')
        
        return []
