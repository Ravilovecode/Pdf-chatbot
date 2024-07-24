from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.core import ServiceContext
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.prompts.prompts import SimpleInputPrompt
from llama_index.core import Settings
from flask import Flask,jsonify
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import huggingface_hub
import posixpath
import torch
from flask import render_template, request,jsonify
import requests
from llama_index.core import PromptTemplate
import base64
import json
import os
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine.context import ContextChatEngine
from llama_index.core.llms import ChatMessage
import markdown
import re
from langchain_core.prompts import ChatPromptTemplate
import os
from werkzeug.utils import secure_filename
from flask import jsonify, request
 
# HF_HUB_CACHE
huggingface_hub.login(token="")
 
 
documents=SimpleDirectoryReader("data").load_data()
app = Flask(__name__)



UPLOAD_FOLDER = 'data'
ALLOWED_EXTENSIONS = {'pdf'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Define upload folder
# Load documents
documents = SimpleDirectoryReader("data").load_data()
 
# System prompt for LLMS
# system_prompt = """<|SYSTEM|>
# Consider yourself as the representative of your company Omaxe Pvt ltd .Given a question input, your task is to identify relevant keywords,sentences,phrases in the question and retrieve corresponding answers from the context. The model should analyze the input question, extract key terms, and search for similar or related questions in the context.The output should provide the answers associated with the identified keywords or closely related topics.
# The model should understand the context of the question, identify relevant keywords,phrases and sentences, and retrieve information from the provided context based on these keywords.It should be able to handle variations in question phrasing and retrieve accurate answers accordingly with smart generative answers like a chatbot answers to users query.Do not show "relevant keyword fetched" or "from the context provided" or "In the context provided" in the answer simply answer the questions in an intelligent manner.If you are unable to answer the question refer to official website omaxe.com also if the question is not related to omaxe notify the user .
#                                           Answer every questions that are asked in max 3 lines.If user greets you then greet them back and if they say goodvye then also say "goodbye".
#                                            If any question is related to owner of omaxe Tell about Rohtas Goel from the context. If questions are related to "chandigarh" give responses related to "new chandigarh" and if related to "new delhi"give responses related to "delhi" with respect to commercial and residential properties from context provided.If you are asked to give list then provide answers in bulleted points.If any one asks about contact infornmation of omaxe then return their email and phone number from context.
#                                           Try not to include phrases like"Based on the context provided" or "In the context provided" instead use "according to my knowledge" or "as a representative of Omaxe" or "as far as I know" give answer in a  more genrative and smart manner like a bot AI agent does
# Context:\n {context}?\n
# Question: \n{question}\n
# """
 
system_prompt = ChatPromptTemplate.from_template("""You are an AI assistant powered by LLaMA 3, specialized in analyzing and discussing PDF documents. Your role is to:

1. Extract and summarize key information from PDFs
2. Answer questions about PDF content
3. Provide insights and analysis related to the document
4. Assist with document navigation and search

Guidelines:
- Always base your responses on the content of the PDF being discussed
- If asked about information not in the PDF, clearly state that it's beyond the scope of the current document
- Maintain a professional and helpful tone
- Respect copyright and confidentiality of document content
- Offer to elaborate or clarify if your initial response might be insufficient

Remember, you can only access and discuss PDFs that have been properly loaded and processed by the system. If a user refers to a PDF that hasn't been loaded, politely ask them to upload it first.

Begin each interaction by confirming which PDF is being discussed or asking the user to specify one if it's unclear.\n\n
    Context:\n {context}?\n
    Question: \n{question}\n . """)
 
# LLMS settings
llm2 = HuggingFaceLLM(
    context_window=8192,
    max_new_tokens=512,
    generate_kwargs={"temperature": 0, "do_sample": False},
    system_prompt="system_prompt",
    query_wrapper_prompt=SimpleInputPrompt("{query_str}"),
    tokenizer_name="meta-llama/Meta-Llama-3-8B-Instruct",
    model_name="meta-llama/Meta-Llama-3-8B-Instruct",
    device_map="auto",
    tokenizer_kwargs={"max_length": 4096},
    model_kwargs={"torch_dtype": torch.bfloat16}
)
 
# Embedding model
embed_model = HuggingFaceEmbedding(model_name="nomic-ai/nomic-embed-text-v1", trust_remote_code=True)
 
# Set settings
Settings.llm = llm2
Settings.embed_model = embed_model
Settings.chunk_size = 1024
 
# Index documents
index = VectorStoreIndex.from_documents(documents)
retrieval = index.as_retriever()
 
# Chat memory buffer
memory = ChatMemoryBuffer.from_defaults(token_limit=20000)
chat_engine = index.as_chat_engine(
    chat_mode="context",
    memory=memory,
    system_prompt=system_prompt,
    llm=llm2,
    verbose=False
)
def gen_response(question):
    # if question=="hi" or question=="Hi" or question =="hi!" or question =="Hi!" or question =="Hello" or question =="hello" or question =="hey" or question =="Hey":
    #     return "Hello! How can I assist you today?"
   
    # history = chat_engine.chat_history
       
    # chat_history = [ChatMessage(role="user",content=history)]  # Replace with your actual chat history
    response = chat_engine.chat(question)
   
    return str(response)
 
def clean_response(response):
    pattern = r'^\s*assistant\s*'
 
    # Use re.sub to replace the pattern with an empty string
    cleaned_response = re.sub(pattern, '', response)
    return cleaned_response
 
# Load suggestions from JSON
def load_suggestions_from_json(json_file_path):
    with open(json_file_path, 'r') as file:
        suggestions = json.load(file)
    return suggestions
 
predefined_suggestions = load_suggestions_from_json('suggestions.json')
 
# Route for suggestion
@app.route('/suggest', methods=['POST'])
def suggest():
    input_text = request.form['input_text'].lower()
    suggestions = get_suggestions(input_text)
    return jsonify(suggestions)
 
# Get suggestions
def get_suggestions(input_text):
    return [suggestion for suggestion in predefined_suggestions if input_text.lower() in suggestion.lower()]
 
# Route to save and send query
@app.route('/save_and_send', methods=['GET', 'POST'])
def save_and_send():
    email = request.form['email']
    category = request.form['category']
    query = request.form['taskdescription']
    send_email(email, category, query)
    return render_template('chat.html')
 
# Function to send email
def send_email(email, category, query):
    subject = f'New query from {category} category'
    sender_name = "User"
    sender_email = email
    recipient_email = "bigsecxxv@gmail.com"
    body = f'New query from {category} category: {query}'
 
    payload = {
        "sender": {
            "name": sender_name,
            "email": sender_email
        },
        "to": [
            {
                "email": recipient_email
            }
        ],
        "subject": subject,
        "htmlContent": body
    }
 
    api_key = ""
    api_url = ""
 
    response = requests.post(api_url, headers={"api-key": api_key}, json=payload)
    if response:
        scroll_to_contact = True if request.path.endswith('/sent') else False
        return render_template("chat.html", status="Successfully", scroll_to_contact=scroll_to_contact)
    else:
        scroll_to_contact = True if request.path.endswith('/sent') else False
        return render_template("chat.html", status="Successfully", scroll_to_contact=scroll_to_contact)
 
# Route for chat
@app.route("/get", methods=["GET", "POST"])
def chat():
   
    question = request.form.get("msg")
    response = gen_response(question)
    # print(response)
    cleaned_response = clean_response(response)
    html_response = markdown.markdown(cleaned_response)
    return jsonify({'response': html_response})


@app.route("/api/question", methods=["POST"])
def api_question():
    data = request.get_json()
    question = data.get("question")
    if not question:
        return jsonify({"error": "Question not provided"}), 400
    
    response = gen_response(question)
    cleaned_response = clean_response(response)
    html_response = markdown.markdown(cleaned_response)
    return jsonify({'response': cleaned_response})


@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # After saving, you might want to update your index
        update_index()
        return jsonify({'message': 'File uploaded successfully'}), 200
    return jsonify({'error': 'File type not allowed'}), 400

def update_index():
    global index, retriever, chat_engine
    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    retriever = index.as_retriever()
    chat_engine = index.as_chat_engine(
        chat_mode="context",
        memory=memory,
        system_prompt=system_prompt,
        llm=llm2,
        verbose=False
    )    


    
        
# Route for index
@app.route("/")
def index():
    return render_template('chat.html')


 
if __name__ == '__main__':
    app.run(debug=False, port=8000)