import gradio as gr
import pandas as pd
import openai
from sentence_transformers import SentenceTransformer, util
import faiss
import os

openai.api_key = os.getenv("API_KEY")

df = pd.read_csv("dayllergist_qa_batch.csv")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("faiss_index.index")

def retrieve_top_k(query, k=5):
    query_embedding = embed_model.encode([query])
    _, indices = index.search(query_embedding, k)
    return df.iloc[indices[0]]

def build_prompt(query, top_qas):
    context = ""
    for _, row in top_qas.iterrows():
        context += f"Q: {row['Question']}\nA: {row['Answer']}\n\n"
    return f"{context}Q: {query}\nA:"

def generate_answer(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": """You are Dayllergist, a medically reliable, compassionate virtual allergy assistant.

                Your job is to:
                - Provide clear, safe, and medically accurate answers about allergy symptoms and management.
                - Tailor responses based on symptom severity (VAS scores), common treatments, and general allergy guidelines.
                - Use a warm and encouraging tone without sounding robotic.
                - Do not give diagnoses — instead, guide users toward appropriate next steps.

                Be concise, helpful, and supportive. If unsure, advise the user to consult a doctor."""
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.4
    )
    return response["choices"][0]["message"]["content"]

def ask_dayllergist(user_query):
    top_qas = retrieve_top_k(user_query)
    prompt = build_prompt(user_query, top_qas)
    return generate_answer(prompt)

base_url = "https://huggingface.co/spaces/WarunaS/Dayllergist/resolve/main"
logo_url = f"{base_url}/logo.jpg"
user_icon_url = f"{base_url}/patients.jpg"
bot_icon_url = f"{base_url}/dayllergist.jpg"
background_url = f"{base_url}/background.jpg"

with gr.Blocks() as demo:
    # Background + buttons style
    gr.HTML(f"""
    <style>
        body {{
            background-image: url("{background_url}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
            font-family: Arial, sans-serif;
        }}
        .gr-button {{
            background-color: #2CA6A4 !important;
            color: white !important;
            font-weight: bold;
        }}
    </style>
    """)

    gr.HTML(f"""
    <div style='display: flex; align-items: center; margin: 20px 40px;'>
        <img src="{logo_url}" width="100px" style='border-radius:12px; margin-right: 20px;'>
        <div style='text-align: left;'>
            <h1 style='font-size: 40px; margin: 0; font-weight: bold; color: #2CA6A4;'>
                DAYLLERGIST
            </h1>


            <p style='color: #444444; font-size: 20px; margin-top: 5px;'>
                Retrieval-Augmented Generation (RAG) Chatbot for Allergy Guidance
            </p>
        </div>
    </div>
    """)

    # Gradient line
    gr.HTML("<div style='height: 3px; background: linear-gradient(to right, #2CA6A4, #70DBB8); margin-bottom: 20px;'></div>")


    with gr.Row():
        with gr.Column():
            gr.HTML(f"<div style='text-align:center;'><img src='{user_icon_url}' width='80px'></div>")
            user_input = gr.Textbox(label="User questions", placeholder="Type your allergy-related question here...", lines=4, elem_id="user-input-box")
            clear_btn = gr.Button("CLEAR")
            submit_btn = gr.Button("ASK")

        with gr.Column():
            gr.HTML(f"<div style='text-align:center;'><img src='{bot_icon_url}' width='80px'></div>")
            bot_output = gr.Textbox(label="Chatbot answer...", lines=10, elem_id="bot-output-box")

    def respond(user_query):
        return ask_dayllergist(user_query)

    submit_btn.click(respond, inputs=[user_input], outputs=[bot_output])
    clear_btn.click(lambda: "", inputs=[], outputs=[user_input, bot_output])

    gr.HTML("<p style='text-align: center; font-size: 12px; margin-top: 40px;'>Made by WarunaS.</p>")

demo.launch()
