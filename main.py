import gradio as gr
import requests
import os

# Your Hugging Face API token
HF_API_TOKEN = "YOUR_HUGGING_FACE_API_TOKEN"  # Replace with your actual token

# Headers for authentication
headers = {
    "Authorization": f"Bearer {HF_API_TOKEN}"
}

# Model endpoints
MODEL_ENDPOINTS = {
    "Llama 2": "https://api-inference.huggingface.co/models/meta-llama/Llama-2-7b-chat-hf",
    "GPT-J": "https://api-inference.huggingface.co/models/EleutherAI/gpt-j-6B"
    # Add more models here if needed
}

def query_model(model_name, prompt):
    api_url = MODEL_ENDPOINTS.get(model_name)
    if not api_url:
        return f"Model '{model_name}' is not supported."

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 512,
            "do_sample": True,
            "top_p": 0.95,
            "top_k": 50,
            "temperature": 0.7,
            "num_return_sequences": 1
        }
    }

    response = requests.post(api_url, headers=headers, json=payload)

    if response.status_code == 200:
        output = response.json()
        # The response format may vary depending on the model
        if isinstance(output, list) and "generated_text" in output[0]:
            return output[0]["generated_text"]
        elif isinstance(output, dict) and "generated_text" in output:
            return output["generated_text"]
        else:
            return str(output)
    else:
        return f"Error {response.status_code}: {response.text}"

# Gradio Interface
model_options = list(MODEL_ENDPOINTS.keys())

def generate_response(prompt, model_name):
    if not HF_API_TOKEN or HF_API_TOKEN == "YOUR_HUGGING_FACE_API_TOKEN":
        return "Please set your Hugging Face API token in the script."
    if not prompt:
        return "Please enter a prompt."
    return query_model(model_name, prompt)

iface = gr.Interface(
    fn=generate_response,
    inputs=[
        gr.Textbox(lines=5, placeholder="Enter your prompt here...", label="Prompt"),
        gr.Dropdown(choices=model_options, label="Choose Model"),
    ],
    outputs="text",
    title="LLM Interface via Hugging Face Inference API",
    description="Interact with Llama 2 and GPT-J models via Hugging Face's Inference API.",
)

if __name__ == "__main__":
    iface.launch()