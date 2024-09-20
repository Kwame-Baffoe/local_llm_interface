import os
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Set the cache directories for models
cache_dir_llama = './models/llama2'
cache_dir_gptj = './models/gptj'

# Ensure the models directories exist
os.makedirs(cache_dir_llama, exist_ok=True)
os.makedirs(cache_dir_gptj, exist_ok=True)

# Detect device (use MPS for Apple Silicon if available)
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# Load Models
def load_models():
    print("Loading Llama 2...")
    # Llama 2
    tokenizer_llama = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-2-7b-chat-hf",
        cache_dir=cache_dir_llama
    )
    model_llama = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-chat-hf",
        cache_dir=cache_dir_llama,
        torch_dtype=torch.float16  # Use float16 for efficiency
    )
    model_llama.to(device)

    print("Loading GPT-J...")
    # GPT-J
    tokenizer_gptj = AutoTokenizer.from_pretrained(
        "EleutherAI/gpt-j-6B",
        cache_dir=cache_dir_gptj
    )
    model_gptj = AutoModelForCausalLM.from_pretrained(
        "EleutherAI/gpt-j-6B",
        cache_dir=cache_dir_gptj,
        torch_dtype=torch.float16  # Use float16 for efficiency
    )
    model_gptj.to(device)

    return {
        "Llama 2": (tokenizer_llama, model_llama),
        "GPT-J": (tokenizer_gptj, model_gptj)
    }

models = load_models()

# Generate Response Function
def generate_response(prompt, model_name):
    try:
        tokenizer, model = models[model_name]
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(
            **inputs,
            max_length=512,
            do_sample=True,
            top_p=0.95,
            top_k=50,
            temperature=0.7,
            num_return_sequences=1
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Gradio Interface
model_options = ["Llama 2", "GPT-J"]

iface = gr.Interface(
    fn=generate_response,
    inputs=[
        gr.inputs.Textbox(lines=5, placeholder="Enter your prompt here...", label="Prompt"),
        gr.inputs.Dropdown(choices=model_options, label="Choose Model"),
    ],
    outputs="text",
    title="Local LLM Interface",
    description="Interact with Llama 2 and GPT-J locally on your MacOS machine.",
)

if __name__ == "__main__":
    iface.launch()