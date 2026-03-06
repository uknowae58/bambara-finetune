"""
Flask API for Bambara Model Inference
"""
from flask import Flask, request, jsonify
from unsloth import FastLanguageModel
import torch

app = Flask(__name__)

# Load model at startup
MODEL_PATH = "./bambara-model-final"

print("Loading model...")
model, tokenizer = FastLanguageModel.from_pretrained(MODEL_PATH, load_in_4bit=True)
FastLanguageModel.for_inference(model)
print("Model loaded!")

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    prompt = data.get('prompt', '')
    max_tokens = data.get('max_tokens', 128)
    temperature = data.get('temperature', 0.7)
    top_p = data.get('top_p', 0.9)
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs, 
        max_new_tokens=max_tokens, 
        temperature=temperature, 
        top_p=top_p,
        do_sample=True
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the response (after the prompt)
    if prompt in response:
        response = response[len(prompt):].strip()
    
    return jsonify({
        "prompt": prompt,
        "response": response
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)