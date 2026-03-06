"""
Flask API for Bambara Model Inference
Run this on a GPU server to test the fine-tuned model
"""
from flask import Flask, request, jsonify
from unsloth import FastLanguageModel
import torch
import os

app = Flask(__name__)

# Configuration
MODEL_PATH = os.environ.get('MODEL_PATH', './bambara-model-final')
MAX_TOKENS = int(os.environ.get('MAX_TOKENS', '256'))

print(f"Loading model from {MODEL_PATH}...")
model, tokenizer = FastLanguageModel.from_pretrained(MODEL_PATH, load_in_4bit=True)
FastLanguageModel.for_inference(model)
print("Model loaded!")

@app.route('/generate', methods=['POST'])
def generate():
    """Generate response for a Bambara prompt"""
    data = request.get_json()
    
    prompt = data.get('prompt', '')
    max_tokens = data.get('max_tokens', MAX_TOKENS)
    temperature = data.get('temperature', 0.7)
    top_p = data.get('top_p', 0.9)
    
    if not prompt:
        return jsonify({'error': 'No prompt provided'}), 400
    
    try:
        # Generate
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True
        )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the response part (after the prompt)
        if prompt in response:
            response = response[len(prompt):].strip()
        
        return jsonify({
            'prompt': prompt,
            'response': response
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'model': 'bambara-qwen2.5-0.5b-v4',
        'loss': '0.7585'
    })

@app.route('/', methods=['GET'])
def index():
    """Simple HTML interface"""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Bambara Model Tester</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
            textarea { width: 100%; height: 100px; margin: 10px 0; }
            button { padding: 10px 20px; background: #4CAF50; color: white; border: none; cursor: pointer; }
            button:hover { background: #45a049; }
            #response { background: #f0f0f0; padding: 20px; margin-top: 20px; white-space: pre-wrap; }
        </style>
    </head>
    <body>
        <h1>🤖 Bambara Model Tester</h1>
        <p>Fine-tuned Qwen2.5-0.5B on 824k Bambara examples</p>
        
        <h3>Enter your prompt in Bambara:</h3>
        <textarea id="prompt" placeholder="xin ni ye..."></textarea>
        
        <h3>Settings:</h3>
        <label>Max tokens: <input type="number" id="max_tokens" value="256"></label>
        <label>Temperature: <input type="number" id="temperature" value="0.7" step="0.1"></label>
        
        <br><br>
        <button onclick="generate()">Generate</button>
        
        <h3>Response:</h3>
        <div id="response">...</div>
        
        <script>
        async function generate() {
            const prompt = document.getElementById('prompt').value;
            const max_tokens = document.getElementById('max_tokens').value;
            const temperature = document.getElementById('temperature').value;
            
            document.getElementById('response').textContent = 'Generating...';
            
            try {
                const response = await fetch('/generate', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        prompt,
                        max_tokens: parseInt(max_tokens),
                        temperature: parseFloat(temperature)
                    })
                });
                
                const data = await response.json();
                document.getElementById('response').textContent = data.response || data.error;
            } catch (e) {
                document.getElementById('response').textContent = 'Error: ' + e;
            }
        }
        </script>
    </body>
    </html>
    '''

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
