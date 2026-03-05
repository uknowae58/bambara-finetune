"""
Inference script for fine-tuned Bambara model
"""
from unsloth import FastLanguageModel
import torch

MODEL_PATH = "./bambara-model-final"

def load_model():
    """Load the fine-tuned model and tokenizer"""
    model, tokenizer = FastLanguageModel.from_pretrained(
        MODEL_PATH,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer

def generate(prompt: str, max_new_tokens: int = 256):
    """Generate response for a prompt in Bambara"""
    model, tokenizer = load_model()
    
    # Format for Qwen3
    formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the assistant response
    if "<|im_start|>assistant\n" in response:
        response = response.split("<|im_start|>assistant\n")[-1]
    
    return response

if __name__ == "__main__":
    # Example prompts in Bambara
    test_prompts = [
        "xin ni ye",
        "i ka ka",
        "a bɛ min"
    ]
    
    print("🤖 Bambara Model Inference")
    print("=" * 50)
    
    for prompt in test_prompts:
        print(f"\n📝 Prompt: {prompt}")
        print("⏳ Generating...")
        response = generate(prompt)
        print(f"✅ Response: {response}")
        print("-" * 50)
