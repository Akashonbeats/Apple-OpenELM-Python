import os
from transformers import AutoModelForCausalLM, AutoTokenizer

# Use environment variable for the API token
api_token = os.getenv("Your_HUGGINGFACE_API_TOKEN")

model_name = "apple/OpenELM-270M"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=api_token)
model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=api_token, trust_remote_code=True)

def generate_response(input_text):
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=50)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage
input_text = "Hello, how are you?"
response = generate_response(input_text)
print(response)