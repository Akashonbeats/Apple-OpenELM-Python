from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "apple/OpenELM-270M"
tokenizer = AutoTokenizer.from_pretrained("gpt2")  # Use GPT-2 tokenizer for compatibility
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

def generate_response(input_text):
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=50)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit']:
        break
    response = generate_response(user_input)
    print(f"Bot: {response}")