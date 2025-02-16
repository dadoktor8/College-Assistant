from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained('./fine_tuned_model')
print("Model Loaded")
tokenizer = AutoTokenizer.from_pretrained('./fine_tuned_model')
print("Tokenizer Loaded")

input_text=input("Ask: ")
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Response: ", response)
