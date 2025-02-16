from flask import Flask, request, jsonify
from flask_cors import CORS
#from transformers import AutoTokenizer, AutoModelForCausalLM
from llama_cpp import Llama

app = Flask(__name__)
CORS(app)
model_path = r"D:\College-Assistant\College-Assistant\data\DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"
#The transformer architceture has been commented out due to gguf support 
llm = Llama(model_path=model_path, n_ctx=2048) #default token length set by n_ctx
#model = AutoModelForCausalLM.from_pretrained("./fine_tuned_model")
#tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_model")

@app.route("/query", methods=["POST"])
def query():
    data = request.json 
    user_query = data.get("query","")
    #inputs = tokenizer(user_query, return_tensors="pt", max_length=512, truncation=True)
    #outputs = model.generate(**inputs, max_length=200)
    outputs = llm(f"User: Remember that in this prompt you only give summaries decrease the overall thinking time and simply give summary for the specific topic, avoid answering questions that can't be summarized default them to this reply $$$This cannot be summarized$$$ {user_query}\nAI",max_tokens=200)

    #response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return jsonify({"response" : outputs["choices"][0]["text"].strip()})

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=5000,debug=True)
