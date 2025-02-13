from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForCausalLM

app = Flask(__name__)
CORS(app)
model = AutoModelForCausalLM.from_pretrained("./fine_tuned_model")
tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_model")

@app.route("/query", methods=["POST"])
def query():
    data = request.json 
    user_query = data.get("query")
    inputs = tokenizer(user_query, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(**inputs, max_length=200)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return jsonify({"response" : response})

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=5000,debug=True)
