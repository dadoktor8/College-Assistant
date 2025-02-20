from flask import Flask, request, jsonify
from flask_cors import CORS
#from transformers import AutoTokenizer, AutoModelForCausalLM
from llama_cpp import Llama
from utils.logger import Logger #This is my log class that handles the error data 
import re

app = Flask(__name__)
CORS(app)
log = Logger()
log.info("Model Starting")
model_path = r"D:\College-Assistant\College-Assistant\data\DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"
#The transformer architceture has been commented out due to gguf support 
llm = Llama(model_path=model_path, n_ctx=2048, n_thread=4) #default token length set by n_ctx, ask the cpu to use 4 cores
#model = AutoModelForCausalLM.from_pretrained("./fine_tuned_model")
#tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_model")


@app.route("/query", methods=["POST"])
def query():
    data = request.json 
    user_query = data.get("query","")
    prompt = f"Please summarize the following text concisely and clearly. Ensure key details are retained while making it as short as possible. If it cannot be summarized, return: $$$This cannot be summarized$$$. \n\nText:{user_query}\nAI:"
    #prompt = f"User: {user_query}\nAI:"
    #inputs = tokenizer(user_query, return_tensors="pt", max_length=512, truncation=True)
    #outputs = model.generate(**inputs, max_length=200)
    #outputs = llm(f"User: Remember that in this prompt you only give summaries decrease the overall thinking time and simply give summary for the specific topic, avoid answering questions that can't be summarized default them to this reply $$$This cannot be summarized$$$ {user_query}\nAI",max_tokens=200)
    outputs = llm(
        prompt,
        #echo=False,
        temperature=0.4,
        max_tokens=512,
        top_p=0.9,
        )
    
    response = outputs["choices"][0]["text"].strip()
    response = response.replace("\u200B","")
    response = response.replace("\u00A0", " ")
    response = re.sub(r"\s+"," ",response)
    response = re.sub(r"[\s\S]*<\/think>\n?","",response).strip()
    try:
        #response = re.sub(r"</think>.*?</think>","",response,flags=re.DOTALL).strip()
        #response = re.sub(r"<think>[\s\S]*?</think>", "", response, flags=re.DOTALL).strip()
        log.info(f"Model Response: {response}")
        if response is None:
            raise ValueError("Empty Response!")
    except Exception as e:
        log.error(f"An error ocurred: {str(e)}")
    
    log.info("Process completed")
    print(outputs)
    print(f"Response is {response}")
    print(f"Raw response is ", outputs["choices"][0]["text"])
    return jsonify({"response":response})
    #response = tokenizer.decode(outputs[0], skip_special_tokens=True)
   # return jsonify({"response" : outputs["choices"][0]["text"].strip()})

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=5000,debug=True)
