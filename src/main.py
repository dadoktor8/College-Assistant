from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
from llama_cpp import Llama
from utils.logger import Logger
import re
import os
import hashlib
import json
import uuid
import PyPDF2
import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph

app = Flask(__name__)
CORS(app)
log = Logger()
log.info("Model Starting")

# Paths for storing uploaded textbooks and their chunks
UPLOAD_FOLDER = 'uploads'
CHUNK_FOLDER = 'chunks'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CHUNK_FOLDER, exist_ok=True)

model_path = r"D:\College-Assistant\College-Assistant\data\DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"
llm = Llama(model_path=model_path, n_ctx=2048, n_thread=4)

# Simple vector store
textbook_chunks = {}
textbook_metadata = {}

@app.route("/")
def home():
    return render_template("index.html")

def extract_text_from_pdf(file_path):
    """Extract text from PDF file"""
    text = ""
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page_num in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page_num].extract_text() + "\n"
    return text

def chunk_text(text, chunk_size=1500, overlap=150):
    """Split text into overlapping chunks"""
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = min(start + chunk_size, text_length)
        if end < text_length and end != text_length:
            # Find the last period or newline to create clean breaks
            while end > start and text[end] not in ['.', '\n']:
                end -= 1
            if end == start:  # If no good break found, just use the chunk size
                end = min(start + chunk_size, text_length)
        
        chunks.append(text[start:end])
        start = end - overlap if end != text_length else text_length
    
    return chunks

def save_chunks(book_id, chunks):
    """Save chunks to files and update the textbook_chunks dictionary"""
    chunk_paths = []
    for i, chunk in enumerate(chunks):
        chunk_file = f"{CHUNK_FOLDER}/{book_id}_{i}.txt"
        with open(chunk_file, 'w', encoding='utf-8') as f:
            f.write(chunk)
        chunk_paths.append(chunk_file)
        
        # Store in memory for searching
        textbook_chunks[f"{book_id}_{i}"] = chunk
    
    return chunk_paths

def find_relevant_chunks(query, book_id=None):
    relevant_chunks = []
    query_keywords = set(query.lower().split())
    
    chunks_to_search = textbook_chunks
    if book_id:
       
        chunks_to_search = {k: v for k, v in textbook_chunks.items() if k.startswith(book_id)}
    
    for chunk_id, chunk_text in chunks_to_search.items():
        chunk_text_lower = chunk_text.lower()
        
       
        matches = sum(1 for keyword in query_keywords if keyword in chunk_text_lower)
        
        if matches > 0:
            relevant_chunks.append({
                'id': chunk_id,
                'text': chunk_text,
                'score': matches / len(query_keywords)  # Simple relevance score
            })
    
    
    relevant_chunks.sort(key=lambda x: x['score'], reverse=True)
    return relevant_chunks[:3]  # Return top 3 most relevant chunks

@app.route("/upload_textbook", methods=["POST"])
def upload_textbook():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    title = request.form.get('title', file.filename)
    author = request.form.get('author', 'Unknown')
    
    
    book_id = hashlib.md5(f"{title}_{author}_{uuid.uuid4()}".encode()).hexdigest()
    
    file_path = os.path.join(UPLOAD_FOLDER, f"{book_id}_{file.filename}")
    file.save(file_path)
    
    
    if file.filename.lower().endswith('.pdf'):
        text = extract_text_from_pdf(file_path)
    else:
    
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    
    chunks = chunk_text(text)
    

    chunk_paths = save_chunks(book_id, chunks)
    
    # Store textbook metadata
    textbook_metadata[book_id] = {
        'title': title,
        'author': author,
        'filename': file.filename,
        'path': file_path,
        'chunks': chunk_paths,
        'num_chunks': len(chunks)
    }
    
    return jsonify({
        "success": True,
        "book_id": book_id,
        "title": title,
        "num_chunks": len(chunks)
    })

@app.route("/list_textbooks", methods=["GET"])
def list_textbooks():
    books = []
    for book_id, metadata in textbook_metadata.items():
        books.append({
            "id": book_id,
            "title": metadata["title"],
            "author": metadata["author"],
            "num_chunks": metadata["num_chunks"]
        })
    return jsonify({"textbooks": books})

@app.route("/query", methods=["POST"])
def query():
    data = request.json
    user_query = data.get("query", "")
    book_id = data.get("book_id", None)  # Optional book_id to search in specific textbook
    
    # Find relevant chunks
    relevant_chunks = find_relevant_chunks(user_query, book_id)
    
    if not relevant_chunks:
        return jsonify({"response": "I couldn't find relevant information in the textbooks. Please try a different query or upload more textbooks."})
    
    # Prepare context from relevant chunks
    context = "\n\n".join([chunk["text"] for chunk in relevant_chunks])
    
    # Prepare prompt with context
    prompt = (
        "You are a helpful teaching assistant. Use the following textbook information to answer the student's question:\n\n"
        f"TEXTBOOK CONTEXT:\n{context}\n\n"
        f"STUDENT QUESTION: {user_query}\n\n"
        "Answer the student's question clearly and concisely, using information from the textbook. "
        "If the textbook doesn't contain enough information to answer, say so honestly.\n\n"
        "ASSISTANT:"
    )
    
    outputs = llm(
        prompt,
        temperature=0.4,
        max_tokens=1024,
        top_p=0.9,
    )
    
    response = outputs["choices"][0]["text"].strip()
    response = response.replace("\u200B", "")
    response = response.replace("\u00A0", " ")
    response = re.sub(r"\s+", " ", response)
    response = re.sub(r"[\s\S]*<\/think>\n?", "", response).strip()
    
    try:
        log.info(f"Model Response: {response}")
        if not response:
            raise ValueError("Empty Response!")
    except Exception as e:
        log.error(f"An error occurred: {str(e)}")
    
    return jsonify({"response": response})

@app.route("/generate_quiz", methods=["POST"])
def generate_quiz():
    data = request.json
    book_id = data.get("book_id")
    num_questions = int(data.get("num_questions", 5))
    topic = data.get("topic", "")
    
    if not book_id or book_id not in textbook_metadata:
        return jsonify({"error": "Invalid or missing book ID"}), 400
    
    # Get book information
    book_info = textbook_metadata[book_id]
    
    # If a topic is specified, find relevant chunks
    relevant_chunks = []
    if topic:
        relevant_chunks = find_relevant_chunks(topic, book_id)
    
    # If no specific topic or no relevant chunks found, use a sample of chunks
    if not relevant_chunks:
        chunk_ids = [k for k in textbook_chunks.keys() if k.startswith(book_id)]
        # Take up to 3 chunks (beginning, middle, end) for variety
        sample_indices = [0]
        if len(chunk_ids) > 2:
            sample_indices.append(len(chunk_ids) // 2)
        if len(chunk_ids) > 1:
            sample_indices.append(len(chunk_ids) - 1)
        
        for idx in sample_indices:
            if idx < len(chunk_ids):
                chunk_id = chunk_ids[idx]
                relevant_chunks.append({
                    'id': chunk_id,
                    'text': textbook_chunks[chunk_id],
                    'score': 1.0
                })
    
    # Combine chunks for context
    context = "\n\n".join([chunk["text"] for chunk in relevant_chunks])
    
    # Generate quiz questions using the model
    prompt = (
        f"You are creating a quiz based on the following textbook content:\n\n"
        f"TEXTBOOK: {book_info['title']} by {book_info['author']}\n\n"
        f"CONTENT:\n{context}\n\n"
        f"Please generate {num_questions} multiple-choice questions based on this content."
        f"Each question should have 4 options (A, B, C, D) with one correct answer."
        f"For each question, also provide an explanation of why the correct answer is right.\n\n"
        f"Format each question as follows:\n"
        f"Question 1: [Question text]\n"
        f"A) [Option A]\n"
        f"B) [Option B]\n"
        f"C) [Option C]\n"
        f"D) [Option D]\n"
        f"Correct Answer: [Letter of correct answer]\n"
        f"Explanation: [Why this answer is correct]\n\n"
        f"IMPORTANT: Do not include any thinking process, planning, or reasoning in your response. Only include the final quiz questions, options, correct answers, and explanations.\n\n"
        f"QUIZ:"
    )
    
    outputs = llm(
        prompt,
        temperature=0.7,  # Higher temperature for more variety
        max_tokens=2048,
        top_p=0.9,
    )
    
    quiz_text = outputs["choices"][0]["text"].strip()
    

    quiz_text = re.sub(r'<think>.*?</think>', '', quiz_text, flags=re.DOTALL)
    quiz_text = re.sub(r'\[All questions and answers\].*?(?=Question)', '', quiz_text, flags=re.DOTALL)

    if "Question" in quiz_text:
        quiz_text = "Question" + quiz_text.split("Question", 1)[1]
    
    # Generate a unique ID for the quiz
    quiz_id = hashlib.md5(f"quiz_{book_id}_{topic}_{uuid.uuid4()}".encode()).hexdigest()
    
    # Store the quiz temporarily
    quiz_data = {
        'id': quiz_id,
        'book_id': book_id,
        'topic': topic,
        'num_questions': num_questions,
        'text': quiz_text,
        'book_title': book_info['title']
    }
    if not hasattr(app, 'quizzes'):
        app.quizzes = {}
    app.quizzes[quiz_id] = quiz_data
    
    return jsonify({
        "success": True,
        "quiz_id": quiz_id,
        "quiz_text": quiz_text
    })

@app.route("/export_quiz_pdf/<quiz_id>", methods=["GET"])
def export_quiz_pdf(quiz_id):
    if not hasattr(app, 'quizzes') or quiz_id not in app.quizzes:
        return jsonify({"error": "Quiz not found"}), 404
    quiz_data = app.quizzes[quiz_id]
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []
   
    title_text = f"Quiz on {quiz_data['book_title']}"
    if quiz_data['topic']:
        title_text += f" - Topic: {quiz_data['topic']}"
    elements.append(Paragraph(title_text, styles['Title']))
    
    
    elements.append(Paragraph(quiz_data['text'].replace('\n', '<br/>'), styles['Normal']))
    
    
    doc.build(elements)
    
    
    buffer.seek(0)
   
    return send_file(
        buffer,
        as_attachment=True,
        download_name=f"Quiz_{quiz_data['book_title'].replace(' ', '_')}.pdf",
        mimetype='application/pdf'
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)