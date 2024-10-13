from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import pytesseract
import cv2
import numpy as np
from pdf2image import convert_from_path
from docx import Document
import os
import openai
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

openai.api_key = "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"  # Replace with your OpenAI API key

# Function to extract text from images
def extract_text_from_image(image: np.ndarray) -> str:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return pytesseract.image_to_string(image).strip()

# Function to extract text from PDF files
def extract_text_from_pdf(file_path: str) -> str:
    images = convert_from_path(file_path)
    text = ""
    for image in images:
        text += pytesseract.image_to_string(np.array(image)) + "\n"
    return text.strip()

# Function to extract text from DOCX files
def extract_text_from_docx(file_path: str) -> str:
    doc = Document(file_path)
    return "\n".join(para.text for para in doc.paragraphs).strip()

# Function to get embedding for a given text
def get_embedding(text: str) -> list:
    response = openai.Embedding.create(
        model="text-embedding-ada-002",  # Change model if necessary
        input=text
    )
    return response['data'][0]['embedding']

@app.post("/upload/teacher")
async def upload_teacher_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        file_path = "teacher_answers.txt"
        with open(file_path, "wb") as f:
            f.write(contents)

        teacher_text = extract_text_from_docx(file_path) if file.filename.endswith('.docx') else extract_text_from_pdf(file_path)
        teacher_embedding = get_embedding(teacher_text)

        with open("teacher_embedding.npy", "wb") as f:
            np.save(f, np.array(teacher_embedding))

        return JSONResponse(content={"message": "Teacher's answers uploaded successfully."})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading teacher's file: {str(e)}")

@app.post("/upload/student")
async def upload_student_file(file: UploadFile = File(...)):
    try:
        # Save the uploaded student file
        contents = await file.read()
        file_path = f"student_answers.{file.filename.split('.')[-1]}"
        with open(file_path, "wb") as f:
            f.write(contents)

        # Extract student text based on file type
        student_text = ""
        if file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image = cv2.imread(file_path)
            student_text = extract_text_from_image(image)
        elif file.filename.lower().endswith('.pdf'):
            student_text = extract_text_from_pdf(file_path)
        elif file.filename.lower().endswith('.docx'):
            student_text = extract_text_from_docx(file_path)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format.")

        # Read teacher's correct answers
        teacher_text = ""
        teacher_file_path = "teacher_answers.txt"
        if os.path.exists(teacher_file_path):
            with open(teacher_file_path, "r", encoding='utf-8', errors='ignore') as f:
                teacher_text = f.read()
        else:
            raise HTTPException(status_code=404, detail="Teacher answers not found.")

        # Generate embeddings for comparison
        teacher_embedding = np.load("teacher_embedding.npy")
        student_embedding = get_embedding(student_text)

        # Calculate similarity score
        score = cosine_similarity([student_embedding], [teacher_embedding])[0][0] * 100

        # Provide feedback
        feedback = generate_feedback(student_text, teacher_text)  # Implement feedback generation logic

        return JSONResponse(content={"student_text": student_text, "score": score, "feedback": feedback})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing student file: {str(e)}")

def generate_feedback(student_text: str, teacher_text: str) -> str:
    # Implement a logic to generate feedback based on the differences
    return "Detailed feedback here based on differences."  # Placeholder
