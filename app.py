from flask import Flask, request, jsonify
import os
import subprocess
from werkzeug.utils import secure_filename
from flask_cors import CORS
import shutil
import asyncio
from pdf2image import convert_from_path
POPPLER_PATH = "/opt/homebrew/bin"
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
SEGMENTS_FOLDER = os.path.join(os.getcwd(), 'segments')
MODELS_FOLDER = os.path.join(os.getcwd(), 'models')

def runOCRPrintedModel(script, args):
    script_path = os.path.join(MODELS_FOLDER, script)
    absolute_args = [os.path.abspath(args[0])]
    
    process = subprocess.Popen(['python', script_path] + absolute_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()
    
    if process.returncode == 0:
        text = stdout.strip()
        return text[6:] if text[:6] == 'text: ' else ""
    else:
        print(f"Error in {script}: {stderr}")
        return None
    
def runHandwrittenModel(script, args):
    script_path = os.path.join(MODELS_FOLDER, script)
    absolute_args = [os.path.abspath(args[0])]
    
    process = subprocess.Popen(['python', script_path] + absolute_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()
    
    if process.returncode == 0:
        text = stdout.strip()
        return  text
    else:
        print(f"Error in {script}: {stderr}")
        return None

def runSimilarityCheckerModel(script, args):
    script_path = os.path.join(MODELS_FOLDER, script)
    process = subprocess.Popen(['python', script_path] + args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()
    
    if process.returncode == 0:
        text = stdout.strip()
        return text[7:-2] if text[:7] == 'score: ' else ""
    else:
        print(f"Error in {script}: {stderr}")
        return None
    
async def runPDFModel(script, args):
    script_path = os.path.join(MODELS_FOLDER, script)
    pdf_path = os.path.abspath(args[0])


    pdf_output_folder = os.path.join(os.getcwd(), 'PDF')
    os.makedirs(pdf_output_folder, exist_ok=True)

    images = convert_from_path(pdf_path, dpi=600, poppler_path=POPPLER_PATH)
    image_paths = []
    pages = [""] * len(images)
    
    for i, image in enumerate(images):
        image_filename = f"{os.path.splitext(os.path.basename(pdf_path))[0]}_page_{i}.png"
        image_path = os.path.join(pdf_output_folder, image_filename)
        image.save(image_path, "PNG")
        image_paths.append(image_path)
        pages[i]=runHandwrittenModel(script_path, [image_path])
    return ' '.join(pages)
    

@app.route('/process-images', methods=['POST'])
async def process_images():
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    if 'student_image' not in request.files or 'answer_key' not in request.files:
        return jsonify({'error': 'Please upload both student and answer key images.'}), 400
    
    student_image = request.files['student_image']
    answer_key = request.files['answer_key']

    student_image_filename = secure_filename(student_image.filename)
    answer_key_filename = secure_filename(answer_key.filename)

    student_image_path = os.path.join(UPLOAD_FOLDER, student_image_filename)
    answer_key_path = os.path.join(UPLOAD_FOLDER, answer_key_filename)
    student_image.save(student_image_path)
    answer_key.save(answer_key_path)
    
    print("Processing uploaded student and answer key images...")
    student_answer=""
    if student_image_path.lower().endswith('.pdf'):
        student_answer = await runPDFModel('handwritten_model.py', [student_image_path, 'PDF'])
    else:
        student_answer = runHandwrittenModel('handwritten_model.py', [student_image_path])
    key = runOCRPrintedModel('ocr_printed_model.py', [answer_key_path])
    print(student_answer, "         - student Answer")
    print(key, "         - Answer Key")
    if student_answer is None or key is None:
        return jsonify({'error': 'Error processing images.'}), 500
    
    score = runSimilarityCheckerModel('similarity_checker_model.py', [student_answer, key])
    print(score, "         - Score")
    if score is None:
        return jsonify({'error': 'Error comparing answers.'}), 500
    
    shutil.rmtree(UPLOAD_FOLDER, ignore_errors=True)
    shutil.rmtree(os.path.join(os.getcwd(), 'PDF'), ignore_errors=True)
    shutil.rmtree(SEGMENTS_FOLDER, ignore_errors=True)
    
    return jsonify({'data': {'score': score}})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
