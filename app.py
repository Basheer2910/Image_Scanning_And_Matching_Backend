from flask import Flask, request, jsonify
import os
import subprocess
from werkzeug.utils import secure_filename
from flask_cors import CORS
import shutil
import asyncio
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
        print(text, " printed text")
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
        return text
    else:
        print(f"Error in {script}: {stderr}")
        return None

def runSimilarityCheckerModel(script, args):
    print(args, " argssss")
    script_path = os.path.join(MODELS_FOLDER, script)
    print(['python', script_path] + args, " runninggggg")
    process = subprocess.Popen(['python', script_path] + args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()
    
    if process.returncode == 0:
        text = stdout.strip()
        return text[7:-2] if text[:7] == 'score: ' else ""
    else:
        print(f"Error in {script}: {stderr}")
        return None

@app.route('/process-images', methods=['POST'])
def process_images():
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    if 'student_image' not in request.files or 'answer_key' not in request.files:
        return jsonify({'error': 'Please upload both student and answer key images.'}), 400
    
    student_image = request.files['student_image']
    answer_key = request.files['answer_key']

    student_image_filename = secure_filename(student_image.filename)
    answer_key_filename = secure_filename(answer_key.filename)
    print(student_image_filename, answer_key_filename, "   images folder name")

    student_image_path = os.path.join(UPLOAD_FOLDER, student_image_filename)
    answer_key_path = os.path.join(UPLOAD_FOLDER, answer_key_filename)
    print(student_image_path, answer_key_path)
    student_image.save(student_image_path)
    answer_key.save(answer_key_path)
    
    print("Processing uploaded student and answer key images...")
    
    student_answer = runHandwrittenModel('handwritten_model.py', [student_image_path])
    key = runOCRPrintedModel('ocr_printed_model.py', [answer_key_path])
    print(student_answer, " student Answer")
    print(key, " keyyy")
    if student_answer is None or key is None:
        return jsonify({'error': 'Error processing images.'}), 500
    
    score = runSimilarityCheckerModel('similarity_checker_model.py', [student_answer, key])
    
    if score is None:
        return jsonify({'error': 'Error comparing answers.'}), 500
    
    shutil.rmtree(UPLOAD_FOLDER, ignore_errors=True)
    shutil.rmtree(SEGMENTS_FOLDER, ignore_errors=True)
    
    return jsonify({'data': {'score': score}})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
