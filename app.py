from flask import Flask, request, jsonify
import os
import subprocess
from werkzeug.utils import secure_filename
from flask_cors import CORS
import shutil
from pdf2image import convert_from_path

POPPLER_PATH = None  # Set your poppler path if on Windows
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
SEGMENTS_FOLDER = os.path.join(os.getcwd(), 'segments')
MODELS_FOLDER = os.path.join(os.getcwd(), 'models')
PDF_FOLDER = os.path.join(os.getcwd(), 'PDF')

def run_model(script_name, args):
    script_path = os.path.join(MODELS_FOLDER, script_name)
    absolute_args = [os.path.abspath(arg) for arg in args]
    
    process = subprocess.Popen(['python3', script_path] + absolute_args,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()

    if process.returncode == 0:
        return stdout.strip()
    else:
        print(f"Error running {script_name}: {stderr}")
        return None

def extract_text_from_pdf(pdf_path, model_script):
    os.makedirs(PDF_FOLDER, exist_ok=True)
    images = convert_from_path(pdf_path, dpi=600, poppler_path=POPPLER_PATH)
    combined_text = []

    for i, image in enumerate(images):
        image_filename = f"{os.path.splitext(os.path.basename(pdf_path))[0]}_page_{i}.png"
        image_path = os.path.join(PDF_FOLDER, image_filename)
        image.save(image_path, "PNG")
        result = run_model(model_script, [image_path])
        if result:
            combined_text.append(result)

    return " ".join(combined_text)

def extract_text(file_path, is_student=True):
    if file_path.lower().endswith('.pdf'):
        return extract_text_from_pdf(file_path, 'handwritten_model.py' if is_student else 'ocr_printed_model.py')
    else:
        return run_model('handwritten_model.py' if is_student else 'ocr_printed_model.py', [file_path])

@app.route('/process-images', methods=['POST'])
def process_images():
    try:
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)

        if 'student_image' not in request.files or 'answer_key' not in request.files:
            return jsonify({'error': 'Please upload both student and answer key files.'}), 400

        student_file = request.files['student_image']
        key_file = request.files['answer_key']

        student_filename = secure_filename(student_file.filename)
        key_filename = secure_filename(key_file.filename)

        student_path = os.path.join(UPLOAD_FOLDER, student_filename)
        key_path = os.path.join(UPLOAD_FOLDER, key_filename)

        student_file.save(student_path)
        key_file.save(key_path)

        print("Extracting student answer...")
        student_answer = extract_text(student_path, is_student=True)

        print("Extracting answer key...")
        answer_key = extract_text(key_path, is_student=False)

        if not student_answer or not answer_key:
            return jsonify({'error': 'Failed to extract text from files.'}), 500

        print("Comparing answers...")
        score_output = run_model('similarity_checker_model.py', [student_answer, answer_key])
        if not score_output or not score_output.startswith("score:"):
            return jsonify({'error': 'Error comparing answers.'}), 500
        print("Score output:", score_output)
        score = int(score_output.split("**")[1].split('/')[0])  # Trimming format

        # Clean up
        shutil.rmtree(UPLOAD_FOLDER, ignore_errors=True)
        shutil.rmtree(PDF_FOLDER, ignore_errors=True)
        shutil.rmtree(SEGMENTS_FOLDER, ignore_errors=True)
        print('Answer Key: ', answer_key)
        print('Student Answer: ', student_answer)
        print("Score: ", score)
        return jsonify({
            'data': {
                'score': score,
                'student_answer': student_answer[6:],
                'answer_key': answer_key[6:]
            }
        })

    except Exception as e:
        print("Exception occurred:", e)
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
