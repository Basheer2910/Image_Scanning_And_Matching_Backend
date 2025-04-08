# import os
# import moondream as md
# from PIL import Image
# import sys

# # Set API key securely
# os.environ["MOONDREAM_API_KEY"] = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJrZXlfaWQiOiI2ZmJlZTFiZS05NTk3LTQ5MGQtOWExMC1hZGQzNDU3NjlmMDEiLCJpYXQiOjE3NDA4NDM4ODd9.EEG73m9h0MFZOF23otGPghwg-JvfADdqUCc0gAKZXNo"

# moondream_api_key = os.getenv("MOONDREAM_API_KEY")
# if not moondream_api_key:
#     raise ValueError("MOONDREAM_API_KEY is not set in environment variables.")

# # Initialize Moondream model
# model = md.vl(api_key=moondream_api_key)

# # Function to process the image with Moondream
# def extract_text(img, prompt):
#     buffer = ""
#     response = model.query(img, prompt, stream=True)
    
#     if "answer" not in response:
#         raise RuntimeError("Invalid response from Moondream API.")
    
#     for chunk in response["answer"]:
#         buffer += chunk
#     return buffer
from PIL import Image
import google.generativeai as genai
import sys

# Use environment variable in production!
genai.configure(api_key="")

# Use updated model
model = genai.GenerativeModel(model_name="gemini-1.5-flash")

def extract_text(image_path, prompt="Extract the text"):
    image = Image.open(image_path)
    response = model.generate_content([prompt, image])
    return response.text

if __name__ == "__main__":
    image_path = sys.argv[1]
    extracted_text = extract_text(image_path)
    print("Extracted Answer:\n", extracted_text)

