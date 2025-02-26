from sentence_transformers import SentenceTransformer, util
from symspellpy import SymSpell
import pkg_resources
import sys

model = SentenceTransformer('all-MiniLM-L6-v2')

sym_spell = SymSpell()
dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

def correct_text(text):
    corrected = sym_spell.word_segmentation(text)
    return corrected.corrected_string

def check_similarity(answer_key, student_answer):
    fixed_key = correct_text(answer_key)
    fixed_student = correct_text(student_answer)

    # Generate embeddings
    embedding_key = model.encode(fixed_key, convert_to_tensor=True)
    embedding_student = model.encode(fixed_student, convert_to_tensor=True)

    # Compute cosine similarity
    similarity_score = util.pytorch_cos_sim(embedding_key, embedding_student).item()

    return similarity_score

if __name__ == "__main__":
    student_answer = sys.argv[1]
    answer_key = sys.argv[2]
    score = check_similarity(answer_key, student_answer)
    print("score:", score)