import pdfplumber
from googletrans import Translator

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()

    return text

def translate_text(text, target_lang='en'):
    translator = Translator()
    max_chars_per_request = 5000  # Adjust the limit based on the translation API's requirements
    text_segments = [text[i:i + max_chars_per_request] for i in range(0, len(text), max_chars_per_request)]

    translated_text = ""
    for segment in text_segments:
        translated_text += translator.translate(segment, dest=target_lang).text

    return translated_text

# Replace 'path_to_pdf' with the actual path to your German PDF file.
pdf_path = r'C:\Users\W1QDP7L\Downloads\11-Z-11-07343-013-023 We-Bu-PST-Verschleiss Bu-PL26_We-14971 (1).pdf'
german_text = extract_text_from_pdf(pdf_path)
english_text = translate_text(german_text)

with open("C:\\Users\\W1QDP7L\\Documents\\New Text Document.txt", "w", encoding="utf-8") as f:
    print(english_text ,file=f)
