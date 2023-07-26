# Llm
import fitz
import pytesseract
from PIL import Image
from googletrans import Translator

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ''
    for page_num in range(doc.page_count):
        page = doc[page_num]
        text += page.get_text()
    doc.close()
    return text

def extract_text_from_image(image_path):
    return pytesseract.image_to_string(Image.open(image_path), lang='deu')

def translate_text(text, source_lang='de', target_lang='en'):
    translator = Translator()
    translated_text = translator.translate(text, src=source_lang, dest=target_lang)
    return translated_text.text

def convert_pdf_to_english(pdf_path):
    german_text = extract_text_from_pdf(pdf_path)
    english_text = translate_text(german_text)
    return english_text

if __name__ == "__main__":
    pdf_path = "path/to/your/pdf/file.pdf"
    english_text = convert_pdf_to_english(pdf_path)
    print(english_text)