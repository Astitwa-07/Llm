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




import PyPDF2
import fitz  # PyMuPDF
import tabula

def extract_text_from_pdf(pdf_file):
    text_content = ""
    with open(pdf_file, 'rb') as file:
        pdf_reader = PyPDF2.PdfFileReader(file)
        num_pages = pdf_reader.numPages

        for page_num in range(num_pages):
            page = pdf_reader.getPage(page_num)
            text_content += page.extractText()

    return text_content

def extract_images_from_pdf(pdf_file):
    image_count = 0
    doc = fitz.open(pdf_file)
    for page_num in range(doc.page_count):
        page = doc[page_num]
        image_list = page.get_images(full=True)
        for image_index, img in enumerate(image_list):
            image_bytes = img[0]
            image_format = img[1]
            image_file = f'image_{image_count}.{image_format}'
            with open(image_file, 'wb') as f:
                f.write(image_bytes)
            image_count += 1

    doc.close()

def extract_tables_from_pdf(pdf_file):
    tables = tabula.read_pdf(pdf_file, pages='all')
    table_data = "\n\n".join([table.to_string() for table in tables])
    return table_data

def save_to_text_file(content, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == "__main__":
    pdf_file = "your_pdf_file.pdf"
    text_content = extract_text_from_pdf(pdf_file)
    extract_images_from_pdf(pdf_file)
    table_data = extract_tables_from_pdf(pdf_file)

    output_file = "output.txt"
    save_to_text_file(text_content + "\n\n" + table_data, output_file)
    
