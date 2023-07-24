#!/usr/bin/env python
# coding: utf-8

# pip install PyMuPDF PyPDF2 pytesseract pillow
# 

# In[6]:


import fitz  # PyMuPDF
import PyPDF2
from PIL import Image
import pytesseract

# Function to extract text from images using OCR (Tesseract)
def extract_text_from_image(image_object):
    try:
        return pytesseract.image_to_string(image_object)
    except pytesseract.TesseractNotFoundError:
        raise Exception("Tesseract OCR is not installed or configured correctly.")

# Function to extract text from a PDF file (including text within images)
def extract_text_from_pdf(pdf_path):
    pdf_document = fitz.open(pdf_path)
    all_text = ""
    
    for page_number in range(pdf_document.page_count):
        page = pdf_document.load_page(page_number)
        
        # Extracting text from the page
        page_text = page.get_text("text")
        all_text += page_text + "\n"
        
        # Extracting text from images on the page
        for img_number, img in enumerate(page.get_images(full=True)):
            try:
                image_text = extract_text_from_image(img[0])
                all_text += "Image {} text: {}\n".format(img_number, image_text)
            except Exception as e:
                # Handle exceptions from OCR errors, if any
                print(f"Error while processing Image {img_number}: {e}")
    
    pdf_document.close()
    return all_text

# Function to extract links from a PDF file
def extract_links_from_pdf(pdf_path):
    links = []
    with open(pdf_path, "rb") as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page in pdf_reader.pages:
            annotations = page.get("/Annots")
            if annotations:
                for annotation in annotations:
                    if "/A" in annotation:
                        link = annotation["/A"]
                        if "/URI" in link:
                            links.append(link["/URI"])
    return links

# Example usage:
pdf_file_path = "C:/Users/Hrithik Kapil/Dropbox/My PC (LAPTOP-AKEUPEMO)/Downloads/RentReceipt.pdf"
text_with_images = extract_text_from_pdf(pdf_file_path)
links = extract_links_from_pdf(pdf_file_path)

print("Extracted Text (including text from images):\n", text_with_images)
print("Extracted Links:\n", links)


# In[ ]:




