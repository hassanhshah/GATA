from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

syllabus = "" # Enter the path to the syllabus pdf file
tb = "" # Enter the path to the textbook pdf file
transcript = "" # Enter the path to the lecture video transcript folder
pdf_path = "" # Enter the path to the lecture slides folder

def extract_files(folder_path):
    files = []
    # Iterate through all the files and subdirectories in the given folder
    for root, _, filenames in os.walk(folder_path):
        for filename in filenames:
            file_path = os.path.join(root, filename)
            files.append(file_path)
    return files


def read_tb_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PdfReader(file)
        num_pages = len(reader.pages)
        text = ''
        for page_num in range(num_pages):
            page = reader.pages[page_num]
            text += str(page_num+1) + '\n' + page.extract_text() + '###'
    return text

def read_syll_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PdfReader(file)
        num_pages = len(reader.pages)
        text = ''
        for page_num in range(num_pages):
            page = reader.pages[page_num]
            text += page.extract_text() +'\n'
    return text

def read_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PdfReader(file)
        num_pages = len(reader.pages)
        text = ''
        for page_num in range(num_pages):
            page = reader.pages[page_num]
            text += pdf_path + '\n' + str(page_num+1) + '\n' + page.extract_text() + '###'
    return text

def extract_timestamps(filename):
    with open(filename, 'r', encoding='utf-8', errors='replace') as file:
        lines = file.readlines()

    # Group lines into chunks of 4 (each subtitle entry)
    chunks = [lines[i:i+4] for i in range(0, len(lines), 4)]
    grouped_chunks = [chunks[i:i+15] for i in range(0, len(chunks), 15)]  # Group chunks into groups of 15
    final_text = ''
    for group in grouped_chunks:
        # Check if the group has at least two elements
        if len(group) >= 2:
            first_timestamp = group[0][1].split(' --> ')[0] if len(group[0]) > 1 else 'Unknown'
            last_timestamp = group[-1][1].split(' --> ')[1] if len(group[-1]) > 1 else 'Unknown'

            text = ''.join(chunk[2] for chunk in group if len(chunk) > 2)
            final_text += f"\n{filename.replace('.txt', '')}\nFirst Timestamp: {first_timestamp} Last Timestamp: {last_timestamp}\nText: {text.strip()}\n###"
        else:
            # Handle the case where group is smaller than expected
            print(f"Warning: Skipping a group in {filename} as it's too small.")
        
    return final_text

def get_text_chunks_advanced(text):
    text_splitter = CharacterTextSplitter(separator="###", chunk_size=300, chunk_overlap=0)
    chunks = text_splitter.split_text(text)
    return chunks

def get_text_chunks_normal(text):
    text_splitter = CharacterTextSplitter(separator = '\n', chunk_size=400, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

ts = extract_files(transcript)
videos_text = ''
for t in ts:
    videos_text = videos_text + extract_timestamps(t)

pdfs = extract_files(pdf_path)
slides_text=''
for p in pdfs:
    slides_text = slides_text + read_pdf(p)

syllabus_text = read_syll_pdf(syllabus)
tb_text = read_tb_pdf(tb)

videos_chunks = get_text_chunks_advanced(videos_text)
slides_chunks = get_text_chunks_advanced(slides_text)
tb_chunks = get_text_chunks_advanced(tb_text)
syl_chunks = get_text_chunks_normal(syllabus_text)

v_store_vids = get_vectorstore(videos_chunks)
v_store_slides = get_vectorstore(slides_chunks)
v_store_tb = get_vectorstore(tb_chunks)
v_store_syl = get_vectorstore(syl_chunks)

v_store_vids.save_local("faiss_vids")
v_store_slides.save_local("faiss_slides")
v_store_tb.save_local("faiss_textbook")
v_store_syl.save_local("faiss_syllabus")


