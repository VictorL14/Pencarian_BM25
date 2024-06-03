import re
import os
import pandas as pd 
import pdfplumber 
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory 
import nltk 
import numpy as np 
from sklearn.feature_extraction.text import TfidfVectorizer 
from collections import Counter

def preprocess_text(text):
    # 1. Case Folding
    text = text.lower()

    # 2. Tokenizing
    tokens = word_tokenize(text)

    # 3. Filtering Stopwords
    stop_words = set(stopwords.words('indonesian'))
    tokens = [word for word in tokens if word.lower() not in stop_words]

    # 4. Stemming
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    tokens = [stemmer.stem(word) for word in tokens]

    # Menggabungkan tokens kembali menjadi teks
    preprocessed_text = ' '.join(tokens)

    return preprocessed_text

def extract_abstract(pdf_file_path):
    with pdfplumber.open(pdf_file_path) as pdf:
        # Tambahkan teks dari halaman kedua
        second_page_text = pdf.pages[1].extract_text()
        second_page_text_lines = second_page_text.split('\n')
        second_page_text = '\n'.join(second_page_text_lines[2:])

        # Pendeteksian untuk mengecek apakah ada kop halaman atau nomor halaman pada baris terakhir halaman pertama
        first_page_text = pdf.pages[0].extract_text()
        first_page_text_lines = first_page_text.split('\n')
        # Hapus baris terakhir pada halaman pertama
        first_page_text = '\n'.join(first_page_text_lines[:-1])

        # Gabungkan teks dari kedua halaman
        combined_text = first_page_text + '\n' + second_page_text

        # Cari abstrak dalam bahasa Indonesia
        abstract_match = re.search(r'(?<=\bABSTRAK\b).*?\b(Kata\s?[kK]unci)', combined_text, re.DOTALL)
        abstract = abstract_match.group(0).strip() if abstract_match else ''

        # Bersihkan karakter '\n' dan '\r'
        abstract = abstract.replace('\n', ' ').replace('\r', '')

        # Hilangkan kata kunci dari abstrak
        abstract = re.sub(r'\b(Kata\s?[kK]unci)\b', '', abstract)

        # Preprocessing teks abstrak
        abstract = preprocess_text(abstract)

        return abstract
    
def extract_raw_title_from_pdf(pdf_file_path):
    raw_title = ""
    with pdfplumber.open(pdf_file_path) as pdf:
        first_page = pdf.pages[0]
        first_page_text = first_page.extract_text()

        # Ambil baris-baris dari halaman pertama
        lines = first_page_text.split('\n')

        # Flag untuk menunjukkan apakah kata dengan angka 1 tanpa spasi telah ditemukan
        found_flag = False

        # Ambil baris-baris yang memenuhi kriteria
        title_lines = []
        for line in lines:
            stripped_line = line.strip()

            # Cek apakah baris mengandung kata dengan angka 1 tanpa spasi
            if re.search(r'\b\w*1,|2,|1 ,|1\*|Institut|Universitas|[A-Za-z]+\d(?:,|\s*,|\s*\*,)?\w*\b', stripped_line):
                found_flag = True

            # Tambahkan baris ke dalam judul selama flag belum ditemukan
            if not found_flag:
                title_lines.append(stripped_line)

        # Gabungkan baris-baris menjadi satu teks
        raw_title = ' '.join(title_lines)

    return raw_title

def extract_title_from_pdf(pdf_file_path):
    title = ""
    with pdfplumber.open(pdf_file_path) as pdf:
        first_page = pdf.pages[0]
        first_page_text = first_page.extract_text()

        # Ambil baris-baris dari halaman pertama
        lines = first_page_text.split('\n')

        # Flag untuk menunjukkan apakah kata dengan angka 1 tanpa spasi telah ditemukan
        found_flag = False

        # Ambil baris-baris yang memenuhi kriteria
        title_lines = []
        for line in lines:
            stripped_line = line.strip()

            # Cek apakah baris mengandung kata dengan angka 1 tanpa spasi
            if re.search(r'\b\w*1,|2,|1 ,|1\*|Institut|Universitas|[A-Za-z]+\d(?:,|\s*,|\s*\*,)?\w*\b', stripped_line):
                found_flag = True

            # Tambahkan baris ke dalam judul selama flag belum ditemukan
            if not found_flag:
                title_lines.append(stripped_line)

        # Gabungkan baris-baris menjadi satu teks
        title = ' '.join(title_lines)

        # Preprocessing teks judul
        title = preprocess_text(title)

    return title


def extract_year(text):
    # Mencari tahun dalam format XXXX (misalnya: 2022)
    match = re.search(r'\b\d{4}\b', text)

    return int(match.group()) if match else None


def extract_year_from_pdf(pdf_file_path):
    with pdfplumber.open(pdf_file_path) as pdf:
        # Tambahkan teks dari baris pertama halaman kedua
        second_page_text = pdf.pages[1].extract_text()
        first_line_second_page = second_page_text.split('\n')[0]

        # Cari tahun dalam teks pada baris pertama halaman kedua
        year = extract_year(first_line_second_page)

        return year


def extract_authors_from_pdf(pdf_file_path):
    with pdfplumber.open(pdf_file_path) as pdf:
        first_page = pdf.pages[0]
        first_page_text = first_page.extract_text()

        # Ambil baris-baris dari halaman pertama
        lines = first_page_text.split('\n')

        # Variabel untuk menyimpan baris yang mengandung nama penulis
        authors_line = ""

        # Flag untuk menunjukkan apakah kata "universitas" atau "institut" ditemukan
        institute_flag = False

        # Loop melalui setiap baris dalam teks
        for line in lines:
            stripped_line = line.strip()

            # Cek apakah baris mengandung kata "universitas" atau "institut"
            if re.search(r'\b(Institut|Universitas|Jurusan|Teknik)\b', stripped_line):
                institute_flag = True

            # Cek apakah baris mengandung pola nama dengan angka yang tidak berurutan
            # atau angka 1 atau 2 diikuti oleh koma tanpa spasi
            if re.search(r'\b[A-Za-z]+\d(?:,|\s*,| \d*,|\s*\*,)', stripped_line) and not institute_flag:
                authors_line = stripped_line
                break

        # Bersihkan spasi di sekitar nama penulis
        cleaned_authors = [author.strip() for author in authors_line.split(',')]

    return cleaned_authors

#r'\b[A-Za-z]+\d(?:,|\s*,| \d*,|\s*\*,)'
# Fungsi untuk menjalani folder
def process_folders_in_directory(folder_path):
    pdf_info = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_file_path = os.path.join(root, file)
                abstract = extract_abstract(pdf_file_path)
                title = extract_title_from_pdf(pdf_file_path)
                
                raw_title = extract_raw_title_from_pdf(pdf_file_path)
                
                # Tambahan: ekstrak tahun dari PDF
                year = extract_year_from_pdf(pdf_file_path)

                # Ekstrak nama penulis dari judul artikel
                authors = extract_authors_from_pdf(pdf_file_path)

                # Rename the file by removing '.pdf' extension
                

                pdf_info.append((raw_title, title, abstract, year, authors))
                af = pd.DataFrame(pdf_info, columns=['judul_awal','judul', 'abstrak', 'tahun', 'nama_penulis'])
    return af

# def process_folders_in_directory(folder_path):
#     pdf_info = []

#     for root, dirs, files in os.walk(folder_path):
#         for file in files:
#             if file.lower().endswith('.pdf'):
#                 pdf_file_path = os.path.join(root, file)
#                 abstract = extract_abstract(pdf_file_path)
#                 title = extract_title_from_pdf(pdf_file_path)
                
#                 raw_title = extract_raw_title_from_pdf(pdf_file_path)
                
#                 # Tambahan: ekstrak tahun dari PDF
#                 year = extract_year_from_pdf(pdf_file_path)

#                 # Ekstrak nama penulis dari judul artikel
#                 authors = extract_authors_from_pdf(pdf_file_path)

#                 # Rename the file by removing '.pdf' extension
                

#                 pdf_info.append((raw_title, title, abstract, year, authors))
#                 af = pd.DataFrame(pdf_info, columns=['judul_awal','judul', 'abstrak', 'tahun', 'nama_penulis'])
                
#                 # Hapus baris duplikat dari DataFrame
#                 pdf_info_cleaned_no_duplicates = pdf_info.dropna(subset=['judul_awal']).drop_duplicates(subset=['judul_awal'], keep='first')
                
#                 # Simpan DataFrame ke dalam tabel MySQL
#                 try:
#                     pdf_info_cleaned_no_duplicates.to_sql(name='artikel', con=engine, if_exists='append', index=False)
#                     print("Data berhasil dimasukkan ke dalam tabel artikel")
#                 except Exception as e:
#                     print("Gagal memasukkan data ke dalam tabel artikel")
#                     print("Error:", e)
#     return pdf_info_cleaned_no_duplicates
