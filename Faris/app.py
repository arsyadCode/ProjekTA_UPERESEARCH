import re
import numpy as np
import csv
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
from nlp_id.lemmatizer import Lemmatizer
from flask import Flask, request, render_template, send_file
import pandas as pd

# Inisialisasi Flask
app = Flask(__name__)

# Fungsi untuk melakukan preprocessing pada teks
def preprocess_text(text):
    text = text.lower()  # Case folding
    text = re.sub(r"[^a-z0-9\s]", "", text)  # Hapus karakter non-alfanumerik
    
    lemmatizer = Lemmatizer()
    words = text.split()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    text = ' '.join(lemmatized_words)
    
    return text

# Fungsi untuk menghitung skor kesamaan
def grade_essay(jawaban_siswa, kunci_jawaban):
    jawaban_siswa = jawaban_siswa.decode('utf-8')
    preprocessed_jawaban_siswa = preprocess_text(jawaban_siswa)
    preprocessed_kunci_jawaban = preprocess_text(kunci_jawaban)
    
    tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-large-p2")
    model = AutoModel.from_pretrained("indobenchmark/indobert-large-p2")
    
    encoded_jawaban_siswa = tokenizer(preprocessed_jawaban_siswa, return_tensors="pt", padding=True, truncation=True)
    encoded_kunci_jawaban = tokenizer(preprocessed_kunci_jawaban, return_tensors="pt", padding=True, truncation=True)
    
    with torch.no_grad():
        jawaban_siswa_embeddings = model(**encoded_jawaban_siswa).last_hidden_state[:, 0, :].numpy()
        kunci_jawaban_embeddings = model(**encoded_kunci_jawaban).last_hidden_state[:, 0, :].numpy()

    similarity_score = cosine_similarity(jawaban_siswa_embeddings, kunci_jawaban_embeddings)[0][0]
    
    # Mengonversi nilai similarity ke dalam rentang 0-100
    similarity_percentage = min(100, max(0, round(similarity_score * 100)))
    
    return similarity_percentage

@app.route('/')
def dashboard():
    return render_template('dashboard.html')
@app.route('/grading', methods=['GET', 'POST'])
def index():
    result_data = []  # List untuk menyimpan informasi nama, kelas, dan nilai similarity

    if request.method == 'POST':
        kunci_jawaban = request.form['kunci_jawaban']
        
        # Mengecek apakah pengguna mengunggah file CSV
        if 'csv_file' in request.files:
            csv_file = request.files['csv_file']
            if csv_file.filename != '':
                # Membaca file CSV dengan pemisah titik koma (;)
                df = pd.read_csv(csv_file, delimiter=';')
                
                # Check if the "Jawaban Siswa" column exists in the DataFrame
                if 'Jawaban Siswa' in df.columns:
                    df['Jawaban Siswa'] = df['Jawaban Siswa'].fillna('')
                    
                    # Convert DataFrame column values to strings
                    jawaban_siswa_values = df['Jawaban Siswa'].astype(str).values
                    
                    # Iterate through each row in the DataFrame
                    for index, row in df.iterrows():
                        jawaban_siswa = jawaban_siswa_values[index]
                        # Konversi teks jawaban_siswa menjadi bytes
                        jawaban_siswa = jawaban_siswa.encode('utf-8')

                        # Hitung skor kesamaan
                        similarity_score = grade_essay(jawaban_siswa, kunci_jawaban)
                        
                        # Menambahkan informasi ke result_data
                        result_data.append({
                            'Nama Siswa': row['Nama Siswa'],  # Mengambil nama siswa dari kolom 'Nama Siswa'
                            'Kelas': row['Kelas'],  # Mengambil kelas dari kolom 'Kelas'
                            'Similarity Score': similarity_score
                        })
                else:
                    print("Column 'Jawaban Siswa' not found in the CSV file.")
        else:
            print("No CSV file uploaded.")
        
        # Jika tidak ada unggahan CSV atau tidak ada kolom 'jawaban', gunakan kunci jawaban manual
        if not result_data:
            kunci_jawaban_manual = request.form['kunci_jawaban']
            # Iterate through each row in the DataFrame
            for index, row in df.iterrows():
                jawaban_siswa = jawaban_siswa_values[index]
                similarity_score = grade_essay(jawaban_siswa, kunci_jawaban_manual)
            similarity_score = round(similarity_score * 100)

                
                # Menambahkan informasi ke result_data
            result_data.append({
                    'Nama Siswa': row['Nama Siswa'],  # Atur nama siswa manual sesuai kebutuhan
                    'Kelas': row['Kelas'],  # Atur kelas manual sesuai kebutuhan
                    'Similarity Score': similarity_score
                })
    
    return render_template('grading.html', result_data=result_data)

# Routing untuk mengunduh template CSV
@app.route('/download_template', methods=['GET'])
def download_template():
    # Data contoh yang akan ditulis ke template CSV
    sample_data = [
        {'Nama Siswa': 'Nama Siswa 1', 'Kelas': 'Kelas 1', 'Jawaban Siswa': 'Jawaban Siswa 1'},
        {'Nama Siswa': 'Nama Siswa 2', 'Kelas': 'Kelas 2', 'Jawaban Siswa': 'Jawaban Siswa 2'}
        # Tambahkan data contoh lainnya sesuai kebutuhan
    ]

    # Header CSV
    csv_header = ['Nama Siswa', 'Kelas', 'Jawaban Siswa']

    # Buat nama file CSV
    csv_filename = 'template.csv'

    # Buat file CSV dengan data contoh
    with open(csv_filename, 'w', newline='') as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=csv_header)
        csv_writer.writeheader()
        csv_writer.writerows(sample_data)

    # Kembalikan file CSV sebagai tautan unduhan
    return send_file(csv_filename, as_attachment=True)

# Menjalankan aplikasi Flask
if __name__ == '__main__':
    app.run(debug=True)
