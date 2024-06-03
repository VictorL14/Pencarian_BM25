
from prepocessing import *

#JUDUL
def calculate_bm25_judul(processed_query, af, k1=1.5, b=0.75):
    # Hitung IDF dan term frequency
    word_freq = Counter(af['judul'].str.split().explode().dropna())
    idf_teks = calculate_idf(af['judul'])
    tfd = calculate_term_frequency(af['judul'])
    avg_dl_teks = sum(len(teks.split()) for teks in af['judul']) / len(af) 
    bm25_scores = {}  # Inisialisasi kamus BM25

    for i, teks in enumerate(af['judul']):
        bm25_judul = 0
        # Hitung panjang judul dari dokumen saat ini
        panjang_judul = len(teks.split())
        panjang_judul_dibagi_avg_dl = panjang_judul / avg_dl_teks
        for term in processed_query.split():
            if term in tfd and term in idf_teks:
                tf = tfd[term][i]  # Frekuensi term di dalam judul, default 0 jika tidak ada
                tf_array = np.array(tf)
                idf = idf_teks[term]  # IDF dari term, default 0 jika tidak ada
                bm25_judul += idf * (tf_array * (k1 + 1)) / (tf_array + (k1 * (1 - b + b)) * (panjang_judul_dibagi_avg_dl))
        bm25_scores[i] = bm25_judul  # Menambahkan skor BM25 ke dalam kamus
    
    return bm25_scores  # Mengembalikan kamus BM25

#abstrak
def calculate_bm25_abstrak(processed_query, af, k1=1.5, b=0.75):
    # Hitung IDF dan term frequency
    word_freq = Counter(af['abstrak'].str.split().explode().dropna())
    idf_teks = calculate_idf(af['abstrak'])
    tfd = calculate_term_frequency(af['abstrak'])
    avg_dl_teks = sum(len(teks.split()) for teks in af['abstrak']) / len(af) 
    bm25_scores = {}  # Inisialisasi kamus BM25

    for i, teks in enumerate(af['abstrak']):
        bm25_abstrak = 0
        # Hitung panjang abstrak dari dokumen saat ini
        panjang_abstrak = len(teks.split())
        panjang_abstrak_dibagi_avg_dl = panjang_abstrak / avg_dl_teks
        for term in processed_query.split():
            if term in tfd and term in idf_teks:
                tf = tfd[term][i]  # Frekuensi term di dalam abstrak, default 0 jika tidak ada
                tf_array = np.array(tf)
                idf = idf_teks[term]  # IDF dari term, default 0 jika tidak ada
                bm25_abstrak += idf * (tf_array * (k1 + 1)) / (tf_array + (k1 * (1 - b + b)) * (panjang_abstrak_dibagi_avg_dl))
        bm25_scores[i] = bm25_abstrak  # Menambahkan skor BM25 ke dalam kamus
    
    return bm25_scores  # Mengembalikan kamus BM25

#GABUNGAN JUDUL DAN abstrak
def calculate_bm25_judul_abstrak(processed_query, af, k1=1.5, b=0.75):
    judul_abstrak = af['judul'] + ' ' + af['abstrak']
    # Hitung IDF dan term frequency untuk judul
    word_freq = Counter(judul_abstrak.str.split().explode().dropna())
    idf_teks = calculate_idf(judul_abstrak)
    tfd = calculate_term_frequency(judul_abstrak)
    avg_dl_teks = sum(len(teks.split()) for teks in judul_abstrak) / len(af) 
    bm25_scores = {}  # Inisialisasi kamus BM25

    for i, teks in enumerate(judul_abstrak):
        bm25_judul_abstrak = 0
        # Hitung panjang judul_abstrak dari dokumen saat ini
        panjang_judul_abstrak = len(teks.split())
        panjang_judul_abstrak_dibagi_avg_dl = panjang_judul_abstrak / avg_dl_teks
        for term in processed_query.split():
            if term in tfd and term in idf_teks:
                tf = tfd[term][i]  # Frekuensi term di dalam judul_abstrak, default 0 jika tidak ada
                tf_array = np.array(tf)
                idf = idf_teks[term]  # IDF dari term, default 0 jika tidak ada
                bm25_judul_abstrak += idf * (tf_array * (k1 + 1)) / (tf_array + (k1 * (1 - b + b)) * (panjang_judul_abstrak_dibagi_avg_dl))
        bm25_scores[i] = bm25_judul_abstrak  # Menambahkan skor BM25 ke dalam kamus
    
    return bm25_scores  # Mengembalikan kamus BM25
