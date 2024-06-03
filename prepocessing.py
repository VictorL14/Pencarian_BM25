from gathering import *

def preprocess_query(query):
    # 1. Case Folding
    query = query.lower()

    # 2. Tokenizing
    tokens = word_tokenize(query)

    # 3. Filtering Stopwords (Opsional)
    stop_words = set(stopwords.words('indonesian'))
    tokens = [word for word in tokens if word.lower() not in stop_words]

    # 4. Stemming (Opsional)
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    tokens = [stemmer.stem(word) for word in tokens]

    # Menggabungkan tokens kembali menjadi teks
    preprocessed_query = ' '.join(tokens)

    return preprocessed_query

def calculate_idf(documents):
    # Inisialisasi objek TfidfVectorizer
    tfidf_vectorizer = TfidfVectorizer()

    # Melakukan transformasi TF-IDF pada dokumen
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

    # Mendapatkan nama fitur (kata) dari vektorizer
    feature_names = tfidf_vectorizer.get_feature_names_out()

    # Menghitung N (jumlah dokumen dalam korpus)
    N = tfidf_matrix.shape[0]

    # Menghitung df(qi) (frekuensi dokumen yang mengandung term qi)
    df_qi = np.array([(tfidf_matrix[:, i] > 0).sum() for i in range(tfidf_matrix.shape[1])])

    # Menghitung IDF menggunakan rumus yang diberikan
    idf_values = np.log((N - df_qi + 0.5) / (df_qi + 0.5))

    # Menyusun hasil ke dalam dictionary
    idf_dict = {feature_names[i]: idf_values[i] for i in range(len(feature_names))}

    return idf_dict

def calculate_term_frequency(documents):
    term_document_matrix = {}
    terms = set()

    # Hitung frekuensi term dan term tersebut termasuk dalam dokumen mana
    for doc_id, document in enumerate(documents):
        terms_in_document = set(document.split())
        for term in terms_in_document:
            if term not in term_document_matrix:
                term_document_matrix[term] = {}
            term_document_matrix[term][doc_id] = document.count(term)
            terms.add(term)

    # Menyusun hasil ke dalam dictionary
    term_frequency_dict = {}
    for term in terms:
        term_frequency_dict[term] = [term_document_matrix.get(term, {}).get(doc_id, 0) for doc_id in range(len(documents))]

    return term_frequency_dict