from rank import *

if __name__ == "__main__":
    folder_path = r'dataset\2021'
    pdf_info = process_folders_in_directory(folder_path)
    # Buat DataFrame
    

    query = ""
    while True:
        print("Sistem Rekomendasi Artikel")
        if not query:
            query = input("Masukkan query (atau ketik 'exit' untuk keluar): ")
            processed_query = preprocess_query(query)
            print("Hasil preprocessing query:", processed_query)
        else:
            print(f"Query yang digunakan: {query}")
            print("Menu:")
            print("1. Berdasarkan Judul")
            print("2. Berdasarkan Abstrak")
            print("3. Berdasarkan Judul dan Abstrak (Gabungan)")
            print("4. Reset Query")
            print("5. Keluar")
            menu_choice = input("Pilih menu (1/2/3/4/5): ")

            if menu_choice == '1':
                print("=====JUDUL=====")
                af_with_bm25 = calculate_bm25_judul(preprocess_query(query), pdf_info)
                for idx, bm25_score in af_with_bm25.items():
                    judul = pdf_info['judul_awal'].iloc[idx]
                    print(f"Judul: {judul}, BM25: {bm25_score}")

            elif menu_choice == '2':
                print("=====ABSTRAK=====")
                af_with_bm25 = calculate_bm25_abstrak(preprocess_query(query), pdf_info)
                for idx, bm25_score in af_with_bm25.items():
                    judul = pdf_info['judul_awal'].iloc[idx]
                    print(f"Judul: {judul}, BM25: {bm25_score}")

            elif menu_choice == '3':
                print("=====Judul_Abstrak=====")
                af_with_bm25 = calculate_bm25_judul_abstrak(preprocess_query(query), pdf_info)
                for idx, bm25_score in af_with_bm25.items():
                    judul = pdf_info['judul_awal'].iloc[idx]
                    print(f"Judul: {judul}, BM25: {bm25_score}")

            elif menu_choice == '4':
                query = ""
                continue

            elif menu_choice == '5':
                break

            else:
                print("Pilihan tidak valid. Silakan pilih menu yang tersedia.")

        if query.lower() == 'exit':
            break


# /: Rute ini dapat menampilkan halaman beranda atau formulir pencarian untuk pengguna memasukkan kueri.
# /search: Rute ini akan menerima kueri pencarian dari pengguna dan mengembalikan hasil rekomendasi berdasarkan kueri tersebut.
# /article/<article_id>: Rute ini akan menampilkan detail artikel berdasarkan ID artikel yang dipilih oleh pengguna.
# /about: Rute ini mungkin menampilkan informasi tentang sistem rekomendasi atau tentang situs web secara umum.