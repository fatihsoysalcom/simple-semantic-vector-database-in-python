import math

# Yapay zeka modellerinin kelimeleri anlamlandırmak için kullandığı 
# çok boyutlu uzayı simüle eden basit bir kelime-vektör (embedding) sözlüğü.
# Boyutlar: [Hız/Performans, Taşıt/Ulaşım, Doğa/Gökyüzü]
SEMANTIC_SPACE = {
    "araba": [0.2, 0.9, 0.0],
    "otomobil": [0.25, 0.9, 0.0],
    "taşıt": [0.1, 0.85, 0.0],
    "hızlı": [0.9, 0.2, 0.0],
    "kırmızı": [0.1, 0.1, 0.1],
    "mavi": [0.0, 0.0, 0.4],
    "gökyüzü": [0.0, 0.0, 0.9],
    "bulut": [0.0, 0.0, 0.8],
    "uçak": [0.8, 0.7, 0.6],
    "deniz": [0.0, 0.1, 0.7]
}

def get_word_vector(word):
    """Kelimenin semantik uzaydaki koordinatını (vektörünü) döner."""
    clean_word = word.lower().strip(",.!?")
    return SEMANTIC_SPACE.get(clean_word, [0.0, 0.0, 0.0])

def get_sentence_embedding(sentence):
    """Cümledeki kelimelerin vektörlerinin ortalamasını alarak cümle vektörü üretir (Mean Pooling)."""
    words = sentence.split()
    vectors = [get_word_vector(w) for w in words]
    valid_vectors = [v for v in vectors if any(v)]
    
    if not valid_vectors:
        return [0.0, 0.0, 0.0]
    
    dim = len(valid_vectors[0])
    avg_vector = [0.0] * dim
    for v in valid_vectors:
        for i in range(dim):
            avg_vector[i] += v[i]
    for i in range(dim):
                avg_vector[i] /= len(valid_vectors)
    return avg_vector

def cosine_similarity(v1, v2):
    """İki vektör arasındaki kosinüs benzerliğini hesaplar (Açısal yakınlık)."""
    dot_product = sum(a * b for a, b in zip(v1, v2))
    norm_v1 = math.sqrt(sum(a * a for a in v1))
    norm_v2 = math.sqrt(sum(b * b for b in v2))
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return dot_product / (norm_v1 * norm_v2)

class SimpleVectorDatabase:
    """Yazıda anlatılan Vektör Veritabanı (Vector Database) mantığının çalışan en basit örneği."""
    def __init__(self):
        self.documents = {}
        self.vectors = {}

    def insert(self, doc_id, text):
        """Veriyi veritabanına ekler ve vektör koordinatlarını hesaplayıp saklar."""
        self.documents[doc_id] = text
        self.vectors[doc_id] = get_sentence_embedding(text)
        print(f"[Eklendi] ID: {doc_id} | Metin: '{text}' | Vektör: {[round(x, 2) for x in self.vectors[doc_id]]}")

    def search(self, query, k=2):
        """Gelen sorguyu vektöre çevirir ve en yakın k adet dökümanı bulur."""
        query_vector = get_sentence_embedding(query)
        results = []
        for doc_id, doc_vector in self.vectors.items():
            sim = cosine_similarity(query_vector, doc_vector)
            results.append((doc_id, self.documents[doc_id], sim))
        # Benzerlik skoruna göre azalan sırada sırala
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:k]

if __name__ == "__main__":
    print("--- VEKTÖR VERİTABANI SİMÜLASYONU ---\n")
    db = SimpleVectorDatabase()
    
    # Veritabanına dökümanları ekleyelim
    db.insert(1, "Hızlı kırmızı otomobil")
    db.insert(2, "Mavi gökyüzü ve bulutlar")
    db.insert(3, "Deniz kenarında bir taşıt")
    
    print("\n--- GELENEKSEL ARAMA VS VEKTÖREL ARAMA ---")
    query = "araba"
    print(f"Aranan Kelime: '{query}'\n")
    
    # 1. Geleneksel Kelime Eşleşmesi (Keyword Search)
    print("1. Geleneksel Kelime Eşleşmesi Sonuçları:")
    found_any = False
    for doc_id, text in db.documents.items():
        if query in text.lower():
            print(f"   -> Eşleşme Bulundu! ID: {doc_id} | Metin: '{text}'")
            found_any = True
    if not found_any:
        print("   -> [Hata] Hiçbir dökümanda birebir 'araba' kelimesi geçmiyor!")

    # 2. Vektörel / Semantik Arama (Vector Search)
    print("\n2. Vektör Veritabanı Semantik Arama Sonuçları:")
    search_results = db.search(query, k=2)
    for doc_id, text, score in search_results:
        print(f"   -> ID: {doc_id} | Metin: '{text}' | Benzerlik Skoru: {score:.4f}")

    print("\nÖzet: Geleneksel arama 'araba' kelimesini bulamadı çünkü veritabanında sadece 'otomobil' yazıyordu.")
    print("Vektör veritabanı ise 'araba' ve 'otomobil' kelimelerinin semantik olarak yakın olduğunu bildiği için doğru sonucu getirdi!")
