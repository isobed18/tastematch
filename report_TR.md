# TasteMatch Öneri Sistemi Raporu 🎬

## 1. Veri Seti Analizi: MovieLens 32M

Sistemimizde, öneri sistemleri dünyasında altın standart olarak kabul edilen **MovieLens 32M** veri setini kullanıyoruz.

*   **Ölçek:**
    *   **32 Milyon Oy (Rating):** Kullanıcılar ve filmler arasındaki yoğun etkileşim matrisi.
    *   **200,948 Kullanıcı:** Kalabalık kitlelerin davranış kalıplarını öğrenmek için geniş bir havuz.
    *   **87,585 Film:** Çok geniş bir içerik kataloğu.
    *   **2 Milyon Etiket (Tag):** Kullanıcılar tarafından oluşturulmuş zengin meta veriler.
*   **Veri Noktaları:**
    *   **Oylar (Ratings):** 1.0 ile 5.0 arasında yıldızlar. (Uygulamamızda Swipe hareketleriyle eşleşir: Superlike=5, Like=4, Dislike=1-2).
    *   **Genom Skorları (Genome Scores):** Her filmin 1,128 benzersiz etiketle (örneğin "atmosferik", "bilim kurgu", "sürpriz son") ne kadar ilgili olduğunu gösteren yoğun bir matris. Bu, bizim için çok güçlü bir **İçerik Gömme (Content Embedding)** vektörüdür.
    *   **Metadata:** Görsel zenginlik ve metin analizi için bu veriyi **TMDB** (Posterler, Özetler) ile zenginleştiriyoruz.

---

## 2. Kullanıcı Verisi Edinimi (Swipe Mekanizması)

Uygulamamız, kullanıcı hareketlerini anlık eğitim sinyallerine dönüştüren bir veri toplama motoru gibi çalışır:

| Kullanıcı Hareketi | Ağırlık | Sinyal Anlamı |
| :--- | :--- | :--- |
| **Superlike** ⭐️ | **2.0** | Güçlü pozitif tercih. Eşleşme önceliği yüksek. |
| **Like** ❤️ | **1.0** | Pozitif tercih. Standart eğitim hedefi. |
| **Dislike** ❌ | **0.5** | Negatif tercih. Two-Tower modelinde "pozitif geçmiş" vektöründen **hariç tutulur** (kullanıcı profili kirlenmesin diye). Sıralama (Ranker) modelinde negatif örnek olarak kullanılır. |
| **Skip/Ignore** | **0.0** | Nötr veya pas geçme. Genellikle eğitimde yoksayılır. |

---

## 3. Uygulanan Mimariler

Basit sezgisel yöntemlerden Derin Öğrenmeye (Deep Learning) uzanan bir dizi gelişmiş teknik denedik.

### A. İşbirlikçi Filtreleme (Collaborative Filtering - CF) - `project/src/train_fast.py`
*   **Yöntem:** **SVD (Singular Value Decomposition)**.
*   **Mantık:** Etkileşim Matrisini iki düşük dereceli matrise (Kullanıcı Faktörleri × Film Faktörleri) ayırır.
*   **Artısı:** Çok hızlıdır, "bunu beğenenler şunu da beğendi" mantığını iyi yakalar.
*   **Eksisi:** Soğuk Başlangıç (Cold Start) sorunu vardır, yeni filmleri öneremez.

### B. İçerik Tabanlı Filtreleme (Content-Based Filtering - CBF) - `project/two_tower/preprocess_content.py`
*   **Yöntem:** **Vektör Benzerliği**.
*   **Mantık:**
    *   **Metin Vektörleri (Text Embeddings):** SBERT (`all-MiniLM-L6-v2`) kullanarak film özetlerini 384 boyutlu vektörlere dönüştürdük.
    *   **Genom Matrisi:** MovieLens'in sunduğu 1128 boyutlu etiket genomunu kullandık.
*   **Kullanımı:** NCF ve Two-Tower modellerinde filmleri daha iyi anlamak için kullanıldı. Hiç kimse oy vermese bile "benzer" filmleri önermemizi sağlar.

### C. Nöral İşbirlikçi Filtreleme (Neural Collaborative Filtering - NCF) - `project/ncf/`
*   **Yöntem:** **Hibrit MLP (Multi-Layer Perceptron)**.
*   **Mimari:**
    *   **Girdi:** Kullanıcı ID + Film ID + **Genom Vektörü**.
    *   **Katmanlar:** Bu girdileri birleştirip (concat), yoğun katmanlardan (Dense Layers) geçirir (örn. 256 -> 128 -> 64).
    *   **Çıktı:** Tahmini Puan (0.5 ile 5.0 arasına ölçeklenmiş Sigmoid çıktısı).
*   **Artısı:** Kullanıcı ve film arasındaki doğrusal olmayan (non-linear) karmaşık ilişkileri yakalar. Genom verisini kullanarak doğruluğu artırır.
*   **Eksisi:** Çıkarım (Inference) yavaştır; her kullanıcı-film çifti için tek tek hesaplama yapması gerekir.

### D. Faktörizasyon Makineleri (Factorization Machines - FM) - `project/fm/`
*   **Yöntem:** **LightFM**.
*   **Mantık:** Matris Faktörizasyonu ile Lineer Regresyonun birleşimidir. Hem ID'ler hem de özellikler (Türler, Etiketler) için vektörler öğrenir.
*   **Bias Yönetimi:** Kullanıcı ve Film yanlılıklarını (bias) açıkça modelleyebilir.
*   **Artısı:** Seyrek verilerde (sparse data) ve yan bilgilerle (metadata) çalışırken çok etkilidir.

### E. Two-Tower Mimarisi (Mevcut Durum) - `project/two_tower/`
*   **Yöntem:** **Bi-Encoder / Retrieval & Ranking**.
*   **Amaç:** 87 bin film arasından en alakalı 100 adayı milisaniyeler içinde bulmak (Retrieval).
*   **Mimari:**
    1.  **Kullanıcı Kulesi (User Tower):** Kullanıcı ID'sini ve **Etkileşim Geçmişini** (beğendiği filmler dizisi) alıp bir `Kullanıcı Vektörü`ne dönüştüren kodlayıcı.
    2.  **Film Kulesi (Item Tower):** Film ID + **Metin Vektörü** + **Genom** + **Türler** verisini alıp bir `Film Vektörü`ne dönüştüren kodlayıcı.
    3.  **Eğitim:**
        *   **InfoNCE Loss:** Kullanıcının vektörünü, beğendiği hedef filmin vektörüne yaklaştırırken, rastgele diğer filmlerden (negatif örnekler) uzaklaştırır.
    4.  **Çıkarım (Inference):**
        *   Tüm Film Vektörlerini önceden hesapla -> FAISS İndeksine koy.
        *   Kullanıcı Vektörünü hesapla -> En Yakın Komşu (Nearest Neighbor) araması yap.
*   **Durum / İyileştirmeler:** Şu anda ayarlanıyor. **Erken Durdurma (Early Stopping)**, **Dislike Filtreleme** ve ağırlıksız **Kayıp Loglaması (Loss Logging)** ekleyerek kararlılığı artırdık.

---

## 4. Gelecek Adımlar

1.  **Zaman Dinamiği (Temporal Dynamics):** Kullanıcı Kulesinde geçmişi işlerken, son beğenilere daha yüksek ağırlık vermek (Time Decay).
2.  **Zor Negatif Madenciliği (Hard Negative Mining):** Sıralama (Ranker) modelini, Retrieval modelinin *yanlışlıkla* getirdiği (yüksek puan verdiği ama kullanıcının aslında sevmediği) filmlerle eğitmek.
3.  **Oturum Bazlı RNN/GRU:** Geçmişi sadece ortalamak yerine, bir GRU (Gated Recurrent Unit) kullanarak beğenme *sırasını* ve örüntüsünü modellemek.
