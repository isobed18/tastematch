# TasteMatch Sistem Mimarisi ve Geliştirme Raporu

Bu belge, TasteMatch öneri sisteminin teknik mimarisini, veri işleme süreçlerini, model eğitim stratejilerini ve final prodüksiyon yapısını detaylandırmaktadır.

## 1. Proje Vizyonu ve Yaklaşım

TasteMatch, kullanıcıların film zevklerini öğrenerek (Swipe mekanizması ile) onlara kişiselleştirilmiş öneriler sunan bir hibrit öneri sistemidir.
**Temel Felsefe:** "Hızlı Aday Belirleme (Retrieval)" ve "Hassas Sıralama (Ranking)" olmak üzere iki aşamalı bir yapı kurarak hem performans hem de kaliteyi maksimize etmek.

## 2. Veri Seti ve Ön İşleme

Sistem, zengin metadata ve kullanıcı-içerik etkileşimleri için iki ana veri setini birleştirmiştir:

1.  **MovieLens 20M:**
    *   **Kullanım:** Kullanıcı puanları (Ratings), Etiketler (Tags) ve en önemlisi **Genome Scores**.
    *   **Genome Scores:** Filmlerin "Atmosferik", "Karanlık", "Komik", "Düşündürücü" gibi 1128 farklı nitelikteki genetik haritasını çıkarır. Bu, içeriğin derinlemesine anlaşılmasını sağlar.
2.  **TMDB (The Movie Database):**
    *   **Kullanım:** Film özetleri (Overview), Posterler ve Güncel Metadata.
    *   **Entegrasyon:** `links.csv` dosyası kullanılarak MovieLens ID'leri ile TMDB ID'leri eşleştirilmiştir.

### Ön İşleme Adımları (`project/two_tower/preprocess_content.py`)
*   **Metin Gömme (Text Embedding):** TMDB'den gelen film özetleri (Overview), `all-MiniLM-L6-v2` (SBERT) modeli ile 384 boyutlu vektörlere dönüştürüldü.
*   **Genome Matrisi:** MovieLens Genome skorları, her film için 1128 boyutlu yoğun (dense) bir vektör olarak `genome_matrix.npy` dosyasına kaydedildi.
*   **ID Eşleme:** Modelin eğitimi için tüm filmlere 0'dan başlayan ardışık `item_idx` atandı ve bu haritalar (`item_map.pkl`) saklandı.

## 3. Hibrit Model Mimarisi

Sistem iki ana modelden oluşur:

### A. Two-Tower Retrieval Model (Aday Belirleme)
*   **Amaç:** 50.000+ film arasından kullanıcıyla en alakalı 100 filmi milisaniyeler içinde bulmak.
*   **Mimari:**
    *   **User Tower:** Kullanıcının geçmişte beğendiği (Like/Superlike) filmlerin vektörlerinin ortalamasını alır (Mean Pooling).
    *   **Item Tower:** Filmin Metin Vektörü (384 dim) + Genome Vektörü (1128 dim) + Meta Veriler birleştirilerek (Fusion MLP) son bir Item Vektörü oluşturulur.
*   **Eğitim:**
    *   **Loss Function:** InfoNCE Loss (Contrastive Learning). Pozitif bir çift (User, Liked Item) ile batch içindeki diğer tüm filmleri (Negatives) kıyaslayarak aradaki açıyı minimize eder.
    *   **Latent Dimension:** 512 (Final vektör boyutu).
    *   **Checkpoint:** `runs_tt/run_20251217_012150`

### B. Neural Collaborative Filtering (NCF) Ranker (Sıralama)
*   **Amaç:** Two-Tower'dan gelen aday havuzunu (100 film), kullanıcının o filme vereceği olası puana (0.5 - 5.0) göre sıralamak.
*   **Mimari:**
    *   Kullanıcı ve Film ID'leri Embedding katmanlarından geçer.
    *   Ayrıca Filmin **Genome Vektörü** de ek girdi olarak verilir (Hybrid NCF).
    *   Multi-Layer Perceptron (MLP) katmanları ile final skoru üretilir.
*   **Checkpoint:** `runs_ncf/run_20251216_154202`

## 4. Backend ve Canlı Akış (Inference Pipeline)

Canlı sistem (`backend/app/inference_service.py` ve `feed.py`) şu adımları izler:

1.  **Tetikleyici:** Kullanıcı `/feed/` endpoint'ine istek atar.
2.  **Geçmiş Analizi:** Veritabanından kullanıcının beğendiği (Like/Superlike) filmlerin `ml_id` listesi çekilir.
3.  **Filtreleme Listesi:** Kullanıcının daha önce gördüğü (Seen) tüm filmler bir "yasaklı liste"ye eklenir (Infinite Feed mantığı).
4.  **Retrieval (Two-Tower):**
    *   Kullanıcı geçmişi `InferenceService`'e gönderilir.
    *   Two-Tower modeli anlık olarak bir Kullanıcı Vektörü oluşturur.
    *   FAISS benzeri bir işlemle (Matrix Multiplication) en yakın 100 film adayı bulunur.
5.  **Ranking (NCF):**
    *   Bu 100 aday, NCF modeline sokulur.
    *   Her birine 0.5 - 5.0 arası bir skor verilir.
6.  **Filtre & Fallback:**
    *   Yasaklı (Seen) filmler elenir.
    *   Kalan filmler Postgres veritabanında var mı diye kontrol edilir.
    *   Eğer liste boşalırsa (Cold Start veya Veri Eksikliği), veritabanından **Popüler** ve **Görülmemiş** filmler eklenerek liste tamamlanır.
7.  **Sunum:** En yüksek skorlu 10 film kullanıcıya json olarak döner.

## 5. Daily Recommendation (Günün Önerisi)

*   **Amaç:** Kullanıcıya her gün 1 adet "Yüksek Kaliteli" öneri sunmak.
*   **Mekanizma:**
    *   NCF Skoru **> 4.0** olan adaylar filtrelenir.
    *   Bu adaylar arasından **TMDB Puanı (Kalite)** en yüksek olan seçilir.
    *   Seçilen film `User` tablosunda `daily_match_ml_id` olarak önbelleğe alınır.
    *   Gün bitene kadar kullanıcıya aynı film gösterilir.
    *   Kullanıcı bu filme etkileşim verirse (Like/Dislike), bu verinin eğitim ağırlığı **5 kat** artırılır (High Priority Feedback).

## 6. Prodüksiyon Notları

*   **Global Init:** Modeller (`app.state.inference_service`) uygulama başlarken bir kere yüklenir ve RAM'de tutulur. Her istekte tekrar yüklenmez.
*   **Fallback:** DB veya Model hatası durumunda sistem asla boş dönmez, popüler içerikle besler.
*   **Git LFS:** Modeller büyük dosyalar (`.pth`) olduğu için Git LFS ile versiyonlanmıştır.

## 7. Geliştirme Yol Haritası (Next Steps)
*   **Online Learning:** Kullanıcı swipe yaptıkça Two-Tower user vektörünün anlık güncellenmesi (Şu an her istekte yeniden hesaplanıyor, bu da bir nevi real-time).
*   **Vektör Veritabanı:** 50k film için NumPy yeterli, ancak 1M+ film için ChromaDB veya Milvus entegrasyonu (kod altyapısı hazır).
