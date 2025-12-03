# TasteMatch Teknik Mimarisi ve Algoritma Detayları 🧠

Bu doküman, TasteMatch uygulamasının kalbinde yer alan öneri sisteminin **matematiksel ve algoritmik detaylarını** en ince ayrıntısına kadar açıklar. Hedef, yeni modeller geliştirmek isteyen mühendisler için tam bir referans sağlamaktır.

---

## 1. Veri Yapısı ve Matrissel Temsil

Öneri sistemimizin temeli, kullanıcıların filmlere verdiği oylardan oluşan devasa bir **User-Item Matrisi**ne ($R$) dayanır.

*   **Veri Seti:** MovieLens 32M (Eğitim için)
*   **Matris ($R$):**
    *   Satırlar: Kullanıcılar ($U$)
    *   Sütunlar: Filmler ($I$)
    *   Değerler ($r_{ui}$): Kullanıcının filme verdiği puan (0.0 - 5.0 arası).

```math
R_{m \times n} = \begin{pmatrix}
r_{11} & r_{12} & \cdots & r_{1n} \\
r_{21} & r_{22} & \cdots & r_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
r_{m1} & r_{m2} & \cdots & r_{mn}
\end{pmatrix}
```
*Bu matris %99 oranında boştur (sparse), çünkü kimse tüm filmleri izleyemez.*

---

## 2. Eğitim Aşaması: SVD (Matrix Factorization)

Modelimiz `project/src/train_fast.py` içinde eğitilen **Truncated SVD** (Singular Value Decomposition) modelidir. Amacımız, bu devasa seyrek matrisi, daha küçük boyutlu ve yoğun (dense) matrislerin çarpımı olarak ifade etmektir.

Bizim yaklaşımımızda SVD, matrisi şu şekilde parçalar:

$$R \approx U \cdot \Sigma \cdot V^T$$

Ancak biz `sklearn.decomposition.TruncatedSVD` kullanarak doğrudan boyut indirgeme yapıyoruz ve tekil değerleri ($ \Sigma $) matrislere yediriyoruz. Sonuçta elimizde şu kalıyor:

1.  **Item Latent Factors ($V$):** Her filmi $k$ boyutlu (bizde $k=64$) bir vektörle temsil eder.
    *   Bu matris (`svd_model.components_.T`), `fast_svd_model.pkl` içinde saklanır.
    *   **Boyut:** $(N_{movies} \times 64)$
    *   **Anlamı:** Her bir boyut, filmin soyut bir özelliğini (latent feature) temsil eder. Örneğin:
        *   Dimension 1: Aksiyon seviyesi
        *   Dimension 2: Romantik/Dram ekseni
        *   Dimension 3: Hedef kitle yaşı (vs. gibi, ancak matematiksel olarak soyuttur)

2.  **User Latent Factors ($U$):** Eğitim setindeki kullanıcıların zevk vektörleri.
    *   **ÖNEMLİ:** Biz bu $U$ matrisini **kullanmıyoruz**. Çünkü uygulamaya gelen kullanıcı (siz) eğitim setinde yoksunuz. Sizin vektörünüz "Anlık" olarak hesaplanmalı.

---

## 3. Inference (Gerçek Zamanlı Çıkarım) ve Vektör Uzayı

Uygulamada önerilerin nasıl üretildiğinin matematiksel ispatı şöyledir (`inference_service.py`):

### Adım 1: Item Vektör Uzayını Yükleme
Uygulama açıldığında, eğitilmiş **Item Factors ($V$)** matrisi belleğe yüklenir. Artık elimizde veritabanındaki her film için 64 boyutlu bir "Kimlik Kartı" (embedding) vardır.

### Adım 2: Dinamik Kullanıcı Vektörü ($u_{new}$) Hesabı
Kullanıcı uygulamada gezindikçe `swipes` tablosuna veriler düşer. Kullanıcının kimliği (User ID ve zevki) statik değildir, dinamiktir.

Kullanıcının beğendiği filmler kümesi $L_u = \{i_1, i_2, ..., i_k\}$ olsun. Her bir filmin vektörü de $v_{i}$ olsun.

Kullanıcının o anki zevk vektörü ($u_{vec}$), beğendiği filmlerin vektörlerinin **ağırlıksız ortalaması (centroid)** olarak hesaplanır:

$$u_{vec} = \frac{1}{|L_u|} \sum_{i \in L_u} v_{i}$$

Bu işlem `InferenceService` içinde şu kodla yapılır:
```python
# inference_service.py:67
for mid in liked_ml_ids:
    idx = self.mappings['movie2idx'][mid]
    user_vector += self.item_factors[idx] # Vektörleri topla
user_vector /= count # Ortalamasını al
```

**Neden böyle yapıyoruz?**
Bu yöntem (Average Embedding), kullanıcının 64 boyutlu uzaydaki "konumunu" belirler. Eğer sürekli Aksiyon filmlerini ($v_{action}$) beğenirseniz, ortalamanız da aksiyon kümesinin merkezine kayar.

### Adım 3: Benzerlik Skoru (Dot Product)
Kullanıcının konumu ($u_{vec}$) belirlendikten sonra, uzaydaki **diğer tüm filmlerle** olan yakınlığına bakılır. Bu yakınlık **Dot Product (İç Çarpım)** ile hesaplanır:

$$Score_{item} = u_{vec} \cdot v_{item}^T$$

Vektörler normalize edilmişse bu Cosine Similarity'dir, değilse büyüklük de skoru etkiler (popüler filmler genelde daha büyük vektör normuna sahip olabilir, bu da onları öne çıkarır).

Kod karşılığı:
```python
# inference_service.py:86
scores = np.dot(self.item_factors, user_vector)
```
Bu işlem sonucunda elimizde 90.000 film için 90.000 adet skor ($S$) oluşur.

### Adım 4: Filtreleme ve Sıralama
1.  $S$ vektörü büyükten küçüğe sıralanır (argsort).
2.  Kullanıcının zaten izlediği ($L_u$) filmler listeden çıkarılır.
3.  En üstteki $N$ film öneri olarak sunulur.

---

## 4. Match Logic: Perfect & Reverse

Sistemimizdeki "Match" mantığı tamamen bu skorlara ($S$) dayanır:

*   **Perfect Match ($Score > 0.75$):**
    *   Eğer $u_{vec} \cdot v_{item} > 0.75$ ise, bu iki vektör uzayda birbirine çok yakındır (açı neredeyse 0).
    *   Bu, kullanıcının geçmiş beğenileriyle bu filmin özelliklerinin matematiksel olarak "mükemmel" örtüştüğü anlamına gelir.

*   **Reverse Match (Keşif):**
    *   Bu, $u_{vec}$ ile **zıt yönlü** veya **ortogonal** (alakasız) vektörlerin seçilmesi işlemidir.
    *   Amaç, kullanıcıyı "Local Minima"dan (sürekli aynı tarz filmler döngüsü) kurtarmaktır.
    *   Şu anki implementasyonda: Rastgele seçimle simüle edilmektedir, ancak teknik olarak en düşük skorlu ($Score \approx 0$ veya negatif) filmler seçilmelidir.

---

## 5. Gelecek Modeller: NCF (Neural Collaborative Filtering)

Projede `ncf_model.pth` ve `project/src/models.py` içinde bir **Derin Öğrenme** modeli de mevcuttur.

**Farkı Nedir?**
SVD, lineer bir modeldir (Matris Çarpımı). Ancak kullanıcı-film ilişkileri bazen lineer değildir.
NCF modelimiz:
1.  User ve Item Embedding katmanları ile başlar.
2.  Bu vektörleri `concatenation` ile birleştirir.
3.  **MLP (Multi-Layer Perceptron)** katmanlarından geçirerek lineer olmayan ilişkileri öğrenir.
4.  Çıktı olarak 0-1 arası bir "Beğeni Olasılığı" verir.

**Nasıl Entegre Edilir?**
Şu an `InferenceService` SVD kullanıyor. NCF'i kullanmak için:
1.  `ncf_model.pth` yüklenmeli.
2.  `user_vector` hesabı yerine, `model(user_id, item_id)` şeklinde forward pass yapılmalı.
3.  Ancak NCF "yeni kullanıcı" (Cold Start) için anlık eğitilemez. NCF kullanmak için kullanıcının ID'sinin eğitim setinde olması gerekir. Bu yüzden hibrit bir yapı (SVD for features + MLP for scoring) daha ileri bir seviye olacaktır.

---

## Özet: Yeni Model Geliştirmek İçin

Yeni bir model yazacaksanız yapmanız gereken tek şey:
1.  **Girdi:** User-Item Interaction Matrix ($R$).
2.  **Çıktı:** Her film için, diğer filmlerle olan ilişkisini veya gizli özelliklerini anlatan bir **Item Embedding Matrix ($V$)** üretmek.
3.  **Inference:** Bu matrisi kullanarak, kullanıcının o anki beğenilerinden bir hedef vektör oluşturup en yakın komşuları (Nearest Neighbors) bulmak.
