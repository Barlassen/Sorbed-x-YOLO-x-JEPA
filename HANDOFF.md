# Devir Teslim — Sorbed × YOLO × JEPA

> Yeni bir sohbette / oturumda kaldığımız yerden devam etmek için özet.
> Son güncelleme: 2026-10-29.

## 🧭 BURADAN DEVAM (2026-10-29) — kısa durum

**Zincir:** maske 🔴 (bası yakın çekimi) → doku 🟡 → evre 🟡. Ürün hattı bağlı ama isteğe bağlı: README'deki 7
`SORBED_*` satırıyla açılıyor; varsayılan hâlâ classical + color_model.
- **Maske:** CWDB'de iyi (old-first 0.865 ≈ insan 0.862), bası yakın çekiminde zayıf (petchs açık yüzey 0.35).
  DeepLab-v4 bası fotoğrafında ~2× (kaynağı eğitimde olmayan 11'de 0.51 → 0.78) ama CWDB'de 0.78 (yara kenarındaki
  pembe epitel halkasını da boyuyor) → ön-kayıtlı ölçüt tutmadı, üretim old-first. DeepLab-v4 petchs train gördü →
  166 kaynaklı evreleme setinde kullanmak için kaynak-katlı yeniden eğitim gerekir.
- **Evre** (166 kaynak, kaynak-katlı CV, 4-sınıf dengeli doğruluk; taban "hep Evre 3" 0.25): A (düzeltilmiş kurallar)
  0.41, ürünün eritem eşiğiyle 0.34; GT bölgeyle bile 0.45; B (doğrudan evre CNN'i) 0.49, A ile fark anlamlı değil.
  Kurallar Evre 4 veremez (166'nın 36'sı), B 22/36'sını buluyor.
- **Ürünün gerçek çıktısı:** güvenlik kapılarıyla 166'nın 141'i belirsiz, 20 U, 5 Evre 3 → gösterilen evrenin dengeli
  doğruluğu ~0.05 (örneklem-içi). Kapı çoğunlukla haklı: 57 görüntüde maske yarayı ıskalıyor (Dice medyan 0.02).
- **İki darboğaz:** (1) kapsam = maske (kaç fotoğrafa evre verilebildiği); (2) doğruluk tavanı = kural ağacı (Evre 4
  yolu yok; mükemmel maskeyle bile ~0.45).
- **Yöntem uyarıları:** CWDB 27 ve petchs-test 26 dört kez "final" olarak kullanıldı (yıprandı, bağımsız test gerekli);
  petchs Roboflow kopya sızıntısı → yalnız `training/petchs_groups.py` katları; tek etiketleyici; pytest/hypothesis
  kurulu değil (testler yerel bir yedek çalıştırıcıyla koşuldu).
- **Sıradaki (öneri, kullanıcı onayı bekliyor):** DeepLab'i kaynak-katlı yeniden eğit → indirme gerektirmeyen maske
  fikirleri ön-kayıtla: doku CNN'inin epitel dediği kenarı kesmek, GrabCut ile kenarı renk sınırına oturtmak, iki modelin
  uyuşmasını güven puanı yapmak → sonra SAM (indirme onayı) → hemşire onaylı maske (Ario ile). Maskeden sonra A+B
  hibriti (Evre 4 için).

**Git:** `claude/sorbed-details-heuk41`; son push 463a6aa, sonrası push edilmedi.

### Önceki durum (2026-10-02, tarihsel)

**Zincir (o tarihte):** maske ✅ (CWDB Dice 0.865 ≈ insan 0.862; bası yakın çekim sorunu sonradan bulundu) → doku 🟡 → evre 🟡.
**Derinlik kolu kapandı** (tek fotoğraftan derinlik + WoundsDB fine-tune negatif; Evre 3↔4 için hemşire girişi
ya da çekim anında gerçek sensör). **Doku:** CWDB 27×4 cetvelinde (G / S / N):
renk yöntemi A 0.71 / 0.46 / **0.70**; çok kaynaklı CNN (DFUTissue + LUTSeg + WoundTissue 14) 0.63 / **0.57** / 0.30;
insan 0.70 / 0.64 / 0.83. G+S'de A ile CNN berabere; nekroz yalnız A'da güçlü → hibrit (CNN G/S + A nekroz).
G zaten insan tavanında; kalan pay S'de az, N'de büyük. 27 görüntüyle GA ≈ ±0.10 → küçük kazançlar ölçülemiyor.
**JEPA:** yöntem (etiketsiz ön-eğitim), model değil; bizim enkoder 2.95M parametreli ViT (224 px, 16 px yama, 196
yama × 192 boyut, 6 katman, 5 kanal). Dizüstü ölçeğinde rastgele başlangıcı geçmedi.

**Açık seçenekler (kullanıcı henüz seçmedi):**
1. **Sözde-etiket (B):** ~3.000 etiketsiz bası fotoğrafına hibrit taslak → CNN'i bunlarla yeniden eğit (~1–1.5 sa).
2. **Üçlü başlangıç karşılaştırması:** aynı doku eğitimi, başlangıç = COCO / bizim JEPA / rastgele (ViT tabanlı
   segmentasyon kurulumu gerekir, ~1–2 sa) → stajın "JEPA katkı sağlar mı" sorusunu doku için temiz kapatır.
3. **WoundTissue tam set** (147; bası + nekroz + kemik/tendon) için yazarlara (Kabir ve ark.) e-posta — kullanıcı/Ario.
4. **Uzman onaylı küçük bası seti** (A/hibrit taslağı → uzman düzeltir; hem cetveli büyütür hem eğitim verisi) — Ario.
5. Ario'ya iletilecek: derinlik sonucu, renk yönteminin CNN kadar iyi olduğu ve nekrozda üstün olduğu, test seti ihtiyacı.

**Git:** `claude/sorbed-details-heuk41`. Staj raporları (`STAJ_RAPORU_*.md/.docx`) bilerek git dışında
(`.git/info/exclude`); `.claude/settings.json` kişisel, commit edilmez.

## ÖN-KAYIT — katmanlı yara haritası v1 + kaynak-katlı DeepLab (2026-10-30, sonuçtan önce)
**Fikir** (kullanıcı; Ario'nun sıralı katmanlarıyla aynı): yara tek maske değil, eş yükselti eğrileri gibi iç içe
katmanlar. v1 yeni etiket istemez, iki mevcut modelden kurulur:
- **D (dış çizgi)** = DeepLab maskesi (eşik 0.5): açık yüzey + iyileşen kenar.
- **E (iyileşen kenar)** = D ∩ doku CNN'i "O" (epitel/diğer).
- **A (açık yara)** = D ∩ doku CNN'i {G, S, N}; içinde doku tipleri.
Doku CNN'i üründeki yolla uygulanır (`crop_resize` → `predict` → `paste`). En dış "kapalı, rengi değişmiş deri"
katmanı v1'de YOK: etiketi yalnız petchs Evre 1'de var (24 görüntü / 13 kaynak) → v2, uzman/hemşire etiketiyle.
**Etiket bulgusu (model çıktısı görülmeden):** petchs'te "evre çizgisi − doku poligonları" kapalı deri DEĞİL: Evre
2'nin %86'sında doku poligonu yok (açık ama doku çizilmemiş), Evre 3'te doku poligonları çizginin medyanda %31'ini
dolduruyor → petchs'ten kapalı-deri katmanı etiketi türetilemez.
**DeepLab kat modelleri** (`data/yolo_seg_v5_fold{k}` → `runs/wound_deeplab/fold{k}.pt`, k = 0, 1, 2):
- Tarif v4'ün aynısı: aynı kod, 25 epoch, AdamW 3e-4 OneCycle, CE + Dice, kırpma 0.45–1, seed 0, eşik 0.5.
- Tek fark petchs kısmı: üç bölümün tümü (train/valid/test), doku poligonu bölgesiyle (v4'le aynı "açık yüzey"
  tanımı; dokusuz görüntüler atlanır). k katının kaynakları tamamen dışarıda.
- Epoch seçimi: kalan kaynakların her 6.'sı (doku CNN'i katlarıyla aynı `kept[5::6]` listesi) + Figshare 20
  (v4 doğrulamasıyla aynı).
- Diğer kaynaklar (FUSeg 300, CO2 300, DFUT 80, segulcer 195) v4'teki dosyaların aynısı. Yakın-kırpmalar
  (1 kopya, seed 0) yeniden üretilir.
- Sızıntı kontrolü: dışarıda tutulan kat ↔ eğitim görüntüleri, dönüş ve çevirme dahil dHash (≤ 6); eşleşen eğitim
  görüntüsü atılır. Kırpılmış kopyaları dHash yakalamaz (sınırlama).
**Değerlendirme** (birim: petchs kaynak başına 1 görüntü, n = 166; her görüntü kendi katının DeepLab'i ve doku
CNN'iyle; `training/eval_layered_map.py`):
- **M1 (birincil, maske):** açık yüzey Dice. GT = petchs doku poligonları; dokusuz kaynaklar hariç. Karşılaştırma:
  old-first, D ve A. Ayrıca ıskalama oranı (Dice < 0.1). Kaynak üzerinden eşli bootstrap, %95 GA.
- **M2 (ikincil, yıpranmış cetvel):** CWDB 27 konsensüs Dice. old-first (0.865), DeepLab-v4 D (0.782), v4 ×
  `tissue_cnn_final` A. Hiçbiri CWDB görmedi.
- **M3 (evre, ürün; karar vermez):** adım-2 A kural motoru, ürün eşikleriyle (eritem 0.15). Bölge = D; doku
  oranları D içinde, ürünle aynı.
  - Kapısız: 4-sınıf dengeli doğruluk.
  - Kapılı: seg güveni = D içindeki ortalama ön-plan olasılığı. Bu tanımla 0.2 kapısı pratikte yalnız boş maskede
    tetiklenir; bunu bilerek ölçüyoruz. Güvenin anlamlılığı (maske Dice'ıyla Spearman) raporlanır; bu adımda eşik
    ayarı YOK.
  - Raporlanan: belirsiz sayısı ve gösterilen evrenin dengeli doğruluğu. old-first ile karşılaştırılır (141
    belirsiz, ~0.05).
**Karar kuralı (maske):** üretim maskesi old-first → katmanlı DeepLab ⇔ M1'de A − old-first > 0 ve GA 0'ı içermiyor
VE M2'de A ≥ 0.835 (old-first'ten en fazla 0.03 düşük; yıpranmış cetvel, yalnız koruma). Ürün için DeepLab aynı
tarifle tüm petchs kaynaklarıyla yeniden eğitilir.
**Sınırlamalar:** tek etiketleyici; petchs doku poligonları açık yüzeyin bir kısmını çiziyor (Dice, çizilmemiş açık
tabanı boyayan maskeyi cezalandırır); CWDB 5. kez kullanılıyor.
**SONUÇ M2 (CWDB 27; `python -m training.eval_layered_map --cwdb --viz` → `runs/layered/m2_cwdb.json`):** old-first
0.865, D 0.782 (iki sayı da eskiyle aynı), **A 0.827**. A − D = +0.045 [0.013, 0.087] → pembe kenarı ayrı katman
yapmak anlamlı iyileşme. A − old-first = −0.038 [−0.075, −0.002]. Koruma eşiği 0.835 TUTMADI (az farkla) → karar
kuralına göre ürün maskesi bu turda değişmez, M1 ne çıkarsa çıksın. Gerçek taşmalar düzeldi: #3 0.33 → 0.79,
#18 0.69 → 0.84. Kullanıcının örneği #9 0.38 → 0.60; D'nin %71'i pembe katman. En büyük kayıp #26 (0.94 → 0.84): CNN
eskarın bir kısmını "O" dedi. Görseller `runs/audit/layered/` (hasta fotoğrafı, yerel).
DeepLab kat eğitimleri 2026-10-30 17:54'te başladı (sırayla, epoch başına ~4–5 dk → ~5.5 sa); log
`logs/wound_deeplab_fold{k}.log`. Bitince: `python -m training.eval_layered_map --petchs` (M1 + M3).

## 🔌 ÜRÜNE BAĞLAMA (2026-10-28) — YOLO old-first maske + CNN doku arka ucu

**Ne yapıldı:** `segmentation_backend=yolo` artık yedek modeli (old-first), güven eşiğini ve "en büyük parça"
filtresini ayardan alıyor (`yolo_seg_*`); retina maskeleri tam çözünürlükte. Yeni `tissue_backend=cnn`
(`src/sorbed/tissue/cnn_model.py`): CNN ham fotoğraftan doku haritası, maroon/eritem renk modelinden
(normalize görüntü) — değerlendirmedeki yolun aynısı. Doku ağırlığının SHA-256'sı provenance'a yazılıyor.
Varsayılanlar ağırlıksız kaldı (classical + color_model; testler ve çevrimdışı çalışma için); doğrulanmış
yapılandırma README'deki 7 `SORBED_*` satırı. Ürün doku modeli: `runs/tissue_cnn_final/seed0.pt` (aynı tarif,
tüm petchs kaynakları, `--petchs-holdout-fold -1`; iç doğrulama Dice 0.761 @30, katlar 0.70–0.74).
**Sadakat kontrolü** (`training/check_product_pipeline.py`, 166 kaynak, her görüntüye kendi katının doku CNN'i):
maske 166/166 birebir aynı, doku oranları farkı 0, kural evresi (seg güveni 1 iken) %100 aynı → bağlantı doğru.
**Bulgu — güvenlik kapıları:** gerçek seg güveniyle ürün 141/166 görüntüde "belirsiz" gösteriyor:
(1) seg güveni < 0.2 → INDETERMINATE kuralı: 123/166 (YOLO güveni = maske pikselleri altındaki tespit skoru
ortalaması; medyan 0.087); (2) güveni ≥ 0.2 olan 43'te kurallar 21 kez Evre 3 diyor, 16'sı "derinliğe dayalı
evre, güven < 0.5" çekimserliğiyle gizleniyor. YOLO güveni anlamlı: maske–anotasyon sınırı Dice'ıyla Spearman
0.45; güven < 0.05 olan 57 görüntüde medyan Dice 0.02 (maske yarayı ıskalıyor), 0.05–0.5 aralığında ~0.3,
> 0.5'te 0.60. **Düzeltmeler:** (a) adım 2'deki "maske maliyeti küçük" seg güveni 1 varsayımıyla ölçülmüştü;
kapı açıkken maske asıl darboğaz (166'nın 57'sinde yara bulunamıyor). (b) Ürün eritem 0.15'te → beklenen ürün
skoru 0.41 değil ~0.34 (iç içe CV 0.336; bu 166'da örneklem-içi 0.349).
**Açık karar:** kapıyı gevşetmek (veriye bakarak eşik seçmek) yerine kök neden: bası fotoğraflarında maske.
DeepLab-v4 petchs açık yüzeyde 0.667 (old-first 0.346; temiz alt küme 0.783 vs 0.514).

## ⚠️ petchs SIZINTISI (2026-10-28) — Roboflow kopyaları bölümler arası
petchs'te Roboflow, artırılmış kopyaları (çevirme, döndürme, kırpma, renk) BÖLMEDEN ÖNCE üretmiş: dosya adındaki
kaynak numarası (`20009_jpg.rf.<hash>`) aynı fotoğrafı gösteriyor (`runs/audit/petchs_split_leak.png`). 310 görüntü
= 166 kaynak; train∩test 17, train∩valid 17, valid∩test 6 kaynak. dHash (döndürme/kırpma görmez) yalnız 2'sini
yakalamıştı. Etkilenen sonuçlar: (a) CNN-P/PS'nin petchs-test doku skorları (0.91–0.93; zaten "şişik" denmişti),
(b) maske v4'te YOLO-v4/DeepLab-v4'ün petchs-test skoru (petchs train'le eğitildiler) — ama kaynağı eğitimde
OLMAYAN 11 görüntüde de: eski 0.514, v1 0.533, YOLO-v4 0.733, **DeepLab-v4 0.783** → ilerleme gerçek (bu 11 daha
kolay; sızıntılı 15'te sırasıyla 0.22/0.41/0.48/0.58). Üretim kararı (CWDB) etkilenmez. Evreleme adım 1 (eğitimsiz
kurallar) sonucu değişmez (sızıntı yalnız iyimser yönde). Bundan sonra petchs = `training/petchs_groups.py`
(kaynağa göre katlar).

## ÖN-KAYIT — evreleme adım 2: A (düzeltilmiş kurallar) ve B (doğrudan evre CNN'i) (2026-10-28, sonuçtan önce)
**Veri ve katlar:** petchs 310 görüntü / 166 kaynak, kaynağa göre 3 kat (evreye göre tabakalı; `petchs_groups`).
Bir kaynağın tüm kopyaları aynı katta. Her kat için yalnız diğer iki katla eğitim/ayar; o kat tahmin edilir (OOF).
**Değerlendirme birimi:** kaynak başına 1 görüntü (alfabetik ilk) → n=166 (1:13, 2:23, 3:77, 4:36, U:17); bootstrap
kaynak üzerinden. Önceki petchs valid/test ayrımı kullanılmaz.
**Doku (A için):** CNN-PS tarifi her kat için yeniden eğitilir (`--petchs-holdout-fold k`: petchs'in o kattaki
kaynakları tamamen dışarıda, iç doğrulama = kalan petchs kaynaklarının her 6.'sı + eski DFUT/LUTSeg iç val). Bir
görüntünün doku haritası, onu görmemiş kat modelinden gelir.
**Maske (A için):** old-first (üretim; petchs görmedi → sızıntısız). DeepLab-v4 petchs train gördüğü için kullanılmaz.
**A — düzeltilmiş kural motoru:** ürün kodunda (1) kural boşluğu kapatılır: açık yara + taban örtülü değil
(slough+eskar < U eşiği) + slough+eskar ≥ `devitalized_max_stage2` → Evre 3 (NPIAP: Evre 3'te slough/eskar görünebilir;
Evre 2'de olmaz); (2) granülasyon eşiği `Thresholds`'a taşınır. Ayarlanan 5 eşik: U eşiği, granülasyon (Evre 3),
slough+eskar (Evre 2/3 sınırı), "deri sağlam" açık-yatak eşiği, eritem (Evre 1). Izgara araması diğer iki katın OOF
özelliklerinde, amaç 4-sınıf dengeli doğruluk {1, 2, 3/4, U} (kurallar 4'ü veremez); eşitlikte varsayılana en yakın.
Ek (ikincil): Sorbed'in GBM başlığı aynı özelliklerle (iç içe CV). Ayrıca karşılaştırma: düzeltilmemiş motor (adım
1 kuralları) aynı OOF özelliklerle; A'nın R1 (GT bölge) hali ayrıştırma için.
**B — doğrudan evre CNN'i:** tüm fotoğraf (maske yok), MobileNetV3 (DeepLab COCO omurgası; ikinci başlangıç: o katın
doku-CNN omurgası), 5 sınıf, ağırlıklı CE, AdamW 3e-4 kosinüs 30 epoch, artırma (kırpma 0.5–1, çevirme, rot90,
parlaklık; renk tonu yok), epoch seçimi eğitim katlarının kaynak-ayrık iç doğrulamasında (her 6. kaynak) dengeli
doğrulukla; seed 0.
**Ölçüler:** 5-sınıf doğruluk ve dengeli doğruluk, **4-sınıf dengeli doğruluk (birincil; 3 ve 4 birleşik)**,
tam/kısmi kat, karışıklık matrisi, %95 GA; A−B ve A−düzeltilmemiş farkı için eşli bootstrap. Taban: hep "Evre 3".
**Karar kuralı:** A'nın eşikleri (tüm OOF'ta ayarlanmış hali) + boşluk düzeltmesi ürüne girer ⇔ A'nın birincil
ölçüsü düzeltilmemiş motordan yüksek VE fark GA'sı 0'ı içermiyor. B ürüne girmez (gerekçe veremez); B−A farkı
"kuralların neyi kaçırdığını" raporlar. Sınırlama: tek etiketleyici; 3 vs 4 fotoğraftan tanım gereği sınırlı.
**SONUÇ** (`python -m training.eval_staging_cv` → `runs/staging/step2.json`; n=166 kaynak, 1:13 2:23 3:77 4:36 U:17;
kural kopyası gerçek motorla her görüntüde eşleşti — assert). 4-sınıf dengeli doğruluk [%95 GA] (birincil) · 5-sınıf
doğruluk · tam/kısmi kat · evresiz:
| | 4-sınıf dengeli | 5-sınıf doğr. | tam/kısmi | evresiz |
|---|---|---|---|---|
| hep "Evre 3" | 0.25 | 0.46 | 0.78 | 0 |
| adım-1 motoru (düzeltilmemiş) × old-first | 0.29 [0.22,0.35] | 0.35 | 0.69 | 0.17 |
| A0 boşluk düzeltmesi, varsayılan eşik | 0.29 | 0.36 | 0.70 | 0.16 |
| **A** düzeltilmiş + ayarlı eşik × old-first | **0.41** [0.32,0.50] | 0.36 | 0.74 | 0.08 |
| A eşikleri × R1 GT bölge (ayrıştırma) | 0.45 [0.36,0.53] | 0.28 | 0.83 | 0.01 |
| Sorbed GBM başlığı (ikincil) | 0.31 | 0.40 | 0.74 | 0 |
| **B-coco** doğrudan evre CNN'i | **0.49** [0.38,0.59] | **0.55** | 0.75 | 0 |
| B-tissue | 0.47 [0.37,0.56] | 0.54 | 0.79 | 0 |
**Karar:** A − adım-1 = +0.126 [0.041, 0.222] → ölçüt tuttu → boşluk düzeltmesi + eşikler ürüne girdi: U eşiği
0.5→0.3, granülasyon (Evre 3) 0.15→0.3, slough+eskar Evre 2/3 sınırı 0.1→0.05 (katlarda eşikler kararlı:
u 0.3 her katta, g 0.2–0.3, d 0.05–0.1). **Ön-kayıttan sapma (güvenlik):** ayar eritem eşiğini 0'a indirdi (kapalı
deri = Evre 1); petchs'te hasarsız deri yok → bu veriyle savunulamaz, ürün 0.15'te tuttu (eritem 0.15 sabitken iç içe
CV'de A = 0.336). B − A = +0.073 [−0.04, 0.19] (B-tissue +0.055 [−0.04, 0.16]) → anlamlı fark YOK.
Ayrıştırma: maske maliyeti küçük (GT bölge 0.45 vs old-first 0.41). A0 = adım-1 → boşluk düzeltmesi CNN dokusuyla
tek başına etkisiz (renk modelinde 42/62 belirsizin kaynağıydı); kazanç eşiklerden.
Hata yapısı: A Evre 4'ü veremez (36'nın 24'ü →3, 11'i →U), Evre 3'ün 24/77'sini U yapıyor, Evre 2'nin 11/23'ünü 3.
B Evre 4'ün 22/36'sını doğru buluyor (kurallar 0) ama U'da zayıf (6/17; A 12/17) → tamamlayıcılar.
**Ürün uyarısı:** ürün hattı hâlâ `segmentation_backend=classical` + `tissue_backend=color_model`; eşikler CNN dokusu +
old-first maskeyle ayarlandı. Ürünün evre vermesi için CNN doku arka ucu + YOLO old-first ürüne bağlanmalı
(→ yapıldı, bkz. "ÜRÜNE BAĞLAMA"; varsayılan hâlâ ağırlıksız, açmak README'de).
Testler: pytest/hypothesis .venv'de yok (kurulmadı); test_staging 11/11, test_report_pdf 3, test_feedback 2 doğrudan
çalıştırıldı, uçtan uca analyze_image çalışıyor.

## ÖN-KAYIT — evreleme adım 1: mevcut Sorbed motorunun ölçümü (2026-10-28, model çıktısı görülmeden)
**Soru:** `src/sorbed/staging` kural motoru (varsayılan eşikler, HİÇ değiştirilmeden) bası fotoğraflarında evreyi
ne kadar doğru veriyor; hata bölgeden (maske) mi, dokudan mı, kurallardan mı geliyor?
**Etiket:** petchs (Roboflow, CC BY 4.0, tek etiketleyici) evre poligonları. Görüntünün evresi = en yüksek evre
(çok-evreli görüntülerin hepsinde poligonlar üst üste → tek yaranın bölgeleri); `unstageable` varsa (evre 4 yoksa) U.
**Değerlendirme seti (birincil):** petchs valid + test (test kopyası olan valid görüntüleri hariç); test ayrıca
raporlanır. Not: valid, DeepLab-v4 ve CNN-PS için epoch seçiminde kullanıldı (eğitimde değil) → hafif iyimser.
**İkincil:** petchs train (test kopyaları hariç) yalnız sızıntısız kombinasyonlarda (renk modeli × {GT bölge, old-first}).
**Kombinasyonlar:** bölge {R1 = GT evre çizgisi ∪ doku poligonları, R2 = old-first (üretim), R2 = DeepLab-v4} ×
doku {Sorbed renk modeli (üretim; gray-world normalize), CNN-PS seed0 (G/S/N/O → granülasyon/slough/eskar/epitel)}.
Renk ipuçları (maroon, eritem) her iki doku kolunda da renk modelinden. seg_confidence = 1 (kural mantığı ölçülür,
güven kapısı değil); çıktı = motorun `rule_stage`'i (kural yoksa "belirsiz").
**Ölçüler:** 5 sınıf (1, 2, 3, 4, U) doğruluk + dengeli doğruluk (sınıf başına doğru oranlarının ortalaması) +
karışıklık matrisi; ikili "tam kat (3/4/U) vs kısmi kat (1/2)" doğruluğu. DTI/belirsiz çıktılar yanlış sayılır, oranı
ayrıca. Önemsiz taban: hep "Evre 3". Görüntü bootstrap %95 GA.
**Bu adım karar vermez** (geçme çizgisi yok): evre hatasını hangi halkanın sürüklediğini gösterir. Adım 2'de (kural
eşikleri / Sorbed'in öğrenilmiş GBM başlığı) her ayar YALNIZ petchs train'de yapılır; test bir kez daha ölçülür.
**Beklenti** (yalnız etiket istatistiğinden, model çıktısı görülmeden): motor Evre 4'ü veremez (kas/tendon/kemik
sınıfı üreten doku modeli yok); granülasyonu Evre 3'e bağlıyor, oysa petchs'te granülasyon Evre 4'ün %98'inde, Evre
3'ün %44'ünde; U'da slough medyanı ~%40 < motorun %50 eşiği.
**SONUÇ — mevcut motor bası fotoğraflarında kullanılamaz** (`runs/staging/step1.json`). valid+test n=62 (1:4, 2:8,
3:36, 4:10, U:4). Doğruluk / dengeli doğruluk [%95 GA] / tam-kısmi kat / evresiz oranı:
| | doğruluk | dengeli | tam/kısmi | evresiz |
|---|---|---|---|---|
| hep "Evre 3" (önemsiz) | 0.58 | 0.20 | 0.81 | 0 |
| R1 GT bölge × renk (üretim dokusu) | 0.02 | 0.01 | 0.11 | **0.85** |
| R1 GT bölge × CNN-PS | 0.29 | 0.23 | 0.81 | 0.02 |
| R2 old-first × renk (= üretim) | 0.07 | 0.07 | 0.27 | 0.68 |
| R2 old-first × CNN-PS | **0.45** [0.32,0.58] | **0.29** [0.16,0.36] | 0.76 | 0.10 |
| R2 DeepLab-v4 × renk | 0.05 | 0.02 | 0.18 | 0.81 |
| R2 DeepLab-v4 × CNN-PS | 0.24 | 0.22 | 0.79 | 0.03 |
Test (31) aynı tablo; train (245, sızıntısız renk kolları) doğruluk 0.04/0.06 → aynı tablo. Hiçbir kombinasyon
önemsiz "hep 3" doğruluğunu geçmiyor; motor 1, 2 ve 4'ü (neredeyse) hiç vermiyor.
**Neden (keşif, sonuçtan sonra):** (1) **Kural ağacında boşluk:** açık yara + granülasyon < %15 + slough/eskar
%10–50 → hiçbir kural tetiklenmiyor ("belirsiz"); renk kolunda 62'nin 42'si bu. NPIAP'e göre görünür ama tabanı
örtmeyen slough/eskar Evre 3 ile uyumlu → kural eksik. (2) **Renk modeli bu fotoğraflarda eskarı abartıyor**
(R1 medyan: eskar %27, granülasyon %1 — gölge/koyu piksel = eskar). (3) **Evre 4'e yol yok:** kas/tendon/kemik
üreten doku modeli yok (NPIAP gereği de fotoğraftan zor). (4) **Evre 1/2 kuralları** eritem/epitel ipuçlarına
dayanıyor; bunlar ~0 çıkıyor → CNN kolunda 1/2 → 3 ya da U. (5) CNN-PS + GT bölge 36 Evre 3'ün 21'ini U yapıyor
(GT evre çizgisi çevre deriyi de içeriyor → slough oranı/eşik etkileşimi).
Not: petchs tek etiketleyici, etiket kalitesi denetlenmedi; Evre 3 vs 4 ayrımı fotoğraftan tanım gereği sınırlı.

## ÖN-KAYIT — maske v4 programı (2026-10-21, her şeyden önce; kullanıcı onayladı)
**Tanım:** yara = AÇIK YÜZEY (CWDB ile aynı). **Bası doğrulama seti (seçim yalnız buna göre):** petchs valid doku
poligonları (24) + Figshare açık ülser (20; Ulceration ∪ doku, Re-ep hariç) = 44. **Final test (bir kez):** CWDB 27
(konsensüs), petchs test doku GT (26), DFUT Test yara yatağı (16). Başarı: CWDB ≥ 0.85 VE petchs-test ≥ 0.75 → üretim.
**Eğitim verisi:** FUSeg train (rastgele 300, seed 0), CO2 train (rastgele 300), DFUT yatak (80), petchs train doku
(184), Segmentation-ulcer train+valid (tüm poligonlar = yara; 195) + yakın-kırpma 1 kopya. Hariç: FUSeg val, CWDB,
petchs test, segulcer test.
**YOLO-v4:** v1 best.pt'den; AdamW lr0 2e-4, warmup 0, cos_lr, 25 epoch, erken durma kapalı, her epoch kaydı; epoch
seçimi bası doğrulamada Dice ile (conf 0.01, tüm parçalar). Sonra çıkarım kuralı ({0.01,0.1,0.25} × {hepsi, top1})
yine bası doğrulamada seçilir.
**DeepLab-maske:** DeepLabV3-MobileNetV3 (COCO), 2 sınıf, aynı eğitim verisi, tam görüntü 512 letterbox, 25 epoch,
epoch seçimi bası doğrulamada Dice; eşik 0.5 (değiştirilmez).
Final: old-first vs YOLO-v4 vs DeepLab (+ v1). Eğitim kaybının düştüğü her eğitimde kontrol edilir.
**Ara sonuç (seçimler, yalnız bası doğrulamada):** YOLO-v4 train kaybı düzenli düştü (seg 2.87→2.16), val kaybı
4.18→3.50. Epoch Dice (conf 0.01, hepsi): v1 0.627, eski 0.480, v4 epoch 1 0.663 … **epoch 19 0.692 (seçildi;
`yolo_v4/weights/epoch18.pt`, 0-tabanlı)**; `runs/segment/yolo_v4_epoch_dice.json`. Çıkarım kuralı: conf 0.01 hepsi 0.692,
top1 0.665; **conf 0.1 hepsi 0.719 (seçildi)**, top1 0.646; conf 0.25 0.663/0.619. Medyan 11.5 parça/görüntü.
DeepLab-v4: train kaybı 0.83→0.35 düzenli düştü; bası doğrulama Dice 0.63 → **0.750 (epoch 19, seçildi)**
(`runs/wound_deeplab/v4.pt`, log `logs/wound_deeplab_v4.log`).
**SONUÇ — BAŞARISIZ (ölçüt tutmadı), ama büyük ilerleme** (`python -m training.eval_wound_masks`,
`runs/segment/mask_v4_final.json`). Dice ort/medyan (isabet · kapsama · alan, medyan):
| | CWDB 27 | petchs-test açık yüzey 26 | DFUT Test yatak 16 |
|---|---|---|---|
| eski-önce (üretim) | **0.865**/0.913 (0.93·0.97·1.04×) | 0.346/0.278 (0.39·0.32·1.00×) | 0.859/0.928 |
| yalnız v1 | 0.747/0.837 (0.73·1.00·1.36×) | 0.463/0.464 (0.37·0.95·2.44×) | 0.509/0.475 |
| YOLO-v4 | 0.786/0.860 (0.81·1.00·1.23×) | 0.589/0.653 (0.70·0.75·1.19×) | **0.933**/0.948 |
| DeepLab-v4 | 0.782/0.876 (0.79·1.00·1.26×) | **0.667**/0.695 (0.68·0.85·1.28×) | 0.855/0.932 |
İkisi de CWDB ≥ 0.85'i tutmuyor, petchs-test ≥ 0.75'i de tutmuyor → **üretim kuralı DEĞİŞMEDİ (old-first)**.
Kazanç: bası açık yüzeyde 0.35 → 0.67 (DeepLab; ~2×), YOLO-v4 DFUT'ta 0.93 (v1'in kallus taşması bitti), eğitim
ilk kez gerçekten öğrendi (teşhis 1 doğrulandı). Kayıp: CWDB'de yeni modeller eski modelden ~0.08 düşük.
Keşif (sonuçtan sonra bakıldı; `runs/audit/mask_v4_cwdb_misses.png`): CWDB kaybı 4 görüntüde (#3, #9, #18, #27;
YOLO'da +#20). Yeni modeller kaçırmıyor (kapsama 1.00), **taşıyor**: uzmanlar yalnız çiğ açık yüzeyi dar çiziyor
(#9'da ayrı adacıklar), modeller yara kenarındaki pembe epitel/maserasyon halkasını da dahil ediyor. Yani kalan
fark çoğunlukla "yara sınırı nereden geçer" tanımı; eğitim etiketleri (petchs doku, segulcer, FUSeg) CWDB'den gevşek.
Keşif 2 (2026-10-29, sonuçtan sonra, kanıt değil; tek tek görüntüler `runs/audit/cwdb_masks/index.html`): kayıpların
çoğunda uzmanlar da anlaşamıyor. #7 uzman çifti Dice ort. 0.28 (2 uzman büyük kırmızı alanı çizmiş, 2'si küçük bir nokta;
3/4 konsensüs noktayı tutuyor; DeepLab uzman 4'e 0.86), #9 0.48 (DeepLab uzman 3'e 0.77), #27 0.54 (DeepLab uzman 1'e
0.69). Uzmanların anlaştığı #3 ve #18'de DeepLab gerçekten taşıyor (#18 alanı 1.9×, pembe halka). İleride ölçüt
olarak "konsensüs" yerine ör. "en yakın uzman" ya da görüntü başına insan tavanı düşünülebilir; ancak önceden kayda yazılarak.

## 🔎 MASKE HATTI TEŞHİS RAPORU (2026-10-20)
1. **Eğitim neden "öğrenmeden" duruyor:** optimizer=auto → AdamW lr0 0.002 (v1'in en iyi epoch'undaki lr 0.0004'ün
   5×), warmup 3 epoch lr'yi 0.0007→0.0018'e çıkarıyor; lr arttıkça TRAIN kaybı da artıyor (v2 box 1.13→1.25, seg
   1.99→2.06) = adım fazla büyük. Epoch 1 gerçekten iyileştiriyor (v1 eğitimsiz val fitness 0.894 → v2 ep1 0.958;
   v3 val 0.638 → 0.686), sonra bozuluyor; patience 8, lr yeniden düşmeden eğitimi kesiyor.
2. **Modeller kendinden emin değil; conf 0.01 + tüm parçaların birleşimi bir koltuk değneği:** petchs dev'de v3
   medyan 45 parça/görüntü; conf 0.25'te v3 0.77→0.21, v1 0.71→0.48. CWDB'de tek parça (top1) v3'ü 0.75→0.82, v1'i
   0.75→0.78 yapıyor → CWDB taşmasının bir kısmı düşük güvenli fazladan parçalar (cetvel, çevre deri).
3. **Veri bileşimi:** v1 eğitiminde HİÇ bası yarası yok (FUSeg ayak 786, CO2 lepra 460, DFUT ayak 80); v3'te bası
   %12. Val (model seçimi) %76 CO2 lepra → seçim bası yarasına göre değil. FUSeg orijinallerinde yara karenin
   medyan %1'i. Not: v1 yakın-kırpma 2541 (≈2 kopya), v2/v3 1446 (1 kopya) → "aynı tarif" birebir değildi.
4. **Etiket tanımları tutarsız:** DFUT (eski: kallus dahil), petchs evre çizgisi (lezyon alanı) vs doku (açık
   yüzey), CWDB açık yüzey → model kaynağa göre farklı "yara" öğreniyor.
5. **Ölçüm kırılgan:** eski FUSeg modeli v2/v3 val'de mask mAP50 0.17–0.19, petchs'te 0.22–0.36 ama CWDB 27'de
   0.865 ve DFUT yatakta 0.859 (ikisi de AZH/FUSeg'e benzer çekim) → maske kararları 27 görüntülük tek cetvele aşırı
   bağlı.
6. **Yapısal:** YOLO örnek-segmentasyonu (kutu + parça) "tek bölge" olan yara için zorlanıyor: kareyi dolduran yarada
   kutu = tüm kare, yara çok parçaya bölünüyor. Piksel-sınıflandırma (semantik; doku CNN'iyle aynı DeepLab) daha
   uygun olabilir.
Öneri sırası: (a) bası maske doğrulama seti (petchs train/valid doku + Figshare; seçim buna göre), (b) eğitim:
lr ~1e-4, warmup 0, kosinüs, patience ≥15, train kaybının düştüğünü kontrol, (c) tek "açık yüzey" tanımı + bası payını
artırma, (d) conf/birleştirme kuralını gel. setinde seçme, (e) semantik maske modeli (DeepLab) karşılaştırması.

## ÖN-KAYIT — segmenter v3 (2026-10-20, eğitimden önce; kullanıcı onayladı)
v2 ile TEK fark: petchs yara bölgesi = yalnız doku poligonları (granulation ∪ slough ∪ eschar = açık yüzey); evre
çizgisinin geri kalanı (kapalı renk değişmiş deri) arka plan. Doku poligonu olmayan petchs görüntüleri (yalnız evre
çizgisi; Evre 1 vb.) DIŞLANIR — negatif örnek yapılmaz, çünkü doku çizilmemiş sığ açık yara içerebilirler. Geri
kalan her şey v2 ile aynı (veri, hariç tutulanlar, yakın-kırpma, combined_v1 best.pt'den, 20 epoch, patience 8,
seçim best.pt). Başarı ölçütü: CWDB ≥ 0.85 VE petchs-test ≥ 0.75, petchs-test GT = **yalnız doku poligonları
(açık yüzey)** — öğretilen tanımla tutarlı olsun diye (sonuç görülmeden, yazım sırasında düzeltildi). Evre çizgisi
∪ doku GT'si ikincil olarak raporlanır. Diğer setler (Figshare, DFUT yatak) v2 ile aynı.
**SONUÇ — BAŞARISIZ** (`combined_v3/weights/best.pt` = yine epoch 1, erken durma epoch 9; `runs/segment/combined_v3_eval.json`).
Dice ort/medyan — eski-önce | v1 | v2 | **v3**: CWDB 0.865 | 0.747 | 0.670 | **0.749**/0.831 · petchs açık yüzey
(birincil) 0.346 | 0.463 | 0.469 | **0.526**/0.509 · petchs lezyon (ikincil) 0.417 | 0.759 | 0.846 | 0.774 · Figshare
0.830 | 0.883 | 0.857 | **0.893** · DFUT yatak 0.859 | 0.509 | 0.618 | **0.850**/0.932.
Teşhis: CWDB'de v3 = v1 (isabet 0.73, alan 1.36× — aynen) → **v2 ve v3'te en iyi epoch 1: ek eğitim neredeyse hiç
işlemedi** (optimizer=auto → SGD lr 0.01 + warmup; sonraki epoch'larda val düşüyor, seçim ilk epoch'u tutuyor).
Tek belirgin kazanç DFUT yatak 0.51→0.85: kallussuz etiket kısa eğitimde bile v1'in kallus taşmasını gideriyor.
petchs açık yüzeyde TÜM modeller ~2.3× alan çiziyor (isabet ≈0.35) → ya modeller kapalı renk değişmiş deriyi de
boyuyor ya da petchs doku poligonları açık alanın tamamını kapsamıyor; ayırt edilemedi.
Üretim kuralı DEĞİŞMEDİ (old-first). Sonraki aday: eğitim yöntemini düzeltmek (düşük lr ince ayar, ör. AdamW
lr 1e-4 / SGD lr 0.001, warmup 0, daha uzun sabır), yine ön-kayıtla.

## ÖN-KAYIT — segmenter v2 (2026-10-17, eğitimden önce; kullanıcı onayladı)
Veri: combined_v1 tarifinin aynısı (FUSeg train, CO2Wounds train/val, DFUTissue TrainVal) **iki değişiklikle**:
(a) DFUTissue seg etiketi yalnız sınıf 1–2 (fibrin+granülasyon = yara yatağı; kallus 3 hariç), (b) + petchs train
(test kopyaları hariç) yara bölgesi = evre çizgisi ∪ doku poligonları; petchs valid → val. Hariç (sızıntı):
FUSeg val, CWDB 27, petchs test. Yakın-kırpma artırması aynı ayarla. Eğitim: combined_v1 best.pt'den fine-tune,
imgsz 640, 20 epoch, patience 8, mps; seçim val (best.pt). Ölçüm (tek sefer): "yalnız v2" maskesi CWDB (konsensüs),
petchs-test, Figshare (gel.-2) ve DFUT Test (yalnız yara yatağı GT) Dice; karşılaştırma old-first ve yalnız v1.
Başarı ölçütü (önceden): CWDB ≥ 0.85 VE petchs-test ≥ 0.75 → üretim kuralı "yalnız v2" olur; değilse olmaz.
**SONUÇ — BAŞARISIZ** (`runs/segment/runs/segment/combined_v2/weights/best.pt` = epoch 1 (YOLO fitness), erken durma
epoch 9; `runs/segment/combined_v2_eval.json`). Dice ort/medyan:
| | CWDB 27 | petchs-test 31 | Figshare 20 | DFUT Test yatak 16 |
|---|---|---|---|---|
| eski-önce (üretim) | **0.865**/0.913 | 0.359/0.353 | 0.830/0.908 | **0.859**/0.928 |
| yalnız v1 | 0.747/0.837 | 0.766/0.815 | **0.883**/0.941 | 0.509/0.475 |
| yalnız v2 | 0.670/0.792 | **0.828**/0.912 | 0.857/0.930 | 0.618/0.681 |
CWDB'de v2 TAŞIYOR (isabet medyan 0.66, kapsama 1.00, alan 1.51×; #7/#27/#9/#3'te neredeyse tüm kare, #27'de cetvel).
**Neden — tanım farkı:** petchs "evre çizgisi" açık yara + çevresindeki kızarık/renk değiştirmiş KAPALI deriyi
kapsıyor (Evre 1 görüntülerinde açık yara yok); CWDB yalnız AÇIK yara yüzeyi. v2 "lezyon alanı"nı öğrendi
(`runs/audit/v2_cwdb_overflow.png`, `petchs_outline_vs_tissue.png`). = 8 Eylül'deki intact_lesion vs open_surface
ayrımı. Üretim kuralı DEĞİŞMEDİ. Not: evre çizgisi (lezyon alanı) Evre 1/DTI için ayrıca değerli olabilir.

## SON DEVAM (2026-10-17) — Maske yakın çekim sorunu: neden + kural seçimi

**Neden (geliştirme = petchs train+valid 277, test kopyaları hariç; yara karenin medyan %71'i):** eski FUSeg modeli
Dice medyan **0.14**, çizdiği alan gerçek yaranın medyan **%8'i** (yalnız kırmızı adacıklar); yeni birleşik model
**0.83**. Şimdiki "eski-önce" kuralı 0.22: eski model küçük bir şey bulunca yeni model hiç çağrılmıyor.
Geliştirme-2 (Figshare 20, uzak çekim, yara payı %13; segmenter eğitiminde yok): eski 0.91, yeni 0.94.
| kural (Dice medyan) | petchs dev | Figshare |
|---|---|---|
| eski-önce (şimdiki) | 0.22 | 0.91 |
| yeni | 0.83 | 0.94 |
| birleşim | 0.83 | 0.94 |
| oran<0.75→yeni (eski alan < 0.75×yeni alan ise yeni, değilse eski) | 0.82 | 0.94 |
**ÖN-KAYIT (cetvellere bakmadan):** birincil kural = `oran<0.75→yeni` (geliştirmede birleşimle berabere; CWDB'de
yeni modelin kallus taşması bilindiği için boyutlar benzerken eski maskeyi koruyan kural seçildi — bu bilgi daha
önce CWDB'den geldi, şeffaflık için not). İkincil = birleşim. Ölçüm: CWDB mask Dice (konsensüse), petchs-test mask
Dice (etiketçi sınırı), ve doku R2 (PS CNN 3 seed + A) iki cetvelde. Kural değiştirilmeyecek.
**SONUÇ (mask Dice ort / medyan):** old-first CWDB 0.865/0.913, petchs-test 0.359/0.353; **ratio(0.75)** CWDB
0.770/0.886, petchs-test **0.761/0.815**; union 0.748/0.837 · 0.769/0.815. → Ön-kayıtlı kural yakın çekimi düzeltti
ama CWDB'yi bozdu. KEŞİF (sonuçtan sonra, kanıt değil): CWDB'de eski/yeni alan oranı medyan 0.74 (yeni model kallus
taşması) → eşik 0.75 15/27 görüntüyü yeniye çeviriyor (ör. #27 eski 0.76 → yeni 0.19). Eşik 0.5: 0.805 · 0.732;
0.25: 0.843 · 0.670 → tek eşik iki tarafı kurtarmıyor. **Teşhis:** eski model yakın çekimde yalnız kırmızıyı çiziyor,
yeni model kallusa taşıyor (DFUTissue seg etiketlerinin %59'u kallus, 2026-09-11 bulgusu, düzeltme onay bekliyor).
**Önerilen kalıcı çözüm (kullanıcı onayı bekliyor):** yeni modeli (a) kallussuz DFUTissue seg etiketleri + (b) petchs
train+valid yara sınırları (test ayrı kalır) ile kısa fine-tune (~2–3 sa), sonra "yalnız yeni model"i CWDB + petchs
test'te ölç. Üretim kuralı DEĞİŞTİRİLMEDİ (old-first); `--mask-rule` yalnız ölçüm betiklerinde seçenek.

## SON DEVAM (2026-10-13, 2) — Roboflow setleri: denetim + ikinci cetvel

Kullanıcı 4 zip indirdi → `data/corpus/rf_{petchs,segulcer,wuxi4,wuxi2}` (hepsi CC BY 4.0). dHash (≤6): cetvellere
(CWDB, DFUT Test) ve eski setlere 0 çakışma; set-içi bölmeler arası: petchs 2, segulcer 2, wuxi4 10.
Görsel denetim `runs/audit/*.png`:
- **wuxi4 (351) + wuxi2 (374, 3× artırılmış): HAYVAN deneyi yaraları** (tüy, cetvel, "G3-test" etiketleri; koyu
  kabuk "granülasyon" denmiş) → DIŞLANDI. wuxi4'te ayrıca 195/351 maske görüntüye göre 90° dönük.
- **petchs (310, gerçek bası, çok-doku poligon + evre çizgisi)**: makul → kullan. 640² gerilmiş.
- **segulcer (217)**: görüntü başına tek doku, tüm yara tek renk → kaba; yalnız ikincil deneme.
**ÖN-KAYIT (eğitimden önce):** İKİNCİ CETVEL = petchs `test` 31 (eğitime hiç girmez; test↔train kopyası eğitimden
çıkarılır). Puanlama yalnız doku-etiketli piksellerde (G/S/N poligon birleşimi; evre çizgisi içi etiketsiz alan
belirsiz olduğu için hariç). Eğitim: multi+fig + petchs train (248 − kopya) ; iç doğrulama + petchs valid 31.
B-CNN-P = bu; B-CNN-PS = + segulcer train+valid (195). 3'er seed. Birincil: CWDB R1 N ve G+S (multi+fig'e karşı) ve
petchs-test G/S/N; A ile fark. Sözlük: granulation→G, slough→S, eschar→N, evre çokgenleri→yok say;
segulcer Granulation→G, Sloughy→S, Necrotic→N, Epithelialising→O.

**SONUÇ** (`runs/tissue_cnn_{P,PS}/seed*.pt`; `runs/tissue_ab_{P,PS}/report.json`, `runs/tissue_petchs/report_{P,PS}.json`).
3-seed ort., G / S / N:
| | CWDB R1 | CWDB R2 | petchs-test R1 (26) |
|---|---|---|---|
| A (renk) | 0.71 / 0.46 / 0.70 | 0.64 / 0.41 / 0.65 | 0.75 / 0.65 / 0.44 |
| CNN multi+fig | 0.62 / 0.61 / 0.48 | 0.59 / 0.55 / 0.54 | 0.74 / 0.65 / 0.29 |
| CNN +petchs (P) | **0.69** / 0.61 / 0.47 | 0.65 / 0.54 / 0.51 | 0.93 / 0.86 / 0.93 |
| CNN +petchs+segulcer (PS) | 0.69 / 0.59 / **0.67** | 0.65 / 0.54 / **0.64** | 0.91 / 0.83 / 0.89 |
| insan | 0.70 / 0.64 / 0.83 | | |
A − CNN (G+S), CWDB: P −0.045..−0.080, PS −0.039..−0.079 (CNN önde, GA'lar 0'a dokunuyor). **PS ile CNN nekrozda
A'ya yetişti** (CWDB R1 0.67 vs 0.70, R2 0.64 vs 0.65). petchs-test skorları aynı kaynak/etiketleyici (muhtemelen
aynı hastalar; set-içi kopya bulundu) → şişkin, bağımsız kanıt DEĞİL; güvenilir cetvel CWDB. DFUT (ayak) PS'te hafif
düştü (G 0.88→0.85). **MASKE BULGUSU:** petchs-test'te YOLO (eski→yeni yedekli) maske etiketçi sınırına Dice medyan
0.41, doku piksellerinin medyan %32'sini kapsıyor (17/26 < %50); `runs/audit/petchs_yolo_mask.png`: yakın çekimde
yara kareyi doldurunca eski FUSeg modeli yalnız kırmızı adacıkları işaretliyor, sarı/siyah alanları kaçırıyor; boş
olmadığı için yedek (yeni) model hiç devreye girmiyor. CWDB'de sorun yok (0.865). Olası düzeltme: yakın çekim
algılanınca yeni modeli kullanmak ya da maskeyi doku modeliyle genişletmek — ölçülmedi.

## SON DEVAM (2026-10-13) — Figshare bası doku poligonları eklendi

Kullanıcının verdiği 4 set projede YOKTU (petchs Pressure-Ulcer-Segmentation ≠ bizdeki fr7kn). Roboflow setleri
(petchs, Segmentation-ulcer, WuXi) giriş gerektiriyor, kullanıcı şu an indiremiyor → beklemede. Figshare 17206940
(CC BY 4.0) indirildi → `data/corpus/figshare_pu2`: **20** Medetec bası yarası, LabelMe poligon: Granulation, slough
tissue, Eschar, Re-ep, Ulceration (755 görüntü iddiası yanlış; makalenin tam korpusu açık değil). dHash: 291
referansla kopya yok. Görsel denetim `runs/figshare_audit.png`: etiketler makul.
**ÖN-KAYIT (eğitimden önce):** B-CNN-multi+fig = çok kaynaklı ayarın aynısı + Figshare 20 (hepsi eğitimde; Eskar→N,
slough→S, Granulation→G, Re-ep→O, Ulceration-yalnız → yok say). 3 seed. Birincil: CWDB R1 N Dice (multi'ye karşı,
3-seed ortalaması); ikincil G+S ve A'ya karşı fark. 20 görüntü az → küçük etki beklenir, GA geniş.
**SONUÇ** (`runs/tissue_cnn_multi_fig/seed{0,1,2}.pt`, `runs/tissue_ab_fig/report.json`). 3-seed ort., multi → +fig:
CWDB R1: G 0.628 → 0.618, **S 0.566 → 0.613** (insan 0.643), **N 0.299 → 0.476** (seed: 0.42/0.53/0.47; A 0.70,
insan 0.83). R2 (bizim maske): S 0.500 → 0.546, **N 0.173 → 0.542** (A 0.645). DFUT Test S 0.64 → 0.72.
A − CNN (G+S): R1 −0.029/−0.011/−0.052, R2 −0.029/−0.031/−0.063 → CNN hafif önde ama GA'lar 0'ı içeriyor.
Yorum: yalnız 20 bası+eskar görüntüsü nekrozu belirgin yükseltti (beklenenden büyük; seed'ler tutarlı, R2'de
3/3 seed eskilerin hepsinin üstünde) → **alan-içi (bası) + hedef sınıf içeren veri, miktardan çok daha değerli.**
Nekrozda A hâlâ önde → hibrit hâlâ mantıklı; Roboflow petchs (bası+eskar, 310) gelirse en güçlü sonraki adım.

## SON DEVAM (2026-09-29, 2) — Doku tespiti A (etiketsiz renk) vs B (DFUTissue etiketli)

`training/tissue_ab_compare.py` → `runs/tissue_ab/report.json`, `montage.png`, `logs/tissue_ab.log`.
A = görüntü başına Lab k-means (k=4) + küme adlandırma (koyu+gri → nekroz, açık+gri → diğer, yoksa renk açısı:
kırmızı → granülasyon, sarı → slough) + süreklilik kuralı (yaranın <%3'ü adacıklar çevreye katılır). Tek serbest
parametre (renk açısı sınırı = 38°) DFUTissue TrainVal'da ayarlandı. B = piksel random forest (Lab, bulanık Lab,
yerel std, HSV, gradyan, sınıra uzaklık), DFUTissue TrainVal 94. CWDB 27 × 4 uzman yalnız sonda bir kez.
Havuzlanmış Dice, 4 uzman ortalaması, görüntü bootstrap %95 GA.

| | G granülasyon | S slough | N nekroz |
|---|---|---|---|
| DFUT Test 16 (ayak) A / B | 0.79 / **0.91** | 0.18 / **0.35** | — |
| CWDB R1 (uzman bölgesi) A | **0.71** | **0.46** | **0.70** |
| CWDB R1 B | 0.67 | 0.15 | 0 (sınıfı yok) |
| CWDB R1 insan tavanı | 0.70 | 0.64 | 0.83 |
| CWDB R1 önemsiz (hepsi G) | 0.58 | 0 | 0 |
| CWDB R2 (bizim maske) A / B | 0.64 / 0.64 | 0.41 / 0.12 | 0.65 / 0 |

A−B (G+S ortalaması): R1 +0.177 [0.089, 0.295], R2 +0.148 [0.068, 0.252], P(A>B)=1.0. B evinde (ayak) kazanıyor,
bası yarasında kaybediyor → alan farkı somut. Süreklilik kuralı etkisiz (A ≈ A-raw), k-means zaten düzgün bölge veriyor.
**Teşhis:** B soluk sarı-beyaz slough'u %41 "kallus" diyor (ayakta soluk = nasır; bası yarasında kallus yok). A'nın
"açık+gri → diğer" kuralı da beyazımsı slough'u kaçırıyor. **SONRADAN (keşif, cetvele bakıldıktan sonra → kanıt değil):**
kallus/diğer → slough eşlemesiyle S: A 0.46→0.55, B 0.15→0.57; G+S: A 0.63, B 0.62. Yani B'nin çöküşü tamamen
sınıf-sözlüğü farkı; ama B nekrozu hiç bilmiyor. Doğrulamak için yeni bir bası doku seti gerekir (kendi 732'den uzman
onaylı küçük set). Olası sonraki: hibrit (A nekroz + soluk→slough kuralı ön-kayıtla), CNN-B (ön-eğitimli ağırlık
indirmesi onay ister).

**ÖN-KAYIT — CNN-B (2026-09-29, CWDB'de koşmadan önce):** B'nin zayıf kurulduğu itirazını kapatmak için
B-CNN = torchvision DeepLabV3-MobileNetV3 (COCO ön-eğitimli, 11M), DFUTissue TrainVal'da eğitim (her 6. görüntü iç
doğrulama; model seçimi iç doğrulama G+S+kallus ortalama Dice ile), 3 seed (0,1,2). Girdi: yara kırpması 256 kare.
Sınıflar fibrin→S, granülasyon→G, kallus→diğer (sözlük eşlemesi YOK; eşlemeli hâl yalnız keşif olarak ayrı raporlanır).
Birincil: A vs B-CNN, G+S Dice ortalaması, CWDB R1 ve R2, 3 seed'in ortalaması ve her biri. Kural sonuç görülünce değişmez.

**SONUÇ — CNN-B** (`training/train_tissue_cnn.py`, `runs/tissue_cnn/seed{0,1,2}.pt`, iç doğrulama 0.63/0.62/0.59;
`runs/tissue_ab_cnn/report.json`). DFUT Test (ayak): B-CNN G 0.87–0.90, S **0.64–0.71** (RF 0.35, A 0.18) → evinde
açık ara en iyi. CWDB R1 (G / S): B-CNN s0 0.67/0.41, s1 0.63/0.46, s2 0.58/0.55; A 0.71/0.46. **A − B-CNN (G+S):
R1 +0.045/+0.037/+0.019, R2 +0.013/−0.012/−0.008; tüm GA'lar 0'ı içeriyor → BERABERE.** "A kazandı" sonucunun
çoğu zayıf RF'ten geliyormuş; düzgün CNN ile bası yarasında G+S'de fark yok. Nekrozu hâlâ yalnız A buluyor (0.70).
Keşif (kanıt değil): B-CNN kallus→S ile CWDB R1 S 0.59–0.63 (insan 0.64), R2 0.52–0.56.
**ÖN-KAYIT — çok kaynaklı CNN (2026-10-01, eğitimden önce):** Yeni veri: LUTSeg (HF `ksanchez84/LUTSeg`, CC BY 4.0,
141 lepra ülseri/39 hasta, kaynak CO2Wounds) + WoundTissue (GitHub akabircs, CC BY-NC-SA, depoda 147'nin yalnız 14'ü,
128 px). Cetvellere dHash sızıntısı (hamming ≤6): 0. Ortak sözlük: S (fibrin/slough), G, O (kallus/epitel/maserasyon),
N (nekroz); LUTSeg "Other", kemik, tendon → yok say. B-CNN-multi: aynı mimari/girdi/kayıp, 4 sınıf, eğitim =
DFUT TrainVal (her 6. hariç) + LUTSeg train + WoundTissue 14; iç doğrulama = DFUT her 6. + LUTSeg val (hasta bazlı);
seçim iç doğrulama sınıf-ortalama Dice; 3 seed. Birincil: A vs B-CNN-multi, G+S, CWDB R1 ve R2. İkincil: N Dice
(A vs B-CNN-multi) ve B-CNN-multi vs B-CNN (yalnız DFUT). Kallus→S yine yalnız keşif.

**SONUÇ — çok kaynaklı CNN** (`runs/tissue_cnn_multi/seed{0,1,2}.pt`, iç doğr. 0.61/0.65/0.62;
`runs/tissue_ab_multi/report.json`). CWDB R1 3-seed ort. (DFUT-only → multi): G 0.628 → 0.628, **S 0.474 → 0.566**
(insan 0.643), **N 0 → 0.30** (A 0.70, insan 0.83). R2: S 0.46 → 0.50, N → 0.17. Ayak (DFUT Test) bozulmadı.
A − multi (G+S): R1 −0.008/−0.012/−0.016, R2 −0.021/−0.020/−0.012, GA'lar 0'ı içeriyor → yine BERABERE (CNN hafif
önde). Kallus→S keşif eşlemesi artık az fark yaratıyor (S +0.01..+0.04): çeşitli veri "soluk = kallus" hatasını büyük
ölçüde kendiliğinden düzeltti. Nekrozda A hâlâ açık ara iyi → hibrit (CNN-multi G/S + A nekroz) mantıklı; CWDB'de
ölçülürse keşif sayılır, kanıt yeni uzman onaylı bası setinde.

**ÖN-KAYIT — hibrit (yeni bası seti için, 2026-09-29):** H = B-CNN (3 seed çoğunluk oyu) G/S + kallus→S, üzerine
A'nın nekroz kuralı (A'nın N dediği pikseller N). Birincil: H vs A vs B-CNN, G+S ortalaması; ikincil N.

**ÖN-KAYIT (2026-09-29, yeni bası doku test seti görülmeden):** Ana doku yöntemi = A, tek değişiklikle:
"açık+gri → diğer" kümeleri **slough** sayılır (bası yarasında kallus yok). Diğer her şey sabit: k=4, renk açısı
sınırı 38°, nekroz L*<30 ve C*<20, süreklilik kuralı açık. Birincil ölçü = G+S Dice ortalaması (havuzlanmış, uzman
ortalaması), ikincil = N Dice. Karşılaştırma: B + "kallus → slough" ve önemsiz taban. Yeni set: kendi 732'den uzman
onaylı küçük bası seti (A taslağı → uzman düzeltir; düzeltilmiş hali cetvel olur, A'nın ham taslağıyla kıyas ayrıca
raporlanır, çünkü taslaktan başlamak uzmanı A'ya doğru çekebilir). Bu set görülmeden kural değiştirilmeyecek.

## SON DEVAM (2026-09-29) — YOLO derinliği WoundsDB gerçek derinliğiyle fine-tune: NEGATİF

(11 Eylül → 29 Eylül arası depoda teknik iş yok; yalnız staj raporu + LinkedIn.)
Araçlar: `training/prepare_woundsdb_depth.py` (hasta bazlı bölme: her 5. vaka test → 9 hasta/14 sahne test,
56 eğitim; hedef = `thermal-stereo.png` şekli, sahne başına temel modelin metre aralığına afin oturtulmuş,
0 = ölçüm yok) ve `training/eval_woundsdb_depth.py` (11 Eylül protokolü, yalnız test hastaları).
Eğitim `yolo26n-depth` AdamW lr 1e-4, mosaic 0, imgsz 320. İlk koşuda (val = test) test silog 8. epoch'a kadar
düştü (0.218→0.155), 9'dan sonra patladı (0.55) → ezber. Temiz seçim için eğitim hastalarından iç doğrulama
(5 vaka) ayrıldı: `runs/depth/runs/depth/woundsdb_ft_inner/weights/best.pt` (epoch 3). Test (tek sefer):

| model | stereo yön uyumu | ToF yön uyumu | |ρ| stereo / ToF | WoundsDB ayırt | bası cetveli ayırt | cetvel çukur |
|---|---|---|---|---|---|---|
| n (temel) | 6/15 | 9/13 | 0.45 / 0.41 | 0.65 | 0.69 | 9/27 |
| l (temel) | 10/15 | 8/13 | 0.46 / 0.44 | 0.70 | 0.71 | 11/27 |
| **n + WoundsDB ft** | 7/15 | 7/13 | **0.56** / 0.38 | 0.64 | **0.60** | 9/27 |

Yorum: model stereo sensörün genel sahne şeklini biraz öğrendi (ρ 0.45→0.56) ama bağımsız ToF'ta iyileşme yok,
yara çukuru yönü hâlâ yazı tura, bası cetvelinde KÖTÜLEŞTİ (0.69→0.60). 56 sahne + 8-bit göreli hedef yara
kabartısını öğretmeye yetmiyor. Not: test 15 sahne → gürültülü. JSON: `runs/depth/woundsdb_eval_{base,ft}.json`.
**Derinlik kolu kapandı:** tek fotoğraftan derinlik ne hazır ne fine-tune ile yara çukurunu görüyor; Evre 3↔4
için gerçek sensör gerekir. Sıradaki: Ario'ya bulgu + renk ağırlıklı etiketsiz doku katmanlaması (CWDB 27 ile ölç).

## SON DEVAM (2026-09-11) — veri haritası, segmenter güçlendirme, insan tavanı

**Veri haritası** (artefakt: https://claude.ai/code/artifact/ce283b86-225e-4df5-9f29-83b87646c433):
7.613 görüntü / 11 kaynak sayıldı. Yeni gelenler: `data/corpus/roboflow_ambatron` (929, yara TÜRÜ
D/V/S/P/N/BG, P=134), `data/corpus/woundsdb` (46 hasta/79 sahne, **bacak/venöz**, bası değil; 64 gerçek
derinlik, etiket yok), `data/corpus/complexwounddb` (bozuk klon düzeltildi: 27 görüntü × 4 uzman doku
maskesi, **çoğu bası**), `data/corpus/co2wounds_v2` (IEEE DataPort'tan elle: 764, 607 maske, resmi bölme
485/122/157, test maskesi ve hasta ID yok). Medetec hâlâ elle indirilmedi (toplu indirme yok, lisans belirsiz).

**Dondurulmuş cetveller** (hiçbir eğitime girmez): seg `azh_fuseg/validation` 200 · doku `DFUTissue
Labeled/Test` **16** · evre `fr7kn/test` (temiz ~189) · tip `ambatron/test` 233 · derinlik WoundsDB 79.
**Sızıntı bulguları:** fr7kn kendi train↔test 17 (valid 45); azh_fuseg train↔kendi val 7; CO2Wounds
train↔val 10; azh_fuseg↔DFUTissue Labeled 7. Havuzlar: `data/pool_foot` 1158, `data/pool_pressure` 3137,
`data/pool_all` 4989 (hepsi cetvellere karşı dHash ile süzülü, hamming 2).

**Segmenter** (commit f7c4a22): yeni araçlar `prepare_co2wounds_seg`, `merge_yolo_seg`,
`prepare_complexwound_ruler`, `augment_tight_crops`. Birleşik set `data/yolo_seg_combined` (1326 +
2541 yakın-kırpma = 3867 train / 136 val). yolo26n-seg 50 epoch → `runs/segment/runs/segment/combined_v1/weights/best.pt`
(val mask mAP50 0.723, epoch 41).
- **İnsan tavanı:** ComplexWoundDB 4 uzman ortalama ikili Dice **0.862 ± 0.19** (`data/ruler_complexwound/agreement.json`).
- Eski segmenter'ın asıl kusuru kadrajdı: yara kareyi doldurunca maske yok (fr7kn %47.5 boş).
- Hizalama: `check_mask_quality` + `precompute_rgbd` artık `retina_masks=True` (COMMIT EDİLMEDİ). Etkisi
  küçük (bası eski 0.848→0.857); 640 px'te 4:3 fotoğraflar dolgu almıyor, 732 gerilemesini açıklamıyor.

| (retina) | Bası CWDB | Ayak azh | Minik DFUT | Havuz boş | Evre I / SDTI bulma |
|---|---|---|---|---|---|
| eski 0.05 | 0.857 | 0.818 | 0.461 | 17.8% | 36% / 65% |
| eski 0.01 | **0.865** | 0.809 | 0.475 | 11.8% | 61% / 83% |
| yeni 0.05 | 0.770 | 0.809 | **0.823** | 7.1% | 75% / 96% |
| yeni 0.01 | 0.747 | 0.773 | 0.778 | **4.0%** | **82% / 100%** |

"Yeni model kapalı lezyonları kaçırıyor" hipotezi veriyle ÇÜRÜDÜ (tersine çok daha iyi buluyor).
Kendi 732'de "gerileme" gürültü: 200 fotoğrafta yeni 12 kaybetti, 16 kazandı.

**Maske kararı — yedekli birleşim** (`logs/fallback_eval.log`): önce eski FUSeg modeli conf 0.01, bulamazsa
yeni birleşik model conf 0.01 → bası Dice **0.865**, havuz boş maske **%1.9** (kendi 732 %5.0, fr7kn %6.7);
minik kırpma 0.567 (hedef alan değil). `precompute_rgbd` artık `--fallback-weights/--fallback-conf` alıyor.

**yolo26s:** `yolo26s-seg.pt` indirildi (22.4 MB, ultralytics/assets v8.4.0) ama EĞİTİLMEDİ — kullanıcı ve
mentör etiketleme yapamıyor, 10–12 saatlik eğitim beklenemiyor. Not: yeni modelin bası gerilemesinin muhtemel
sebebi model seçiminin lepra/ayak val'i ile yapılması; ileride bası val seti olmadan tekrarlanmamalı.

**Evre cetveli** `data/ruler_stage_fr7kn/` (sızıntısız): test **187** (E1 39·E2 61·E3 52·E4 35), probe-train
1671 (fr7kn train+valid, test ikizleri atıldı). Yeni prob `training/validate_stage_probe.py` (+6 test):
doku probuyla aynı çok-seed+bootstrap protokolü, macro-F1 + QWK, pre-training girdisini birebir taklit eder,
checkpoint anahtarları uyuşmazsa DURUR (eski probların `strict=False` sessiz-rastgele riskine karşı).

**Koşan orkestrasyon** (2026-09-11 ~11:48 başladı; loglar `logs/`): pool_pressure derinlik+maske →
`train_jepa` (tuned: crop+depth+relief, vic 4, lr 3e-4, 30 epoch) → `runs_jepa/jepa_pressure.pt` →
`runs_jepa/stage_validation_pressure.json`. Paralelde kontrol: `jepa_best_c05` ve `jepa_v2` aynı evre
cetvelinde → `runs_jepa/stage_validation_{jepa_best_c05,jepa_v2}.json`. Bu üçü alan-uyumu matrisinin evre sütunu.

**Kontrol probları (evre cetveli, test 187, ortalama havuzlama):** rastgele 5 seed F1 0.611±0.012 / QWK
0.727±0.007; `c05` 0.542/0.545, `v2` 0.534/0.555 → JEPA rastgeleden ANLAMLI kötü (QWK Δ≈−0.18, P(Δ>0)=0.001).
Teşhis (`logs/collapse_diag.log`): çökme DEĞİL (efektif rank JEPA ≥ rastgele), standartlaştırma değil. JEPA
yama vektörleri görüntü içinde neredeyse dik (cos 0.04–0.12; rastgele 0.84) → ortalama havuzlama bilgiyi
götürüyor. Havuzlama teşhisi (`logs/pooling_diag.log`, SADECE geliştirme checkpoint'leri + 3 rastgele seed):

| havuz (F1/QWK) | rastgele | c05 Δ | v2 Δ |
|---|---|---|---|
| mean | 0.612/0.731 | −0.071/−0.186 | −0.078/−0.176 |
| meanstd | 0.667/0.766 | −0.052/−0.032 | −0.086/−0.097 |
| max | 0.588/0.722 | −0.039/−0.096 | −0.020/+0.002 |

Havuzlama QWK farkının çoğunu açıklıyor, ama JEPA hiçbir havuzlamada rastgeleyi geçmiyor.

**ÖN-KAYIT (pressure JEPA sonucu görülmeden, 2026-09-11 ~12:35):** birincil = `mean` (baştan sabit);
ikincil = `meanstd` (geliştirme setinde rastgele için ve c05 için en iyi → rakibe en güçlü tabanı veren seçim).
Pressure JEPA'ya başka havuzlama denenmeyecek. `validate_stage_probe.py --pool` eklendi (varsayılan mean).

**SONUÇ — bası JEPA'sı** (`runs_jepa/jepa_pressure.pt`, pool_pressure 3137, 30 epoch, kayıp 0.396→0.211):

| JEPA | Ayak doku (F1; rastgele ~0.52) | Bası evre, mean (F1/QWK; rastgele 0.611/0.727) | Bası evre, meanstd (rastgele ~0.65/0.76) |
|---|---|---|---|
| c05 (ayak-ağırlıklı 2142) | **0.545** | 0.542/0.545 | 0.615/0.733* |
| v2 (karma 5096) | 0.464 | 0.534/0.555 | 0.581/0.668* |
| **pressure (3137)** | 0.461 | 0.539/0.572 | 0.565/0.667 |

(* geliştirme teşhisi: 3 seed, bootstrap yok. pressure satırı tam protokol: 5 seed + 2000 bootstrap;
JSON'lar `runs_jepa/stage_validation_pressure{,_meanstd}.json`, `runs_jepa/tissue_validation_pressure.json`.)
Bası JEPA'sı birincil: F1 Δ −0.077 [−0.151, −0.001] P(Δ>0)=0.025, QWK Δ −0.152 P=0.002. İkincil: F1 Δ
−0.088 [−0.169, −0.007] P=0.015, QWK Δ −0.091 [−0.206, +0.018] P=0.052. İkisinde de JEPA rastgeleden kötü.

**Yorum:** Ayak doku probunda alan etkisi var (en iyisi ayak-ağırlıklı JEPA). Bası evrelemesinde alan uyumu
YETMİYOR: eşleşen havuzla eğitilen JEPA bile dondurulmuş lineer probda rastgele kodlayıcıyı geçemiyor.
Ön-kayıtlı, temiz bir negatif sonuç. Sıradaki aday: fine-tune merdiveni (önceki bulgu: JEPA'nın avantajı
fine-tune'da görünüyordu) ve/veya dikkat-tabanlı prob (yama dikliği sorununu doğrudan ele alır).

**KALLUS BULGUSU (segmenter):** DFUTissue etiketli alanının **%59'u kallus** (sınıf 3 = yara çevresindeki
nasırlı deri, yara yatağı DEĞİL); `prepare_dfutissue_seg.py` `ann > 0` ile kallusu da "yara" saymış. Kanıt,
DFUTissue Test (16): eski model kallus-dahil GT 0.475 → sadece-yatak GT **0.790** (çizim/yatak 0.98×); yeni
model 0.823 → **0.557** (çizim/yatak **2.71×**). Yani eski modelin DFUT "0.46"sı minik-fotoğraf sorunu DEĞİL,
tanım farkıymış (önceki açıklama yanlıştı). Yeni model kallusu yara diye öğrenmiş → bası cetvelinde taşma
(isabet 0.680, kapsama 0.982, alan medyan 1.32×; tek büyük maske, fazladan aday değil — sadece en güvenli adayı
almak 0.770→0.777). Düzeltme planı: DFUTissue seg etiketlerini sadece sınıf 1–2 ile yeniden üret → yakın
kırpmaları yeniden üret → yeni modeli düzeltilmiş veriyle kısa fine-tune et (~1–1.5 saat) → bası/ayak/DFUT-yatak
cetvelleri + havuz boş oranıyla ölç. KULLANICI ONAYI BEKLİYOR.

**MENTÖR (Ario) YÖNTEMİ — derinlik kontrolü:** Ario etiketsiz maskeleme önerdi (derinlik + renk gradyanı +
dıştan içe sıralı, kesintisiz katman kuralları; önce etiketsiz dene → doğruysa devam, değilse etiketle). İlk
kontrol, ComplexWoundDB 27 (derinlik `data/ruler_complexwound/depth`): yara içi vs hemen dışı ayırt edicilik
(yönden bağımsız AUC medyan) **derinlik 0.60** (≥0.80: 4/27; yön tutarsız, 19/27; sınırda basamak yok, 1.09×).
Vücut kıvrımı ikinci derece yüzeyle çıkarılınca **0.69** (≥0.80: 3/27; yön 18/27; etki 0.76). **Renk (Lab a\*)
0.80** (≥0.80: 14/27; sınırda belirgin değişim, 1.93×). Görsel: YOLO26n-depth haritaları bulanık; vücut şeklini
ve cetvel/çarşaf gibi nesneleri görüyor, mm ölçekli yara çukurunu görmüyor, koyu eskarı "uzak" sanıyor.
Sonuç: bu tahmini derinlik yara sınırı için güvenilir değil.

**Gerçek derinlik testi** (WoundsDB, bacak/venöz ülser, 79 sahne; termal çerçevede hizalı, montajla doğrulandı):
YOLO26n-depth ↔ gerçek derinlik sıra korelasyonu |ρ| medyan **0.48** (stereo) / **0.44** (ToF); iki gerçek sensör
birbirine **0.70** (sağlama). Kıvrım çıkarılmış derinlikte yara içi vs hemen dışı (maskeler bizim sistemden):
**stereo 0.78** (≥0.80: 32/74, yön tutarlı **68/74**, etki 1.02), **ToF 0.76** (21/61, yön 50/61, etki 0.93),
**YOLO 0.65** (12/77, yön 44/77 ≈ yazı tura, etki 0.42). Görsel: gerçek derinlik yara yüzeyinin kabartısını
gösteriyor, YOLO haritası sadece kaba bölgeler. **Sonuç: derinlik fikri geçerli; zayıf halka YOLO26n-depth (nano)
modeli.**

**yolo26l-depth denendi** (`yolo26l-depth.pt`, 55.9 MB, ultralytics/assets v8.4.0): genel geometri gerçek derinliğe
daha yakın (|ρ| stereo 0.48→**0.59**, ToF 0.44→**0.53**; gerçek↔gerçek 0.70) AMA yara çukuru hâlâ görünmüyor:
WoundsDB ayırt etme 0.65→0.67, yön 57%→64% (gerçek stereo 92%); bası cetveli 0.69→0.71, yön 67%→59%. Sonuç:
büyük YOLO derinliği sahneyi daha iyi görüyor, mm ölçekli çukuru görmüyor. Sıradaki aday: Depth Anything V2
(`transformers` kurulumu + 94 MB) ya da Marigold V2 (yalnız Linux+CUDA, ~17 GB VRAM → bulut GPU). Aksi hâlde
etiketsiz maske renk ağırlıklı kurulmalı.

**Depth Anything V2 Small denendi** (`transformers 5.17.0` kuruldu; `depth-anything/Depth-Anything-V2-Small-hf`,
24.8M parametre, 94 MB, MPS'te çalışıyor): WoundsDB'de YOLO-L ile aynı (ayırt etme 0.65, yön 64%); bası cetvelinde
**0.84** (≥0.80: 15/27, etki 2.88; renk 0.80). Ayırıcı test (`logs/depth_geometry_check.log`):
(a) WoundsDB'de yara içinin çukur/tümsek yönünde gerçek sensörle uyum **L 45–49%, DA-V2 38–48% = yazı tura** →
modellerin gerçek yara geometrisini gördüğüne dair kanıt yok. (b) Parlaklık bağı testi ayırıcı çıkmadı: gerçek
ToF'un kendisi de parlaklıkla 0.50 bağlı (çukur gerçekten koyu ve/veya ToF'un koyu yüzey hatası). (c) Bası
cetvelinde DA-V2 yara içini 20/27 "daha uzak" gösteriyor, ama gerçek derinlik olmadan doğrulanamaz.
**Sonuç:** fotoğraftan derinlik tahmini yara kabartısı için güvenilir bir ölçüm değil. DA-V2 haritası bası'da
yarayı ayırt eden bir *özellik* olarak işe yarayabilir (kaynağı görünüş de olabilir). Gerçek derinlik (Evre
3↔4) için gerçek sensör (telefon LiDAR/stereo) gerekir; bu bulgu Ario'ya iletilecek. Maske için yol: renk ağırlıklı
etiketsiz yöntem (+ isteğe bağlı DA-V2 haritası ek ipucu), cetvelde ölçülerek.
Not: WoundsDB bası değil; maskeler uzman değil (üç kaynakta aynı maske → karşılaştırma adil).
Commit edilmemiş: precompute/check_mask_quality değişiklikleri, validate_stage_probe + testi.

## SON DEVAM (2026-09-10) — havuz büyütüldü (2142 → 5096), yeni veriler geldi

**Bağlam:** Kullanıcı Gemini araştırmasıyla veri setleri getirdi. Denetim: Gemini'nin
"bulduğu"nun çoğu zaten `training/fetch_corpus.py` registry'sinde/diskinde. Asıl
kaçırılan kaldıraç: diskteki veriyi havuza tam katmamışız (piid 1091 hiç girmemiş).

**Yeni indirilen veriler** (`~/Downloads`, detay hafıza notu `sorbed-new-datasets-2026-09`):
- `results.zip` = **WoundsDB** (chronicwounddatabase.eu, lisanslı): 46 vaka, 158 RGB +
  **64 gerçek derinlik** + 79 termal + stereo/mesh. Derinlik kalibrasyonu için altın
  standart (İş Kolu D). **Git'e girmez, dağıtılmaz.** Henüz işlenmedi.
- **roboflow fr7kn** (CC BY 4.0): 2013 bası yarası + evre kutusu (stage1-4, DTI/unstage YOK).
  `data/corpus/roboflow_fr7kn/` altına açıldı. Etiket kalitesi belirsiz (eskara stage2).
- **ambatron** wound-classification: 929, yara TÜRÜ (BG,D,N,P,S,V), çoğu alan-dışı → düşük öncelik.

**Yeni araç `training/build_pool.py` (+ tests/test_build_pool.py, 5 test geçti):**
Dağınık kaynakları tek havuza indirir. `training/dedup.py`'nin dHash'ini yeniden kullanır.
İki AYRI eşik: `--leak-hamming` (sıkı, test sızıntısı için "aynı fotoğraf") ve `--hamming`
(dedup). Symlink'leri tarama dışı bırakır (eski havuzun 732 gerçek dosyasını almak için).
`Labeled` (doku test, 110) dHash ile dışlanır → sızıntı 0 doğrulandı.

**Kurulan yeni havuz `data/rgbd_pool_v2/images` (5096, hamming 2 = neredeyse-birebir dedup):**
fr7kn 1874 + piid 1086 + azh_fuseg 985 + pressure 575 + dfutissue 372 + kaggle 204.
→ **Alan-içi (bası yarası) ağırlıklı: ~3739 bası vs 1357 ayak** (eski havuz tersineydi).
Dedup duyarlılığı: hamming 0→264, 2→561, 4→757, 6→921 elenen. Havuz-içi bölme olmadığı
+ probe testi zaten dışlandığı için sıkı eşik (2) seçildi = veri kaybını en aza indir.
Eski `data/rgbd_pool` (2142) ve checkpoint'ler DOKUNULMADI (geri dönülebilir).

**KOŞULDU — SONUÇ NEGATİF (dürüst).** precompute (conf 0.05) + train_jepa (30 epoch,
loss 2.0→~0.15) + validate koşuldu. `runs_jepa/jepa_v2.pt`, `tissue_validation_v2.json`.

| | Eski havuz (2142, ayak-ağırlıklı) | Yeni havuz (5096, bası-ağırlıklı) |
|---|---|---|
| JEPA macro-F1 | 0.545 | **0.464** ⬇️ |
| JEPA − random | +0.027 (P=0.93) | **−0.054** (z=−1.38, P(Δ>0)=0.034) |

**Havuzu büyütmek doku probunu KÖTÜLEŞTİRDİ.** Neden (+önemli düzeltme): doku test seti
**ayak yarası** (DFUTissue); yeni havuz ise **bası-ağırlıklı** (ayak 1357 vs bası 3739;
piid ve fr7kn+kaggle hepsi BASI — bu oturumda "piid ayak/karışık" dediğim YANLIŞTI, piid bası).
Eski havuz ~%66 ayaktı, yeni ~%73 bası → ayak-yarası sinyali seyreldi → ayak-probu düştü.
Skor tek yönlü (bası oranı arttıkça düştü) = gürültü değil, alan-uyumsuzluğunun EN güçlü kanıtı.

**Yayınlanabilir bulgu:** alan-dışı ön-eğitim verisi eklemek alan-içi probu bozar; ham havuz
boyutundan çok ALAN UYUMU belirleyici. Sorbed bası sistemi ama JEPA hep ayak probuyla ölçülüyor
= yanlış cetvel. **Asıl kaldıraç: bası yarası için etiketli değerlendirme seti (İş Kolu B).**
Eski checkpoint'ler + rgbd_pool DOKUNULMADI. Plan artefaktı:
https://claude.ai/code/artifact/ce283b86-225e-4df5-9f29-83b87646c433
build_pool.py + testi COMMIT edildi (9bc218c). tissue_validation_v2.json henüz commit'lenmedi.

## SON DEVAM (2026-09-08) — ayrı yüzey sınıfları, anotasyon v2

Kullanıcı açık/kapalı ayrımını kabul etti. Sekiz pilot görüntü yeniden görsel
incelendi ve **bütünlüğü korunmuş lezyonlu deri / açık yüzey / slough-eskarla
örtülü yüzey / belirsiz** olarak ayrı hedefler üretildi.

- v2: `data/annotations/pressure_injury_v2/`; v1 sonuçları korunuyor.
- İnceleme: `pilot/review.html`, `pilot/surface_qa.png`; kararlar
  `pilot/surface_drafts.json`, orijinal piksel maskeleri `pilot/surface_masks/`.
- ID'ler: 0=background, 1=intact_lesion, 2=open_surface, 3=covered_surface,
  255=unknown/ignore. Kapalı deri **lezyon eksi açık yara** ile hesaplanmıyor.
- Pilot: kapalı 2, açık 7, örtülü 1, belirsiz 7 görüntüde var (çoklu bölgeler).
  case_002 sarı tabakası bakım materyali olabileceği için v2'de belirsiz bırakıldı.
- Yeni kod: `scripts/export_surface_annotations.py`,
  `training/{surface_labels,train_surface_seg,predict_surface_seg,evaluate_surface_seg}.py`.
  Tek ortak encoder/decoder ve 4 kanallı yüzey başlığı; 255 kayıptan dışlanır.
- 11 sentetik test geçti: bilinmeyen pikseller, sınıf çakışması, boş Dice,
  abstention cezalandırması, hasta/kopya sızıntısı, eğitim→tahmin→değerlendirme.
- **Hasta verisiyle eğitim yapılmadı.** Uzman onayı ve hasta bazında bölünme yok;
  eğitim/değerlendirme araçlarının bu koşullarda durduğu kontrol edildi.
  Yeni yüzey sınıfları için gerçek performans/Dice sonucu henüz yok.
- Klinik runtime veya evreleme kuralları değiştirilmedi. Yüzey etiketi evreye
  otomatik çevrilmiyor; mevcut YOLO da bu üç sınıfı henüz tahmin etmiyor.
- Protokol ve sonraki çalıştırma komutları: `training/SURFACE_ANNOTATIONS.md`.
  Kalan 724 fotoğrafın anotasyonu hâlâ bekliyor.

## SON OTURUM (2026-09-08) — görsel anotasyon pilotu

Kullanıcı mentorunun önerisiyle Codex'ten **yara sınırı + doku + evre**
anotasyonu istedi. İlk pilot: **8 farklı vakadan 8 görüntü**, tek tek görsel
inceleme ve kaba poligon taslakları. **724 görüntü henüz anotasyon bekliyor.**
YOLO/Sorbed otomatik etiketleri bu pilotta kullanılmadı. Kaynak fotoğraflar değişmedi.

- Yerel veri: `data/annotations/pressure_injury_v1/` (gitignore kapsamında).
- İnceleme: `pilot/review.html`; statik görsel kontrol: `pilot/qa_overview.png`.
- Protokol: `PROTOCOL.md`; kaynak taslaklar: `pilot/visual_drafts.json`.
- Çıktılar: orijinal boyutta lezyon, yara yüzeyi ve sınıf-ID doku PNG'leri;
  gerekçeli evre JSON'ları ve düzenlenebilir LabelMe poligonları.
- Exporter: `scripts/export_visual_annotations.py`; boyut, hash, koordinat,
  yara yüzeyinin lezyon içinde kalması ve PNG round-trip doğrulamaları var.
- **Tüm etiketler AI taslağı**, uzman onayı yok; eğitim ve ground-truth uygunluğu
  false. Belirsiz doku 255. `red_surface_unspecified` granülasyon değildir.
  Anotasyon sınıf-ID'leri DFUTissue ile farklı; doğrudan eğitime bağlanmamalı.
- Evre adayları: 1 deep_tissue_injury, 2 unstageable, 1 stage_2, 4 indeterminate.
  Kesin klinik tanı/evre olarak kullanılamaz. Gerekçe ve alternatifler JSON'da.

**Veri bütünlüğü bulgusu:** 732 dosya / 32 vaka klasörü içinde 577 benzersiz
SHA-256; 87 birebir tekrar grubu, 155 ek kopya. Birebir tekrarlar vaka klasörleri
arasında geçmiyor. Benzer/yeniden sıkıştırılmış kareler henüz denetlenmedi;
vaka-hasta eşlemesi de doğrulanmalı. 732 kaynak dosyanın tamamının aynı hash'li
kopyası `data/rgbd_pool/images` içinde mevcut. Bu nedenle mevcut JEPA
checkpoint'leriyle bu veriyi hiç görülmemiş bağımsız test diye sunmayın.

**Önceki handoff'a düzeltme:** 6-blok fine-tune zaten koşulmuş:
`tissue_finetune_b6.json`, JEPA 0.500 / random 0.527. Yalnız bası yarasıyla
ön-eğitim sonucu `tissue_validation_pi_only.json`: 0.505 / 0.518.
`seg_ft.log` tamamlanmış DFUTissue fine-tune içeriyor; best checkpoint son
doğrulama mask mAP50=0.169. Aşağıdaki eski "sıradaki" listeleri tarihsel.
Sweep ayarları Test skoruyla seçiliyor; aynı Test'te bootstrap seçim yanlılığını
gidermiyor. Dondurulmuş probe'daki 5 seed, 5 ayrı JEPA ön-eğitimi değil.

**Devam:** pilot konturlarını/doku belirsizliklerini ve evre adaylarını mentorla
incelemek, düzeltmeleri ayrı bir anotasyon sürümünde korumak; kalan 724 dosyada
ilerlemeyi inventory üzerinden takip etmek. Henüz veri bölünmesi yapılmadı.

## 🆕 SON OTURUM (2026-09-02) — daha çok + alan-içi veri denendi, fine-tuning aracı eklendi

**Yeni veri katıldı (bası yarası):** `anonymized_wound_images` (32 vaka, **732 gerçek
bası yarası fotoğrafı** — koksiks/gluteal/sırt; projedeki en alan-içi veri, etiketsiz).
`precompute_rgbd` ile maske+derinlik üretildi (**732'nin 524'ünde yara**, %72 — ayak
yarası segmenter'ının bası yaralarına iyi genellediğini gösterir). Ön-eğitim havuzu
**1410 → 2142** (+%52). Eski 1410 sonuçları `runs_jepa/sweep_1410.*` olarak yedeklendi.

**Tuned JEPA yeniden eğitildi** (2142 havuz, vic=4.0 lr=3e-4, 30 epoch, MPS) →
`runs_jepa/jepa_best.pt`. Doğrulama (`validate_tissue_probe`, 5 seed + 2000 bootstrap)
→ `runs_jepa/tissue_validation.json`.

**Sonuç (dürüst NEGATİF):** daha çok + alan-içi veri, dondurulmuş doku probunda farkı
kapatmadı, hatta sildi:

| | Önce (1410) | Sonra (2142, +732 bası yarası) |
|---|---|---|
| JEPA macro-F1 | 0.537 | 0.521 |
| random ort. | 0.514 | 0.518 |
| JEPA − random | +0.023 (z≈0.59) | **+0.003 (z≈0.07)** |
| eşleştirilmiş Δ %95 CI | +0.028 [−0.009,+0.063] | **+0.002 [−0.041,+0.045]** |
| P(Δ>0) | 0.932 | **0.539** |
| Verdict | NOT established | NOT established |

**Neden:** eklenen veri **bası yarası**, ama doku test seti **ayak yarası** (DFUTissue)
→ test-alan uyumsuzluğu; ayrıca JEPA 0.537→0.521 gürültü içinde. Temel bulgu (linear-probe
+ minik model + küçük/alan-dışı test → JEPA ≈ random) pekişti. Yayınlanabilir dürüst bir
negatif sonuç.

**Yeni araç: `training/finetune_tissue_probe.py`** (push'landı). Dondurulmuş probe'un
adil olmayabileceğini test eder: encoder'ın son N bloğunu **çözüp** uçtan uca fine-tune
eder, JEPA-init vs random-init'i **aynı** ön-işleme + çok-seed + bootstrap ile kıyaslar.
SSL'in faydası çoğu zaman ancak fine-tune'da görünür. Koş (laptop'ta):
```bash
python -m training.finetune_tissue_probe \
    --weights runs_jepa/jepa_best.pt --crop --depth data/rgbd_tissue_labeled/depth \
    --relief --unfreeze-blocks 1 --epochs 40 --seeds 0 1 2 3 4 --bootstrap 2000 \
    --out runs_jepa/tissue_finetune.json
```
Not: Test seti ~16 görüntü olduğu için `--unfreeze-blocks` küçük tutuldu (varsayılan 1;
0=sadece head, 6=tam).

**ÖNEMLİ — ölçüm aleti düzeltmesi:** ilk sürüm skoru fine-tune'un kendi torch başlığıyla
hesaplıyordu ve doğrulanmış dondurulmuş probe'u yeniden üretemiyordu (unfreeze=0'da
JEPA 0.28/random 0.43 verip JEPA'yı haksız cezalandırıyordu → sahte "JEPA daha kötü"
sonucu). Düzeltildi: encoder fine-tune edilir (torch başlığı sadece gradyan taşıyıcı),
skor **her zaman** `validate_tissue_probe`'un sklearn probe'undan gelir. Artık
`--unfreeze-blocks 0` yerleşik bir sağlama: encoder değişmediği için tam olarak
dondurulmuş probe'u (JEPA 0.521 / random 0.518) yeniden üretir — doğrulandı ✅.

**Fine-tune merdiveni sonucu (2026-09-02, 5 seed + 2000 bootstrap, sklearn ölçümü):**

| Çözülen blok | JEPA-init | random-init | Δ (JEPA−random) | Δ>0 seed | P(Δ>0) |
|---|---|---|---|---|---|
| 0 (dondurulmuş) | 0.521 (std 0.000) | 0.518 (std 0.039) | +0.003 | 3/5 | — |
| 1 | 0.531 (std 0.008) | 0.515 (std 0.039) | +0.017 | 3/5 | 0.77 |
| 2 | 0.525 (std 0.014) | 0.496 (std 0.052) | +0.029 | 4/5 | 0.76 |

Desen: **JEPA'nın öne geçmesi uyarladıkça artıyor** (+0.003→+0.029) ve JEPA **çok daha
kararlı** (std ~0.01 vs 0.04–0.05); random, fazla uyarlamada (az veri) ezberleyip
bozuluyor. Yani JEPA az-veri fine-tune'unda daha iyi + daha güvenilir bir başlangıç.
**AMA** her basamakta bootstrap %95 CI hâlâ 0'ı içeriyor, P(Δ>0)~0.76 → istatistiksel
olarak **kesinleşmedi**; ölçek sınırı bulgusu sürüyor. Tutarlı ama ispatlanmamış olumlu
sinyal — dürüst ve yayınlanabilir. (JSON: `runs_jepa/tissue_finetune_b{0,1,2}.json`.)

**Sonraki olası adımlar:** (1) tam fine-tune (`--unfreeze-blocks 6`) merdivenin ucu için;
(2) asıl kaldıraç — bası-yarası için **etiketli** değerlendirme seti (şu an test ayak
yarası/DFUTissue, eklenen veri bası yarası → alan uyumsuzluğu); (3) evrelemede derinlik/relief.

## 🔴 MASKE KALİTESİ BULGUSU (2026-09-02) — Ario haklı çıktı, JEPA yükseldi

Ario "maskelere bak, Dice'larda sıkıntı var" dedi. Yeni araç `training/check_mask_quality.py`
(görsel montaj + GT'ye karşı Dice) ile YOLO yara maskelerini denetledik. **Maskeler kötü:**

| DFUTissue | boş tahmin | ort. Dice | Dice<0.5 |
|---|---|---|---|
| Test (conf 0.25) | %31 | 0.32 | %56 |
| TrainVal (conf 0.25) | %31 | 0.27 | %78 |
| Test (**conf 0.05**) | **%12** | **0.46** | %44 |

conf'u 0.25→0.05 yapmak boş tahmini yarıya indirdi, Dice'ı belirgin yükseltti. `precompute_rgbd`'ye
`--no-depth` eklendi (sadece maskeyi hızlı yeniden üretmek için). Havuz maskeleri conf 0.05 ile
yeniden üretildi (`data/rgbd_pool_c05/masks`), tuned JEPA temiz maskeyle yeniden eğitildi
(`runs_jepa/jepa_best_c05.pt`), doku probu tekrar koşuldu:

| Havuz maskesi | JEPA | random ort | Δ | P(Δ>0) |
|---|---|---|---|---|
| conf 0.25 (2142) | 0.521 | 0.518 | +0.002 | 0.54 |
| **conf 0.05 (2142)** | **0.545** | 0.518 | **+0.027** | **0.928** |

**Sadece maske kalitesini düzeltince** (aynı veri/model/test), JEPA 0.521→**0.545** (şimdiye kadarki
en iyi), JEPA−random farkı +0.002→**+0.027**, P(Δ>0) 0.54→**0.93**. Yani "JEPA ≈ random"ın bir
kısmı **ölçek değil, bozuk maske (bug)** kaynaklıymış — mask-güdümlü JEPA aç kalıyordu. Yine de
formel anlamlılık kılpayı geçilmedi (CI [−0.012,+0.062], z=0.68); maske hâlâ Dice ~0.46, daha iyi
segmentasyon çizgiyi geçirebilir. (JSON: `runs_jepa/tissue_validation_c05.json`.)

**Denenen sıradaki adımlar ve sonuçları:**
- **(a) Daha düşük conf → tükendi.** conf 0.02 ve 0.01, DFUTissue Test'te 0.05 ile aynı (%12 boş,
  Dice ~0.47'de plato). 0.05'in altında kazanç yok; conf lever'ı bitti, en iyi hali `jepa_best_c05` (0.545).
- **(b) Segmenter'ı DFUTissue ile fine-tune → BAŞARISIZ (bozdu).** `prepare_dfutissue_seg.py` ile
  DFUTissue TrainVal'i YOLO-seg formatına çevirip mevcut ağırlığı fine-tune ettik (`yolo segment train`,
  80 epoch). Sonuç maske **kötüleşti**: DFUTissue Test'te boş %12→**%50**, Dice 0.46→**0.27** (medyan 0).
  **Sebep:** DFUTissue etiketleri **doku bölgeleri** (fibrin/granülasyon), temiz **yara sınırı** değil;
  yanlış türde hedefle + 94 küçük görüntü FUSeg'in iyi yara-bulmasını bozdu. Bu segmenter **atıldı**
  (`runs/segment/runs/segment/.../yolo26n_dfu_ft-2`). Dürüst negatif bulgu: doku-etiketiyle yara
  segmenter'ı eğitilmez.

**Gerçek kalan kaldıraç — SAM (sıradaki büyük iş, henüz yapılmadı):** maske Dice tavanı (~0.47) minik
FUSeg-YOLO'nun **yetenek sınırı**, özellikle etiketsiz bası yaralarında. Doğru araç: **düz SAM**
(doğal RGB, 2B, kutu-promptlu) ile **YOLO kutusu → SAM hassas maske** boru hattı — etiket gerektirmez.
(MedSAM/MedSAM2 uygun değil: MedSAM radyoloji-ağırlıklı → RGB yara fotoğrafında domain gap; MedSAM2'nin
gücü 3B/video yayma → bizde 2B tek fotoğraf, işe yaramaz.) Havuzu SAM ile yeniden maskele → JEPA'yı
tekrar dene → hedef: farkı anlamlılık çizgisinin üstüne taşımak.

## 📋 OTURUM SKOR ÖZETİ (2026-09-02) — hepsi bir arada
**Doku probu, JEPA vs random (dondurulmuş, 5 seed + 2000 bootstrap):**

| Ön-eğitim | JEPA | random ort | Δ | P(Δ>0) |
|---|---|---|---|---|
| 1410 (ayak) | 0.537 | 0.514 | +0.028 | 0.93 |
| 2142 karışık, conf 0.25 | 0.521 | 0.518 | +0.002 | 0.54 |
| 732 sadece-bası, conf 0.25 | 0.505 | 0.518 | −0.009 | 0.43 |
| **2142 karışık, conf 0.05** | **0.545** | 0.518 | **+0.027** | **0.928** |

**Fine-tune merdiveni, JEPA-init vs random-init:** unfreeze 0/1/2 → Δ +0.003 / +0.017 / +0.029
(JEPA tutarlı önde + çok daha kararlı std ~0.01 vs 0.04-0.05; anlamlılık geçilmedi).

**Maske Dice (DFUTissue Test):** YOLO conf0.25 → 0.32 (%31 boş); conf0.05 → 0.46 (%12 boş, en iyi);
DFUTissue-fine-tune → 0.27 (%50 boş, atıldı).

**Genel dürüst sonuç:** Bu ölçekte JEPA ≈ random; en güçlü tek hamle **maske kalitesini düzeltmek**
oldu (JEPA 0.521→0.545, P 0.54→0.93). Sınır kısmen bug (maske), kısmen ölçek. Anlamlılık henüz yok;
bir sonraki kaldıraç SAM ile daha iyi maske.

## Proje
Bası yarası (bedsore) fotoğrafından **evre** ve **doku** analizi yapan **Sorbed**
sistemine **YOLO26** (yara tespiti + monoküler derinlik) ve **mask-güdümlü JEPA**
(etiketsiz temsil öğrenme) entegrasyonu. Nihai hedef: bir **manuscript**.
Mentör: **Ariorad Moniri**. Her şey **dizüstünde Apple MPS** ile çalışıyor (AWS yok).

## Ortam (devam için)
```bash
cd /Users/mericsen/Downloads/Sorbed-claude-bedsore-grading-system-wcd4ol
source .venv/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1
export SSL_CERT_FILE=$(python -c 'import certifi;print(certifi.where())')
```
- Python **3.14**, venv `.venv`, `sorbed[ml,pdf]` + `ultralytics` + `torch 2.13`.
- Veri: `data/corpus/` (GitHub'a gitmez, .gitignore). Modeller: `runs_jepa/`, `runs/segment/`.
- Repo: `github.com/Barlassen/Sorbed-x-YOLO-x-JEPA` (kod push'lu, senkron).

## ✅ Tamamlananlar
- **YOLO fine-tune** → mask mAP50 **0.911** (`runs/segment/runs_wound/yolo26n_fuseg/weights/best.pt`).
  Hazır COCO modeli yarayı bulamıyordu; fine-tune sonrası buluyor.
- **Mask-güdümlü JEPA** uçtan uca çalışıyor (YOLO otomatik maske → JEPA öğrenir).
- **Evreleme ablasyonu (PIID + Kaggle):** çok-seed'le random 0.806 > in-domain JEPA 0.740
  (fark gerçek). Ama alan-içi veri artınca fark 0.066 → 0.013'e düşüyor (kapanıyor).
- **Doku tespiti (DFUTissue):** patch-probu. Hyperparameter taraması → en iyi ayar
  **vic=4.0, lr=3e-4 → macro-F1 0.517**, ilk kez random ortalamasını (0.508) geçti.
- **Kod push'lu:** `train_jepa.py` (crop, `--depth` 2.5D, `--relief` MinIP/MIP,
  `--vic` anti-collapse/VICReg, `--step-log`), `precompute_rgbd.py`,
  `train_tissue_probe.py`, `train_grade_jepa.py`, `prepare_yolo_seg.py`,
  `yolo_backend.py`, `yolo_depth.py` (Sorbed'de derinlik varsayılanı artık "yolo").

## 📊 SONUÇ (2026-09-01) — doğrulama koşuldu, dürüst cevap alındı
**Tarama (`sweep_jepa.py`, 9 hücre, 30 epoch, 2.5D+relief, seed 0):** en iyi
**lr=3e-4, vic=4.0 → macro-F1 0.542** (tek random referansı 0.484, Δ +0.058).
Kritik desen: **vic (çökme freni) belirleyici** — üç `vic=0` satırı dibin dibinde
(0.42–0.48), vic=4 en iyiyi veriyor. Tek başına lr ya da vic yetmiyor; lr=3e-4+vic=0
en kötü (0.420), lr=3e-4+vic=4 en iyi (0.542). → *"JEPA'nın faydası çökme-önleyici
düzenlileştirmeye bağlı."* (`runs_jepa/sweep.json` + `.csv`)

**Doğrulama (`validate_tissue_probe.py`, jepa_best.pt, 5 seed, bootstrap 2000):**
- JEPA = **0.537** (deterministik).
- random 5 seed: **ort=0.514, std=0.039, min=0.477, max=0.564** → random tek sayı değil,
  bir bulut; en iyi random çekilişleri (0.552, 0.564) JEPA'yı geçiyor. Taramadaki 0.484
  şanssız-düşük bir çekilişmiş.
- JEPA vs random ort: **+0.023, z ≈ 0.59** (eşik z>1 sağlanmadı).
- Eşleştirilmiş bootstrap: **Δ = +0.028, %95 CI [−0.009, +0.063], P(Δ>0)=0.932**.
- **VERDICT: NOT established.** JEPA pozitife eğiliyor (%93 önde) ama bu ölçekte kanıt
  yok — CI 0'ı içeriyor, z<1. (`runs_jepa/tissue_validation.json`)

**Yorum:** Tek şanslı sayı (0.542 vs 0.484) bizi yanıltabilirdi; doğru istatistik
"neredeyse anlamlı ama değil" diyor. Sınır bir bug değil, **ölçek** — HANDOFF'un ana
bulgusunu doğruluyor. Yayınlanabilir, dürüst bir sonuç. Bunu sınırdan çıkaracak tek
şey **daha fazla veri** → yeni yara fotoğrafları işleniyor (aşağıya bak).

## 🔸 Yarım kalan / sıradaki iş
**Ana yarım iş — tuned ayarı doğrulama:** vic=4, lr=3e-4 ile çıkan 0.517 gerçek mi,
yoksa 6 denemenin en iyisini seçmekten gelen iyimser yanlılık mı? Doğrulama gerek:
1. **Çok-seed:** tuned ayarı 3-5 seed ile eğit, random'la dağılım kıyasla (eğitim rastgeleliği).
2. **Bootstrap (Ario istedi):** 16 görüntülük küçük test setini tekrarlı yeniden örnekleyip
   macro-F1 güven aralığı çıkar (test-seti belirsizliği).
3. İkisi birlikte → "JEPA random'ı gerçekten geçti mi" istatistiksel cevap.

> **DURUM (kod hazır, laptop'ta koşulacak):** `training/validate_tissue_probe.py`
> yazıldı ve push'landı. Üç maddeyi tek komutta yapıyor: çok-seed random dağılımı +
> JEPA'nın z-skoru, **görüntü-seviyesi** bootstrap %95 CI, ve aynı yeniden-örneklenmiş
> görüntülerde **eşleştirilmiş** (JEPA−random) delta CI + P(Δ>0). Not: `--init jepa`
> deterministik (sabit ağırlık → sabit özellik → lbfgs deterministik), o yüzden JEPA
> tek nokta; random ise seed'e göre dağılım. Verdict: JEPA>random ancak z>1 **ve**
> eşleştirilmiş Δ CI'sı 0'ın üstündeyse "desteklenmiş" sayılıyor. Koş:
> ```bash
> python -m training.validate_tissue_probe --weights runs_jepa/jepa_best.pt \
>     --crop --depth data/rgbd_tissue_labeled/depth --relief \
>     --seeds 0 1 2 3 4 --bootstrap 2000 --out runs_jepa/tissue_validation.json
> ```
> Çıkan JSON'u mentör raporuna ekle. (Ön-işleme bayrakları JEPA ön-eğitimiyle
> **birebir** aynı olmalı — probe ile aynı `--crop/--depth/--relief`.)

> **HYPERPARAMETER TARAMASI (kod hazır, laptop'ta koşulacak):**
> `training/sweep_jepa.py` yazıldı ve push'landı. "6 elle deneme"yi tek, kayıtlı,
> tekrar-üretilebilir artefakta çeviriyor: bir `(lr, vic)` ızgarasındaki her kombinasyon
> için JEPA'yı eğitir, doku probuyla (`train_tissue_probe`'un birebir eval'i) macro-F1
> ölçer, random-init referansıyla birlikte CSV+JSON'a en-iyi-önce sıralı döker, en iyi
> encoder'ı kaydeder. (lr, vic) dışındaki her şey + seed sabit → satırlar arası fark
> hyperparameter'a atfedilebilir. Kazanan yine ızgara maksimumu — güvenmeden önce
> `validate_tissue_probe.py` (çok-seed + bootstrap) ile doğrula. Koş:
> ```bash
> python -m training.sweep_jepa \
>     --images data/rgbd_pool/images --masks data/rgbd_pool/masks --depth data/rgbd_pool/depth \
>     --crop --relief --tissue-depth data/rgbd_tissue_labeled/depth \
>     --lrs 1e-3 3e-4 1e-4 --vics 0.0 1.0 4.0 --epochs 30 \
>     --out runs_jepa/sweep.json --save-best runs_jepa/jepa_best.pt
> ```

**Diğer açık başlıklar:**
- Derinlik/relief'i **evrelemede** dene (dokuda değil — derinlik oraya daha uygun,
  çünkü Evre 3↔4 derinliğe bağlı; doku ise renk işi).
- **Fine-tune** (enkoderi dondurmayı kaldır) — linear-probe minik SSL'e haksız bir test.
- Mentör raporunu relief + hyperparameter sonuçlarıyla güncelle.

## Faydalı komutlar
```bash
# Doku ablasyonu (relief'li)
python -m training.train_tissue_probe --init jepa --weights runs_jepa/jepa_relief.pt \
    --depth data/rgbd_tissue_labeled/depth --crop --relief
python -m training.train_tissue_probe --init random --depth data/rgbd_tissue_labeled/depth --crop --relief --seed 0

# 2.5D + relief JEPA ön-eğitim (tuned ayar)
python -m training.train_jepa --images data/rgbd_pool/images --masks data/rgbd_pool/masks \
    --depth data/rgbd_pool/depth --relief --crop --vic 4.0 --lr 3e-4 --epochs 30 \
    --device mps --workers 0 --out runs_jepa/jepa_best.pt --step-log runs_jepa/step_loss.csv
```

## Ana bulgu (dürüst)
Laptop ölçeğinde JEPA random'ı zorlu geçiyor; sınır **ölçek** (minik model + az veri +
linear-probe). Bir **bug değil** — tüm mekanik şüpheler (maske, kanal, collapse, crop,
derinlik ölçeği) düzeltilip doğrulandı. Doğru ayar + veri + görevle açık kapanıyor.
Gerçek test büyük ölçekte (büyük enkoder, çok veri).

## Referanslar
- Detaylı hafıza: `sorbed-internship-task` notu (tüm rakamlar/yollar).
- Mentör raporu: https://claude.ai/code/artifact/c0c284b5-6a86-4888-8e1a-aa788d238847
- Yolculuk belgesi (sade anlatım): https://claude.ai/code/artifact/86b7bd90-d975-4393-bbcb-fb890c3b1697
