# Açık / kapalı / örtülü yüzey segmentasyonu

2026-09-08: Kullanıcının onayıyla tek bir modelin birden fazla yüzey durumunu aynı
fotoğrafta ayırabilmesi için v2 anotasyon, eğitim, tahmin ve değerlendirme hattı eklendi.

## Etiket sözleşmesi

| Maske ID | Ad | Anlam |
|---|---|---|
| 0 | background | Taslak lezyon dışında kalan arka plan; uzman doğrulaması bekler |
| 1 | intact_lesion | Bütünlüğü görünürde korunmuş, lezyonlu deri |
| 2 | open_surface | Görünür açık, örtüsüz yara yüzeyi |
| 3 | covered_surface | Slough/eskar benzeri doku ile örtülü yara yüzeyi |
| 255 | unknown | Yüzey durumu belirsiz veya değerlendirilemiyor |

Modelin dört çıkış kanalı 0/1/2/3'tür. Belirsizliği bağımsız bir doku gibi öğrenmesini
istemiyoruz: 255 eğitimde `ignore_index`, değerlendirmede referans dışlama maskesidir.
Tahminde isteğe bağlı düşük güven alanları 255 olabilir; güven eşiği varsayılan 0
(abstention yok), kalibre edilmiş değildir. Eşik kullanılacaksa doğrulama verisinde
seçilmeli, test verisinde ayarlanmamalı.

**Lezyon eksi açık yüzey = kapalı deri yapılmaz.** Sadece ayrı görsel poligonla
belirlenen bölge kapalıdır. Sınıflandırılmamış lezyon pikselleri 255 kalır. Bir
fotoğraf aynı anda 1, 2 ve 3 sınıflarını içerebilir. Farklı kesin yüzey sınıfları
çakışırsa exporter hata verir; aynı sınıfın poligonları birleşebilir. Belirsiz
poligonlar lezyon dışında kalan örtülü/görünmeyen bölgeleri de işaretleyebilir.
Pilot bölge sınırları kaba taslaktır; çevredeki arka plan etiketleri de onaysızdır.

Örtülü yüzey, kapalı/sağlam deri değildir. Pansuman, krem veya materyal kalıntısı
slough/eskar olduğu varsayılarak 3'e çevrilmez. Örneğin pilot case_002'nin sarı-tan
örtüsü, v1'de düşük kanaatli slough taslağı olmasına rağmen, v2 yüzey hedefinde
materyal ayrımı belirsiz kaldığı için 255 olarak tutuldu. Alt doku türleri v1'de
kayıtlı ayrı taslaklardır; v2 yüzey değişikliği onları uzman onayına dönüştürmez.

Yüzey etiketi klinik evre değildir. Bül, derin doku yaralanması, örtülü taban ve
önceki evre gibi bilgiler kesin bir açık/kapalı→evre eşlemesini engeller. Evre
adayları ayrı kayıttır; bu araçlar otomatik evre üretmez. Kaynak:
[NPIAP tanımları](https://cdn.ymaws.com/npiap.com/resource/resmgr/online_store/npiap_pressure_injury_stages.pdf).

## Pilot ve dosyalar

Yerel kök: `data/annotations/pressure_injury_v2/`. v1 korunur.

- `pilot/surface_drafts.json`: 8 görüntünün açıkça yazılmış görsel yüzey kararları.
  Önceki poligon yeniden kullanılmışsa bu kayıt doğrudan hangi poligonu seçtiğini
  ve görsel gerekçeyi belirtir; otomatik doku→yüzey dönüştürme uygulanmaz.
- `pilot/records/*.json`: Kaynak fotoğraf ve önceki anotasyon hash'leri, yüzey
  poligonları, ayrı evre adayı, nitel belirsizlik, maske yolu/hash'i ve inceleme durumu.
- `pilot/surface_masks/*.png`: Orijinal çözünürlükte 0/1/2/3/255 sınıf-ID maskeleri.
- `pilot/validity_masks/*.png`: 255=bilinen taslak piksel, 0=belirsiz. Buradaki
  255 geçerlilik kodudur; sınıf maskesindeki 255 ise belirsizdir. Karıştırmayın.
- `pilot/labelme/*.json`: Düzenlenebilir yüzey poligonları; klinik karar içermez.
- `pilot/review.html`: Fotoğraflı yerel inceleme raporu.
- `pilot/surface_qa.png`: Mavi=kapalı lezyon; pembe=açık; sarı=örtülü; gri=belirsiz.
- `surface_classes.json`, `summary.json`: Şema ve kapsam.

8 görüntü / 8 vaka: kapalı 2, açık 7, örtülü 1, belirsiz 7 görüntüde bulunuyor.
Bu sayılar toplanmaz; aynı görüntü birden çok sınıf içerir. 724 kaynak dosya
henüz anotasyon bekliyor. Özellikle örtülü yüzey tek vaka ile temsil ediliyor;
eğitim ve bağımsız değerlendirme için daha fazla uzman etiketli vaka gerekiyor.
Daha yüksek Dice veya klinik başarı gösterilmiş değildir.

Tüm kayıtlar `ai_draft_pending_clinician_review`, uygunluk bayrakları false,
`patient_group_id=null`, `split=unassigned`. Belirsizliğin low/medium değerleri
kalibre edilmiş olasılık değildir. v1 doku ID'leri, v2 yüzey ID'leri ve DFUTissue
ID'leri farklı sözleşmelerdir; tek bir etiket haritasında karıştırılamaz.

## Yeniden üretim ve uzman düzeltmesi

Komutları proje kökünden çalıştırın:

```sh
MPLCONFIGDIR=/tmp/sorbed_annotation_mpl .venv/bin/python -m scripts.export_surface_annotations --render
.venv/bin/python -m tests.test_surface_segmentation
```

Exporter yalnız v2 pilot taslaklarını yeniden üretir. Uzman düzeltmeleri ve onayı
ayrı sürümde korunmalı. LabelMe düzeltmesi otomatik raster/record güncellemez;
düzeltilen poligonlar yeniden rasterize edilmeli ve `surface_mask_sha256`
güncellenmeli. Onay yalnız gerçek incelemeyi kaydeder; idari bayrak değiştirmek
bağımsız referans üretmez. Uzman sürümünde `review_status=clinician_reviewed`,
`reviewer={id,date}` ve kullanım amacına göre uygunluk bayrağı gereklidir.

## Tek modelle eğitim ve değerlendirme

`training/train_surface_seg.py`: küçük bir encoder-decoder, ortak encoder ve
4 kanallı softmax başlık. Hazır ağırlık indirmez, sıfırdan başlar; bu başlangıç
modeli YOLO'nun veya JEPA'nın eğitilmiş alternatifi olarak sunulmaz. Giriş RGB,
varsayılan 256×256; maskeler nearest-neighbor ile ölçeklenir, 255 korunur. Kayıp
bilinen pikseller üzerinde cross-entropy'dir. Validation loss checkpoint seçer;
validation Dice ayrıca raporlanır ve eğitim çözünürlüğünde ölçüldüğü belirtilir.

Eğitim sadece gerçek uzman onaylı kayıtları kabul eder. Doğrulanmış hasta grubu,
vaka veya birebir fotoğraf kopyası bölümler arasında geçerse durur. Aynı bölümde
aynı görüntünün birebir kopyası bir kez yüklenir; çelişkili maskeler hata verir.
Test görüntüleri ve maskeleri eğitimde açılmaz. JSON içindeki evre alanı eğitim
girdisi değildir. Pilotla aşağıdaki eğitim komutu bilinçli olarak veri hazırlığı
hatası verir; hasta görüntüleriyle eğitim başlatılmadı.

Uzman sürümü ve hasta bazında bölümler hazır olduğunda örnek kullanım:

```sh
.venv/bin/python -m training.train_surface_seg \
  --records data/annotations/pressure_injury_reviewed/records \
  --out runs_surface/experiment_001 --device mps

.venv/bin/python -m training.predict_surface_seg \
  --weights runs_surface/experiment_001/best.pt \
  --records data/annotations/pressure_injury_reviewed/records \
  --out runs_surface/experiment_001/test_predictions --split test

.venv/bin/python -m training.evaluate_surface_seg \
  --records data/annotations/pressure_injury_reviewed/records \
  --predictions runs_surface/experiment_001/test_predictions \
  --prediction-schema runs_surface/experiment_001/test_predictions/surface_classes.json \
  --split test --out runs_surface/experiment_001/test_metrics.json
```

Tahmin logits'leri orijinal boyuta döndürülür; çıktı sınıf-ID PNG'dir. Evre
üretilmez. Değerlendirici maske boyutlarını sessizce değiştirmez. Her yüzey sınıfı
için pooled Dice, görüntü ortalama Dice, TP/FP/FN ve kaç görüntünün ölçüldüğü ayrı
raporlanır. Referans ve tahmin birlikte boşsa Dice `null` olur ve ortalamaya
katılmaz; referans boşken yanlış pozitif tahmin varsa Dice 0 olur. Referansta
255 olan pikseller dışlanır; bilinen pozitif pikselde tahminin 255 olması FN'dir.
Bu, tahminden kaçınarak Dice'ın yapay yükselmesini önler. Kısmi etiketlerde
bilinmeyen bölgedeki hatalar ölçülmez; etiket kapsamı sonuçlarla birlikte okunmalı.

Varsayılan değerlendirme uzman onayı ister. Yalnız taslaklar arası teknik uyum
incelemesi için `--allow-draft-reference --split all` kullanılabilir; rapor açıkça
`unreviewed_draft_agreement` yazar. Eski tek sınıflı YOLO maskeleri yeni üç yüzey
sınıfını tahmin etmez; bunları yüzey sınıflarına otomatik ayırıp yeni Dice değeri
üretmeyin. Şimdilik yeni sınıflar için gerçek model performansı raporu yok.

Bu 732 fotoğrafın mevcut JEPA havuzunda bulunduğu daha önce hash ile doğrulandı.
Gelecekte JEPA başlangıcı eklenecekse test hastalarını ön-eğitim havuzundan da
çıkarmak gerekir. Mevcut klinik runtime/evreleme renk kuralları değiştirilmedi;
bu sürüm veri ve araştırma hattıdır. Eğitilmiş, doğrulanmış modelin runtime'a
bağlanması ayrı bir sonraki adımdır.
