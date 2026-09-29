# Devir Teslim — Sorbed × YOLO × JEPA

> Yeni bir sohbette / oturumda kaldığımız yerden devam etmek için özet.
> Son güncelleme: 2026-09-11.

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
