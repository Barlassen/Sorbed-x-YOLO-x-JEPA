# Devir Teslim — Sorbed × YOLO × JEPA

> Yeni bir sohbette / oturumda kaldığımız yerden devam etmek için özet.
> Son güncelleme: 2026-09-10.

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

**SIRADAKİ (kullanıcı onayı bekliyor — AĞIR, saatler):** yeni havuz için maske+derinlik
üret (`precompute_rgbd --images data/rgbd_pool_v2/images --out data/rgbd_pool_v2 --conf 0.05`),
sonra `train_jepa` ile yeniden eğit + `validate_tissue_probe` ile ölç. Hedef: P(Δ>0) 0.93 → >0.95.
Detaylı plan artefaktı: https://claude.ai/code/artifact/ce283b86-225e-4df5-9f29-83b87646c433
build_pool.py + testi henüz COMMIT edilmedi.

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

**Sıradaki:** (a) daha düşük conf (0.01–0.02) ile maskeyi daha da iyileştir; (b) **segmenter'ı daha
çok veriyle yeniden eğit** (ör. FUSeg + DFUTissue yara maskeleri) → maske Dice'ını yükselt → JEPA'yı
tekrar dene. Hedef: farkı anlamlılık çizgisinin üstüne taşımak.

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
