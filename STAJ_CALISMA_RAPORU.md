# Sorbed × YOLO × JEPA — Gün Gün Çalışma Raporu

**Proje:** Bası yaralarının (bedsore) tek fotoğraftan analizi — evre ve doku tipi tahmini
**Alt çalışma:** YOLO26 (yara tespiti + monoküler derinlik) ve mask-güdümlü JEPA (kendinden-denetimli temsil öğrenme) entegrasyonu
**Danışman:** Ariorad Moniri
**Ortam:** MacBook Air · Apple MPS · Python 3.14 · PyTorch 2.13 · Ultralytics — bulut yok (veri gerçek anonim hasta fotoğrafları olduğu için tüm işlem yerelde)
**Kod deposu:** github.com/Barlassen/Sorbed-x-YOLO-x-JEPA · geliştirme dalı: `claude/sorbed-details-heuk41`
**Çalışma dönemi:** 26 Ağustos – 4 Eylül 2026

> Not: Tarihler sürüm kontrolündeki (git) gerçek commit tarihlerinden alınmıştır. Her gün "Yapılan iş → Neden → Sonuç → Öğrenilen" akışıyla yazılmıştır.

---

## Amaç ve araştırma sorusu

Etiketli bası-yarası verisi az ve pahalıdır (bir klinisyenin elle işaretlemesi gerekir). Kendinden-denetimli öğrenme, **bol miktardaki etiketsiz** yara fotoğrafından faydalanmayı vaat eder. Bu stajdaki temel araştırma sorusu:

> **Mask-güdümlü JEPA ön-eğitimi, eğitilmemiş (rastgele ağırlıklı) bir encoder'dan ölçülebilir biçimde daha iyi bir görüntü encoder'ı üretiyor mu — bir dizüstünün izin verdiği ölçekte?**

Nihai hedef: bulguları bir makaleye (manuscript) dönüştürmek.

---

## Gün 1 — 26 Ağustos 2026
### YOLO26 + mask-güdümlü JEPA entegrasyonu ve evreleme ablasyonu

**Yapılan iş:**
- Projeye YOLO26 ve mask-güdümlü JEPA'yı bağlayan ilk entegrasyon iskeleti kuruldu.
- İlk "evreleme ablasyonu" (staging ablation) deneyi kuruldu: alan-içi veri miktarını değiştirip evre tahmininin nasıl etkilendiğine bakma.

**Neden:** Önce iki modelin (YOLO tespit/derinlik + JEPA temsil) tek bir boru hattında birlikte çalıştığını göstermek gerekiyordu.

**Sonuç / gözlem:** Evre etiketlerinde çok-seed rastgele taban (0.806) alan-içi JEPA'yı (0.740) geçti; ama alan-içi veri arttıkça aradaki fark **0.066'dan 0.013'e kapandı**. "Ölçek arttıkça JEPA yakınsıyor" hikâyesinin ilk işareti.

**Öğrenilen:** Rastgele bir ağ bile sanılandan güçlü bir taban oluşturur; JEPA'nın onu geçtiğini iddia etmek için dikkatli bir kıyas şart.

---

## Gün 2 — 27 Ağustos 2026
### Proje sözlüğü, teknik dokümantasyon ve monoküler derinlik

**Yapılan iş:**
- Proje için bir terimler sözlüğü ve teknik dokümantasyon yazıldı.
- YOLO26 monoküler derinlik, sistemin varsayılan derinlik motoru yapıldı.

**Neden:** Ekip/danışman için ortak bir kavram zemini oluşturmak; derinlik kanalının (evre 3↔4 ayrımı için kritik) tutarlı biçimde üretilmesini sağlamak.

**Sonuç:** Her fotoğraf için tek kameradan (monoküler) göreli derinlik haritası üretilebilir hale geldi.

**Öğrenilen:** "Monoküler derinlik" = tek fotoğraftan tahmini derinlik; iki kamera gerektirmez, yüzeyin çukur/tümsek bilgisini modele taşır.

---

## Gün 3 — 30 Ağustos 2026
### 2.5D girdi, doku probu ve ilk istatistiksel doğrulama altyapısı

**Yapılan iş:**
- **2.5D girdi** eklendi: RGB + derinlik + yüzey şeklini vurgulayan MinIP/MIP "relief" kanalı.
- **Çökme-önleyici (anti-collapse / VICReg)** terimi eklendi — modelin "her şeye aynı cevabı verme" tuzağına düşmesini engeller.
- **Doku probu** (tissue probe) kuruldu: dondurulmuş encoder + basit sınıflandırıcı → macro-F1. Encoder kalitesini ölçen ana termometre.
- **Çok-seed + bootstrap doğrulama** altyapısı yazıldı (`validate_tissue_probe`).
- İlk `HANDOFF.md` (oturum devir notu) oluşturuldu.

**Neden:** Encoder'ın "gerçekten bir şey öğrenip öğrenmediğini" ölçmenin dürüst bir yolu gerekiyordu. Test seti çok küçük (16 görüntü) olduğu için istatistiksel titizlik şart.

**Sonuç:** İki belirsizlik ekseni ölçülebilir hale geldi: (1) 5-seed ile eğitim/başlangıç gürültüsü, (2) 2000× bootstrap ile test-seti gürültüsü.

**Öğrenilen:**
- **macro-F1:** her doku sınıfını eşit sayan, dengesiz veride adil başarı ölçüsü.
- **Bootstrap:** elimizdeki 16 görüntüyü yerine koyarak tekrar tekrar örnekleyip "sonuç ne kadar şansa bağlı" ölçme yöntemi.
- **VICReg:** temsillerin çökmesini (tek noktaya toplanmasını) engelleyen düzenlileştirme.

---

## Gün 4 — 31 Ağustos 2026
### Hiperparametre taraması (lr × vic)

**Yapılan iş:**
- Yeniden üretilebilir bir JEPA hiperparametre taraması yazıldı (`sweep_jepa`): 9 hücrelik `lr × vic` ızgarası.
- Kod kalite düzeltmeleri (ruff satır uzunluğu, `zip strict=`).

**Neden:** JEPA'nın faydasının hangi ayarlara bağlı olduğunu görmek; en iyi ayarı bulmak.

**Sonuç:** Desen keskin çıktı: her `vic=0` satırı dipte (0.42–0.48); `vic=4` en iyi (**0.542**, lr 3e-4). Yani JEPA'nın faydası **çökme-önleyici VICReg terimine bağlı**.

**Öğrenilen:** Çökme freni olmadan (vic=0) model işe yaramaz temsiller öğreniyor. Doğru ayar: `lr=3e-4`, `vic=4.0`.

**Teknik not (terminal deneyimi):** Tarama ilk denemede yanlışlıkla CPU'da başladı ve çok yavaştı (`--device mps` unutulmuştu). Cihaz MPS'e alınıp tekil tuned modele geçilerek çözüldü.

---

## Gün 5 — 1 Eylül 2026
### Bootstrap / çok-seed doğrulama sonucunun kaydı

**Yapılan iş:**
- Tuned JEPA (`vic=4.0`, `lr=3e-4`, 30 epoch) ön-eğitildi, dondurulup problandı; sonuç HANDOFF'a işlendi.

**Neden:** Karar kuralını uygulamak: "JEPA > random" ancak z > 1 **ve** eşleştirilmiş Δ %95 güven aralığı 0'ın üstündeyse *kesinleşmiş* sayılır.

**Sonuç:** 1410 (ayak-yarası) havuzunda JEPA **0.537** vs random **0.514**, Δ **+0.028**, P(Δ>0) **0.93** — umut verici ama güven aralığı 0'ı içerdiği için "kesinleşmedi".

**Öğrenilen:**
- **Güven aralığı (CI) 0'ı içeriyor** = "gerçekte fark yok" ihtimalini eleyemedik → bilimsel olarak "kanıtlandı" diyemeyiz.
- **P(Δ>0)** bir olasılıktır (skor değil): 0.50 = yazı-tura, 0.93 = büyük olasılıkla önde ama %95 eşiğinin altında.

---

## Gün 6 — 2 Eylül 2026
### 732 gerçek bası-yarası fotoğrafının işlenmesi + fine-tuning değerlendirmesi

**Yapılan iş:**
- **732 gerçek anonim bası-yarası fotoğrafı** (32 vaka; koksiks, gluteal, sırt) işlendi ve JEPA ön-eğitim havuzuna eklendi.
- Ön-işleme her fotoğraf için maske + derinlik üretti; **524'ünde (%72) yara bulundu**. Havuz **1410 → 2142'ye (+%52)** büyüdü.
- **Fine-tuning (ince ayar) değerlendirmesi** yazıldı (`finetune_tissue_probe`): encoder'ı dondurmak yerine son N transformer bloğunu çözüp uçtan uca uyarlama.
- Bir **enstrüman (ölçüm) hatası** bulundu ve düzeltildi: fine-tune skorlaması torch başlığı yerine doğrulanmış sklearn probe üzerinden yönlendirildi.

**Neden:** Alan-içi (bası yarası) veriyle havuzu büyütmenin skoru yükseltip yükseltmediğini ve encoder'ın uçtan uca uyarlanınca öne geçip geçmediğini görmek.

**Sonuç:**
- **Negatif sonuç:** 2142 karışık havuz (conf 0.25) JEPA'yı yükseltmedi — JEPA **0.521** vs random **0.518**, Δ **+0.002**, P **0.54** (yazı-tura). Daha çok veri tek başına dondurulmuş ölçütte farkı açmadı.
- **Fine-tune merdiveni:** çözülen blok arttıkça JEPA-init'in random-init'e üstünlüğü **büyüdü** (Δ +0.003 → +0.017 → +0.029) ve JEPA çok daha **kararlıydı** (std ~0.01 vs 0.04–0.05).

**Öğrenilen:**
- **Enstrüman hatası:** yanlış skorlama yolu JEPA'yı 0.28 gösteriyordu (gerçekte 0.52). Ölçüm aracının kendisini doğrulamak, sonucu doğrulamak kadar önemli.
- Fine-tune altında JEPA daha iyi *ve* daha kararlı bir başlangıç sağlıyor; random-init az veride ezberleyip düzensizleşiyor.

---

## Gün 7 — 3 Eylül 2026
### Maske-kalitesi bulgusu (stajın en büyük kaldıracı)

**Yapılan iş:**
- Danışman Ario'nun sezgisi üzerine ("maskelere bak, Dice'lar tuhaf") bir maske-kalitesi kontrol aracı yazıldı (`check_mask_quality`): YOLO maskesini gerçek işaretlemeye karşı Dice ile ölçer + görsel montaj üretir.
- `precompute_rgbd`'ye `--no-depth` seçeneği eklendi (maskeleri yeni bir conf ile hızlı yeniden üretmek için).
- Havuz maskeleri düşük güven eşiğiyle (conf 0.05) yeniden üretilip **başka hiçbir şey değişmeden** JEPA yeniden eğitildi.

**Neden:** "JEPA ≈ random" sonucunun ölçek sınırından mı yoksa bozuk maskelerden mi kaynaklandığını ayırt etmek. Mask-güdümlü JEPA nereye bakacağını maskelerden öğrenir; maske bozuksa model yanlış yere bakar.

**Sonuç (kilit bulgu):**
- Maskeler gerçekten kötüydü: conf 0.25'te segmenter fotoğrafların **%31'inde yarayı tamamen kaçırıyordu**, ort. Dice **0.32**.
- Güven eşiğini düşürmek (0.25 → 0.05) boş tahminleri **%12'ye** indirdi, Dice'ı **0.46'ya** çıkardı (0.02/0.01'de 0.47 platosu).
- Sadece maske kalitesini düzeltmek JEPA'yı **0.521 → 0.545'e**, JEPA−random farkını **+0.002'den (P 0.54) +0.027'ye (P 0.93)** taşıdı.

**Öğrenilen:** "JEPA ≈ random"ın bir kısmı ölçek değil, modelin ana mekanizmasını aç bırakan bir **maske bug'ıydı.** Bu turun en büyük tek kazanımı veri değil, **maske kalitesi** oldu.

---

## Gün 8 — 4 Eylül 2026
### Segmenter iyileştirme denemesi (dürüst negatif), belgeleme ve yol haritası

**Yapılan iş:**
- Maskeleri ~0.47 Dice tavanının üstüne taşımak için segmenter, DFUTissue verisiyle fine-tune edildi (`prepare_dfutissue_seg` ile YOLO seg formatına dönüştürüldü, Test bölümü sızıntıya karşı ayrı tutuldu).
- HANDOFF güncellendi: conf platosu, başarısız DFUTissue segmenter fine-tune, sıradaki kaldıraç (SAM), tüm oturum skor özeti.

**Neden:** Maske kalitesinin en büyük kaldıraç olduğu kanıtlandığına göre, segmenter'ı daha çok veriyle güçlendirmeyi denemek mantıklıydı.

**Sonuç (dürüst negatif):** İşler kötüleşti — boş tahmin **%12 → %50**, Dice **0.46 → 0.27**. Neden: DFUTissue etiketleri *doku bölgelerini* (fibrin, granülasyon) işaretler, temiz *yara sınırını* değil. Yanlış hedefle (üstüne 80 epoch boyunca sadece 94 küçük görüntü) eğitmek segmenter'ın iyi yara-bulmasını bozdu. Model atıldı.

**Öğrenilen:** **Doku-sınıfı etiketleri, yara-segmentasyon etiketleri değildir.** Negatif sonuç başlı başına bir bulgudur ve dürüstçe raporlanır. Sıradaki doğru kaldıraç: etiket gerektirmeden maskeyi inceltmek için **SAM** (düz SAM, doğal RGB + 2B kutu-prompt; MedSAM radyoloji-ağırlıklı, MedSAM2 3B/video için — ikisi de bu 2B fotoğraflara uymaz).

---

## Genel sonuç tablosu (doku probu, macro-F1)

| Ön-eğitim havuzu | JEPA | Random | Δ | P(Δ>0) | Karar |
|---|---|---|---|---|---|
| 1410 · ayak yarası | 0.537 | 0.514 | +0.028 | 0.93 | kesinleşmedi |
| 2142 · karışık · conf 0.25 | 0.521 | 0.518 | +0.002 | 0.54 | kesinleşmedi |
| 732 · sadece-bası · conf 0.25 | 0.505 | 0.518 | −0.009 | 0.43 | kesinleşmedi |
| **2142 · karışık · conf 0.05** | **0.545** | 0.518 | **+0.027** | **0.93** | kesinleşmedi |

**Fine-tune merdiveni:** çözülen blok 0 → 1 → 2 arttıkça Δ = +0.003 → +0.017 → +0.029; JEPA std ~0.01 (random ~0.04–0.05).

**Maske kalitesi:** conf 0.25 → Dice 0.32 (%31 boş); conf 0.05 → Dice 0.46 (%12 boş); plato 0.47.

**Segmenter mAP50 (FUSeg):** 0.911.

---

## Üretilen araçlar (hepsi `claude/sorbed-details-heuk41` dalına push'landı)

| Araç | İşlevi |
|---|---|
| `validate_tissue_probe` | Çok-seed + bootstrap ile doku probu doğrulama (macro-F1, Δ, P, CI) |
| `sweep_jepa` | Yeniden üretilebilir `lr × vic` hiperparametre taraması |
| `finetune_tissue_probe` | Encoder'ın son N bloğunu çözüp uçtan uca fine-tune; doğrulanmış probe ile skorlama |
| `check_mask_quality` | YOLO maskesi vs gerçek yara Dice ölçümü + görsel montaj |
| `precompute_rgbd --no-depth` | Maskeleri yeni bir conf ile hızlı yeniden üretme |
| `prepare_dfutissue_seg` | DFUTissue → YOLO wound-seg dönüşümü (Test sızıntısız ayrılır) |

---

## Kazanımlar ve öğrenilen kavramlar

**Teknik kavramlar:** kendinden-denetimli öğrenme, JEPA, mask-güdümlü ön-eğitim, Vision Transformer (ViT), 2.5D girdi, monoküler derinlik, VICReg (çökme-önleme), doku probu, lineer probe, dondurulmuş encoder, fine-tune / blok çözme, Dice katsayısı, mAP50, güven eşiği (conf), macro-F1.

**İstatistiksel titizlik:** çok-seed, z-skoru, bootstrap, güven aralığı (CI), eşleştirilmiş fark (paired Δ), P(Δ>0), önceden belirlenmiş karar kuralı.

**Araştırma disiplini:**
- Negatif sonuçları saklamamak, dürüstçe raporlamak (2142 havuz, DFUTissue segmenter).
- "Kodda var" ile "kanıtlandı"yı ayırmak.
- Ölçüm aracının kendisini doğrulamak (enstrüman bug'ı).
- Danışman sezgisini deneyle sınamak (maske bulgusu).
- Veri mahremiyeti: gerçek hasta fotoğrafları yerelde tutuldu, buluta gönderilmedi.

---

## Sonuç (dürüst özet)

> Dizüstü ölçeğinde, mask-güdümlü JEPA yara-doku sınıflandırmasında rastgele bir encoder'ı **henüz istatistiksel olarak anlamlı biçimde geçmiyor** (tüm güven aralıkları 0'a değiyor); ancak JEPA tutarlı biçimde pozitife eğiliyor ve fine-tune altında hem önde hem çok daha kararlı. Bu dönemin en güçlü tek hamlesi bir yara-maskesi kalite kusurunu düzeltmek oldu; bu, farkı yazı-turadan (P 0.54) %93 güvene taşıdı. Tavanı belirleyen şey yöntemin kendisi değil, **maske kalitesi ve ölçek.**

**Sıradaki adımlar (kanıta göre öncelikli):** (1) SAM ile daha iyi maske, (2) ölçeği büyütme (daha büyük encoder/epoch, GPU), (3) daha çok alan-içi etiketsiz veri, (4) JEPA ayar taraması.
