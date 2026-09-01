# Devir Teslim — Sorbed × YOLO × JEPA

> Yeni bir sohbette / oturumda kaldığımız yerden devam etmek için özet.
> Son güncelleme: 2026-08-25.

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
