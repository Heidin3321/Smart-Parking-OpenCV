# Parking V3 — Manual Labelling, KNN Adaptation, and Honest Evaluation

**Tujuan:** memperbaiki kasus mobil terlihat tetapi diprediksi `EMPTY`, dengan data label manual khusus video Parking V3.

JANGAN mengganti ROI lagi ketika sudah mulai melabel. **File `parking_v3_slots_REVIEWED.json` harus tetap dan berisi 14 ROI valid.**

## Pemasangan

```powershell
cd C:\Smart-Parking-OpenCV
Expand-Archive -LiteralPath "$env:USERPROFILE\Downloads\SmartParking_V3_Retraining_Kit.zip" -DestinationPath . -Force
python -c "import json;print('ROI:',len(json.load(open('config/parking_v3_slots_REVIEWED.json'))['slots']))"
```

Paket hanya menambah script dan dokumen. **Tidak menimpa `main.py`, `detector.py`, `run_parking_v3.py`, ataupun konfigurasi ROI.** Membutuhkan `dataset/parking_v3_stabilized.mp4` dan `config/parking_v3_slots_REVIEWED.json`.

## 1. Label sampel manual (Ichsan/Lenno)

```powershell
python scripts/v3_adapt.py label
```

GUI memperlihatkan satu kotak ROI pada salah satu dari **7 frame**. Total **98 crop (14 ROI x 7 timestamp)**.

- `O` = **OCCUPIED** jika kendaraan berada di slot
- `E` = **EMPTY** jika slot kosong
- `K` = unclear/unknown (tidak dipakai untuk training atau evaluasi)
- `B` = kembali untuk mengoreksi label sebelumnya
- `Q` = simpan dan keluar. Menjalankan lagi akan **resume**, bukan mengulang.

Label manusia tersimpan di `evaluation/parking_v3_manual_labels.csv`. Jangan menyalin status dari output KNN! Idealnya dua anggota saling review label.

## 2. Buat training dan holdout yang tidak bocor antar-slot

```powershell
python scripts/v3_adapt.py prepare
```

**Split telah ditentukan sebelum label dibuat:** test slots = `P003`, `P010`, `P012`. Crop dari slot-slot tersebut tidak masuk `dataset/v3_calibrated/train` yang digunakan untuk evaluasi. Sisanya untuk training. Ini lebih jujur daripada split acak gambar berulang dari slot sama. Training dibalance antar kelas; test tidak dibalance.

- `dataset/v3_calibrated/train/empty|occupied` = pelatihan adaptasi, tanpa slot test.
- `dataset/v3_calibrated/test/empty|occupied` = holdout hanya dari 3 slot test.
- `dataset/v3_calibrated/deploy_train/empty|occupied` = **semua sampel berlabel**, *khusus membuat video demo setelah evaluasi*, BUKAN evaluasi holdout.

**Kelemahan:** 7 frame dari video pendek saling berkorelasi, dan belum membuktikan kemampuan pada lokasi parkir lain.

## 3. Bandingkan model lama vs model V3

```powershell
python scripts/v3_adapt.py evaluate
```

Menghasilkan:

- `results/v3_adaptation/v3_evaluation.json` (accuracy, precision, recall, F1, confusion matrix)
- `results/v3_adaptation/v3_model_comparison.csv`

Perbandingan menggunakan crop holdout yang sama, dan kedua model tidak memakai label holdout untuk training evaluasi. **Jangan mengasumsikan model V3 pasti lebih akurat.**

## 4. Video hasil adaptasi

Gunakan script V3 yang sudah ada di proyek. Untuk hasil video pada **seluruh 14 slot** (menggunakan train_all / deploy_train):

```powershell
python scripts/run_parking_v3_adapted.py --save-video
```

```powershell
ii .\results\parking_v3_adapted\annotated_video.mp4
```

Perhatian: video demo dilatih menggunakan contoh dari **semua** slot, jadi hasil visualnya tidak dapat digunakan untuk melaporkan nilai akurasi holdout. Gunakan hasil `evaluate` untuk metrik uji.

Untuk menjalankan dengan model adaptasi *tanpa* slot holdout (lebih fair sebagai inspeksi):

```powershell
python scripts/run_parking_v3.py --config config/parking_v3_slots_REVIEWED.json --train-dir dataset/v3_calibrated/train --output-dir results/parking_v3_holdout_model --save-video
```

## 5. GitHub

Jangan commit dataset/video atau hasil label tanpa alasan dan memeriksa lisensinya. Tambahkan ke `.gitignore`:

```gitignore
/dataset/v3_calibrated/
/results/v3_adaptation/
/results/parking_v3_adapted/
/results/parking_v3_holdout_model/
/evaluation/parking_v3_manual_labels.csv
```

Lalu:

```powershell
git status
git add scripts/v3_adapt.py docs/V3_RETRAIN_GUIDE.md .gitignore
git commit -m "Add Parking V3 manual labels and domain adaptation workflow"
git push -u origin feature/parking-v3
```

Jika belum berada di branch `feature/parking-v3`, lakukan `git status` dahulu dan pilih branch kerja yang benar. Jangan mengubah `main` tanpa review.

## Catatan untuk laporan Week 8

Cantumkan sumber CNRPark untuk baseline, video Pexels V3 sesuai sumber aktual jika diketahui, waktu/slot split, jumlah label nyata, confusion matrix, dan keterbatasan evaluasi pada cuplikan tunggal. Jangan sebut akurasi CNRPark 96,56% sebagai akurasi video V3.
