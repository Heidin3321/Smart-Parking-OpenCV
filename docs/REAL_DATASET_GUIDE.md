# REAL CNRPark — Method and provenance (Week 8)

**Sumber:** https://cnrpark.it/ — subset original CNRPark, 12.584 crop gambar dari dua kamera (A, B) pada Juli 2015. Berlabel `free` (empty) dan `busy` (occupied). Arsip resmi: `CNRPark-Patches-150x150.zip` (36,6 MB).

**Tujuan:** Klasifikasi *status satu slot parkir per gambar*. Tidak sama dengan deteksi posisi semua slot atau tracking kendaraan pada video. Posisi ROI video masih harus disediakan Rifqi.

## Preprocessing dan klasifikasi

Menggunakan fitur yang sudah diterapkan dalam `detector.py`: resize 16x20 BGR, konversi HSV, normalisasi pixel, gabungan BGR+HSV, dan KNN OpenCV dengan `k=3` (dapat diubah). Dengan demikian hasil *real data* bisa dibandingkan terhadap baseline sintetis, tetapi performanya belum tentu bagus. KNN sederhana rentan ke perubahan sudut kamera dan pencahayaan.

## Metode pembagian data

Skrip membaca label dari direktori asli `A/free`, `A/busy`, `B/free`, `B/busy` dalam ZIP. Kelompok split ditentukan dari kombinasi kamera dan timestamp `YYYYMMDD_HHMM` pada nama gambar. Seluruh crop dari satu timestamp/kamera selalu masuk ke split yang sama. Seed 2026; 80% kelompok train dan 20% test (estimasi kelompok), sampling per kelas dilakukan **setelah** pemisahan.

**Limitasi metodologis:** Ini masih *within-dataset temporal holdout*, bukan kamera/tempat yang sama sekali baru. Tidak boleh melaporkan akurasi yang didapat sebagai 'akurasi sistem parkir dunia nyata'. Untuk evaluasi generalisasi, sebaiknya uji video dari lokasi lain dan tandai ground truth manual atau gunakan split antarkamera.

## Bukti yang boleh masuk laporan

- `dataset/real_cnrpark/dataset_summary.json` — jumlah data terambil, distribusi kelas, grup pengambilan.
- `results/real_cnrpark/evaluation.json` — akurasi, precision, recall, F1 pada crop *test* CNRPark.
- `results/real_cnrpark/predictions.csv` — prediksi aktual terhadap gambar uji.
- `results/real_cnrpark/real_test_gallery.jpg` — ilustrasi hasil klasifikasi pada crop foto asli.
- `results/real_video/annotated_video.mp4` hanya sesudah tes footage asli dengan ROI yang benar.

## Penggunaan data yang bertanggung jawab

Sebutkan penulis dataset, sumber, metode split, jumlah sampel, dan kesulitan generalisasi. Hindari menyebarkan foto asli atau arsip penuh tanpa mengecek ketentuan redistribusi. Dataset asli tidak disertakan dalam ZIP upgrade ini.
