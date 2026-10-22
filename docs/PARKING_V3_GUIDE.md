# Parking V3 — video asli, ROI 14 slot, dan evaluasi

## Apa yang ada di paket

- `dataset/parking_v3_original.mp4` — salinan video dari pengguna (380 × 452, 30 FPS, sekitar 28,3 detik).
- `dataset/parking_v3_stabilized.mp4` — video yang disejajarkan ke frame awal melalui ORB + RANSAC (380 × 452, 10 FPS, 283 frame).
- `config/parking_v3_slots_DRAFT.json` — **14 kandidat** ROI untuk dua kolom × tujuh baris. Bukan hasil anotasi tervalidasi.
- `scripts/stabilize_parking_v3.py` — mengulang stabilisasi dari video original.
- `scripts/preview_parking_v3.py` — pratinjau ROI pada beberapa waktu.
- `scripts/edit_parking_v3.py` — mengoreksi atau menambah/menghapus ROI; simpan ke file REVIEWED.
- `scripts/run_parking_v3.py` — inferensi KNN memakai `detector.py`, `preprocessing.py`, dan `main.read_slots` milik kelompok.
- `scripts/evaluate_parking_v3.py` — evaluasi hasil memakai label manual terpisah.
- `evaluation/parking_v3_ground_truth_template.csv` — 42 label kosong yang harus diisi manual (frame 10, 100, 200 × 14 slot).

## Kenapa mengganti V2?

V2 memiliki parkiran ratusan slot dengan kamera jauh; sebagian ROI draft sulit diverifikasi. V3 memperlihatkan dua deret dengan pemisah dan garis putih jelas, sehingga 14 kandidat slot dapat dicek manual lebih cepat. Video masih memiliki pergeseran horizontal di sepanjang rekaman, sehingga gunakan versi stabilized dengan ROI di paket ini. ROI ini **tidak boleh** dipasangkan dengan video V2 atau video original V3 tanpa registrasi ulang.

## Prasyarat

Proyek sebelumnya harus mempunyai `main.py`, `detector.py`, `preprocessing.py`, dan environment `opencv-python==4.12.0.88`, serta folder `dataset/real_cnrpark/train/empty` dan `dataset/real_cnrpark/train/occupied`. **Tidak ada berkas lama yang ditimpa oleh ZIP ini.**

## Langkah menjalankan

Dari PowerShell di `C:\Smart-Parking-OpenCV`:

```powershell
python scripts/preview_parking_v3.py
python scripts/run_parking_v3.py --max-frames 20 --save-video
python scripts/run_parking_v3.py --save-video
ii .\results\parking_v3\annotated_video.mp4
```

Output: `results/parking_v3/first_frame.jpg`, `annotated_video.mp4`, dan `predictions.csv`.

## Review dan koreksi ROI

```powershell
python scripts/edit_parking_v3.py --time 10
```

Kontrol: drag kiri untuk tambah ROI, tekan M untuk mode pindah ROI lalu drag, klik kanan untuk hapus ROI terdekat, U undo, I tampilkan/sembunyikan ID, **S simpan**, Q keluar. Editor membuka video pada ukuran tampilan 2× agar garis kotak lebih mudah terlihat.

File yang tersimpan: `config/parking_v3_slots_REVIEWED.json`. Nama REVIEWED menunjukkan edit manual sudah disimpan, **bukan jaminan semua ROI pasti tepat**. Periksa di awal, tengah, dan akhir video:

```powershell
python scripts/preview_parking_v3.py --config config/parking_v3_slots_REVIEWED.json
python scripts/run_parking_v3.py --config config/parking_v3_slots_REVIEWED.json --save-video
```

## Evaluasi objektif

1. Jalankan inferensi V3 agar `results/parking_v3/predictions.csv` terbuat.
2. Buka `evaluation/parking_v3_ground_truth_template.csv`.
3. Buka frame pada index `10`, `100`, dan `200` di video stabilized (10 FPS). Tentukan secara **manual** `true_status` tiap ROI (`empty` / `occupied`). Jangan salin prediksi KNN.
4. Jalankan:

```powershell
python scripts/evaluate_parking_v3.py
```

Jika belum ada label manual, program sengaja berhenti tanpa menampilkan angka akurasi. Tabel evaluasi menghasilkan confusion matrix dan F1 untuk kelas occupied hanya pada sampel yang diberi label; tidak boleh digeneralisasi sebagai akurasi CCTV keseluruhan.

## Catatan metodologi

- Metode: OpenCV (ORB + RANSAC untuk stabilisasi), ROI manual, KNN (k=3) yang dilatih pada crop CNRPark untuk klasifikasi awal.
- Perubahan kelas pada video harus diverifikasi karena model CNRPark mungkin tidak akurat pada orientasi kendaraan horizontal dalam V3.
- Durasi 28,3 detik tidak menjamin semua slot mengalami pergantian kondisi; satu slot kanan terlihat berubah dalam preview, tetapi status akhir harus diverifikasi manual.
- Laporkan sumber footage sesuai asal nyata (termasuk lisensi bila bersumber dari platform), serta jangan klaim seluruh gambar/rekaman dibuat sendiri jika bukan.
- Batasan: ROI draft, occlusion, bayangan, resolusi 380x452 yang cukup kecil, domain shift CNRPark ke video V3.

## Kolaborasi

Rifqi: validasi ROI dan dokumentasi video/stabilisasi. Fachriza: cek dan kalibrasi prediksi KNN. Lenno: buat ground truth dan evaluasi, screenshot, slide/video demo. Ichsan: integrasi, GitHub, dan pelaporan.
