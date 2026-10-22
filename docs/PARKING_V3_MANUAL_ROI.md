# Smart Parking V3 — Manual ROI dari nol

Tujuan: mengganti 14 kotak ROI *draft* dengan gambar ROI buatan kamu sendiri tanpa menghapus file lama dan tanpa menyentuh `main.py`.

## Persiapan

Di direktori `C:\Smart-Parking-OpenCV`, pastikan file `dataset/parking_v3_stabilized.mp4` sudah ada. Jangan pakai ROI video V3 untuk footage V2, dan jangan gunakan ROI ini dengan video V3 original tanpa stabilisasi.

## Membuat ROI baru

```powershell
python scripts/draw_parking_v3_rois.py --time 10
```

Di jendela yang terbuka, belum ada kotak sama sekali. Tarik mouse kiri mengelilingi satu slot (usahakan garis batas slot masuk kotak). Ulangi untuk seluruh slot yang ingin diuji.

- Klik kiri + drag: **tambah ROI baru**.
- Klik kanan di dalam ROI: **hapus satu ROI**.
- `U`: undo.
- `I`: tampilkan/sembunyikan ID.
- `C` dua kali: bersihkan semua ROI.
- `S`: simpan saat itu juga.
- `Q`: **simpan dan keluar**.
- `Esc`: keluar tanpa menyimpan perubahan yang terakhir.

Hasil simpan: `config/parking_v3_slots_REVIEWED.json`. Kalau kamu membuka editor lagi, ROI yang sudah tersimpan akan dimuat otomatis; gunakan `--reset` bila benar-benar ingin menggambar ulang dari nol.

## Cek ROI pada beberapa frame

```powershell
python scripts/preview_parking_v3.py --config config/parking_v3_slots_REVIEWED.json
ii .\results\parking_v3\roi_review\roi_0s.jpg
ii .\results\parking_v3\roi_review\roi_10s.jpg
ii .\results\parking_v3\roi_review\roi_20s.jpg
```

Kalau video bergerak, jangan ubah koordinat hanya berdasarkan satu waktu. Gunakan **stabilized.mp4** untuk pembuatan ROI dan untuk prediksi.

## Jalankan KNN dengan ROI buatanmu

```powershell
python scripts/run_parking_v3.py --config config/parking_v3_slots_REVIEWED.json --max-frames 30 --save-video
python scripts/run_parking_v3.py --config config/parking_v3_slots_REVIEWED.json --save-video
ii .\results\parking_v3\annotated_video.mp4
```

**Penting:** Ketepatan kotak ROI **tidak otomatis memperbaiki akurasi KNN**. Model dilatih pada CNRPark yang berbeda sudut dan domain dengan V3. Ada mobil yang diprediksi `empty` meskipun ROI sudah benar. Itu masalah klasifikasi; perlu data crop V3 yang diberi label manual untuk kalibrasi atau model yang berbeda, lalu evaluasi pada frame/video yang independen. Jangan mengubah `status` secara manual untuk menyamarkan hasil prediksi.

## Git workflow

Jalankan dari branch `feature/parking-v3` (bukan `main`) dan commit file baru sesudah dicek:

```powershell
git add config/parking_v3_slots_EMPTY.json scripts/draw_parking_v3_rois.py docs/PARKING_V3_MANUAL_ROI.md
git add config/parking_v3_slots_REVIEWED.json
git commit -m "Add manually reviewed Parking V3 ROI configuration"
git push -u origin feature/parking-v3
```

Jangan commit hasil deteksi, video asli, atau dataset mentah ke GitHub tanpa alasan dan izin lisensi yang sesuai.
