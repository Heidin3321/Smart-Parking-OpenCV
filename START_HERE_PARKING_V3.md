# Smart Parking V3 — mulai di sini

Paket ini mengganti **dataset demo aktif**, bukan menghapus data V2/250-ROI. Kode lama (`main.py`, `detector.py`, `preprocessing.py`) tidak ditimpa.

1. Extract ZIP ke root `C:\Smart-Parking-OpenCV`.
2. Jalankan `python scripts/preview_parking_v3.py`.
3. Jalankan `python scripts/run_parking_v3.py --max-frames 20 --save-video`.
4. Jalankan `python scripts/run_parking_v3.py --save-video`.
5. Lihat `results/parking_v3/annotated_video.mp4`.

Diperlukan folder `dataset/real_cnrpark/train` yang sudah dibuat di tahap sebelumnya. Jika tidak tersedia, program akan berhenti, bukan diam-diam melatih dari data sintetis.

**Penting:** ROI ada 14, masih DRAFT. Angka hijau/merah = prediksi yang belum divalidasi. Dilarang mengutip akurasi video tanpa label manual. Baca `docs/PARKING_V3_GUIDE.md` untuk metode dan evaluasi.

Video original pengguna juga ada di ZIP sebagai `dataset/parking_v3_original.mp4`. Kalau ingin menyalin langsung dari Downloads, gunakan (opsional):

```powershell
Copy-Item -LiteralPath "$env:USERPROFILE\Downloads\parkingv3.mp4" -Destination ".\dataset\parking_v3_original.mp4" -Force
```

Jika nama file di Downloads berbeda, sesuaikan dengan nama yang benar. Tidak perlu memindahkan file jika sudah extract paket.
