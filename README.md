# Smart Parking Occupancy Detection System — Final Project

**Mata kuliah:** Teknologi Berkembang (A), Institut Teknologi Sepuluh Nopember  
**Kelompok:** 02  
**Repository:** https://github.com/Heidin3321/Smart-Parking-OpenCV

## 1. Deskripsi proyek

Smart Parking adalah prototipe pengolahan video parkiran menggunakan Python dan OpenCV untuk mengidentifikasi status setiap petak sebagai **EMPTY**, **OCCUPIED**, atau **UNCERTAIN**. Pengguna memilih video, menetapkan *Region of Interest* (ROI) untuk petak yang akan dianalisis, dan memberi label manual pada contoh data. Program mengekstraksi fitur dari crop ROI, melatih model scene-specific K-Nearest Neighbors (KNN), mengevaluasi prediksi pada sampel akhir video yang dipisahkan secara temporal, kemudian menghasilkan video dengan kotak warna dan ringkasan jumlah slot.

Sistem juga memakai estimasi gerak kamera berbasis ORB/RANSAC, dengan fallback *phase correlation*, untuk membantu menyesuaikan koordinat ROI antarfame. Ini adalah **prototipe berbasis video**, bukan sistem CCTV produksi atau model yang sudah teruji pada seluruh lokasi parkir.

## 2. Perangkat lunak dan dependensi

- Windows 10/11 dengan PowerShell.
- **Python 3.13** (disarankan agar sesuai dependensi yang dikunci).
- Paket dalam `requirements.txt`: `numpy==2.2.6` dan `opencv-python==4.12.0.88`.
- Video input dalam format yang didukung OpenCV (contoh: MP4).
- Jendela GUI OpenCV dan pemilih berkas Windows harus tersedia untuk workflow interaktif.

Gunakan virtual environment **di dalam folder proyek** agar paket tidak mengganggu instalasi Python/OpenCV global:

```powershell
cd C:\Smart-Parking-OpenCV-FINAL
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install --only-binary=:all: -r requirements.txt
.\.venv\Scripts\python.exe -c "import cv2, numpy; print('OpenCV:', cv2.__version__, '| NumPy:', numpy.__version__)"
```

Ganti path direktori pada baris pertama apabila proyek berada di folder lain. Semua perintah berikut dijalankan **dari direktori root proyek** dan memakai `\.venv\Scripts\python.exe`, bukan Python global.

## 3. Struktur penting proyek

```text
Smart-Parking-OpenCV/
  main.py                  # Entry point prototipe integrasi awal
  detector.py              # Ekstraksi fitur dan classifier KNN
  preprocessing.py         # Validasi/persiapan frame
  visualization.py         # Visualisasi
  scripts/
    parking_roi_wizard.py  # Workflow utama video baru: ROI -> label -> train/test -> MP4
    video_label_flow.py    # Human labeling, calibration, holdout, report
    run_motion_aware_parking.py  # Estimasi pergerakan kamera dan rendering
    test_new_parking_video.py    # Modul pendukung video/ROI/evaluasi
  config/new_parking_video/       # ROI tersimpan berbentuk JSON
  evaluation/new_parking_video/   # Label manusia untuk calibration dan test
  tests/                          # Automated unit/integration tests
  requirements.txt
  README.md
```

Folder `.venv/`, `dataset/`, dan `results/` merupakan lingkungan/data/hasil lokal; tidak perlu dimasukkan dalam arsip **source code**. Materi sample input dan/atau petunjuk memperolehnya disediakan terpisah dalam paket pengumpulan.

## 4. Menyiapkan input

Untuk eksperimen utama, kelompok menggunakan `Vidtest1.mp4`, yaitu video hasil pemrosesan dari footage aerial parkiran yang dicantumkan pada bagian **Sumber dataset**. Letakkan berkas `Vidtest1.mp4` di `demo_inputs/`, atau pilih lokasinya lewat dialog Windows.

**Catatan reproduksibilitas:** Konfigurasi `Vidtest1_a34ca0eb1f00_rois.json` dan label yang cocok disimpan menggunakan identitas yang mempertimbangkan **nama file dan fingerprint isi video**. Gunakan file `Vidtest1.mp4` yang sama tanpa mengganti nama atau mengubah isinya apabila ingin menggunakan ROI dan label hasil eksperimen utama. File footage baru dapat memakai ROI serta label baru.

## 5. Cara menjalankan aplikasi

### A. Demo dari awal: pilih video, gambar ROI dan label secara manual

```powershell
.\.venv\Scripts\python.exe scripts\parking_roi_wizard.py --pick --roi-mode manual --detector scene_knn --no-open
```

1. Dialog Windows muncul; pilih **video input mentah** (bukan video output anotasi).
2. Pada jendela ROI, seret mouse kiri untuk menandai tiap petak parkir. Tekan **Enter/S** untuk menyimpan, **U** untuk membatalkan aksi terakhir, atau **C** untuk menghapus semua ROI.
3. Pada jendela pelabelan, tekan **O** = OCCUPIED, **E** = EMPTY, atau **U** = UNKNOWN jika kondisi tidak jelas/tidak bisa dinilai; **B** untuk kembali; **Q** untuk menyimpan progres dan keluar.
4. Pastikan data **kalibrasi** memiliki contoh EMPTY dan OCCUPIED yang benar agar KNN khusus scene dapat dibentuk. Jangan mengubah label yang sebenarnya hanya untuk menyeimbangkan kelas.
5. Setelah pelabelan lengkap, program menjalankan training, evaluasi, dan rendering hasil secara otomatis.

Jika memakai **file yang sebelumnya sudah pernah diberi label**, identitas yang sama dapat memuat ROI/label tersimpan. Untuk mendemonstrasikan seluruh proses pelabelan dari awal tanpa mengganggu eksperimen Week 8, gunakan **salinan video dengan nama baru** dan lakukan pelabelan ulang.

### B. Reproduksi eksperimen utama Vidtest1 dari ROI/label tersimpan

```powershell
.\.venv\Scripts\python.exe scripts\parking_roi_wizard.py --video ".\demo_inputs\Vidtest1.mp4" --roi-mode existing --detector scene_knn --no-open
```

Perintah ini melatih ulang model berdasarkan sampel kalibrasi yang telah dilabel, menguji terhadap sampel holdout, lalu membuat video output. **Tidak perlu menggambar ROI atau memberi label ulang** bila berkas video dan JSON cocok.

### C. Buka hasil eksperimen utama

```powershell
ii ".\results\new_parking_video\Vidtest1_a34ca0eb1f00\report.html"
ii ".\results\new_parking_video\Vidtest1_a34ca0eb1f00\annotated_video_motion_roi.mp4"
```

Untuk run dengan nama video lain, lihat folder terbaru di `results/new_parking_video/`.

## 6. Output yang dihasilkan

Pada folder `results/new_parking_video/<identitas_video>/`, program dapat menyimpan:

- `annotated_video_motion_roi.mp4`: video dengan kotak ROI dan label EMPTY/OCCUPIED/UNCERTAIN serta penghitung per frame;
- `report.html`: ringkasan hasil pengujian yang dapat dibuka di browser;
- `report.json`: metrik dan metadata terstruktur;
- `holdout_predictions.csv`: detail hasil klasifikasi tiap sampel uji.

Kotak **hijau** menunjukkan EMPTY, **merah** menunjukkan OCCUPIED, dan **kuning** menunjukkan UNCERTAIN. Jumlah status pada satu frame merupakan **prediksi saat itu**, bukan akurasi model.

## 7. Evaluasi dan batas interpretasi

Pada pengujian terdokumentasi dengan `Vidtest1.mp4`, kelompok menggunakan **14 ROI manual**. Sampel kalibrasi diambil dari frame 107 dan 198; sampel pengujian dari frame 599 dan 715. Total terdapat **28 sampel uji berlabel** (22 OCCUPIED, 6 EMPTY):

| Pendekatan | Benar / diuji | Akurasi | UNCERTAIN |
| --- | ---: | ---: | ---: |
| `new_scene_knn` | 28/28 | 100,00% | 0 |
| `new_scene_reference` | 26/28 | 92,86% | 2 |
| `cnrpark_pretrained` | 12/28 | 42,86% | 0 |

Angka ini berasal dari **temporal holdout pada satu video yang sama**, bukan pengujian video/lokasi yang independen. Frame yang berdekatan dapat memiliki kemiripan yang tinggi. Karena itu, **100% bukan jaminan kinerja di semua situasi**. Baseline `cnrpark_pretrained` hanya muncul apabila dataset pelatihan CNRPark tersedia di `dataset/real_cnrpark/train/{empty,occupied}`; dataset tersebut **tidak termasuk dalam clone GitHub standar**.

Untuk menjalankan pengujian kode:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -t . -v
```

Perhatikan bahwa hasil unit test memeriksa fungsi kode dan alur, **bukan** pengganti pengukuran akurasi klasifikasi pada video independen.

## 8. Keterbatasan dan pengembangan lanjutan

- ROI manual dan label manusia masih diperlukan; usulan ROI otomatis belum dianggap selalu akurat.
- Hasil eksperimen utama berasal dari video dan kamera yang sama; performa pada scene lain belum terbukti.
- Sistem sensitif pada perubahan cahaya, oklusi, perbedaan sudut kamera, dan kegagalan estimasi gerak.
- Kualitas hasil dipengaruhi jumlah/keseimbangan sampel EMPTY dan OCCUPIED dalam data kalibrasi.
- Sistem belum diintegrasikan ke dashboard real-time, sensor lapangan, atau pengelolaan parkir produksi.

Pengembangan yang relevan meliputi evaluasi video independen, perluasan dataset, penyeimbangan kelas, otomatisasi ROI yang tervalidasi, dan peningkatan ketahanan tracking.

## 9. Sumber dataset dan referensi

- **Footage utama:** Shtohryn, *Aerial Footage of a Parking Lot*, Pexels, https://www.pexels.com/video/aerial-footage-of-a-parking-lot-5587732/ . Video `Vidtest1.mp4` merupakan video olahan/cuplikan yang digunakan kelompok. Hak atas footage sumber tetap mengikuti ketentuan pemilik/platform.
- **Baseline pembanding (opsional):** CNRPark dataset, apabila disediakan secara terpisah oleh peneliti/pengguna.
- **Pustaka utama:** Python, OpenCV, NumPy.
- **Kode kolaboratif kelompok:** https://github.com/Heidin3321/Smart-Parking-OpenCV .

## 10. Anggota kelompok

- Lenno Andhika Pramudya Arkadewa — 5026241057
- Fachriza Rizky Susanto — 5026241092
- Muhammad Rifqi Ahnaf Novanto — 5026241118
- Ichsan Ramadhan — 5026241126
- Muhammad Hanif Syahnandra — 5026241176

Rincian kontribusi aktual masing-masing anggota tercantum dalam dokumen terpisah `Appendix/Team_Contributions.pdf` pada paket pengumpulan final.
