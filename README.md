# Smart Parking OpenCV — Week 8 Integration Starter

**PIC Integrasi:** Ichsan  
**Proyek:** Smart Parking Occupancy Detection System  
**Status:** Prototipe integrasi pada **data sintetis**. Bukan sistem parkir siap produksi dan **belum teruji pada CCTV nyata**.

## Fungsionalitas saat ini

- Membaca video atau webcam (`--mode input`), tanpa memaksa webcam memakai ROI contoh.
- Memuat koordinat slot dari JSON dengan pemeriksaan ukuran frame.
- Menjalankan klasifikasi **KNN OpenCV** berdasarkan foto patch sintetis berlabel `empty` / `occupied` (`--mode detect`).
- Menampilkan kotak hijau/merah, ID slot, dan total status slot per frame.
- Menyimpan screenshot, video beranotasi (opsional), dan prediksi per-frame ke CSV.
- Memisahkan `main.py`, `preprocessing.py`, `detector.py`, `visualization.py` sehingga modul tim bisa diganti tanpa menulis ulang entry point.

## Kebutuhan

- Python 3.13.x pada Windows (yang sudah dipakai kelompok).
- `opencv-python==4.12.0.88`; NumPy versi kompatibel (sebelumnya dipakai `numpy==2.2.6`).
- **Dataset sintetis pendamping yang sudah diekstrak:**

```text
C:\Smart-Parking-OpenCV\
  main.py
  preprocessing.py
  detector.py
  visualization.py
  README.md                           # copy from README_INTEGRATION.md
  README_INTEGRATION.md
  requirements.txt                     # file existing user
  SmartParking_TestDataset\
    config\parking_slots.json
    dataset\parking_sample.mp4
    dataset\patches\train\empty\*.jpg
    dataset\patches\train\occupied\*.jpg
  docs\
  tests\
  results\
```

Aktifkan `.venv` pada terminal PowerShell yang ada di `C:\Smart-Parking-OpenCV`:

```powershell
.\.venv\Scripts\Activate.ps1
python -c "import cv2; print(cv2.__version__, hasattr(cv2, 'ml'))"
```

Jika `opencv-python` ternyata versi 5.0, gunakan `python -m pip install "opencv-python==4.12.0.88"` dahulu.

## Jalankan prototipe terintegrasi

**Deteksi otomatis (default, pada data simulasi):**

```powershell
python main.py --mode detect
```

**Simpan anotasi video dan CSV tanpa GUI:**

```powershell
python main.py --mode detect --no-window --save-video
```

**Uji cepat hanya 3 frame:**

```powershell
python main.py --mode detect --no-window --max-frames 3
```

**Uji webcam tanpa deteksi:**

```powershell
python main.py --mode input --source 0
```

**Video baru + koordinat baru:**

```powershell
python main.py --mode detect --source dataset/parking_real.mp4 --config config/parking_slots.json
```

**PERINGATAN:** Perintah video baru di atas baru sah setelah Rifqi menyediakan ROIs untuk video tersebut **dan** Fachriza mengadaptasi/melatih ulang detektor dengan data nyata. Jangan menafsirkan output classifier yang dilatih pada sintetis sebagai akurasi nyata.

**Shortcut:** Tekan `S` untuk screenshot, `Q` untuk berhenti saat jendela aplikasi terbuka.

**Output default:**

```text
results/integration/
  first_frame.jpg
  predictions.csv
  annotated_video.mp4             # hanya jika --save-video
  screenshot_frame_XXXX.jpg       # hanya jika pengguna menekan S
```

Ini folder output integrasi baru, jadi **tidak menimpa** `results/detection_demo.mp4` dan `results/detection_predictions.csv` dari prototipe sebelumnya.

## Pengujian otomatis

```powershell
python -m unittest discover -s tests -v
```

Tes unit berlaku tanpa dataset. Tes *pipeline* tiga frame akan dilewati jika `SmartParking_TestDataset` belum tersedia.

## Kontrak anggota

Baca [docs/TEAM_INTERFACE_CONTRACT.md](docs/TEAM_INTERFACE_CONTRACT.md). Format ROI harus memiliki `frame_width`, `frame_height`, dan `slots` (setiap slot `id`, `x`, `y`, `w`, `h`). Sistem **mengharuskan resolusi video sama dengan resolusi ROI**. `ParkingDetector.predict(frame_bgr, slots)` mengembalikan daftar hasil per slot dengan `status` berupa string `empty` atau `occupied`.

## Apa yang sudah / belum

| Aspek | Status |
|---|---|
| Webcam/video input | Sudah diuji oleh tim |
| Prediksi pada simulasi | Sudah diuji oleh tim (dataset sintetis) |
| Integrasi kode modular baru | Siap untuk diuji pada laptop tim |
| Video parkir nyata & ROI | Menunggu Rifqi |
| Adaptasi metode deteksi nyata | Menunggu Fachriza |
| Pengukuran akurasi footage nyata | Menunggu pengujian Lenno |

**Penting untuk ETS:** Jangan menyebut agreement `576/576` pada scene sintetis sebagai akurasi parkir nyata. Nilai baseline `48/48` pada patch sintetis juga bukan validasi independent dunia nyata. Cantumkan sumber asli data dan bukti pengujian yang benar-benar ada.

## Sinkronisasi GitHub

Repository kelompok: https://github.com/Heidin3321/Smart-Parking-OpenCV

Sebelum `git add .`, tambahkan isi `docs/GITIGNORE_APPEND.txt` ke `.gitignore` agar video/data dan file hasil tidak ikut di-push secara tidak sengaja. Jika video/dataset perlu dibagikan, cantumkan cara mendapatkannya di README atau gunakan penyimpanan yang disepakati kelompok.

Disarankan commit integrasi:

```powershell
git status
git add main.py detector.py preprocessing.py visualization.py README.md README_INTEGRATION.md docs tests .gitignore
git commit -m "Add modular Week 8 integration starter"
git push origin main
```

Lakukan `git status` sebelum dan setelah stage; jangan menimpa file teman yang sudah memiliki perubahan tanpa pengecekan.
