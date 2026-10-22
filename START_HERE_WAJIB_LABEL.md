# FIX: Pilihan ROI (Manual / Auto / Existing) + **WAJIB LABELING** + VIDEO MP4

Paket ini menggantikan jalur lama `parking_roi_wizard.py` yang langsung memproses video setelah ROI tanpa meminta label. Ini adalah pipeline lengkap yang sesuai untuk demo kelompok Smart Parking.

## Instalasi Windows PowerShell

```powershell
cd C:\Smart-Parking-OpenCV
Expand-Archive -LiteralPath "C:\Users\lenovo\Downloads\SmartParking_ROI_Label_Video_Flow_FIX.zip" -DestinationPath "C:\Smart-Parking-OpenCV" -Force
Test-Path .\scripts\parking_roi_wizard.py
```

Tutup dulu video MP4 di Media Player sebelum membuat output baru.

## 1 command utama — menu interaktif

```powershell
python scripts/parking_roi_wizard.py --pick
```

Pilih video parkiran ASLI, bukan video hasil anotasi. Terminal akan menampilkan pilihan:

```text
1. MANUAL   - gambar ROI sendiri dari awal
2. AUTO     - ROI dibuat otomatis (TANPA DRAG), tekan ENTER untuk setuju
3. EXISTING - ROI tersimpan, langsung lanjut LABELING (hanya jika pernah tersimpan)
0. Batal
KETIK ANGKA 1 atau 2 atau 3 lalu ENTER:
```

### Shortcut / tombol untuk ROI

- **MANUAL (ketik `1`, Enter):** drag kiri untuk tambah kotak, klik kanan untuk hapus kotak, `U` undo, `C` hapus semua, **`S` atau `Enter` simpan dan lanjut labeling**.
- **AUTO (ketik `2`, Enter):** kotak *SUDAH DIGAMBAR PROGRAM* ketika window terbuka. **TIDAK PERLU DRAG**. Periksa apakah setiap kotak tepat, lalu **tekan `Enter` atau `S`** untuk menyetujui dan lanjut labeling. Bila ada kotak salah, pengguna *boleh* koreksi dengan klik kanan lalu drag manual (opsional). `Q` batal.
- **EXISTING (ketik `3`, Enter):** langsung menggunakan ROI tersimpan, tanpa window gambar ulang; lanjut/resume labeling.
- **AUTO BENAR-BENAR TANPA WINDOW REVIEW** (lewati konfirmasi ROI saja, labeling tetap wajib):

```powershell
python scripts/parking_roi_wizard.py --pick --roi-mode auto --auto-accept --max-rois 12
```

Opsi Auto adalah heuristik garis marka OpenCV, **bukan jaminan** semua petak otomatis benar. Periksa gambar `auto_roi_preview.jpg` jika memakai `--auto-accept`; untuk pengujian yang valid, kotak yang keliru harus dikoreksi sebelum pemberian label.

## PELABELAN WAJIB: berlaku pada MANUAL / AUTO / EXISTING

Begitu ROI selesai, editor labeling tampil otomatis, **sebelum prediksi model ditampilkan**. Tiap sampel frame menyorot **satu ROI** untuk diberi status aktual:

| Tombol | Fungsi |
| --- | --- |
| **O** | OCCUPIED — terisi mobil |
| **E** | EMPTY — kosong |
| **U** | UNKNOWN — tidak jelas / tracking hilang |
| **B** | Kembali untuk revisi label |
| **Q** | Simpan progres, berhenti sementara |

Default **2 frame awal untuk kalibrasi** dan **2 frame akhir untuk pengujian** (tidak berarti harus memberi label untuk setiap frame video).

Contoh: 12 ROI × 4 frame = **48 jawaban label**. Sampel frame awal dipakai untuk melatih model scene baru, frame akhir *hanya* untuk evaluasi temporal holdout. Jika ROI dan label pada video itu telah lengkap dan tidak berubah, file label lama boleh digunakan ulang.

Jika terhenti di tengah, ulangi command yang sama dan pilih **`3` EXISTING**; label terakhir tersimpan. Kalau camera tracking LOST pada suatu sampel, gunakan `U`—jangan menebak O/E.

## Setelah label lengkap, tidak perlu command lagi

Program otomatis:

1. Membuat training patches dari *calibration* dengan ROI disesuaikan pergerakan kamera.
2. Melatih **KNN** untuk scene bila kedua kelas ada; reference classifier memakai contoh per-slot.
3. Menguji dengan sampel *test* yang dilabel manusia sebelum melihat prediksi.
4. Menghasilkan video anotasi dengan **motion-aware ROI**, label merah/hijau/kuning, dan penghitung slot.
5. Menyimpan accuracy, confusion matrix dan CSV hasil uji.

**Lokasi hasil utama:**

```text
results/new_parking_video/<nama_video_hash>/
  annotated_video_motion_roi.mp4  <-- OUTPUT VIDEO UTAMA
  preview_motion_roi.jpg
  auto_roi_preview.jpg           <-- hanya untuk Auto
  report.html                    <-- laporan akurasi
  report.json
  holdout_predictions.csv
  motion_roi_predictions.csv
  camera_motion_tracking.csv
```

Buka video terakhir:

```powershell
$vid = Get-ChildItem .\results\new_parking_video -Recurse -Filter annotated_video_motion_roi.mp4 | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($vid) { ii $vid.FullName }
```

Buka laporan:

```powershell
$rep = Get-ChildItem .\results\new_parking_video -Recurse -Filter report.html | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($rep) { ii $rep.FullName }
```

## Jika bingung, pilih ini

```powershell
python scripts/parking_roi_wizard.py --pick
```

- Tekan `2` lalu `Enter` di terminal.
- Di jendela Auto ROI tekan `Enter` untuk menerima kotak tanpa drag.
- Di jendela Label tekan O/E/U **pada tiap sampel**, sampai selesai.
- Video anotasi muncul otomatis setelah selesai.

## Batasan ilmiah

- Sistem mengompensasi *gerak kamera global* (ORB + RANSAC dengan phase-correlation fallback); kamera yang berputar jauh, zoom besar, atau tracking gagal masih dapat menimbulkan UNCERTAIN.
- Evaluasi dari frame akhir pada **video yang sama**, sehingga hanya *temporal holdout*, tidak membuktikan generalisasi ke video/sesi lapangan baru.
- Satu kelas saja pada frame awal tidak cukup untuk training KNN biner. Model reference masih dapat digunakan terbatas; untuk hasil representatif butuh contoh EMPTY dan OCCUPIED pada kalibrasi.
- Jika ROI Auto salah posisi, jangan mengandalkan hasil meskipun tampil angka akurasi; periksa dan koreksi ROI dulu.
