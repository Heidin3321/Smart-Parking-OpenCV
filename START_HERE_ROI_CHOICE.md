# Smart Parking — Manual / Auto / Existing ROI (Video Mode)

Satu command baru untuk memilih bagaimana menentukan ROI pada **video parkiran baru**. Tetap mengekspor **MP4 anotasi dengan ROI dan status**, memakai motion compensation yang sudah ada.

## Instalasi

Extract ZIP ini langsung ke folder `C:\Smart-Parking-OpenCV` (Merge/Overwrite boleh). **Tidak menimpa script lamamu.**

Dibutuhkan script dari dua paket sebelumnya:

- `scripts/test_new_parking_video.py` (New Video Annotated FIX)
- `scripts/run_motion_aware_parking.py` (Camera Motion ROI FIX)

Uji keberadaan dua file tersebut dengan `Test-Path` di PowerShell.

## Menampilkan MENU INTERAKTIF

```powershell
python scripts/parking_roi_wizard.py --pick
```

Pilih:
1. **MANUAL:** gambar semua ROI dari awal.
2. **AUTO:** program mencari garis petak parkir top-down dan mengusulkan kotak ROI, kemudian kamu review dan boleh menghapus/menambah.
3. **EXISTING:** pakai ROI video yang sebelumnya sudah disimpan (muncul jika ada).

**AUTO bukan jaminan semua slot ditemukan.** Algoritma memakai citra awal dan garis putih parkir, sehingga tidak cocok untuk marka pudar, sudut ekstrem, dan petak tidak beraturan. User **wajib** memeriksa kotak di GUI sebelum menyimpan.

### Tombol editor

| Input | Tindakan |
| --- | --- |
| Drag mouse kiri | Tambah ROI |
| Klik kanan pada kotak | Hapus ROI salah |
| `U` | Batalkan perubahan terakhir |
| `C` | Kosongkan semua ROI |
| `S` | Setujui ROI + mulai proses video |
| `Q` atau `Esc` | Batalkan tanpa menyimpan |

### Langsung pilih mode lewat terminal (tanpa menu)

```powershell
python scripts/parking_roi_wizard.py --pick --roi-mode manual
python scripts/parking_roi_wizard.py --pick --roi-mode auto --max-rois 12
python scripts/parking_roi_wizard.py --pick --roi-mode existing
```

Jika gambar top-down memiliki 2 kolom mobil tetapi Auto gagal mengenali separator vertikal, gunakan `--columns 2`.

```powershell
python scripts/parking_roi_wizard.py --pick --roi-mode auto --columns 2 --max-rois 14
```

Setelah ROI disimpan/dipilih, program otomatis menggunakan detektor yang tersedia dan melakukan **motion-aware ROI tracking** (ORB + phase correlation fallback).

### OUTPUT MP4 — BUKAN CSV DOANG

`results/new_parking_video/<nama_video_hash>/annotated_video_motion_roi.mp4`

Video ini memuat ROI yang bergerak menyesuaikan kamera dan label EMPTY/OCCUPIED/UNCERTAIN. File gambar preview, CSV prediksi, dan CSV tracking di folder yang sama.

Cari dan buka video terakhir dari PowerShell:

```powershell
$mp4 = Get-ChildItem .\results\new_parking_video -Recurse -Filter annotated_video_motion_roi.mp4 | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($mp4) { ii $mp4.FullName }
```

**Perhatian label lama:** Ketika kamu mengubah posisi/jumlah ROI, label kalibrasi sebelumnya dibackup dan tidak dipakai ulang karena terkait koordinat lama. Jika belum ada label baru, `--detector auto` mencoba KNN yang sudah dilatih sebelumnya. Gunakan `test_new_parking_video.py wizard` untuk membuat label kalibrasi baru sebelum memilih `--detector scene_reference`.

## Pengujian

```powershell
python -m unittest discover -s tests -p "test_parking_roi_choice.py" -v
```

Contoh frame hasil AUTO ada di `auto_roi_preview_v3.jpg` (preview dari contoh video V3, belum dikoreksi secara manual). Setelah memakai kamera baru, Auto harus diuji/ditinjau ulang; ini bukan object detection model yang sempurna.
