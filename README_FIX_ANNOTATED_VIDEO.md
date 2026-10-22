# FIX: VIDEO DENGAN KOTAK ROI + DETEKSI (PARKIRAN BARU)

## Tujuan
**Masukkan VIDEO parkiran BARU -> gambar ROI sekali -> output MP4 berisi kotak ROI merah/hijau, ID slot, status EMPTY/OCCUPIED/UNCERTAIN, penghitung slot dan video bergerak.** Tidak hanya CSV.

Paket ini menggantikan **satu script** (`scripts/test_new_parking_video.py`) dari kit sebelumnya. Config ROI, label manual, dataset, dan output lama **tidak dihapus dan tidak di-reset**.

## Install di Windows PowerShell

```powershell
cd C:\Smart-Parking-OpenCV
Expand-Archive -LiteralPath "C:\Users\lenovo\Downloads\SmartParking_New_Video_Annotated_FIX.zip" -DestinationPath "C:\Smart-Parking-OpenCV" -Force
```

## Cara cepat: langsung VIDEO OUTPUT (TANPA harus label semua frame)

```powershell
python scripts/test_new_parking_video.py video --pick
```

1. Pilih file VIDEO (.mp4) parkiran baru, bukan gambar JPG.
2. Gambar ROI manual di window OpenCV, tekan **S**. Kalau ROI untuk video yang sama sudah pernah disimpan, ROI itu akan **dipakai ulang secara otomatis**.
3. Program memakai **model KNN CNRPark yang sudah dilatih**. Bila referensi kalibrasi untuk video baru sudah lengkap, mode `auto` memilih **reference classifier** untuk video tersebut.
4. Program mengolah video dan membuat **`annotated_video.mp4`** berisi video asli dengan **kotak, label dan penghitung**. Di Windows video akan dicoba **dibuka otomatis**.

Gunakan file video yang sama lagi untuk memakai ROI yang sudah dibuat:

```powershell
python scripts/test_new_parking_video.py video --video "C:\Users\lenovo\Downloads\NAMA_VIDEO_KAMU.mp4"
```

Paksa menggunakan CNRPark KNN (untuk mendemokan generalisasi tanpa kalibrasi dari video baru):

```powershell
python scripts/test_new_parking_video.py video --pick --detector cnrpark
```

Pakai model referensi setelah CALIBRATION label selesai (tanpa perlu semua label testing):

```powershell
python scripts/test_new_parking_video.py video --pick --detector scene_reference
```

## Cara buka MP4 kalau player tidak muncul otomatis

```powershell
$video = Get-ChildItem .\results\new_parking_video -Recurse -Filter annotated_video.mp4 | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($video) { ii $video.FullName } else { Write-Host "Belum ada MP4. Periksa output perintah video sebelumnya." }
```

Direktori output untuk setiap video: `results/new_parking_video/<identitas_video>/`

- **`annotated_video.mp4`** <- **INI OUTPUT UTAMA VIDEO YANG DIMINTA**
- `preview_annotated_video.jpg` <- screenshot hasil frame pertama
- `video_predictions.csv` <- file pendamping untuk evaluasi, **bukan** pengganti video

## Kalau sekaligus mau kalibrasi, evaluasi dan video final

```powershell
python scripts/test_new_parking_video.py wizard --pick
```

Setelah ROI dan labeling calibration+testing selesai, script menulis:
- `annotated_video.mp4` (video utama, otomatis dibuka)
- `annotated_new_scene_reference.mp4` (video referensi)
- `annotated_new_scene_knn.mp4` (video KNN kalibrasi, jika kedua kelas ada)
- `annotated_cnrpark_pretrained.mp4` (video baseline, jika data CNRPark tersedia)
- `report.html`, `report.json` dan CSV prediksi untuk evaluasi

Kalau label sudah lengkap dan cuma mau generate ulang video **tanpa jendela**:

```powershell
python scripts/test_new_parking_video.py run --video "C:\Users\lenovo\Downloads\NAMA_VIDEO_KAMU.mp4"
```

**Catatan akademik:** Mode cepat `video` menampilkan prediksi model yang tersedia, bukan skor akurasi. Untuk akurasi diperlukan label manual terpisah. Kalibrasi dari video yang sama bukan bukti generalisasi ke parkiran independen. Error kotak hijau pada mobil harus dilaporkan, bukan disembunyikan.

**Kondisi input:** ROI tetap memerlukan kamera relatif diam (tanpa pan/zoom/pergeseran signifikan). Kalau video drone bergerak, perlu stabilisasi atau tracking ROI.

**Bila error `Access is denied` saat menyimpan MP4:** tutup Windows Media Player/Movies & TV yang sedang membuka file `annotated_video.mp4`, lalu jalankan ulang.
