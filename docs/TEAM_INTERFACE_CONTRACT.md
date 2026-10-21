# Kontrak Integrasi Tim — Smart Parking Week 8

**Tujuan:** Semua anggota dapat bekerja paralel tanpa saling menulis ulang `main.py`.

## Alur sistem

`main.py` (Ichsan) → `preprocess_frame()` (Rifqi) → `ParkingDetector.predict()` (Fachriza) → `draw_detection()` (Fachriza) → output / CSV / demo / evaluasi (Lenno).

### Rifqi — format ROI dan preprocessing

File yang dikirim:

- `dataset/parking_real.mp4` (atau link lokasi video beserta izin/sumber)
- `config/parking_slots.json`
- `preprocessing.py` berisi fungsi sesuai tanda tangan di bawah.

**Format JSON:**

```json
{
  "frame_width": 1280,
  "frame_height": 720,
  "slots": [
    {"id": "A1", "x": 101, "y": 108, "w": 138, "h": 174},
    {"id": "A2", "x": 273, "y": 108, "w": 138, "h": 174}
  ]
}
```

Koordinat ini hanya **contoh dari video simulasi**. Tidak boleh dipakai ulang pada video CCTV lain tanpa penentuan ROI baru. Semua ukuran piksel pada **frame original** sebelum resized/cropped. ROI persegi panjang sumbu sejajar; untuk tempat miring dapat ditambahkan poligon pada iterasi berikutnya, dengan kontrak baru yang disepakati.

```python
def preprocess_frame(frame_bgr: np.ndarray) -> np.ndarray:
    """Return BGR uint8 frame dengan dimensi (height, width, 3) TETAP."""
```

Saat ini default preprocessing adalah identitas (tidak mengubah frame), karena baseline memakai resize ROI ke 16×20 dan konversi HSV pada langkah *feature extraction*. Jika Rifqi menambah blur/threshold, Fachriza harus memastikan training dan prediksi menggunakan transformasi yang konsisten.

### Fachriza — format classifier dan visualisasi

File yang boleh diperbarui:

- `detector.py` — `ParkingDetector(train_dir).predict(frame_bgr, slots)`
- `visualization.py` — `draw_detection(frame_bgr, results, frame_index, dataset_label=...)`

**Contoh output `predict`:**

```python
[
    {"id": "A1", "status": "empty", "x": 101, "y": 108, "w": 138, "h": 174},
    {"id": "A2", "status": "occupied", "x": 273, "y": 108, "w": 138, "h": 174}
]
```

Semua slot harus memperoleh satu hasil. `id` dan koordinat harus sesuai ROI input, `status` **persis** `empty` / `occupied` (huruf kecil). `predict` jangan membaca `ground_truth.csv` atau menentukan status dari indeks frame. Output harus berasal dari pengolahan frame. Jika metode nyata tidak membutuhkan folder training, komunikasikan perubahan constructor ke Ichsan, bukan langsung mengganti `main.py` tanpa koordinasi.

**Prasyarat untuk nyata:** Algoritma KNN awal dilatih pada gambar sintetis, belum mewakili pencahayaan/perspektif video asli. Lakukan adaptasi model/fitur dan kalibrasi untuk footage yang dipilih; jika perlu gunakan baseline ROI thresholding atau model pre-trained setelah disepakati tim.

### Ichsan — integrasi `main.py`

- Memeriksa kontrak fungsi sebelum menggabungkan modul.
- Menjamin video dan JSON cocok ukurannya.
- Mencegah file video/hasil tes tak sengaja masuk repo.
- Menyimpan screenshot/log hasil uji, menghubungkan kode tim.

### Lenno — evaluasi

Minimal ground truth: buat tabel `(frame, slot_id, actual_status)` untuk sejumlah frame yang dipilih secara jelas. Bandingkan dengan `results/integration/predictions.csv` berisi `(frame, slot_id, predicted_status)`. Hitung yang benar/salah untuk sampel itu dan jelaskan jumlah, sumber serta batasannya. Jangan gunakan label simulasi sebagai hasil evaluasi CCTV nyata.

**Koordinasi branch GitHub:** Branch dapat menggunakan nama `feature/roi-rifqi`, `feature/detector-fachriza`, `feature/testing-lenno`; integrasi di `main` melalui review bersama. Upload hanya data yang aman serta berizin.
