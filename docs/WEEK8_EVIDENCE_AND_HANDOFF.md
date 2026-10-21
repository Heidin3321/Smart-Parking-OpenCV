# Checklist Bukti Week 8 & Serah Terima

> Ini dokumen *template kerja*, bukan klaim bahwa item di bawah sudah selesai.

## Bukti teknis yang perlu disiapkan

- [ ] Terminal memperlihatkan `python main.py --mode detect` berhasil dan versi OpenCV sesuai.
- [ ] Screenshot satu frame output KNN pada **simulasi**, diberi label "synthetic demonstration".
- [ ] Rekaman/video output sintetis dilengkapi penjelasan keterbatasan.
- [ ] Sumber video nyata dan izin/ketentuan pemakaian dicatat (oleh Rifqi).
- [ ] File ROI dan contoh gambar yang memperlihatkan batas masing-masing slot (oleh Rifqi).
- [ ] Penjelasan metode dan parameter deteksi berdasarkan kode final (oleh Fachriza).
- [ ] Screenshot hasil deteksi pada video nyata *jika sudah ada* (oleh Ichsan/Lenno).
- [ ] Hasil uji manual untuk beberapa frame: prediksi, ground truth, jumlah benar/salah (oleh Lenno).
- [ ] README dapat diikuti anggota lain untuk menjalankan program (oleh Ichsan).
- [ ] Semua source, tutorial, dataset, dan referensi eksternal dicatat (semua anggota).

## Rencana report (9 bagian + appendix)

1. Introduction — Ichsan: masalah pencarian slot parkir, tujuan yang diajukan Week 7.
2. Proposed Solution — Ichsan: proses input→ROI→preprocessing→klasifikasi→visualisasi.
3. CV Method — Fachriza: KNN sintetis awal dan metode aktual yang terpakai, alasan serta batasan.
4. Dataset & Preprocessing — Rifqi: sumber, resolusi, jenis gambar/video, tahapan olah citra.
5. Implementation — Ichsan + Fachriza: modul, dependency, command running, flowchart.
6. Preliminary Results — Lenno: screenshot **aktual** dan label sumber simulasi/nyata.
7. Evaluation — Lenno: metode pengujian, ukuran sampel, error dan kendala.
8. Progress & Next Steps — Ichsan: selesai vs belum, target Week 9.
9. References — seluruh anggota: dokumentasi OpenCV, dataset, tutorial/paket pihak ketiga.

**Appendix A:** Seluruh anggota sesuai daftar kelompok Week 7; isi kontribusi aktual, bukan rencana.  
**Appendix B:** Pernyataan orisinalitas dan rujukan sumber, disetujui seluruh anggota.  
**Appendix C:** Platform GitHub + URL + bagaimana digunakan untuk version control.

## Materi Week 8 yang harus dikumpulkan

- `GroupXX_OpenCV_Progress_Report.pdf`
- `GroupXX_OpenCV_Progress_Slides.pdf`
- `GroupXX_OpenCV_Progress_Demo.mp4` (disarankan 3–5 menit; **rekam aplikasi berjalan**)
- `GroupXX_OpenCV_Source_Code.zip` (README, dependensi, source dan petunjuk dataset)

**Deadline resmi:** 24 Oktober 2026, pukul 23.59 WIB. **Target internal:** 24 Oktober, pukul 18.00 WIB.
