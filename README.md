# 🚀 Aplikasi Pengolahan Citra Digital (PCD) Berbasis FastAPI & OpenCV

Aplikasi web interaktif untuk berbagai tugas pengolahan citra digital, mulai dari operasi dasar, histogram, filtering, kompresi, hingga analisis warna dan tekstur. Dibangun dengan **FastAPI**, **OpenCV**, dan **Bootstrap** untuk tampilan modern dan responsif.

---

## ✨ Fitur Utama

- **Modul 1: Array RGB**  
  Tampilkan array mentah RGB dari gambar yang diunggah.

- **Modul 2: Operasi Dasar & Histogram**  
  - Grayscale, Histogram, Histogram Equalization, Histogram Specification.
  - Operasi aritmatika dan logika pada citra.

- **Modul 3: Filtering & Transformasi**  
  - Convolution, Zero Padding, Filter (Low/High Pass), Fourier Transform, Reduce Periodic Noise.

- **Modul 4: Dataset Wajah**  
  - Tambah dataset wajah dari webcam.

- **Modul 5: Deteksi & Analisis Kontur**  
  - Canny Edge Detection, Chaincode, Proyeksi Integral.

- **Modul 6: Kompresi Gambar**  
  - **Lossy (JPEG):** Kompresi dengan pengaturan kualitas, metrik PSNR & SSIM.
  - **Lossless (PNG):** Kompresi tanpa kehilangan data, cek identik/tidak.

- **Modul 7: Analisis Warna & Tekstur**  
  - Konversi ke berbagai ruang warna (RGB, XYZ, Lab, YCbCr, HSV, YIQ).
  - Analisis tekstur: statistik lokal, GLCM, LBP, Gabor.

- **Kompilasi Keseluruhan**  
  Proses semua fitur sekaligus dan tampilkan hasilnya dalam satu halaman.


---

## 📂 Struktur Project

```
.
├── main.py                # Entry point FastAPI
├── templates/             # Template HTML (Bootstrap, Jinja2)
│   ├── base.html
│   ├── compile.html
│   ├── ... (modul1-7, hasil, dsb)
├── static/                # File statis (hasil upload, gambar, dsb)
├── dataset/               # Dataset wajah (ignored by git)
├── venv/                  # Virtual environment (ignored by git)
└── .gitignore
```

---

## ⚙️ Cara Instalasi & Menjalankan

1. **Clone repository**
   ```bash
   git clone <repo-url>
   cd Tugas_FastApi-OpenCV
   ```

2. **Buat virtual environment & aktifkan**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependensi**
   ```bash
   pip install fastapi uvicorn opencv-python scikit-image matplotlib jinja2
   ```

4. **Jalankan aplikasi**
   ```bash
   uvicorn main:app --reload
   ```
   Buka browser ke [http://localhost:8000](http://localhost:8000)

---

## 📝 Catatan

- Folder `static/` dan `dataset/` otomatis dibuat saat aplikasi berjalan.
- Semua file hasil upload dan proses disimpan di `static/uploads/`.
- Untuk fitur webcam/dataset, pastikan device memiliki kamera dan izin akses.


---

## 👨‍💻 Author

**Geraldin Gysrawa**  
231511011 