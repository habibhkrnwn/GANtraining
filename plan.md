Langkah 1 — Struktur Folder Proyek

Buat struktur folder ini di komputer/server Anda:

text
carotid-diffusion-augmentor/
│
├── data/
│   ├── cubs_2021/          ← folder CUBS clinical (images, segmentations, CF, CSV)
│   ├── cubs_2022/          ← folder CUBS technical (images, LIMA-Profiles, SNR xlsx)
│   └── processed/          ← hasil preprocessing (masks, label CSV, split JSON)
│
├── src/
│   ├── data/
│   │   ├── dataset.py      ← PyTorch Dataset class untuk CUBS
│   │   ├── preprocess.py   ← buat IMC mask dari LI/MA profiles
│   │   └── augment.py      ← classical augmentation (flip, rotate, dll)
│   │
│   ├── model/
│   │   ├── diffusion.py    ← CDA (fine-tune LDM + ControlNet)
│   │   └── segmentation.py ← SAMA-UNet dan baseline (U-Net)
│   │
│   ├── train/
│   │   ├── train_cda.py    ← training Carotid Diffusion Augmentor
│   │   └── train_seg.py    ← training segmentasi (S0, S1, S2)
│   │
│   ├── eval/
│   │   ├── metrics.py      ← Dice, IoU, IMT error, FID, SSIM
│   │   └── evaluate.py     ← evaluasi model segmentasi
│   │
│   └── utils/
│       ├── io.py           ← baca .tiff, .txt profiles, CF
│       └── visualize.py    ← plot overlay segmentasi di citra
│
├── configs/
│   └── config.yaml         ← semua hyperparameter
│
├── notebooks/
│   └── 01_eda.ipynb        ← exploratory data analysis CUBS
│
└── requirements.txt

Langkah 2 — Prerequisites yang Harus Anda Install

Sebelum menulis kode apapun, pastikan environment ini siap:

bash
# Environment
python >= 3.10
pytorch >= 2.0 (dengan CUDA jika pakai GPU)

# Library utama
pip install torch torchvision
pip install diffusers accelerate          ← untuk Stable Diffusion / LDM
pip install controlnet-aux                ← untuk ControlNet preprocessing
pip install transformers
pip install monai                         ← medical image augmentation
pip install scikit-image pillow tifffile  ← baca/olah .tiff
pip install numpy pandas openpyxl         ← data processing
pip install scipy                         ← spline interpolation profil LI/MA
pip install pytorch-fid                   ← evaluasi FID synthetic vs real
pip install matplotlib seaborn            ← visualisasi
pip install pyyaml                        ← config management

Langkah 3 — Yang Harus Dibuat Pertama Kali (Urutan)

Ini urutan file Python yang harus dikerjakan dari awal sebelum training apapun:
Urutan prioritas:

1. src/utils/io.py — baca semua format data CUBS

    Baca .tiff → numpy array

    Baca .txt profil LI/MA (koordinat x, y tiap baris)

    Baca CF (mm/pixel) dari .txt

    Baca split_info.json

2. src/data/preprocess.py — bangun IMC mask dari profil

    Input: profil LI (batas atas IMC) + MA (batas bawah IMC) + ukuran gambar

    Interpolasi profil → kurva kontinu menggunakan spline

    Isi area antara LI dan MA → binary mask IMC

    Hitung IMT per kasus (rata-rata jarak LI–MA × CF mm/pixel)

    Tandai kasus "thin IMC" (IMT < 0.5 mm) dan "ambiguous" (STD IMT antar annotator tinggi)

3. notebooks/01_eda.ipynb — eksplorasi dataset

    Jumlah gambar per center

    Distribusi IMT (histogram)

    Distribusi SNR dari CUBS-tech xlsx

    Contoh visualisasi: gambar + overlay mask IMC

4. src/data/dataset.py — PyTorch Dataset

    CUBSDataset class

    Load gambar + mask + label (center ID, SNR level, thin/ambiguous flag)

    Normalisasi gambar ke [-1, 1] (standar diffusion)

    Crop/resize ke ukuran standar (mis. 256×256 atau 512×512)

    Return: {image, mask, center_id, snr_label, split}

Langkah 4 — Yang Perlu Anda Tahu Sebelum Mulai Coding

Tentang format profil CUBS:

    File profil LI/MA di CUBS adalah koordinat (x, y) per gambar, bukan mask langsung.

    Anda perlu menginterpolasi profil ini menjadi kurva kontinu, lalu mengisinya menjadi area mask — ini bagian terpenting dari preprocess.py.

Tentang ukuran gambar:

    Gambar CUBS berformat .tiff, ukuran bervariasi antar center — perlu distandarisasi sebelum masuk ke diffusion model.

Tentang CF mm/pixel:

    Setiap gambar punya faktor konversi mm/pixel yang berbeda.

    Ini wajib dipakai saat menghitung IMT dalam mm (bukan piksel) untuk clinical metric.

Ringkasan: Apa yang Harus Disiapkan Sekarang
Prioritas	Yang Disiapkan	Tujuan
🔴 1	Buat folder struktur proyek	Fondasi semua file
🔴 2	Install requirements	Pastikan library tersedia
🔴 3	Tulis io.py	Bisa baca semua format CUBS
🔴 4	Tulis preprocess.py	Bangun IMC mask dari profil
🟠 5	Buat EDA notebook	Pahami distribusi data sebelum training
🟠 6	Tulis dataset.py	DataLoader siap untuk training
🟡 7	Siapkan config.yaml	Semua hyperparameter di satu tempat