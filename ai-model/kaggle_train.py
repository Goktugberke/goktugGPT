"""
kaggle_train.py — GoktugGPT Kaggle Training Script

Bu script Kaggle notebook'unda çalışmak üzere tasarlanmıştır.
Kaggle'da yeni bir notebook aç ve aşağıdaki hücreleri yapıştır.

=====================================================================
HÜCRE 1 — Repo'yu klonla ve bağımlılıkları kur
=====================================================================

!git clone https://github.com/goktugberke/goktugGPT.git
%cd goktugGPT/goktugGPT
!pip install -q tqdm datasets

=====================================================================
HÜCRE 2 — Dataset'ten dosyaları kopyala
=====================================================================

Kaggle'da Dataset olarak yüklediğin dosyalar /kaggle/input/ altında gelir.
Dataset adını kendi yüklediğin isimle değiştir (aşağıda DATASET_NAME).

import shutil, os

DATASET_NAME = "goktugGPT-data"   # <-- Kaggle'da oluşturduğun dataset'in adı
INPUT_DIR = f"/kaggle/input/{DATASET_NAME}"

os.makedirs("data", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

shutil.copy(f"{INPUT_DIR}/train_clean.txt", "data/train_clean.txt")
shutil.copy(f"{INPUT_DIR}/val_clean.txt",   "data/val_clean.txt")
shutil.copy(f"{INPUT_DIR}/tokenizer.json",  "checkpoints/tokenizer.json")

print("Files copied:")
print(f"  data/train_clean.txt  — {os.path.getsize('data/train_clean.txt') / 1e6:.1f} MB")
print(f"  data/val_clean.txt    — {os.path.getsize('data/val_clean.txt') / 1e6:.1f} MB")
print(f"  checkpoints/tokenizer.json — ready")

=====================================================================
HÜCRE 3 — Eğitimi başlat
=====================================================================

!python train.py \\
    --config medium \\
    --data data/train_clean.txt \\
    --val-data data/val_clean.txt \\
    --checkpoint-dir checkpoints

=====================================================================
HÜCRE 4 — Modeli output klasörüne kaydet (Kaggle'dan indirme için)
=====================================================================

import shutil
shutil.copy("checkpoints/best_model.pt", "/kaggle/working/best_model.pt")
print("best_model.pt saved to /kaggle/working/ — indirmek için Output sekmesine git.")

=====================================================================
NOTLAR:
- Kaggle'da GPU seç: Settings > Accelerator > GPU T4 x2
- Session süre limiti: 12 saat (free tier), yetmezse --resume ile devam edilebilir
- Checkpoint adımları: her 1000 step'te otomatik kaydedilir
- Eğitim bitmeden notebook kapanırsa: checkpoint_step_XXXXX.pt dosyasıyla devam et
=====================================================================
"""

# Bu dosya doğrudan çalıştırılmak için değil, Kaggle talimatları içindir.
# Yukarıdaki hücreleri Kaggle notebook'una kopyala.
print(__doc__)
