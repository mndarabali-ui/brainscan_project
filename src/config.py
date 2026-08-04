"""
src/config.py — Modul Konfigurasi Terpusat
Brain Disease Classification Pipeline
Vision Transformer (google/vit-base-patch16-224) Pretrained
"""

import os
from pathlib import Path

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  HYPERPARAMETER UTAMA & SEED
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SEED        = 42
IMG_SIZE    = 224
BATCH_SIZE  = 16
NUM_CLASSES = 5
LR          = 5e-5
EPOCHS      = 30

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  GEMINI API KEY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Bisa pakai:
# - GEMINI_API_KEY   untuk local/server biasa
# - GEMINIAPIKEY     untuk FastAPI Cloud kalau underscore tidak diterima
GEMINI_API_KEY = (
    os.environ.get("GEMINI_API_KEY")
    or os.environ.get("geminiapikey")
    or ""
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  KELAS PENYAKIT OTAK (5 KELAS)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CLASSES = [
    "Alzheimer",
    "Intracranial_Hemorrhage",
    "Normal",
    "Stroke_Iskemik",
    "Tumor",
]

CLASS_DISPLAY = {
    "Alzheimer": "Alzheimer",
    "Intracranial_Hemorrhage": "ICH",
    "Normal": "Normal",
    "Stroke_Iskemik": "Ischemic Stroke",
    "Tumor": "Brain Tumor",
}

CLASS_COLORS = ["#4E79A7", "#F28E2B", "#59A14F", "#E15759", "#B07AA1"]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STRUKTUR DIREKTORI PROYEK
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BASE_DIR = Path(__file__).resolve().parent.parent

# ── Data Directories ──────────────────────────────────────────────────────
DATA_DIR           = BASE_DIR / "data"
RAW_DIR            = DATA_DIR / "raw"
INTERIM_DIR        = DATA_DIR / "interim"
PROCESSED_DIR      = DATA_DIR / "processed"
SPLITS_DIR         = DATA_DIR / "splits"

# ── Output Directories ────────────────────────────────────────────────────
OUTPUT_DIR         = BASE_DIR / "outputs"
CHECKPOINT_DIR     = OUTPUT_DIR / "checkpoints"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  HUGGING FACE MODEL REPO
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Bisa pakai:
# - HF_REPO_ID   untuk local/server biasa
# - HFREPOID     untuk FastAPI Cloud kalau underscore tidak diterima
HF_REPO_ID = (
    os.environ.get("HF_REPO_ID")
    or os.environ.get("HFREPOID")
    or "Marksnb/brain-hybrid-efficientnet-vit"
)

def download_model_from_hf(filename: str):
    """
    Download file model dari Hugging Face Hub kalau belum ada lokal.
    Kalau gagal, return None supaya aplikasi bisa fallback.
    """
    from huggingface_hub import hf_hub_download

    local_path = CHECKPOINT_DIR / filename

    if local_path.exists():
        return str(local_path)

    try:
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)

        return hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=filename,
            local_dir=str(CHECKPOINT_DIR),
        )

    except Exception as e:
        print(f"⚠️ Gagal download '{filename}' dari Hugging Face: {e}")
        return None

LOGS_DIR           = OUTPUT_DIR / "logs"
FIGURES_DIR        = OUTPUT_DIR / "figures"
TABLES_DIR         = OUTPUT_DIR / "tables"
REPORTS_DIR        = OUTPUT_DIR / "reports"

# ── File Paths Penting ────────────────────────────────────────────────────
TRAINING_LOG_FILE    = LOGS_DIR        / "training.log"
SPLITS_CSV_FILE      = SPLITS_DIR      / "train_val_test.csv"
BEST_MODEL_PATH      = CHECKPOINT_DIR  / "hybrid_vit_efficientnet_brain_best.pth"

# ─── Output Figures ───────────────────────────────────────────────────────
FIG_TRAINING_PERF      = FIGURES_DIR / "training_performance.png"
FIG_CONFUSION_MATRIX   = FIGURES_DIR / "confusion_matrix.png"
FIG_TRAINING_SUMMARY   = FIGURES_DIR / "training_summary.png"
FIG_CLASS_DIST         = FIGURES_DIR / "class_distribution_before.png"
FIG_AUGMENT_COMP       = FIGURES_DIR / "augmentation_comparison.png"

# ─── Output Tables & Reports ──────────────────────────────────────────────
TABLE_CLASSIF_REPORT   = TABLES_DIR  / "classification_report.csv"
TABLE_DATASET_DIST     = TABLES_DIR  / "dataset_distribution.csv"
TABLE_TRAINING_HISTORY = TABLES_DIR  / "training_history.csv"
TABLE_TEST_EVAL        = TABLES_DIR  / "test_evaluation_results.csv"
REPORT_AUDIT_SUMMARY   = REPORTS_DIR / "audit_summary.json"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MODE TRAINING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
QUICK_TEST = False

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TARGET AUGMENTASI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AUGMENT_TARGET_PER_CLASS = 19097

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  INIT_FOLDERS — Buat Semua Direktori Otomatis
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def init_folders() -> None:
    """
    Membuat seluruh struktur folder proyek yang diperlukan jika belum ada.
    """
    all_dirs = [
        RAW_DIR,
        INTERIM_DIR,
        PROCESSED_DIR,
        SPLITS_DIR,
        CHECKPOINT_DIR,
        LOGS_DIR,
        FIGURES_DIR,
        TABLES_DIR,
        REPORTS_DIR,
    ]

    for d in all_dirs:
        os.makedirs(d, exist_ok=True)
