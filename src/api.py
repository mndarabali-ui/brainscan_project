import os
import io
import base64
import shutil
import datetime

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from fpdf import FPDF

from src.config import CLASSES, OUTPUT_DIR, CHECKPOINT_DIR, download_model_from_hf
from src.preprocess import val_transforms, precheck_transforms
from src.models.precheck_model import BrainPreCheckModel
from src.models.classifier_model import BrainHybridModel
from src.gemini_client import generate_radiology_report
from src.explainability import generate_attention_heatmap

router = APIRouter()

# Pakai GPU kalau tersedia, kalau tidak fallback ke CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────
# Util tanggal Indonesia (dipakai di laporan PDF)
# ─────────────────────────────────────────────────────────────
_HARI_ID = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
_BULAN_ID = ["", "Januari", "Februari", "Maret", "April", "Mei", "Juni",
             "Juli", "Agustus", "September", "Oktober", "November", "Desember"]


def tanggal_indonesia_sekarang() -> str:
    """Format tanggal sekarang, misal: 'Selasa, 7 Juli 2026' — bukan tanggal tetap."""
    now = datetime.datetime.now()
    return f"{_HARI_ID[now.weekday()]}, {now.day} {_BULAN_ID[now.month]} {now.year}"


# ─────────────────────────────────────────────────────────────
# Helper: load checkpoint .pth ke dalam arsitektur model PyTorch
# ─────────────────────────────────────────────────────────────
def _load_pth_weights(model: torch.nn.Module, pth_path: str) -> bool:
    """
    Muat bobot dari file .pth ke sebuah model PyTorch yang sudah dibuat.
    Mendukung 2 format checkpoint:
      - langsung state_dict
      - dict dengan key "model_state_dict" (format checkpoint training)
    Return True kalau sukses, False kalau gagal (model tetap pakai bobot
    pretrained bawaan sebagai fallback).
    """
    try:
        ckpt = torch.load(pth_path, map_location=device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt)
        return True
    except RuntimeError as e:
        print(f"Warning: Checkpoint '{pth_path}' tidak kompatibel dengan arsitektur model: {e}")
        return False


# ─────────────────────────────────────────────────────────────
# Muat model .pth via PyTorch saat modul ini diimpor
# ─────────────────────────────────────────────────────────────
precheck_model: torch.nn.Module | None = None
hybrid_model: torch.nn.Module | None = None

try:
    print("Memuat model Precheck (.pth)...")
    precheck_model = BrainPreCheckModel().to(device)
    precheck_pth = (
        download_model_from_hf("best_precheck_model.pth")
        or os.path.join(CHECKPOINT_DIR, "best_precheck_model.pth")
    )
    if os.path.exists(precheck_pth):
        if _load_pth_weights(precheck_model, precheck_pth):
            print(f"Sukses memuat Precheck dari: {precheck_pth}")
    else:
        print("Warning: best_precheck_model.pth tidak ditemukan, menggunakan bobot pretrained bawaan.")
    precheck_model.eval()

    print("Memuat model Hybrid (.pth)...")
    hybrid_model = BrainHybridModel().to(device)
    hybrid_pth = (
        download_model_from_hf("hybrid_vit_efficientnet_brain_best.pth")
        or os.path.join(CHECKPOINT_DIR, "hybrid_vit_efficientnet_brain_best.pth")
    )
    if os.path.exists(hybrid_pth):
        if _load_pth_weights(hybrid_model, hybrid_pth):
            print(f"Sukses memuat Hybrid dari: {hybrid_pth}")
    else:
        print("Warning: hybrid_vit_efficientnet_brain_best.pth tidak ditemukan, menggunakan bobot pretrained bawaan.")
    hybrid_model.eval()

    print("Seluruh model AI berhasil dimuat.")
except Exception as e:
    print(f"Gagal memuat model AI: {str(e)}")


# ─────────────────────────────────────────────────────────────
# Helper: jalankan inference PyTorch
# ─────────────────────────────────────────────────────────────
@torch.no_grad()
def _run_precheck(tensor_image: torch.Tensor) -> tuple[int, float]:
    """
    Jalankan inference model Precheck.
    Return: (is_valid_idx, probability_score)
      is_valid_idx == 1  => gambar Valid (brain scan)
      is_valid_idx == 0  => gambar Invalid
    """
    logits = precheck_model(tensor_image)
    probs = F.softmax(logits, dim=1)
    pred_idx = int(torch.argmax(probs, dim=1).item())
    prob_score = float(probs[0][pred_idx].item())
    return pred_idx, prob_score


@torch.no_grad()
def _run_hybrid(tensor_image: torch.Tensor) -> tuple[int, float, np.ndarray | None]:
    """
    Jalankan inference model Hybrid.
    Return: (predicted_class_idx, confidence_score, attention_map_array)
    Attention diambil dari layer Transformer terakhir lewat
    forward_with_attention(), lalu dikonversi ke numpy supaya kompatibel
    dengan generate_attention_heatmap().
    """
    logits, last_attn = hybrid_model.forward_with_attention(tensor_image)

    if logits.ndim != 2 or logits.shape[0] < 1:
        raise RuntimeError(
            f"Output logits model Hybrid tidak valid. Shape: {tuple(logits.shape)}"
        )

    probs = F.softmax(logits, dim=1)
    pred_idx = int(torch.argmax(probs, dim=1).item())
    confidence = float(probs[0][pred_idx].item()) * 100

    attention = last_attn.detach().cpu().numpy() if last_attn is not None else None

    print("========================================")
    print("HYBRID INFERENCE")
    print("Logits shape        :", tuple(logits.shape))
    print("Attention tersedia  :", attention is not None)
    print("Predicted index     :", pred_idx)
    print("Confidence          :", confidence)
    print("========================================")

    return pred_idx, confidence, attention


# ─────────────────────────────────────────────────────────────
# Pydantic models
# ─────────────────────────────────────────────────────────────
class PatientCreate(BaseModel):
    nik: str
    name: str
    age: int = None
    birth_date: str = None
    gender: str = None
    address: str = None
    phone: str = None


class PDFDownloadRequest(BaseModel):
    patient_name: str
    patient_age: str
    patient_gender: str
    patient_nik: str
    patient_birth_date: str = ""
    patient_address: str = ""
    patient_phone: str = ""
    report_text: str


# ─────────────────────────────────────────────────────────────
# Endpoint: status server
# ─────────────────────────────────────────────────────────────
@router.get("/api/status")
def get_status():
    """Mengecek status online server dan ketersediaan model AI"""
    return {
        "status": "Online",
        "precheck_model_loaded": precheck_model is not None,
        "classifier_model_loaded": hybrid_model is not None,
        "device": str(device),
    }


# ─────────────────────────────────────────────────────────────
# Endpoint: data pasien
# ─────────────────────────────────────────────────────────────
from src.database import upsert_patient, get_patient, add_scan_record, get_patient_history

@router.post("/api/patients/")
def register_patient(patient: PatientCreate):
    """Menyimpan atau memperbarui data profil pasien"""
    try:
        upsert_patient(
            nik=patient.nik,
            name=patient.name,
            age=patient.age,
            birth_date=patient.birth_date,
            gender=patient.gender,
            address=patient.address,
            phone=patient.phone,
        )
        return {"status": "Success", "message": "Data pasien berhasil disimpan."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal menyimpan data pasien: {str(e)}")


@router.get("/api/patients/{nik}")
def get_patient_info(nik: str):
    """Mengambil data pasien berdasarkan NIK"""
    patient = get_patient(nik)
    if not patient:
        raise HTTPException(status_code=404, detail="Pasien tidak ditemukan.")
    return {"status": "Success", "patient": patient}


@router.get("/api/patients/{nik}/history")
def get_patient_scans_history(nik: str):
    """Mengambil riwayat scan pasien berdasarkan NIK"""
    try:
        history = get_patient_history(nik)
        return {"status": "Success", "history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal mengambil riwayat scan: {str(e)}")


# ─────────────────────────────────────────────────────────────
# Endpoint utama: analisis gambar scan otak
# ─────────────────────────────────────────────────────────────
@router.post("/api/analyze/")
async def analyze_brain_image(file: UploadFile = File(...), patient_nik: str = Form(None)):
    """
    Endpoint utama untuk mengunggah gambar scan otak, menjalankan pre-check,
    menjalankan klasifikasi penyakit, memvisualisasikan atensi model (XAI),
    dan menghasilkan laporan radiologi AI.
    """
    # 1. Validasi Ekstensi File
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Format file harus berupa gambar (PNG, JPG, JPEG).")

    try:
        # 2. Simpan file unggahan sementara untuk visualisasi heatmap
        os.makedirs("temp_uploads", exist_ok=True)
        temp_file_path = os.path.join("temp_uploads", file.filename)
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 3. Baca gambar dan konversi ke tensor PyTorch
        image = Image.open(temp_file_path).convert("RGB")

        # Precheck: normalisasi [0.5, 0.5, 0.5]
        precheck_tensor = precheck_transforms(image).unsqueeze(0).to(device)
        # Hybrid: normalisasi ImageNet
        hybrid_tensor = val_transforms(image).unsqueeze(0).to(device)

        # 4. TAHAP 1: Precheck
        is_valid = True
        precheck_prob_val = 0.99
        if precheck_model is not None:
            is_valid_idx, precheck_prob_val = _run_precheck(precheck_tensor)
            is_valid = (is_valid_idx == 1)

        if not is_valid:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            return {
                "status": "Invalid",
                "filename": file.filename,
                "message": "Gambar tidak dikenali sebagai scan otak yang valid (CT-Scan/MRI). Hubungi Administrator.",
                "precheck_confidence": f"{precheck_prob_val * 100:.2f}%",
            }

        # 5. TAHAP 2: Klasifikasi Utama
        if hybrid_model is None:
            raise HTTPException(status_code=500, detail="Model utama klasifikasi tidak termuat di server.")

        predicted_idx, confidence_score, attention = _run_hybrid(hybrid_tensor)

        print("========================================")
        print("DEBUG HYBRID")
        print("predicted_idx :", predicted_idx)
        print("confidence    :", confidence_score)
        print("CLASSES       :", CLASSES)
        print("jumlah kelas  :", len(CLASSES))
        print("attention shape:", getattr(attention, "shape", None))
        print("========================================")

        # Pastikan indeks hasil model sesuai dengan jumlah kelas di config.py.
        if predicted_idx < 0 or predicted_idx >= len(CLASSES):
            raise ValueError(
                f"Index prediksi tidak valid: {predicted_idx}. "
                f"Jumlah CLASSES hanya {len(CLASSES)}. "
                f"CLASSES={CLASSES}"
            )

        predicted_class = CLASSES[predicted_idx]

        # 6. TAHAP 3: Eksplanabilitas AI (XAI)
        heatmap_filename = f"heatmap_{os.path.splitext(file.filename)[0]}.png"
        heatmap_path = os.path.join(OUTPUT_DIR, "figures", heatmap_filename)

        os.makedirs(os.path.dirname(heatmap_path), exist_ok=True)

        if attention is not None:
            # Model memiliki output attention.
            generate_attention_heatmap(
                image_path=temp_file_path,
                save_name=heatmap_filename,
                attention_override=attention,
            )
            heatmap_available = True
        else:
            # Kondisi ini seharusnya jarang terjadi (forward_with_attention
            # selalu mengembalikan attention layer terakhir), tapi tetap
            # dijaga sebagai fallback supaya endpoint tidak gagal total.
            # Gunakan gambar asli sebagai fallback heatmap.
            # Ini BUKAN heatmap AI sungguhan; heatmap_available=False memberi
            # tanda ke frontend bahwa attention map tidak tersedia.
            shutil.copyfile(temp_file_path, heatmap_path)
            heatmap_available = False

            print(
                "WARNING: Attention map tidak tersedia dari model Hybrid. "
                "Menggunakan gambar asli sebagai fallback."
            )

        # 7. TAHAP 4: Laporan radiologi AI
        modality = "CT" if "ct" in file.filename.lower() else "MRI"
        report_text = generate_radiology_report(predicted_idx, confidence_score, modality)

        # 8. Encode heatmap dan gambar asli ke base64
        heatmap_path = os.path.join(OUTPUT_DIR, "figures", heatmap_filename)
        with open(heatmap_path, "rb") as img_file:
            heatmap_base64 = base64.b64encode(img_file.read()).decode("utf-8")

        with open(temp_file_path, "rb") as img_file:
            original_base64 = base64.b64encode(img_file.read()).decode("utf-8")

        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

        # 9. Simpan ke database (opsional)
        if patient_nik:
            try:
                add_scan_record(
                    patient_nik=patient_nik,
                    filename=file.filename,
                    modality=modality,
                    predicted_class=predicted_class,
                    confidence=confidence_score,
                    report_text=report_text,
                    original_b64=f"data:image/png;base64,{original_base64}",
                    heatmap_b64=f"data:image/png;base64,{heatmap_base64}",
                )
            except Exception as db_err:
                print(f"Gagal menyimpan riwayat scan ke database: {str(db_err)}")

        # 10. Kembalikan respons JSON
        return {
            "status": "Valid",
            "filename": file.filename,
            "modality_detected": modality,
            "prediction": {
                "class_name": predicted_class,
                "class_index": predicted_idx,
                "confidence": f"{confidence_score:.2f}%",
            },
            "precheck_confidence": f"{precheck_prob_val * 100:.2f}%",
            "radiology_report": report_text,
            "original_image_b64": f"data:image/png;base64,{original_base64}",
            "heatmap_image_b64": f"data:image/png;base64,{heatmap_base64}",
            "heatmap_available": heatmap_available,
        }

    except Exception as e:
        if "temp_file_path" in locals() and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        print("========================================")
        print("ERROR /api/analyze/")
        print(f"{type(e).__name__}: {e}")
        print("========================================")
        raise HTTPException(
            status_code=500,
            detail=f"Terjadi kesalahan internal analisis: {type(e).__name__}: {str(e)}"
        )


# ─────────────────────────────────────────────────────────────
# Endpoint: download laporan PDF
# ─────────────────────────────────────────────────────────────
@router.post("/api/download-pdf/")
def download_pdf(data: PDFDownloadRequest):
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("helvetica", size=10)

        # 1. Header (Kop Surat)
        pdf.set_font("helvetica", "B", 14)
        pdf.cell(0, 8, "PUSAT RADIOLOGI DIGITAL & DIAGNOSTIK AI", new_x="LMARGIN", new_y="NEXT", align="C")
        pdf.set_font("helvetica", size=9)
        pdf.cell(0, 5, "Jl. Semilasari Barat No. 88, Sektor Kecerdasan Buatan, Denpasar", new_x="LMARGIN", new_y="NEXT", align="C")
        pdf.cell(0, 5, "Email: support@brainscan.ai | Telp: (021) 555-2026", new_x="LMARGIN", new_y="NEXT", align="C")

        pdf.ln(3)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(5)

        # 2. Document Title
        pdf.set_font("helvetica", "B", 12)
        pdf.cell(0, 7, "DOKUMEN LAPORAN HASIL PEMERIKSAAN RADIOLOGI (OPINI AI)", new_x="LMARGIN", new_y="NEXT", align="C")
        pdf.ln(4)

        # 3. Patient Details
        pdf.set_font("helvetica", "B", 10)
        pdf.cell(0, 6, "I. IDENTITAS PASIEN & PEMERIKSAAN", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("helvetica", size=9)

        details = [
            ("Nama Pasien", data.patient_name, "Jenis Kelamin", data.patient_gender),
            ("Umur", f"{data.patient_age} Tahun", "Tanggal Lahir", data.patient_birth_date),
            ("NIK Pasien", data.patient_nik, "No. Telepon", data.patient_phone),
            ("Alamat", data.patient_address, "Tanggal Analisis", tanggal_indonesia_sekarang()),
        ]

        col_width = 40
        val_width = 55
        for row in details:
            pdf.set_font("helvetica", "B", 9)
            pdf.cell(col_width, 6, f"{row[0]}:", border=0)
            pdf.set_font("helvetica", "", 9)
            pdf.cell(val_width, 6, str(row[1]), border=0)

            pdf.set_font("helvetica", "B", 9)
            pdf.cell(col_width, 6, f"{row[2]}:", border=0)
            pdf.set_font("helvetica", "", 9)
            pdf.cell(val_width, 6, str(row[3]), border=0, new_x="LMARGIN", new_y="NEXT")

        pdf.ln(3)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(5)

        # 4. Report Text Content
        pdf.set_font("helvetica", "B", 10)
        pdf.cell(0, 6, "II. LAPORAN PEMERIKSAAN (RADIOLOGY REPORT)", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

        pdf.set_font("helvetica", "", 9.5)
        lines = data.report_text.split("\n")
        for line in lines:
            stripped = line.strip()
            if stripped.startswith(("1. ", "2. ", "3. ", "4. ")):
                pdf.ln(2)
                pdf.set_font("helvetica", "B", 10)
                pdf.multi_cell(0, 6, line, new_x="LMARGIN", new_y="NEXT")
                pdf.set_font("helvetica", "", 9.5)
            elif stripped.startswith(("* ", "- ")):
                pdf.set_font("helvetica", "", 9.5)
                pdf.set_x(15)
                pdf.multi_cell(0, 5, line, new_x="LMARGIN", new_y="NEXT")
            elif stripped.startswith(("*Catatan:", "Catatan:")):
                pdf.ln(4)
                pdf.set_font("helvetica", "I", 8.5)
                pdf.multi_cell(0, 4.5, line, new_x="LMARGIN", new_y="NEXT")
            else:
                pdf.multi_cell(0, 5, line, new_x="LMARGIN", new_y="NEXT")

        # 5. Signatures
        pdf.ln(15)
        current_y = pdf.get_y()

        if current_y > 240:
            pdf.add_page()
            current_y = pdf.get_y()

        pdf.set_font("helvetica", "", 9.5)
        pdf.set_xy(130, current_y)
        pdf.cell(60, 5, f"Denpasar, {tanggal_indonesia_sekarang()}", new_x="LMARGIN", new_y="NEXT", align="C")
        pdf.set_x(130)
        pdf.cell(60, 5, "Pusat Radiologi Digital & Diagnostik AI", new_x="LMARGIN", new_y="NEXT", align="C")

        pdf.ln(10)
        pdf.set_x(130)
        pdf.set_font("helvetica", "B", 9.5)
        pdf.cell(60, 5, "dr. _________________________, Sp.Rad", new_x="LMARGIN", new_y="NEXT", align="C")
        pdf.set_x(130)
        pdf.set_font("helvetica", "", 8.5)
        pdf.cell(60, 5, "NIP. ___________________________", new_x="LMARGIN", new_y="NEXT", align="C")

        pdf_bytes = bytes(pdf.output())

        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=Laporan_Radiologi_BrainScan.pdf"},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal memproses PDF: {str(e)}")