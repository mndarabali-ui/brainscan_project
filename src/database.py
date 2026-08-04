import sqlite3
import os
from src.config import DATA_DIR

DB_PATH = os.path.join(DATA_DIR, "brainscan.db")

def get_db_connection():
    """Membuka koneksi ke database SQLite dan mengembalikan objek koneksi"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Mengembalikan hasil query sebagai dict-like object
    return conn

def init_db():
    """Menginisialisasi tabel-tabel di database jika belum ada"""
    # Pastikan direktori database ada
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 1. Tabel Pasien (Patients)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS patients (
        nik TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        age INTEGER,
        birth_date TEXT,
        gender TEXT,
        address TEXT,
        phone TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    # 2. Tabel Riwayat Scan (Scans)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS scans (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_nik TEXT NOT NULL,
        filename TEXT,
        modality TEXT,
        predicted_class TEXT NOT NULL,
        confidence REAL NOT NULL,
        radiology_report TEXT,
        original_image_b64 TEXT,
        heatmap_image_b64 TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (patient_nik) REFERENCES patients(nik) ON DELETE CASCADE
    )
    """)
    
    # 3. Tabel Distribusi Dataset
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS dataset_distribution (
        kelas TEXT PRIMARY KEY,
        sebelum_balancing INTEGER,
        setelah_balancing INTEGER
    )
    """)

    # 4. Tabel Histori Pelatihan
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS training_history (
        epoch INTEGER PRIMARY KEY,
        train_loss REAL,
        val_loss REAL,
        train_acc REAL,
        val_acc REAL,
        train_f1 REAL,
        val_f1 REAL,
        epoch_time_seconds REAL
    )
    """)

    # 5. Tabel Hasil Evaluasi Uji (Confusion Matrix)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS test_evaluation_results (
        actual_class TEXT,
        predicted_class TEXT,
        count INTEGER,
        PRIMARY KEY (actual_class, predicted_class)
    )
    """)
    
    conn.commit()
    conn.close()
    print(f"Database berhasil diinisialisasi di: {DB_PATH}")

def save_dataset_distribution_to_db(df):
    """Menyimpan data distribusi dataset ke database"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM dataset_distribution")
    for _, row in df.iterrows():
        cursor.execute("""
        INSERT INTO dataset_distribution (kelas, sebelum_balancing, setelah_balancing)
        VALUES (?, ?, ?)
        """, (row["Kelas"], int(row["Sebelum_Balancing"]), int(row["Setelah_Balancing"])))
    conn.commit()
    conn.close()

def save_training_history_to_db(df):
    """Menyimpan data histori training ke database"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM training_history")
    for _, row in df.iterrows():
        cursor.execute("""
        INSERT INTO training_history (epoch, train_loss, val_loss, train_acc, val_acc, train_f1, val_f1, epoch_time_seconds)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (int(row["Epoch"]), float(row["Train_Loss"]), float(row["Val_Loss"]), float(row["Train_Acc"]), float(row["Val_Acc"]), float(row["Train_F1"]), float(row["Val_F1"]), float(row["Epoch_Time_Seconds"])))
    conn.commit()
    conn.close()

def save_test_evaluation_to_db(cm, classes):
    """Menyimpan data confusion matrix ke database"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM test_evaluation_results")
    for i, act_cls in enumerate(classes):
        for j, pred_cls in enumerate(classes):
            cursor.execute("""
            INSERT INTO test_evaluation_results (actual_class, predicted_class, count)
            VALUES (?, ?, ?)
            """, (act_cls, pred_cls, int(cm[i, j])))
    conn.commit()
    conn.close()

def upsert_patient(nik, name, age=None, birth_date=None, gender=None, address=None, phone=None):
    """Menyisipkan atau memperbarui data pasien berdasarkan NIK"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO patients (nik, name, age, birth_date, gender, address, phone)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(nik) DO UPDATE SET
        name = excluded.name,
        age = excluded.age,
        birth_date = excluded.birth_date,
        gender = excluded.gender,
        address = excluded.address,
        phone = excluded.phone
    """, (nik, name, age, birth_date, gender, address, phone))
    conn.commit()
    conn.close()

def get_patient(nik):
    """Mengambil informasi pasien berdasarkan NIK"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM patients WHERE nik = ?", (nik,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return dict(row)
    return None

def add_scan_record(patient_nik, filename, modality, predicted_class, confidence, report_text, original_b64=None, heatmap_b64=None):
    """Menyimpan data riwayat pemeriksaan scan otak"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO scans (patient_nik, filename, modality, predicted_class, confidence, radiology_report, original_image_b64, heatmap_image_b64)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (patient_nik, filename, modality, predicted_class, confidence, report_text, original_b64, heatmap_b64))
    conn.commit()
    conn.close()

def get_patient_history(nik):
    """Mengambil riwayat scan dari pasien tertentu berdasarkan NIK"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT s.*, p.name, p.age, p.gender, p.birth_date, p.address, p.phone 
    FROM scans s
    JOIN patients p ON s.patient_nik = p.nik
    WHERE s.patient_nik = ?
    ORDER BY s.created_at DESC
    """, (nik,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

# Inisialisasi DB saat modul di-import pertama kali
init_db()
