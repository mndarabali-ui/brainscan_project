import os
import datetime
import libsql_client

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TURSO CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TURSO_DATABASE_URL = (
    os.environ.get("TURSO_DATABASE_URL")
    or os.environ.get("tursodatabaseurl")
    or ""
)

TURSO_AUTH_TOKEN = (
    os.environ.get("TURSO_AUTH_TOKEN")
    or os.environ.get("tursoauthtoken")
    or ""
)

_client = None


def _get_client():
    """Buat koneksi ke Turso sekali saja, lalu dipakai ulang (lazy init)."""
    global _client
    if _client is None:
        if not TURSO_DATABASE_URL:
            raise RuntimeError(
                "TURSO_DATABASE_URL belum di-set di environment variable."
            )
        _client = libsql_client.create_client_sync(
            url=TURSO_DATABASE_URL,
            auth_token=TURSO_AUTH_TOKEN,
        )
        _init_tables(_client)
    return _client


def _init_tables(client):
    client.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            nik TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            age INTEGER,
            birth_date TEXT,
            gender TEXT,
            address TEXT,
            phone TEXT,
            updated_at TEXT
        )
    """)
    client.execute("""
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_nik TEXT NOT NULL,
            filename TEXT,
            modality TEXT,
            predicted_class TEXT,
            confidence REAL,
            report_text TEXT,
            original_b64 TEXT,
            heatmap_b64 TEXT,
            created_at TEXT
        )
    """)


def _row_to_dict(rs, row):
    return dict(zip(rs.columns, row))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FUNGSI YANG DIPAKAI src/api.py
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def upsert_patient(nik, name, age=None, birth_date=None, gender=None, address=None, phone=None):
    client = _get_client()
    client.execute(
        """
        INSERT INTO patients (nik, name, age, birth_date, gender, address, phone, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(nik) DO UPDATE SET
            name=excluded.name, age=excluded.age, birth_date=excluded.birth_date,
            gender=excluded.gender, address=excluded.address, phone=excluded.phone,
            updated_at=excluded.updated_at
        """,
        [nik, name, age, birth_date, gender, address, phone,
         datetime.datetime.now().isoformat()],
    )


def get_patient(nik):
    client = _get_client()
    rs = client.execute("SELECT * FROM patients WHERE nik = ?", [nik])
    if not rs.rows:
        return None
    return _row_to_dict(rs, rs.rows[0])


def add_scan_record(patient_nik, filename, modality, predicted_class, confidence,
                     report_text, original_b64, heatmap_b64):
    client = _get_client()
    client.execute(
        """
        INSERT INTO scans (patient_nik, filename, modality, predicted_class,
                            confidence, report_text, original_b64, heatmap_b64, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [patient_nik, filename, modality, predicted_class, confidence,
         report_text, original_b64, heatmap_b64, datetime.datetime.now().isoformat()],
    )


def get_patient_history(nik):
    client = _get_client()
    rs = client.execute(
        "SELECT * FROM scans WHERE patient_nik = ? ORDER BY created_at DESC", [nik]
    )
    return [_row_to_dict(rs, row) for row in rs.rows]
