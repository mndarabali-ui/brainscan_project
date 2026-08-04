import os
import libsql_client

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TURSO CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Mendukung 2 gaya nama ENV:
# - TURSO_DATABASE_URL / TURSO_AUTH_TOKEN
# - TURSODATABASEURL / TURSOAUTHTOKEN

TURSO_DATABASE_URL = (
    os.environ.get("TURSO_DATABASE_URL")
    or os.environ.get("TURSODATABASEURL")
    or ""
)

TURSO_AUTH_TOKEN = (
    os.environ.get("TURSO_AUTH_TOKEN")
    or os.environ.get("TURSOAUTHTOKEN")
    or ""
)
