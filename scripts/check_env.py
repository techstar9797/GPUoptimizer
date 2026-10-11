"""Check .env without printing any secret values, and test each connection.

Usage: python scripts/check_env.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backfill.config import SETTINGS  # noqa: E402

OK, BAD, WARN = "ok  ", "FIX ", "note"


def check_uri(uri: str) -> bool:
    if not uri:
        print(f"{BAD} MONGODB_URI is not set in .env")
        return False
    good = True
    if not uri.startswith(("mongodb+srv://", "mongodb://")):
        print(f"{BAD} MONGODB_URI should start with mongodb+srv://")
        good = False
    if uri[0] in "'\"" or uri[-1] in "'\"":
        print(f"{BAD} MONGODB_URI is wrapped in quotes; remove them")
        good = False
    if "<" in uri or ">" in uri:
        print(f"{BAD} MONGODB_URI still contains a <placeholder>; replace <db_password> (brackets included) with the real password")
        good = False
    m = re.match(r"^mongodb(?:\+srv)?://([^@/]*)@([^/?]+)", uri)
    if not m:
        print(f"{BAD} MONGODB_URI has no user:password@host part")
        return False
    creds, host = m.group(1), m.group(2)
    if ":" not in creds:
        print(f"{BAD} MONGODB_URI has a username but no password (expected user:password@...)")
        return False
    user, pw = creds.split(":", 1)
    print(f"{OK} MONGODB_URI user '{unquote(user)}' at host {host}")
    if uri.count("@") > 1:
        print(f"{BAD} the password contains '@'; write it as %40")
        good = False
    raw_special = [c for c in ":/?#[]" if c in pw]
    if raw_special:
        print(f"{BAD} the password contains {' '.join(raw_special)}; URL-encode them or choose a letters-and-digits password")
        good = False
    if "%" in pw and not re.search(r"%[0-9A-Fa-f]{2}", pw):
        print(f"{BAD} the password contains a bare '%'; write it as %25")
        good = False
    if pw != pw.strip():
        print(f"{BAD} the password has leading or trailing spaces")
        good = False
    print(f"{OK} password is {len(unquote(pw))} characters (not shown)")
    return good


def ping_atlas() -> None:
    from pymongo import MongoClient
    from pymongo.errors import OperationFailure, ServerSelectionTimeoutError
    try:
        MongoClient(SETTINGS.mongodb_uri, serverSelectionTimeoutMS=10000).admin.command("ping")
        print(f"{OK} connected to Atlas")
    except OperationFailure as e:
        if e.code == 18:
            print(f"{BAD} Atlas rejected the username or password. Check Security > Database Access: the user exists, "
                  "its password matches, and it has been deployed (allow about a minute after creating or editing it).")
        else:
            print(f"{BAD} Atlas error {e.code}: {e.details.get('errmsg') if e.details else e}")
    except ServerSelectionTimeoutError:
        print(f"{BAD} could not reach the cluster. Add your IP under Security > Network Access, or 0.0.0.0/0.")


def main():
    print(f"Reading {Path(__file__).resolve().parent.parent / '.env'}")
    uri_ok = check_uri(SETTINGS.mongodb_uri)
    for name, val in [("OPENAI_API_KEY", SETTINGS.openai_api_key), ("VOYAGE_API_KEY", SETTINGS.voyage_api_key),
                      ("RAINDROP_WRITE_KEY", SETTINGS.raindrop_write_key)]:
        print(f"{OK if val else WARN} {name} {'is set' if val else 'is not set'}")
    print(f"{OK} OPENAI_MODEL = {SETTINGS.openai_model}")
    if uri_ok:
        ping_atlas()
    if SETTINGS.voyage_api_key:
        try:
            from backfill import embed
            v = embed.embed_one("test")
            print(f"{OK} Voyage embeddings work ({len(v)} dims, via {embed._working})")
        except Exception as e:
            print(f"{BAD} Voyage: {e}")
    if SETTINGS.raindrop_write_key:
        from backfill import observe
        print(f"{OK if observe.ENABLED else BAD} Raindrop SDK {'initialized' if observe.ENABLED else 'failed to start'}")


if __name__ == "__main__":
    main()
