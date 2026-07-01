"""Test the document upload API by sending real files."""

import sys
import json
import requests
from pathlib import Path

BASE_URL = "http://127.0.0.1:8000"


def upload_files(paths: list[str]):
    files = []
    handles = []
    for path in paths:
        p = Path(path)
        f = open(p, "rb")
        handles.append(f)
        files.append(("files", (p.name, f, "application/octet-stream")))

    print(f"Uploading {len(files)} file(s)...")
    response = requests.post(f"{BASE_URL}/upload", files=files)

    for f in handles:
        f.close()

    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))


def list_documents():
    response = requests.get(f"{BASE_URL}/documents")
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))


def get_document(filename: str):
    response = requests.get(f"{BASE_URL}/documents/{filename}")
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python -m scripts.test_api upload <path1> [path2 ...]")
        print("  python -m scripts.test_api list")
        print("  python -m scripts.test_api get <filename-without-extension>")
        sys.exit(1)

    command = sys.argv[1]

    if command == "upload":
        upload_files(sys.argv[2:])
    elif command == "list":
        list_documents()
    elif command == "get":
        get_document(sys.argv[2])
    else:
        print(f"Unknown command: {command}")