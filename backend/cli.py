"""Local ingestion CLI, using the same pipeline as the API."""

import argparse
import json
from pathlib import Path

from backend.config import Settings
from backend.service import RAGService


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path)
    args = parser.parse_args()
    service = RAGService(Settings.from_env())
    for path in args.files:
        result = service.ingest(path.name, path.read_bytes())
        print(json.dumps(result.model_dump(), ensure_ascii=False))


if __name__ == "__main__":
    main()
