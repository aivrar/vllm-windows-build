"""Validate the assembled Windows wheel before publishing it."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import zipfile
from collections import Counter
from email.parser import BytesParser
from pathlib import Path


REQUIRED_FLASH_ATTN_FILES = {
    "vllm/vllm_flash_attn/_vllm_fa2_C.pyd",
    "vllm/vllm_flash_attn/layers/__init__.py",
    "vllm/vllm_flash_attn/layers/rotary.py",
    "vllm/vllm_flash_attn/ops/__init__.py",
    "vllm/vllm_flash_attn/ops/triton/__init__.py",
    "vllm/vllm_flash_attn/ops/triton/rotary.py",
    "vllm/vllm_flash_attn/cute/__init__.py",
    "vllm/vllm_flash_attn/cute/interface.py",
}

REQUIRED_RELEASE_FILES = REQUIRED_FLASH_ATTN_FILES | {
    "vllm/_C_stable_libtorch.pyd",
    "vllm/_moe_C_stable_libtorch.pyd",
    "vllm/_rust_tool_parser.pyd",
    "vllm/cumem_allocator.pyd",
    "vllm/spinloop.pyd",
    "vllm/vllm-rs.exe",
}

REQUIRED_NONEMPTY_FILES = REQUIRED_RELEASE_FILES - {
    "vllm/vllm_flash_attn/layers/__init__.py",
    "vllm/vllm_flash_attn/ops/__init__.py",
    "vllm/vllm_flash_attn/ops/triton/__init__.py",
    "vllm/vllm_flash_attn/cute/__init__.py",
}


def sha256_record(data: bytes) -> str:
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
    return "sha256=" + digest.decode("ascii")


def validate_wheel(wheel_path: Path) -> None:
    with zipfile.ZipFile(wheel_path) as wheel:
        bad_member = wheel.testzip()
        assert bad_member is None, f"ZIP CRC check failed: {bad_member}"
        names = wheel.namelist()
        duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
        assert not duplicates, f"duplicate archive members: {duplicates}"

        missing = sorted(REQUIRED_RELEASE_FILES - set(names))
        assert not missing, f"missing release payloads: {missing}"
        for name in REQUIRED_NONEMPTY_FILES:
            assert wheel.getinfo(name).file_size > 0, f"empty release payload: {name}"

        metadata_names = [name for name in names if name.endswith(".dist-info/METADATA")]
        assert len(metadata_names) == 1, f"expected one METADATA, found {metadata_names}"
        metadata = BytesParser().parsebytes(wheel.read(metadata_names[0]))
        assert metadata.get("Name", "").lower() == "vllm"
        assert metadata.get("Version") == "0.24.0+cu128"

        cute_files = [
            name
            for name in names
            if name.startswith("vllm/vllm_flash_attn/cute/") and name.endswith(".py")
        ]
        assert len(cute_files) >= 40, f"incomplete CuteDSL payload: {len(cute_files)} files"
        for name in cute_files:
            data = wheel.read(name).replace(b"vllm.vllm_flash_attn.cute", b"")
            assert b"flash_attn.cute" not in data, (
                f"unrewritten CuteDSL import in {name}"
            )

        record_names = [name for name in names if name.endswith(".dist-info/RECORD")]
        assert len(record_names) == 1, f"expected one RECORD, found {record_names}"
        record_name = record_names[0]
        rows = {
            row[0]: (row[1], row[2])
            for row in csv.reader(io.StringIO(wheel.read(record_name).decode("utf-8")))
        }
        assert set(rows) == set(names), "RECORD does not cover every archive member"

        for name in names:
            digest, size = rows[name]
            if name == record_name:
                assert digest == "" and size == "", "RECORD must hash neither itself nor its size"
                continue
            data = wheel.read(name)
            assert digest == sha256_record(data), f"RECORD hash mismatch: {name}"
            assert size == str(len(data)), f"RECORD size mismatch: {name}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()
    validate_wheel(args.wheel.resolve())
    print(f"wheel content validation passed: {args.wheel.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
