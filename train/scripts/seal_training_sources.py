#!/usr/bin/env python3
"""Seal the exact source set used by a staged K-SOVA training release.

Raw source bodies stay in the ignored, versioned runtime directory.  A compact
report containing counts and hashes can be published in the release directory.
The tool never reads ``work/current`` implicitly: the legacy registry and
snapshot root must be supplied as explicit inputs.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.parse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
V2_ROOT = REPO_ROOT / "benchmark/v2.0"
sys.path.insert(0, str(V2_ROOT / "tools"))

from discover_calibration_sources import extract_html_text, fetch_source_once  # noqa: E402


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: expected an object")
            rows.append(value)
    return rows


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp-{os.getpid()}")
    temp.write_text(value, encoding="utf-8")
    temp.replace(path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    atomic_text(
        path,
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
    )


def write_json(path: Path, value: dict[str, Any]) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def copy_exact(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_digest = sha256(source)
    if destination.exists():
        if not destination.is_file() or sha256(destination) != source_digest:
            raise RuntimeError(f"refusing to overwrite divergent snapshot: {destination}")
        return
    temp = destination.with_name(destination.name + f".tmp-{os.getpid()}")
    shutil.copyfile(source, temp)
    if sha256(temp) != source_digest:
        temp.unlink(missing_ok=True)
        raise RuntimeError(f"snapshot copy verification failed: {destination}")
    temp.replace(destination)


def source_ids_from_examples(rows: list[dict[str, Any]]) -> set[str]:
    source_ids: set[str] = set()
    for index, row in enumerate(rows, 1):
        metadata = row.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError(f"training example {index} has no metadata object")
        values = metadata.get("source_ids")
        if not isinstance(values, list) or not values:
            raise ValueError(f"training example {index} has no source_ids")
        for value in values:
            if not isinstance(value, str) or not value:
                raise ValueError(f"training example {index} has invalid source_id")
            source_ids.add(value)
    return source_ids


def safe_legacy_path(root: Path, relative: str) -> Path:
    value = Path(relative)
    if value.is_absolute():
        raise ValueError(f"absolute legacy snapshot path is forbidden: {relative}")
    resolved_root = root.resolve()
    resolved = (root / value).resolve()
    if not resolved.is_relative_to(resolved_root):
        raise ValueError(f"legacy snapshot escapes its root: {relative}")
    return resolved


def archive_existing(
    source: dict[str, Any], legacy_root: Path, snapshot_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    files: list[dict[str, Any]] = []
    for relative in source.get("raw_snapshot_files") or []:
        if not isinstance(relative, str):
            raise ValueError(f"invalid raw snapshot path for {source['source_id']}")
        origin = safe_legacy_path(legacy_root, relative)
        if not origin.is_file():
            raise FileNotFoundError(origin)
        destination = snapshot_dir / origin.name
        copy_exact(origin, destination)
        digest = sha256(destination)
        files.append({
            "path": rel(destination),
            "bytes": destination.stat().st_size,
            "sha256": digest,
            "registry_content_sha256_match": (
                digest == source.get("content_sha256")
            ),
        })
    return ({
        "status": "copied_legacy_snapshot",
        "retrieved_at": source.get("retrieved_at"),
        "files": files,
    }, files)


def curl_snapshot_fallback(
    source: dict[str, Any], snapshot_dir: Path, previous_error: str,
) -> dict[str, Any]:
    """Archive a source that needs cookies, a longer timeout, or JSON support."""
    parsed = urllib.parse.urlsplit(str(source.get("url") or ""))
    hostname = (parsed.hostname or "").casefold()
    if parsed.scheme not in {"http", "https"} or hostname == "kp" or hostname.endswith(".kp"):
        return {
            "status": "failed",
            "error": f"{previous_error}; curl fallback blocked unsafe URL",
        }
    download = snapshot_dir / f"{source['source_id']}.download"
    cookie = snapshot_dir / f"{source['source_id']}.cookies"
    command = [
        "curl", "--location", "--fail", "--silent", "--show-error",
        "--http1.1",
        "--max-time", "120", "--max-redirs", "5", "--max-filesize", "41943040",
        "--user-agent", "K-SOVA-Research/2.0",
        "--cookie-jar", str(cookie), "--cookie", str(cookie),
        "--output", str(download),
        "--write-out", "%{http_code}\n%{url_effective}\n%{content_type}\n",
        str(source["url"]),
    ]
    completed = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        timeout=135,
    )
    cookie.unlink(missing_ok=True)
    if completed.returncode != 0 or not download.is_file():
        download.unlink(missing_ok=True)
        return {
            "status": "failed",
            "error": (
                f"{previous_error}; curl fallback: "
                f"{completed.stderr.strip() or f'exit {completed.returncode}'}"
            ),
        }
    if download.stat().st_size > 40 * 1024 * 1024:
        download.unlink(missing_ok=True)
        return {"status": "failed", "error": "curl fallback exceeded 40 MiB"}
    lines = completed.stdout.splitlines()
    http_status = int(lines[0]) if lines and lines[0].isdigit() else None
    final_url = lines[1] if len(lines) > 1 else str(source["url"])
    content_type = (lines[2] if len(lines) > 2 else "").split(";", 1)[0].strip()
    prefix = download.read_bytes()[:8]
    if content_type == "application/pdf" or prefix.startswith(b"%PDF"):
        suffix = ".pdf"
    elif content_type in {"application/json", "application/ld+json"}:
        suffix = ".json"
    else:
        suffix = ".html"
    snapshot = snapshot_dir / f"{source['source_id']}{suffix}"
    download.replace(snapshot)
    if suffix == ".pdf":
        text_path = snapshot.with_suffix(".txt")
        converted = subprocess.run(
            ["pdftotext", "-layout", str(snapshot), str(text_path)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=90,
        )
        text = (
            text_path.read_text(encoding="utf-8", errors="replace")
            if converted.returncode == 0 and text_path.is_file() else ""
        )
    elif suffix == ".json":
        try:
            payload = json.loads(snapshot.read_text(encoding="utf-8", errors="replace"))
            text = json.dumps(payload, ensure_ascii=False)
        except json.JSONDecodeError as exc:
            snapshot.unlink(missing_ok=True)
            return {"status": "failed", "error": f"invalid JSON snapshot: {exc}"}
    else:
        text = extract_html_text(snapshot.read_bytes())
    return {
        "status": "fetched",
        "http_status": http_status,
        "final_url": final_url,
        "content_type": content_type,
        "text_chars": len(text),
        "relevance_pass": None,
        "transport": "curl_cookie_fallback",
    }


def fetch_for_seal(
    source: dict[str, Any], snapshot_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with tempfile.TemporaryDirectory(prefix="ksova-source-seal-") as temporary:
        temporary_path = Path(temporary)
        request = {
            "source_id": source["source_id"],
            "title": source.get("title"),
            "url": source["url"],
            "retrieval_context": [
                value for value in (
                    source.get("title"),
                    source.get("evidence_summary_ko"),
                ) if isinstance(value, str) and value.strip()
            ],
        }
        fetched = fetch_source_once(request, temporary_path)
        retrieval = fetched.get("retrieval") or {}
        if retrieval.get("status") != "fetched":
            retrieval = curl_snapshot_fallback(
                request, temporary_path,
                str(retrieval.get("error") or "unknown fetch failure"),
            )
            if retrieval.get("status") != "fetched":
                return ({
                    "status": "fetch_failed",
                    "retrieved_at": now(),
                    "error": retrieval.get("error") or "unknown fetch failure",
                    "files": [],
                }, [])
        files: list[dict[str, Any]] = []
        for origin in sorted(temporary_path.glob(f"{source['source_id']}.*")):
            if not origin.is_file():
                continue
            destination = snapshot_dir / origin.name
            copy_exact(origin, destination)
            digest = sha256(destination)
            files.append({
                "path": rel(destination),
                "bytes": destination.stat().st_size,
                "sha256": digest,
                "registry_content_sha256_match": (
                    digest == source.get("content_sha256")
                ),
            })
        if not files:
            return ({
                "status": "fetch_failed",
                "retrieved_at": now(),
                "error": "fetch returned no snapshot file",
                "files": [],
            }, [])
        return ({
            "status": "refetched_for_seal",
            "retrieved_at": now(),
            "final_url": retrieval.get("final_url"),
            "http_status": retrieval.get("http_status"),
            "content_type": retrieval.get("content_type"),
            "text_chars": retrieval.get("text_chars"),
            "relevance_pass": retrieval.get("relevance_pass"),
            "transport": retrieval.get("transport") or "urllib",
            "files": files,
        }, files)


def reusable_manifest_entry(
    row: dict[str, Any], expected_source_id: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
    if row.get("source_id") != expected_source_id:
        return None
    if row.get("status") not in {
        "copied_legacy_snapshot", "refetched_for_seal",
    }:
        return None
    files = row.get("files")
    if not isinstance(files, list) or not files:
        return None
    for file in files:
        if not isinstance(file, dict):
            return None
        path = Path(str(file.get("path") or ""))
        if not path.is_absolute():
            path = REPO_ROOT / path
        if (
            not path.is_file()
            or path.stat().st_size != file.get("bytes")
            or sha256(path) != file.get("sha256")
        ):
            return None
    provenance = {
        key: value for key, value in row.items()
        if key not in {
            "source_id", "title", "url", "authority_type", "content_sha256",
        }
    }
    provenance["reused_verified_seal"] = True
    return provenance, files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--legacy-snapshot-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--release-report", type=Path, required=True)
    parser.add_argument("--fetch-missing", action="store_true")
    parser.add_argument("--url-overrides", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    examples = read_jsonl(args.examples)
    needed_ids = source_ids_from_examples(examples)
    registry_rows = read_jsonl(args.registry)
    registry_by_id: dict[str, dict[str, Any]] = {}
    duplicate_ids: list[str] = []
    for row in registry_rows:
        source_id = row.get("source_id")
        if not isinstance(source_id, str) or not source_id:
            continue
        if source_id in registry_by_id:
            duplicate_ids.append(source_id)
        registry_by_id[source_id] = row
    missing_registry = sorted(needed_ids - set(registry_by_id))
    if duplicate_ids or missing_registry:
        raise RuntimeError(
            f"source registry failed: duplicate={sorted(set(duplicate_ids))}; "
            f"missing={missing_registry}"
        )

    selected = [registry_by_id[source_id] for source_id in sorted(needed_ids)]
    snapshot_dir = args.output_dir / "raw_snapshots"
    results: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
    existing_manifest_path = args.output_dir / "snapshot_manifest.jsonl"
    existing_manifest = {
        row["source_id"]: row for row in read_jsonl(existing_manifest_path)
    } if existing_manifest_path.is_file() else {}
    url_overrides: dict[str, str] = {}
    if args.url_overrides:
        value = json.loads(args.url_overrides.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or any(
            not isinstance(key, str) or not isinstance(url, str)
            for key, url in value.items()
        ):
            raise ValueError("url overrides must be a JSON object of strings")
        url_overrides = value
    fetch_rows: list[dict[str, Any]] = []
    for source in selected:
        reusable = reusable_manifest_entry(
            existing_manifest.get(source["source_id"], {}), source["source_id"],
        )
        if reusable is not None:
            results[source["source_id"]] = reusable
            continue
        if source.get("raw_snapshot_files"):
            results[source["source_id"]] = archive_existing(
                source, args.legacy_snapshot_root, snapshot_dir,
            )
        elif args.fetch_missing:
            override = url_overrides.get(source["source_id"])
            fetch_rows.append({
                **source,
                **({"url": override, "original_url": source.get("url")}
                   if override else {}),
            })
        else:
            results[source["source_id"]] = ({
                "status": "legacy_excerpt_only",
                "retrieved_at": source.get("retrieved_at"),
                "files": [],
            }, [])

    if fetch_rows:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(fetch_for_seal, source, snapshot_dir): source["source_id"]
                for source in fetch_rows
            }
            for future in concurrent.futures.as_completed(futures):
                source_id = futures[future]
                try:
                    results[source_id] = future.result()
                except Exception as exc:
                    results[source_id] = ({
                        "status": "fetch_failed",
                        "retrieved_at": now(),
                        "error": f"{type(exc).__name__}: {exc}",
                        "files": [],
                    }, [])

    manifest: list[dict[str, Any]] = []
    sealed_registry: list[dict[str, Any]] = []
    for source in selected:
        provenance, files = results[source["source_id"]]
        manifest.append({
            "source_id": source["source_id"],
            "title": source.get("title"),
            "url": source.get("url"),
            "authority_type": source.get("authority_type"),
            "content_sha256": source.get("content_sha256"),
            **provenance,
        })
        if source["source_id"] in url_overrides:
            manifest[-1]["fetch_url_override"] = url_overrides[source["source_id"]]
        sealed_registry.append({
            **source,
            "sealed_provenance": {
                key: value for key, value in provenance.items() if key != "files"
            },
            "sealed_snapshot_files": [row["path"] for row in files],
        })

    registry_output = args.output_dir / "source_registry.jsonl"
    manifest_output = args.output_dir / "snapshot_manifest.jsonl"
    write_jsonl(registry_output, sealed_registry)
    write_jsonl(manifest_output, manifest)
    statuses = Counter(row["status"] for row in manifest)
    snapshot_files = [file for row in manifest for file in row.get("files", [])]
    failed_ids = sorted(
        row["source_id"] for row in manifest if row["status"] == "fetch_failed"
    )
    report = {
        "schema_version": "ksova-training-source-seal-report-1.0",
        "created_at": now(),
        "pass": not failed_ids,
        "training_examples": len(examples),
        "training_sources": len(needed_ids),
        "registry_sources_selected": len(selected),
        "status_counts": dict(sorted(statuses.items())),
        "snapshot_files": len(snapshot_files),
        "snapshot_bytes": sum(row["bytes"] for row in snapshot_files),
        "failed_source_ids": failed_ids,
        "inputs": {
            "examples": {"path": rel(args.examples), "sha256": sha256(args.examples)},
            "registry": {"path": rel(args.registry), "sha256": sha256(args.registry)},
            "url_overrides": ({
                "path": rel(args.url_overrides),
                "sha256": sha256(args.url_overrides),
            } if args.url_overrides else None),
        },
        "runtime_artifacts": {
            "source_registry": {
                "path": rel(registry_output),
                "rows": len(sealed_registry),
                "sha256": sha256(registry_output),
            },
            "snapshot_manifest": {
                "path": rel(manifest_output),
                "rows": len(manifest),
                "sha256": sha256(manifest_output),
            },
            "raw_snapshot_directory": rel(snapshot_dir),
        },
        "policy": {
            "implicit_work_current_lookup": False,
            "network_attempts_per_missing_source": 1 if args.fetch_missing else 0,
            "transport_fallback_attempts_max": 1 if args.fetch_missing else 0,
            "raw_snapshots_git_published": False,
            "legacy_refetch_is_retroactive_provenance": False,
        },
    }
    runtime_report = args.output_dir / "source_seal_report.json"
    write_json(runtime_report, report)
    report["runtime_artifacts"]["source_seal_report"] = {
        "path": rel(runtime_report),
        "sha256": sha256(runtime_report),
    }
    write_json(args.release_report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
