#!/usr/bin/env python3
"""Block unsafe or prohibited URLs before K-SOVA item authoring."""

from __future__ import annotations

import argparse
import ipaddress
import json
import urllib.parse
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def host_matches(host: str, suffix: str) -> bool:
    value = suffix.casefold().strip().removeprefix(".")
    return host == value or host.endswith("." + value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = read_jsonl(args.sources)
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    blocked = {str(value).casefold() for value in policy.get("blocked_hosts", [])}
    blocked.add("kp")
    findings: list[dict[str, Any]] = []
    hosts: Counter[str] = Counter()

    for row in rows:
        source_id = str(row.get("source_id") or "")
        url = str(row.get("url") or "").strip()
        try:
            parsed = urllib.parse.urlsplit(url)
            hostname = (parsed.hostname or "").casefold().strip(".")
        except ValueError as exc:
            findings.append({"source_id": source_id, "url": url, "code": "malformed_url", "detail": str(exc)})
            continue
        hosts[hostname or "missing"] += 1
        codes = []
        if parsed.scheme.casefold() not in {"http", "https"}:
            codes.append("non_http_scheme")
        if not hostname:
            codes.append("missing_host")
        if parsed.username is not None or parsed.password is not None:
            codes.append("embedded_credentials")
        if hostname == "localhost" or hostname.endswith(".localhost"):
            codes.append("localhost")
        if any(host_matches(hostname, suffix) for suffix in blocked):
            codes.append("blocked_host")
        try:
            address = ipaddress.ip_address(hostname)
        except ValueError:
            address = None
        if address is not None and not address.is_global:
            codes.append("non_global_ip_literal")
        if any(ord(character) < 32 for character in url):
            codes.append("control_character")
        for code in dict.fromkeys(codes):
            findings.append({"source_id": source_id, "url": url, "host": hostname, "code": code})

    report = {
        "schema_version": "ksova-v2-source-security-audit-1.0",
        "pass": not findings,
        "sources": len(rows),
        "unique_hosts": len(hosts),
        "finding_count": len(findings),
        "findings_by_code": dict(sorted(Counter(row["code"] for row in findings).items())),
        "findings": findings[:500],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "findings"}, ensure_ascii=False))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
