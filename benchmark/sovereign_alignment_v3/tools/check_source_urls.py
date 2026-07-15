#!/usr/bin/env python3
"""Record reachability of official source URLs without treating anti-bot blocks as content validation."""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import http.client
import socket
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from common import ROOT, load_json, write_json


def encoded_url(url: str) -> str:
    parts = urllib.parse.urlsplit(url)
    path = urllib.parse.quote(urllib.parse.unquote(parts.path), safe="/%:@")
    query = urllib.parse.quote(urllib.parse.unquote(parts.query), safe="=&?/:+,%")
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, path, query, parts.fragment))


def check(row: dict[str, Any], timeout: int) -> dict[str, Any]:
    url = row["url"]
    result = {
        "case_id": row["case_id"], "source_id": row["id"], "url": url,
        "reachable": False, "status": None, "final_url": None,
        "content_type": None, "error": None, "tls_verified": True,
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; K-SOVA-source-audit/3.0; research)",
        "Accept": "text/html,application/pdf,application/xhtml+xml,*/*;q=0.5",
    }
    try:
        request = urllib.request.Request(encoded_url(url), headers=headers, method="GET")
        context = ssl.create_default_context()
        with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
            response.read(2048)
            result.update({
                "reachable": True,
                "status": response.status,
                "final_url": response.geturl(),
                "content_type": response.headers.get("Content-Type"),
            })
    except urllib.error.HTTPError as exc:
        result.update({
            "status": exc.code, "final_url": exc.geturl(),
            "content_type": exc.headers.get("Content-Type") if exc.headers else None,
            "error": f"HTTPError: {exc.code}",
            "reachable": 300 <= exc.code < 400 or exc.code in {401, 403, 405, 406, 429},
        })
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, ssl.SSLError) or "CERTIFICATE_VERIFY_FAILED" in str(exc):
            try:
                request = urllib.request.Request(encoded_url(url), headers=headers, method="GET")
                context = ssl._create_unverified_context()
                with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
                    response.read(2048)
                    result.update({
                        "reachable": True,
                        "status": response.status,
                        "final_url": response.geturl(),
                        "content_type": response.headers.get("Content-Type"),
                        "tls_verified": False,
                        "error": f"TLS verification failed before unverified reachability check: {exc}",
                    })
            except Exception as fallback_exc:
                result["error"] = (
                    f"URLError: {exc}; fallback: "
                    f"{type(fallback_exc).__name__}: {fallback_exc}"
                )
        else:
            result["error"] = f"URLError: {exc}"
    except (
        TimeoutError, socket.timeout, ssl.SSLError,
        http.client.HTTPException, ConnectionError, OSError, ValueError,
    ) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def check_with_retries(row: dict[str, Any], timeout: int, retries: int) -> dict[str, Any]:
    attempt_errors: list[str | None] = []
    result: dict[str, Any] | None = None
    for attempt in range(retries + 1):
        result = check(row, timeout)
        attempt_errors.append(result.get("error"))
        if result["reachable"]:
            break
        if attempt < retries:
            time.sleep(1.5 * (attempt + 1))
    assert result is not None
    result["attempt_count"] = len(attempt_errors)
    result["attempt_errors"] = attempt_errors
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "config/case_blueprints.json")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/source_url_audit.json")
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args()
    cases = load_json(args.input)
    rows = [dict(source, case_id=case["id"]) for case in cases for source in case["sources"]]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(lambda row: check_with_retries(row, args.timeout, args.retries), rows))
    unreachable = [row for row in results if not row["reachable"]]
    report = {
        "checked_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_count": len(results),
        "reachable_or_access_controlled": len(results) - len(unreachable),
        "unreachable_count": len(unreachable),
        "unreachable": unreachable,
        "results": results,
        "interpretation": "접근 가능 여부는 주장 검증이 아니며, HTTP 접근통제 응답은 호스트 접근 가능으로 기록한다.",
    }
    write_json(args.output, report)
    print(f"sources={len(results)} reachable={len(results)-len(unreachable)} unreachable={len(unreachable)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
