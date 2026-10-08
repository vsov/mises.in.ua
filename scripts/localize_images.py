#!/usr/bin/env python3
"""Localize article images using Hugo's rendered HTML as the inventory."""

import argparse
import concurrent.futures
import csv
import hashlib
import html
from html.parser import HTMLParser
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qsl, quote, unquote, urlencode, urlsplit, urlunsplit
import xml.etree.ElementTree as ET

from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "work/image-localization-manifest.json"
REPORT = ROOT / "work/unresolved-images.md"
ASSETS = ROOT / "static/images/articles"
HUGO = shutil.which("hugo") or "/opt/homebrew/bin/hugo"
REMOTE = re.compile(r"^(https?:)?//", re.I)
IMAGE_PATH = re.compile(r"\.(?:jpe?g|png|gif|webp|svg|avif|bmp|tiff?|ico)$", re.I)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save(data):
    temporary = MANIFEST.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(MANIFEST)


class ArticleImages(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.depth = 0
        self.anchors = []
        self.images = set()
        self.embedded = set()
        self.local_references = set()

    def add(self, value):
        if value and REMOTE.match(value):
            self.images.add(value)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "div":
            if self.depth or "article-body" in attrs.get("class", "").split():
                self.depth += 1
        if not self.depth:
            return
        for attribute in ("src", "href"):
            if attrs.get(attribute, "").startswith("/images/articles/"):
                self.local_references.add(attrs[attribute])
        if tag == "a":
            self.anchors.append(attrs.get("href", ""))
            if IMAGE_PATH.search(unquote(urlsplit(attrs.get("href", "")).path)):
                self.add(attrs.get("href"))
        if tag == "img":
            self.add(attrs.get("src"))
            if REMOTE.match(attrs.get("src", "")):
                self.embedded.add(attrs["src"])
            for attribute in ("data-src", "data-original"):
                self.add(attrs.get(attribute))
                if REMOTE.match(attrs.get(attribute, "")):
                    self.embedded.add(attrs[attribute])
            if self.anchors:
                self.add(self.anchors[-1])
        if tag in ("img", "source"):
            for attribute in ("srcset", "data-srcset"):
                for candidate in re.findall(r"(?:^|,\s+)(\S+)(?:\s+\d+(?:\.\d+)?[wx])?", attrs.get(attribute, "")):
                    self.add(candidate.rstrip(","))
                    if REMOTE.match(candidate):
                        self.embedded.add(candidate.rstrip(","))
                    elif candidate.startswith("/images/articles/"):
                        self.local_references.add(candidate.rstrip(","))

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_endtag(self, tag):
        if self.depth and tag == "a" and self.anchors:
            self.anchors.pop()
        if self.depth and tag == "div":
            self.depth -= 1


def source_spans(source, urls):
    candidates = []
    for url in urls:
        variants = {url, html.escape(url, quote=False), unquote(url)}
        variants |= {value.replace("(", r"\(").replace(")", r"\)") for value in variants}
        for value in variants:
            for match in re.finditer(re.escape(value), source):
                if match.end() < len(source) and not (source[match.end()].isspace() or source[match.end()] in "<>\"')],"):
                    continue
                candidates.append((match.start(), match.end(), url))
    chosen = []
    end = -1
    for start, stop, url in sorted(candidates, key=lambda item: (item[0], -(item[1] - item[0]))):
        if start >= end:
            chosen.append((start, stop, url))
            end = stop
    return chosen


def inventory(render_root):
    if MANIFEST.exists():
        raise ValueError("Inventory already exists; preserve it for provenance and resumable downloads")
    listing = subprocess.run([HUGO, "list", "all"], cwd=ROOT, check=True, capture_output=True, text=True)
    rows = list(csv.DictReader(io.StringIO(listing.stdout)))
    articles = []
    for row in rows:
        if row["kind"] != "page" or row["section"] != "article":
            continue
        path = ROOT / row["path"]
        source = path.read_text()
        output = render_root / unquote(urlsplit(row["permalink"]).path).lstrip("/") / "index.html"
        parser = ArticleImages()
        parser.feed(output.read_text())
        if not parser.images:
            continue
        matched = {url for _, _, url in source_spans(source, parser.images)}
        articles.append({"path": row["path"], "title": row["title"], "date": row["date"],
                         "permalink": row["permalink"],
                         "embedded": sorted(parser.embedded),
                         "source_sha256": sha(source.encode()), "urls": sorted(parser.images),
                         "unmatched": sorted(parser.images - matched)})
    data = {"version": 1, "articles": articles, "results": {}}
    save(data)
    print(f"Inventory: {len(articles)} articles, {len({u for a in articles for u in a['urls']})} URLs", flush=True)
    return data


def request(url, timeout, attempts=2, connect_timeout=8):
    url = "https:" + url if url.startswith("//") else url
    url = quote(url, safe="%:/?&=+$,;@!~*'()-._#[]")
    last = ""
    for attempt in range(attempts):
        with tempfile.TemporaryDirectory(prefix="mises-image-") as temporary:
            output = Path(temporary) / "response"
            result = subprocess.run(["/usr/bin/curl", "-sS", "-L", "--connect-timeout", str(connect_timeout),
                                     "--max-time", str(timeout), "--max-filesize", str(50 * 1024 * 1024),
                                     "--proto", "=http,https", "--proto-redir", "=http,https",
                                     "-A", "Mozilla/5.0 MisesImageLocalization/1.0", "-o", str(output),
                                     "-w", "%{json}", url], capture_output=True, text=True)
            try:
                meta = json.loads(result.stdout)
            except ValueError:
                meta = {}
            status = meta.get("http_code", 0)
            if result.returncode == 0 and 200 <= status < 300:
                return output.read_bytes(), meta
            last = f"HTTP {status}; curl {result.returncode}: {result.stderr.strip()[:250]}"
            if status not in (0, 408, 429, 500, 502, 503, 504):
                break
        if attempt + 1 < attempts:
            time.sleep(1 + attempt)
    raise ValueError(last)


def store_image(body):
    stripped = body.lstrip()
    if stripped.startswith((b"<svg", b"<?xml")):
        if ET.fromstring(body).tag.split("}")[-1] != "svg":
            raise ValueError("Response is not an SVG image")
        extension = "svg"
    else:
        with Image.open(io.BytesIO(body)) as image:
            extension = {"JPEG": "jpg", "TIFF": "tif"}.get(image.format, image.format.lower())
            image.verify()
    digest = sha(body)
    destination = ASSETS / f"{digest}.{extension}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        destination.write_bytes(body)
    return {"local_url": f"/images/articles/{destination.name}", "sha256": digest, "bytes": len(body)}


def fetch_live(url, timeout):
    try:
        body, meta = request(url, timeout)
        return {"live": {**store_image(body), "source": "live", "download_url": meta["url_effective"]}}
    except UnidentifiedImageError:
        return {"live_error": "Response is not a decodable image", "response_type": meta.get("content_type", ""), "archives": {}}
    except Exception as error:
        return {"live_error": f"{type(error).__name__}: {error}", "archives": {}}


def cutoff(date):
    value = datetime.fromisoformat(date.replace("Z", "+00:00"))
    if value.tzinfo is None:
        raise ValueError("Hugo date has no timezone")
    return (value.astimezone(timezone.utc) - timedelta(seconds=1)).strftime("%Y%m%d%H%M%S")


def fetch_archive(url, date, timeout, replay_timeout=None, fetcher=None):
    fetcher = fetcher or request
    before = cutoff(date)
    candidates = [url]
    parts = urlsplit("https:" + url if url.startswith("//") else url)
    parameters = parse_qsl(parts.query, keep_blank_values=True)
    # Strip only delivery/cache tokens; parameters such as doc/id identify the image.
    without_tokens = [(key, value) for key, value in parameters if key not in {"itok", "_", "cachebuster"}]
    if len(without_tokens) < len(parameters):
        candidates.append(urlunsplit(parts._replace(query=urlencode(without_tokens), fragment="")))
    attempts = []
    for original in candidates:
        query = "https://web.archive.org/cdx/search/cdx?" + urlencode([
            ("url", original), ("output", "json"), ("fl", "timestamp,original,mimetype,statuscode"),
            ("filter", "statuscode:200"), ("to", before), ("limit", "-5")])
        try:
            body, _ = fetcher(query, timeout, attempts=1)
            if body.lstrip().startswith(b"<"):
                raise ValueError("Archive returned an HTML service/error page instead of CDX JSON")
            records = json.loads(body)
            if not isinstance(records, list):
                raise ValueError("Unexpected CDX response")
        except Exception as error:
            return {"status": "archive_service_error", "cutoff_utc": before, "query": query,
                    "error": f"{type(error).__name__}: {error}"}
        for timestamp, captured_url, mime, status in reversed(records[1:]):
            if timestamp > before:
                continue
            replay = f"https://web.archive.org/web/{timestamp}id_/{captured_url}"
            try:
                body, meta = fetcher(replay, replay_timeout if replay_timeout is not None else min(timeout, 12), attempts=1)
                actual = re.search(r"/web/(\d{14})", meta["url_effective"])
                if not actual or actual.group(1) > before:
                    raise ValueError("Replay redirected outside the allowed capture date")
                return {**store_image(body), "source": "archive", "capture": actual.group(1),
                        "cutoff_utc": before, "download_url": meta["url_effective"], "query": query}
            except Exception as error:
                attempts.append(f"{timestamp}: {type(error).__name__}: {error}")
    return {"status": "no_usable_capture", "cutoff_utc": before, "error": "; ".join(attempts) or "No capture found before Hugo date"}


def asset(result, date):
    return result.get("live") or result.get("archives", {}).get(date, {})


def download(data, workers, timeout, archive_only=False):
    urls = sorted({u for a in data["articles"] for u in a["urls"]})
    pending = [] if archive_only else [u for u in urls if not data["results"].get(u, {}).get("live")]
    print(f"Fetching {len(pending)} live URLs", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_live, u, timeout): u for u in pending}
        for number, future in enumerate(concurrent.futures.as_completed(futures), 1):
            url = futures[future]
            previous = data["results"].get(url, {}).get("archives", {})
            data["results"][url] = future.result()
            if previous and "live" not in data["results"][url]:
                data["results"][url]["archives"] = previous
            if number % 25 == 0 or number == len(pending):
                save(data)
                count = sum("live" in value for value in data["results"].values())
                print(f"Live {number}/{len(pending)}; {count} valid images", flush=True)
    save(data)
    embedded = {u for a in data["articles"] for u in a.get("embedded", a["urls"])}
    for url, result in data["results"].items():
        if url not in embedded and result.get("response_type", "").startswith("text/html"):
            result["excluded"] = "linked_page_not_image"
    archive_jobs = sorted({(u, a["date"]) for a in data["articles"] for u in a["urls"] if "live" not in data["results"][u] and not data["results"][u].get("excluded")})
    pending_archives = [(u, date) for u, date in archive_jobs if not data["results"][u]["archives"].get(date, {}).get("local_url")]
    print(f"Archive recovery: {len(pending_archives)} URL/date pairs", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(workers, 4)) as pool:
        futures = {pool.submit(fetch_archive, u, date, timeout): (u, date) for u, date in pending_archives}
        for number, future in enumerate(concurrent.futures.as_completed(futures), 1):
            url, date = futures[future]
            data["results"][url]["archives"][date] = future.result()
            save(data)
            if number % 10 == 0 or number == len(pending_archives):
                restored = sum(bool(a.get("local_url")) for r in data["results"].values() for a in r.get("archives", {}).values())
                print(f"Archive {number}/{len(pending_archives)}; restored {restored}", flush=True)
    save(data)
    report(data)


def report(data):
    lines = ["# Unresolved article images", "", "Archive cutoff: strictly before the article's Hugo date (converted to UTC).",
             "Original URLs are preserved when downloads fail. Service errors do not prove that no archive capture exists.", ""]
    success = failed = 0
    for article in data["articles"]:
        errors = []
        for url in article["urls"]:
            result = data["results"].get(url, {})
            if result.get("excluded"):
                continue
            resolved = asset(result, article["date"])
            application_error = article.get("application_errors", {}).get(url)
            if resolved.get("local_url") and url not in article["unmatched"] and not application_error:
                success += 1
                continue
            failed += 1
            errors.append((url, "downloaded_not_applied" if application_error else resolved.get("status", "not_downloaded"), result.get("live_error", ""), application_error or resolved.get("error", "")))
        if errors:
            lines.extend([f"## {article['title']}", "", f"Article: `{article['path']}`", f"Hugo date: `{article['date']}`", ""])
            for url, status, live_error, archive_error in errors:
                lines.extend([f"- URL: <{url}>", f"  Status: `{status}`", f"  Live: {live_error}", f"  Archive: {archive_error}", ""])
    lines[2:2] = [f"Localized article/URL pairs: {success}. Unresolved pairs: {failed}.", ""]
    REPORT.write_text("\n".join(line.rstrip() for line in lines).rstrip() + "\n")
    print(f"Localized {success} article/URL pairs; unresolved {failed}", flush=True)


def apply(data, only_paths=None):
    changed = 0
    selected = [a for a in data["articles"] if only_paths is None or a["path"] in only_paths]
    for article in selected:
        current = sha((ROOT / article["path"]).read_text().encode())
        if current not in (article["source_sha256"], article.get("rewritten_sha256")):
            raise ValueError(f"Article changed since inventory: {article['path']}")
        for url in article["urls"]:
            resolved = asset(data["results"].get(url, {}), article["date"])
            if resolved.get("local_url"):
                body = (ROOT / "static" / resolved["local_url"].lstrip("/")).read_bytes()
                if sha(body) != resolved["sha256"]:
                    raise ValueError(f"Asset missing or changed: {resolved['local_url']}")
    for article in selected:
        path = ROOT / article["path"]
        source = path.read_text()
        replacements = {u: asset(data["results"].get(u, {}), article["date"]).get("local_url") for u in article["urls"]}
        spans = source_spans(source, [u for u, local in replacements.items() if local])
        for start, end, url in reversed(spans):
            source = source[:start] + replacements[url] + source[end:]
        local_spans = source_spans(source, {local for local in replacements.values() if local})
        wrapped = False
        for start, end, local in reversed(local_spans):
            prefix = source[:start]
            if start > 0 and source[start - 1] == "<" and source[end:end + 1] == ">":
                source = source[:start - 1] + f'<a href="{local}">{local}</a>' + source[end + 1:]
                wrapped = True
            elif not (prefix.rfind("<") > prefix.rfind(">")
                      or re.search(r"\]\(\s*<?$|\]:\s*<?$|\[[^\]\n]*$", prefix)
                      or prefix.rfind("<a ") > prefix.rfind("</a>")):
                source = source[:start] + f'<a href="{local}">{local}</a>' + source[end:]
                wrapped = True
        if spans or wrapped:
            temporary = path.with_name(f".{path.name}.localize-{os.getpid()}.tmp")
            temporary.write_text(source)
            temporary.chmod(path.stat().st_mode & 0o7777)
            if sha(path.read_text().encode()) not in (article["source_sha256"], article.get("rewritten_sha256")):
                temporary.unlink()
                raise ValueError(f"Article changed during image application: {article['path']}")
            temporary.replace(path)
            article["rewritten_sha256"] = sha(source.encode())
            changed += 1
    save(data)
    report(data)
    print(f"Rewrote {changed} articles", flush=True)


def verify(data, render_root):
    seen = set()
    listing = subprocess.run([HUGO, "list", "all"], cwd=ROOT, check=True, capture_output=True, text=True)
    permalinks = {row["path"]: row["permalink"] for row in csv.DictReader(io.StringIO(listing.stdout))}
    for article in data["articles"]:
        source = (ROOT / article["path"]).read_text()
        if render_root:
            relative = unquote(urlsplit(permalinks[article["path"]]).path).lstrip("/")
            parser = ArticleImages()
            parser.feed((render_root / relative / "index.html").read_text())
            expected = {url for url in article["urls"] if not asset(data["results"].get(url, {}), article["date"]).get("local_url") or url in article["unmatched"] or url in article.get("application_errors", {})}
            if parser.images != expected:
                raise ValueError(f"Unexpected remaining external image URLs: {article['path']} {parser.images ^ expected}")
        for url in article["urls"]:
            result = asset(data["results"].get(url, {}), article["date"])
            if not result.get("local_url") or url in article["unmatched"] or url in article.get("application_errors", {}):
                continue
            local = result["local_url"]
            if local not in source:
                raise ValueError(f"Missing local URL: {article['path']} {local}")
            if render_root and local not in parser.local_references:
                raise ValueError(f"Local URL is not clickable or displayed: {article['path']} {local}")
            if local not in seen:
                body = (ROOT / "static" / local.lstrip("/")).read_bytes()
                if sha(body) != result["sha256"]:
                    raise ValueError(f"Checksum mismatch: {local}")
                if render_root and (render_root / local.lstrip("/")).read_bytes() != body:
                    raise ValueError(f"Hugo output mismatch: {local}")
                seen.add(local)
            if result["source"] == "archive" and result["capture"] >= datetime.fromisoformat(article["date"]).astimezone(timezone.utc).strftime("%Y%m%d%H%M%S"):
                raise ValueError("Archive capture does not precede publication")
    print(f"Verified {len(seen)} local assets and article references", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["inventory", "download", "recover", "apply", "verify", "report"])
    parser.add_argument("--render-root", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=25)
    args = parser.parse_args()
    if args.action == "inventory":
        if not args.render_root:
            parser.error("inventory requires --render-root from hugo --buildDrafts --buildFuture")
        inventory(args.render_root)
        return
    data = json.loads(MANIFEST.read_text())
    if args.action in ("download", "recover"):
        download(data, args.workers, args.timeout, archive_only=args.action == "recover")
    elif args.action == "apply":
        apply(data)
    elif args.action == "verify":
        verify(data, args.render_root)
    else:
        report(data)


if __name__ == "__main__":
    main()
