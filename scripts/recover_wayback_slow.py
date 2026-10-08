#!/usr/bin/env python3
"""Resume serial Wayback recovery with persisted retry limits and request spacing."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import threading
import time
from datetime import datetime, timezone

import localize_images as images

RUNTIME = images.ROOT / ".image-localization"
STOP = threading.Event()


def now():
    return datetime.now(timezone.utc).isoformat()


def write_state(state):
    temporary = RUNTIME / "state.tmp"
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(RUNTIME / "state.json")


def pending(data, limit):
    jobs = {(url, a["date"]) for a in data["articles"] for url in a["urls"]
            if not data["results"][url].get("excluded") and not images.asset(data["results"][url], a["date"]).get("local_url")}
    return sorted((url, date) for url, date in jobs
                  if len(data["results"][url].get("slow_attempts", {}).get(date, [])) < limit)


class PoliteRequests:
    def __init__(self, pause, state):
        self.pause = pause
        self.state = state
        self.next_allowed = state.get("next_request_epoch", 0)

    def __call__(self, url, timeout, attempts=1):
        delay = max(0, self.next_allowed - time.time())
        self.state.update(phase="waiting" if delay else "request", next_request_epoch=self.next_allowed)
        write_state(self.state)
        if STOP.wait(delay) or STOP.is_set():
            raise InterruptedError("Recovery stopped")
        self.state.update(phase="request", request_url=url, request_started=now())
        write_state(self.state)
        print(f"{now()} Request: {url}", flush=True)
        try:
            return images.request(url, timeout, attempts=1, connect_timeout=min(timeout, 30))
        finally:
            self.next_allowed = time.time() + self.pause
            self.state.update(next_request_epoch=self.next_allowed, request_finished=now())
            write_state(self.state)


def run(args):
    RUNTIME.mkdir(parents=True, exist_ok=True)
    with (RUNTIME / "worker.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("A slow recovery worker is already running")
        previous = json.loads((RUNTIME / "state.json").read_text()) if (RUNTIME / "state.json").exists() else {}
        state = {"pid": os.getpid(), "started": now(), "status": "running", "timeout": args.timeout,
                 "pause": args.pause, "attempt_limit": args.attempts, "next_request_epoch": previous.get("next_request_epoch", 0)}
        write_state(state)
        fetcher = PoliteRequests(args.pause, state)
        try:
            while not STOP.is_set():
                data = json.loads(images.MANIFEST.read_text())
                jobs = pending(data, args.attempts)
                if not jobs:
                    state.update(status="completed", finished=now(), phase="idle")
                    break
                print(f"{now()} Pass: {len(jobs)} unresolved URL/date pairs", flush=True)
                for url, date in jobs:
                    if STOP.is_set():
                        break
                    data = json.loads(images.MANIFEST.read_text())
                    result = data["results"][url]
                    if images.asset(result, date).get("local_url"):
                        continue
                    history = result.setdefault("slow_attempts", {}).setdefault(date, [])
                    if len(history) >= args.attempts:
                        continue
                    entry = {"started": now(), "status": "in_progress"}
                    history.append(entry)
                    state.update(url=url, hugo_date=date, attempt=len(history), phase="lookup")
                    images.save(data)
                    write_state(state)
                    print(f"{now()} Attempt {len(history)}/{args.attempts}: {url} (Hugo {date})", flush=True)
                    recovered = images.fetch_archive(url, date, args.timeout, replay_timeout=args.timeout, fetcher=fetcher)
                    result.setdefault("archives", {})[date] = recovered
                    entry.update(finished=now(), status=recovered.get("source", recovered.get("status")), error=recovered.get("error"))
                    images.save(data)
                    if recovered.get("local_url"):
                        affected = [a for a in data["articles"] if url in a["urls"] and a["date"] == date]
                        for article in affected:
                            try:
                                images.apply(data, only_paths={article["path"]})
                                article.get("application_errors", {}).pop(url, None)
                            except (ValueError, OSError) as error:
                                article.setdefault("application_errors", {})[url] = str(error)
                                print(f"{now()} Saved image; article needs manual application: {article['path']}: {error}", flush=True)
                        images.save(data)
                    images.report(data)
                    print(f"{now()} Outcome: {entry['status']}", flush=True)
            if STOP.is_set():
                state.update(status="stopped", finished=now(), phase="idle")
        except BaseException as error:
            state.update(status="failed", error=f"{type(error).__name__}: {error}", finished=now())
            raise
        finally:
            write_state(state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempts", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--pause", type=int, default=60)
    args = parser.parse_args()
    if not (1 <= args.attempts <= 10) or args.timeout <= 0 or args.pause < 0:
        parser.error("Use 1..10 attempts, a positive timeout, and a nonnegative pause")
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: STOP.set())
    run(args)


if __name__ == "__main__":
    main()
