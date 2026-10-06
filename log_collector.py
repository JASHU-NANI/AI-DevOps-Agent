"""Fetch logs of ALL failed jobs and reduce them to the part that matters."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from github_client import GitHubClient

MAX_JOBS = 5            # analyze at most this many failed jobs per run
MAX_EXCERPT_CHARS = 65_000
TAIL_LINES = 120
WINDOW = 12             # lines of context around an error hit
MAX_HITS = 5

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z ")
ERROR_HINT = re.compile(
    r"##\[error\]|\berror\b|\bfailed\b|\bexception\b|\bfatal\b|traceback|exit code [1-9]", re.I)

SECRET_PATTERNS = [
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"AIza[0-9A-Za-z_\-]{30,}"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]{20,}"),
]


def redact(text: str) -> str:
    for pattern in SECRET_PATTERNS:
        text = pattern.sub("***REDACTED***", text)
    return text


@dataclass
class JobLog:
    name: str
    job_id: int
    url: str
    failed_steps: list[str] = field(default_factory=list)
    excerpt: str = ""
    logs_available: bool = True


def _excerpt(raw: str) -> str:
    lines = [TIMESTAMP.sub("", ANSI.sub("", ln)) for ln in raw.splitlines()]
    hits = [i for i, ln in enumerate(lines) if ERROR_HINT.search(ln)][:MAX_HITS]

    keep = set(range(max(0, len(lines) - TAIL_LINES), len(lines)))
    for i in hits:
        keep.update(range(max(0, i - WINDOW), min(len(lines), i + WINDOW + 1)))

    out, prev = [], None
    for i in sorted(keep):
        if prev is not None and i != prev + 1:
            out.append("...")
        out.append(lines[i])
        prev = i
    text = redact("\n".join(out))
    return text[-MAX_EXCERPT_CHARS:]


def collect_failed_job_logs(client: GitHubClient, repo: str, run_id: str) -> tuple[list[JobLog], int]:
    """Returns (job logs, total number of failed jobs)."""
    failed = client.list_failed_jobs(repo, run_id)
    results: list[JobLog] = []
    for job in failed[:MAX_JOBS]:
        steps = [s["name"] for s in job.get("steps", []) if s.get("conclusion") == "failure"]
        raw = client.get_job_logs(repo, job["id"])
        results.append(JobLog(
            name=job["name"], job_id=job["id"], url=job.get("html_url", ""),
            failed_steps=steps,
            excerpt=_excerpt(raw) if raw else "",
            logs_available=raw is not None,
        ))
    return results, len(failed)
