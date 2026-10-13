from guardrails import redact
from __future__ import annotations

import re
from dataclasses import dataclass, field

from github_client import GitHubClient

MAX_JOBS = 5

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
TIMESTAMP = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z "
)

ERROR_HINT = re.compile(
    r"##\[error\]|\berror\b|\bfailed\b|\bexception\b|\bfatal\b|traceback|exit code [1-9]",
    re.I
)#used by falback analysis

@dataclass
class JobLog:  #this data class stores the job name, id, url ...in it.
    name: str
    job_id: int
    url: str
    failed_steps: list[str] = field(default_factory=list)
    excerpt: str = ""
    logs_available: bool = True



#this is not just storing the error lines. it keeps complete job log
def _excerpt(raw: str) -> str:
    lines = [
        TIMESTAMP.sub("", ANSI.sub("", ln))
        for ln in raw.splitlines()
    ]

    #remove formatting and time then redact matching secrets
    return redact("\n".join(lines))


def collect_failed_job_logs(
    client: GitHubClient,
    repo: str,
    run_id: str
) -> tuple[list[JobLog], int]:

    #Returns (job logs, total number of failed jobs.

    failed = client.list_failed_jobs(repo, run_id)

    results: list[JobLog] = []
 
    for job in failed[:MAX_JOBS]:
        steps = [
            s["name"]
            for s in job.get("steps", [])
            if s.get("conclusion") == "failure"
        ]

        raw = client.get_job_logs(repo, job["id"])

        results.append(
            JobLog(
                name=job["name"],
                job_id=job["id"],
                url=job.get("html_url", ""),
                failed_steps=steps,
                excerpt=_excerpt(raw) if raw else "",
                logs_available=raw is not None,
            )
        )

    return results, len(failed)