from __future__ import annotations

import logging
import os
import re
import sys

from analyzer import Analysis, analyze_job
from commenter import RunContext, build_markdown, post_pr_comment, write_job_summary
from github_client import GitHubClient, GitHubError
from log_collector import collect_failed_job_logs
from source_collector import collect_relevant_source

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("agent")

REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
SHA_RE = re.compile(r"[0-9a-fA-F]{40}")


def _env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        sys.exit(f"Missing required environment variable: {name}")
    return value


def _report_error(repo: str, run_id: str, message: str) -> int:
    log.error(message)
    write_job_summary(
        f"##AI DevOps Agent: could not analyze `{repo}` run `{run_id}`\n\n{message}"
    )
    return 1


def _collect_source(gh, repo, ctx, job_logs):
    """Best-effort source collection; never fails the analysis."""
    if not (job_logs and SHA_RE.fullmatch(ctx.sha)):
        log.info("Skipping source collection: no valid commit SHA.")
        return []
    try:
        logs = "\n\n".join(j.excerpt for j in job_logs if j.excerpt)
        files = collect_relevant_source(gh, repo, ctx.sha, logs)
        log.info("Collected %d source files from %s", len(files), ctx.sha)
        return files
    except GitHubError as exc:
        log.warning("Could not collect source files: %s", exc)
        return []


def main() -> int:
    # 1. Validate inputs
    repo, run_id = _env("TARGET_REPO"), _env("TARGET_RUN_ID")
    if not REPO_RE.match(repo) or not run_id.isdigit():
        sys.exit("Invalid TARGET_REPO or TARGET_RUN_ID")

    allowed = [o.strip().lower() for o in os.getenv("ALLOWED_OWNERS", "").split(",") if o.strip()]
    if allowed and repo.split("/")[0].lower() not in allowed:
        sys.exit("Repository owner is not in ALLOWED_OWNERS")

    gh = GitHubClient(_env("AGENT_GITHUB_TOKEN"))

    # 2. Fetch run; skip anything that isn't failed or in progress
    try:
        run = gh.get_run(repo, run_id)
    except GitHubError as exc:
        return _report_error(repo, run_id, f"Cannot read run: {exc}")

    conclusion = run.get("conclusion")
    in_progress = conclusion is None
    if not in_progress and conclusion != "failure":
        log.info("run %s/%s conclusion=%s, nothing to do", repo, run_id, conclusion)
        return 0

    ctx = RunContext(
        repo=repo,
        run_id=run_id,
        workflow=run.get("name", "?"),
        branch=run.get("head_branch", "?"),
        sha=run.get("head_sha", "?" * 7),
        run_url=run.get("html_url", ""),
    )
    log.info("Analyzing %s run=%s workflow=%s branch=%s sha=%s",
             repo, run_id, ctx.workflow, ctx.branch, ctx.sha)

    # 3. Collect logs
    try:
        job_logs, total_failed = collect_failed_job_logs(gh, repo, run_id)
    except GitHubError as exc:
        return _report_error(repo, run_id, f"Cannot fetch job logs: {exc}")

    if in_progress and not job_logs:
        log.info("run %s/%s in progress with no failed jobs, nothing to do", repo, run_id)
        return 0

    # 4. Analyze
    if job_logs:
        source_files = _collect_source(gh, repo, ctx, job_logs)
        analyses = [
            analyze_job(job, repo, ctx.workflow, ctx.branch, source_files=source_files)
            for job in job_logs
        ]
    else:
        analyses = [
            Analysis(
                failed_job="(none)",
                failed_step="(none)",
                category="configuration",
                confidence="low",
                root_cause=(
                    "The run failed but no failed job was found. "
                    "This usually means a workflow syntax/startup problem."
                ),
                suggested_fix=f"Open the run page and check the workflow file: {ctx.run_url}",
                evidence=["(no job logs available)"],
            )
        ]

    markdown = build_markdown(ctx, analyses, total_failed)

    # 5. Publish: PR comment if possible, otherwise Job Summary
    pr_number = gh.find_pr_number(repo, run)
    if not pr_number:
        log.info("no PR for %s run %s, publishing to Job Summary", repo, run_id)
        write_job_summary(markdown)
        return 0

    try:
        post_pr_comment(gh, ctx, pr_number, markdown)
    except GitHubError as exc:
        write_job_summary(markdown)
        return _report_error(
            repo, run_id,
            f"Analysis done but PR comment failed: {exc}. Analysis is shown above.",
        )

    log.info("posted analysis on %s#%s", repo, pr_number)
    server = os.getenv("GITHUB_SERVER_URL", "https://github.com")
    write_job_summary(
        f"Also posted as a PR comment: [{repo}#{pr_number}]({server}/{repo}/pull/{pr_number})\n\n"
        f"{markdown}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())