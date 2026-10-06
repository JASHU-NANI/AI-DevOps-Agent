
from __future__ import annotations

import logging
import os
import re
import sys

from analyzer import Analysis, analyze_job
from commenter import RunContext, build_markdown, post_pr_comment, write_job_summary
from github_client import GitHubClient, GitHubError
from log_collector import collect_failed_job_logs

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("agent")

REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


def _env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        sys.exit(f"Missing required environment variable: {name}")  # prints the name only
    return value


def _report_error(repo: str, run_id: str, message: str) -> int:
    log.error(message)
    write_job_summary(f"## 🤖 AI DevOps Agent: could not analyze `{repo}` run `{run_id}`\n\n{message}")
    return 1


def main() -> int:
    repo, run_id = _env("TARGET_REPO"), _env("TARGET_RUN_ID")
    if not REPO_RE.match(repo) or not run_id.isdigit():
        sys.exit("Invalid TARGET_REPO or TARGET_RUN_ID")

    allowed = [o.strip().lower() for o in os.getenv("ALLOWED_OWNERS", "").split(",") if o.strip()]
    if allowed and repo.split("/")[0].lower() not in allowed:
        sys.exit("Repository owner is not in ALLOWED_OWNERS")

    gh = GitHubClient(_env("AGENT_GITHUB_TOKEN"))

    try:
        run = gh.get_run(repo, run_id)
    except GitHubError as exc:
        return _report_error(repo, run_id, f"Cannot read run: {exc}")

    if run.get("conclusion") != "failure":
        log.info("run %s/%s conclusion=%s, nothing to do", repo, run_id, run.get("conclusion"))
        return 0

    ctx = RunContext(repo=repo, run_id=run_id, workflow=run.get("name", "?"),
                     branch=run.get("head_branch", "?"), sha=run.get("head_sha", "?" * 7),
                     run_url=run.get("html_url", ""))

    try:
        job_logs, total_failed = collect_failed_job_logs(gh, repo, run_id)
    except GitHubError as exc:
        return _report_error(repo, run_id, f"Cannot fetch job logs: {exc}")

    if job_logs:
        analyses = [analyze_job(j, repo, ctx.workflow, ctx.branch) for j in job_logs]
    else:  # e.g. startup_failure / workflow-level failure with no failed job
        analyses = [Analysis(
            failed_job="(none)", failed_step="(none)", category="configuration", confidence="low",
            root_cause="The run failed but no failed job was found. This usually means a workflow "
                       "syntax/startup problem.",
            suggested_fix=f"Open the run page and check the workflow file: {ctx.run_url}",
            evidence=["(no job logs available)"])]

    markdown = build_markdown(ctx, analyses, total_failed)
    pr_number = gh.find_pr_number(repo, run)

    if pr_number:
        try:
            post_pr_comment(gh, ctx, pr_number, markdown)
            log.info("posted analysis on %s#%s", repo, pr_number)
            pr_url = f"{os.getenv('GITHUB_SERVER_URL', 'https://github.com')}/{repo}/pull/{pr_number}"
            write_job_summary(f"Also posted as a PR comment: [{repo}#{pr_number}]({pr_url})\n\n{markdown}")
            return 0
        except GitHubError as exc:
            write_job_summary(markdown)
            return _report_error(repo, run_id,
                                 f"Analysis done but PR comment failed: {exc}. Analysis is shown above.")

    log.info("no PR for %s run %s, publishing to Job Summary", repo, run_id)
    write_job_summary(markdown)
    return 0


if __name__ == "__main__":
    sys.exit(main())
