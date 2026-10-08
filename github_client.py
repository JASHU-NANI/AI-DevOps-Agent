"""Thin GitHub REST client: retries, rate-limit handling, clear errors, no secret leakage."""
from __future__ import annotations

import logging
import os
import time

import requests

log = logging.getLogger("agent.github")

MAX_RETRIES = 3
MAX_WAIT_SECONDS = 90


class GitHubError(Exception):
    pass

class AuthError(GitHubError):
    pass

class PermissionDenied(GitHubError):
    pass

class NotFound(GitHubError):
    pass

class RateLimited(GitHubError):
    pass

class GitHubClient:
    def __init__(self, token: str):
        
        self.api = os.getenv("GITHUB_API_URL", "https://api.github.com").rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": os.getenv("GITHUB_API_VERSION", "2022-11-28"),
            "User-Agent": "ai-devops-agent",
        })

    
    def _request(self, method: str, path: str, **kwargs) -> requests.Response:
        url = f"{self.api}{path}"
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = self.session.request(method, url, timeout=30, **kwargs)
            except requests.RequestException as exc:
                if attempt == MAX_RETRIES:
                    raise GitHubError(f"network error: {type(exc).__name__}") from None
                time.sleep(2 ** attempt)
                continue

            if resp.status_code < 400:
                return resp

            if resp.status_code in (403, 429) and self._is_rate_limited(resp):
                wait = self._wait_seconds(resp)
                if wait > MAX_WAIT_SECONDS or attempt == MAX_RETRIES:
                    raise RateLimited(f"GitHub rate limit hit; resets in ~{wait}s")
                log.warning("rate limited, sleeping %ss", wait)
                time.sleep(wait)
                continue

            if resp.status_code >= 500 and attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)
                continue

            self._raise_for(resp)
        raise GitHubError("request failed after retries")  # pragma: no cover

    @staticmethod
    def _is_rate_limited(resp: requests.Response) -> bool:
        h = resp.headers
        return (resp.status_code == 429
                or h.get("x-ratelimit-remaining") == "0"
                or "retry-after" in h)

    @staticmethod
    def _wait_seconds(resp: requests.Response) -> int:
        h = resp.headers
        if "retry-after" in h:
            return int(h["retry-after"]) + 1
        reset = int(h.get("x-ratelimit-reset", time.time() + 60))
        return max(1, int(reset - time.time()) + 1)

    @staticmethod
    def _raise_for(resp: requests.Response) -> None:
        code = resp.status_code
        try:
            msg = resp.json().get("message", "")
        except ValueError:
            msg = ""
        if code == 401:
            raise AuthError("GitHub token is invalid or expired")
        if code == 403:
            needed = resp.headers.get("X-Accepted-GitHub-Permissions", "unknown")
            raise PermissionDenied(
                f"token lacks permission ({msg}). Accepted permissions for this endpoint: {needed}")
        if code in (404, 410):
            raise NotFound(f"not found / gone (HTTP {code}). For private repos GitHub also "
                           "returns 404 when the token has no access to the repository")
        raise GitHubError(f"HTTP {code}: {msg}")

    # ---------- workflow data ----------
    def get_run(self, repo: str, run_id: str) -> dict:
        return self._request("GET", f"/repos/{repo}/actions/runs/{run_id}").json()

    def list_failed_jobs(self, repo: str, run_id: str) -> list[dict]:
        """All failed jobs of the latest attempt (a run can have several)."""
        jobs, page = [], 1
        while True:
            data = self._request(
                "GET", f"/repos/{repo}/actions/runs/{run_id}/jobs",
                params={"filter": "latest", "per_page": 100, "page": page}).json()
            batch = data.get("jobs", [])
            jobs.extend(batch)
            if len(batch) < 100:
                break
            page += 1
        return [j for j in jobs if j.get("conclusion") in ("failure", "timed_out")]

    def get_job_logs(self, repo: str, job_id: int) -> str | None:
        """Plain-text log of one job. None if expired/missing."""
        try:
            return self._request("GET", f"/repos/{repo}/actions/jobs/{job_id}/logs").text
        except NotFound:
            return None
        # ---------- repository source ----------
    def get_file_content(self, repo: str, path: str, ref: str) -> str | None:
        import base64

        try:
            data = self._request(
                "GET",
                f"/repos/{repo}/contents/{path}",
                params={"ref": ref},
            ).json()

            if data.get("type") != "file":
                return None

            content = data.get("content", "")
            if not content:
                return None

            return base64.b64decode(content).decode("utf-8", errors="replace")

        except NotFound:
            return None
        
    # ---------- pull requests ----------
    def find_pr_number(self, repo: str, run: dict) -> int | None:
        prs = run.get("pull_requests") or []
        if prs:
            return prs[0]["number"]
        # Fallback: payload list is empty for fork PRs / some events.
        try:
            resp = self._request("GET", f"/repos/{repo}/commits/{run['head_sha']}/pulls")
            open_prs = [p for p in resp.json() if p.get("state") == "open"]
            return open_prs[0]["number"] if open_prs else None
        except GitHubError as exc:
            log.warning("PR lookup failed, treating as no PR: %s", exc)
            return None

    def post_pr_comment(self, repo: str, pr_number: int, body: str) -> None:
        # PR comments use the issues endpoint.
        self._request("POST", f"/repos/{repo}/issues/{pr_number}/comments", json={"body": body})
