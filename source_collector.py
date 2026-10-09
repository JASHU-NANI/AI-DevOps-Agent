from __future__ import annotations

import os
import re
from dataclasses import dataclass

from github_client import GitHubClient


MAX_FILES = 8
MAX_FILE_CHARS = 20_000
MAX_TOTAL_CHARS = 80_000

IMPORTANT_FILES = {
    ".java",
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".xml",
    ".yml",
    ".yaml",
    ".json",
    ".properties",
    ".gradle",
    ".kts",
    ".sql",
    ".sh",
}


@dataclass
class SourceFile:
    path: str
    content: str


def _extract_paths_from_logs(logs: str) -> list[str]:

    candidates = set()

    patterns = [
        #using regex pattern to find the files 
        r"(?:^|[\s:(])([A-Za-z0-9_.\-/\\]+\.(?:java|py|js|ts|tsx|jsx|xml|yml|yaml|json|properties|gradle|kts|sql|sh))(?:[:)\s]|$)",

        #source file path
        r"([A-Za-z0-9_.\-/\\]+\.java):\[\d+,\d+\]",

        # Maven/Gradle paths
        r"(?:file:)?([A-Za-z0-9_.\-/\\]+/(?:src|test)/[A-Za-z0-9_.\-/\\]+)",
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, logs, re.IGNORECASE):
            path = match.group(1).replace("\\", "/")

            path = re.sub(r"^[A-Za-z]:/", "", path)

            if "://" in path:
                continue

            if path.startswith("/"):
                path = path.lstrip("/")

            candidates.add(path)

    return list(candidates)


def _looks_like_source_file(path: str) -> bool:
    lower = path.lower()

    if any(lower.endswith(ext) for ext in IMPORTANT_FILES):
        return True

    return False


def collect_relevant_source(
    client: GitHubClient,
    repo: str,
    sha: str,
    logs: str,
) -> list[SourceFile]:

    paths = _extract_paths_from_logs(logs)

    paths = [
        p for p in paths
        if _looks_like_source_file(p)
    ]

    paths = paths[:MAX_FILES]

    results: list[SourceFile] = []
    total_chars = 0

    for path in paths:
        content = client.get_file_content(repo, path, sha)

        if not content:
            continue

        content = content[:MAX_FILE_CHARS]

        if total_chars + len(content) > MAX_TOTAL_CHARS:
            break

        results.append(
            SourceFile(
                path=path,
                content=content,
            )
        )

        total_chars += len(content)

    return results