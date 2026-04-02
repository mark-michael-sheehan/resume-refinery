"""Career repository storage — CRUD for structured career data.

Repositories are stored at: ~/.resume_refinery/careers/<repo_id>/
Override the root with the RESUME_REFINERY_CAREERS_DIR env var.

Layout:
    <repo_id>/
        career.json          — Full CareerRepository state (single source of truth)
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from .models import CareerRepository


def _careers_root() -> Path:
    override = os.environ.get("RESUME_REFINERY_CAREERS_DIR")
    if override:
        return Path(override)
    return Path.home() / ".resume_refinery" / "careers"


def _slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    return text[:40] or "unnamed"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class CareerRepoStore:
    """Read/write career repositories to disk."""

    def __init__(self) -> None:
        self.root = _careers_root()
        self.root.mkdir(parents=True, exist_ok=True)

    def create(self, name: str) -> CareerRepository:
        """Create a new career repository with a slug derived from the name."""
        base_slug = _slugify(name)
        repo_id = base_slug
        suffix = 2
        while (self.root / repo_id).exists():
            repo_id = f"{base_slug}-{suffix}"
            suffix += 1

        repo_dir = self.root / repo_id
        repo_dir.mkdir(parents=True)

        now = _now_iso()
        repo = CareerRepository(
            repo_id=repo_id,
            created_at=now,
            updated_at=now,
        )
        repo.identity.name = name
        self._write(repo)
        return repo

    def save(self, repo: CareerRepository) -> CareerRepository:
        """Persist the current state of a career repository."""
        repo.updated_at = _now_iso()
        self._write(repo)
        return repo

    def get(self, repo_id: str) -> CareerRepository:
        """Load a career repository by ID."""
        meta_file = self.root / repo_id / "career.json"
        if not meta_file.exists():
            raise FileNotFoundError(f"Career repository not found: {repo_id}")
        return self._read(repo_id)

    def list_repos(self) -> list[CareerRepository]:
        """List all career repositories."""
        repos: list[CareerRepository] = []
        for path in sorted(self.root.iterdir()):
            meta_file = path / "career.json"
            if path.is_dir() and meta_file.exists():
                try:
                    repos.append(self._read(path.name))
                except Exception as exc:
                    logging.warning("Skipping unreadable career repo '%s': %s", path.name, exc)
        return repos

    def delete(self, repo_id: str) -> None:
        """Delete a career repository from disk."""
        import shutil
        repo_dir = self.root / repo_id
        if repo_dir.exists():
            shutil.rmtree(repo_dir)

    def _write(self, repo: CareerRepository) -> None:
        path = self.root / repo.repo_id / "career.json"
        path.write_text(repo.model_dump_json(indent=2), encoding="utf-8")

    def _read(self, repo_id: str) -> CareerRepository:
        path = self.root / repo_id / "career.json"
        try:
            return CareerRepository(**json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Career repo '{repo_id}' metadata is corrupted (invalid JSON): {exc}"
            ) from exc
