"""Session management — persistence, versioning, and retrieval.

Sessions are stored at: ~/.resume_refinery/sessions/<session_id>/
Override the root with the RESUME_REFINERY_SESSIONS_DIR env var.

Layout:
    <session_id>/
        session.json            — Session metadata
        inputs/
            career_profile.md
            voice_profile.md
            job_description.md
        v1/
            cover_letter.md
            resume.md
            interview_guide.md
            voice_review.json   (optional)
            ai_review.json      (optional)
        v2/
            ...
"""

from __future__ import annotations

import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .models import (
    AIDetectionResult,
    CareerProfile,
    DocumentKey,
    DocumentSet,
    DraftingContext,
    EvidencePack,
    ExemptedPhrases,
    JobDescription,
    ReviewBundle,
    Session,
    TruthfulnessResult,
    VersionInfo,
    VoiceProfile,
    VoiceReviewResult,
    VoiceStyleGuide,
)


def _sessions_root() -> Path:
    override = os.environ.get("RESUME_REFINERY_SESSIONS_DIR")
    if override:
        return Path(override)
    return Path.home() / ".resume_refinery" / "sessions"


def _slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    return text[:40]


def _make_session_id(job: JobDescription) -> str:
    company = _slugify(job.company or "unknown-company")
    title = _slugify(job.title or "unknown-role")
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    base = f"{company}_{title}_{date}"

    root = _sessions_root()
    candidate = base
    suffix = 2
    while (root / candidate).exists():
        candidate = f"{base}_{suffix}"
        suffix += 1
    return candidate


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class SessionStore:
    """Read/write sessions to disk."""

    def __init__(self) -> None:
        self.root = _sessions_root()
        self.root.mkdir(parents=True, exist_ok=True)

    # --- Create ------------------------------------------------------------

    def create(
        self,
        job: JobDescription,
        career: CareerProfile,
        voice: VoiceProfile,
    ) -> Session:
        """Initialise a new session directory and return the Session object."""
        session_id = _make_session_id(job)
        session_dir = self.root / session_id
        session_dir.mkdir(parents=True)
        (session_dir / "inputs").mkdir()

        # Copy input files into the session
        (session_dir / "inputs" / "career_profile.md").write_text(
            career.raw_content, encoding="utf-8"
        )
        (session_dir / "inputs" / "voice_profile.md").write_text(
            voice.raw_content, encoding="utf-8"
        )
        (session_dir / "inputs" / "job_description.md").write_text(
            job.raw_content, encoding="utf-8"
        )

        now = _now_iso()
        session = Session(
            session_id=session_id,
            job_description=job,
            created_at=now,
            current_version=0,  # bumped to 1 on first save_documents call
            versions=[],
        )
        self._write_metadata(session)
        return session

    # --- Documents ---------------------------------------------------------

    def save_documents(
        self,
        session: Session,
        docs: DocumentSet,
        feedback: str | None = None,
        docs_regenerated: list[DocumentKey] | None = None,
    ) -> Session:
        """Save a new version of documents and update session metadata."""
        session.current_version += 1
        version = session.current_version
        version_dir = self.root / session.session_id / f"v{version}"
        version_dir.mkdir(parents=True, exist_ok=True)

        if docs.cover_letter:
            (version_dir / "cover_letter.md").write_text(docs.cover_letter, encoding="utf-8")
        if docs.resume:
            (version_dir / "resume.md").write_text(docs.resume, encoding="utf-8")
        if docs.interview_guide:
            (version_dir / "interview_guide.md").write_text(
                docs.interview_guide, encoding="utf-8"
            )

        session.versions.append(
            VersionInfo(
                version=version,
                created_at=_now_iso(),
                feedback=feedback,
                docs_regenerated=docs_regenerated or list(_ALL_DOC_KEYS),
                has_reviews=False,
            )
        )
        self._write_metadata(session)
        return session

    def load_documents(self, session: Session, version: int | None = None) -> DocumentSet:
        """Load documents for a given version (defaults to current)."""
        v = version or session.current_version
        version_dir = self.root / session.session_id / f"v{v}"
        return DocumentSet(
            cover_letter=_read_opt(version_dir / "cover_letter.md"),
            resume=_read_opt(version_dir / "resume.md"),
            interview_guide=_read_opt(version_dir / "interview_guide.md"),
        )

    # --- Reviews -----------------------------------------------------------

    def save_reviews(self, session: Session, reviews: ReviewBundle) -> Session:
        """Persist review results for the current version."""
        version_dir = self.root / session.session_id / f"v{session.current_version}"
        if reviews.voice:
            (version_dir / "voice_review.json").write_text(
                reviews.voice.model_dump_json(indent=2), encoding="utf-8"
            )
        if reviews.ai_detection:
            (version_dir / "ai_review.json").write_text(
                reviews.ai_detection.model_dump_json(indent=2), encoding="utf-8"
            )
        if reviews.truthfulness:
            (version_dir / "truth_review.json").write_text(
                reviews.truthfulness.model_dump_json(indent=2), encoding="utf-8"
            )

        # Mark current version as reviewed
        for v in session.versions:
            if v.version == session.current_version:
                v.has_reviews = True
                break

        self._write_metadata(session)
        return session

    def load_reviews(self, session: Session, version: int | None = None) -> ReviewBundle:
        """Load reviews for a given version (defaults to current)."""
        v = version or session.current_version
        version_dir = self.root / session.session_id / f"v{v}"
        voice = _load_model_opt(version_dir / "voice_review.json", VoiceReviewResult)
        ai = _load_model_opt(version_dir / "ai_review.json", AIDetectionResult)
        truth = _load_model_opt(version_dir / "truth_review.json", TruthfulnessResult)
        return ReviewBundle(voice=voice, ai_detection=ai, truthfulness=truth)

    # --- Context (EvidencePack + VoiceStyleGuide) --------------------------

    def save_context(self, session: Session, context: DraftingContext) -> None:
        """Persist the EvidencePack and VoiceStyleGuide for the current version."""
        version_dir = self.root / session.session_id / f"v{session.current_version}"
        version_dir.mkdir(parents=True, exist_ok=True)
        (version_dir / "evidence_pack.json").write_text(
            context.evidence_pack.model_dump_json(indent=2), encoding="utf-8"
        )
        (version_dir / "voice_guide.json").write_text(
            context.voice_style_guide.model_dump_json(indent=2), encoding="utf-8"
        )

    def load_context(self, session: Session, version: int | None = None) -> DraftingContext | None:
        """Load persisted EvidencePack and VoiceStyleGuide for a version."""
        v = version or session.current_version
        version_dir = self.root / session.session_id / f"v{v}"
        ep = _load_model_opt(version_dir / "evidence_pack.json", EvidencePack)
        vg = _load_model_opt(version_dir / "voice_guide.json", VoiceStyleGuide)
        if ep is None or vg is None:
            return None
        return DraftingContext(evidence_pack=ep, voice_style_guide=vg)

    # --- Suppression / exempted phrases ------------------------------------

    def save_suppressions(self, session: Session, phrases: ExemptedPhrases) -> None:
        """Persist the cumulative set of repair-loop exempted phrases for the current version."""
        version_dir = self.root / session.session_id / f"v{session.current_version}"
        version_dir.mkdir(parents=True, exist_ok=True)
        (version_dir / "exempted_phrases.json").write_text(
            phrases.model_dump_json(indent=2), encoding="utf-8"
        )

    # --- Repair pass snapshots ---------------------------------------------

    def save_repair_pass(
        self, session: Session, pass_num: int, docs: DocumentSet,
        reviews: ReviewBundle | None = None,
    ) -> None:
        """Snapshot documents and reviews after a repair pass for auditing."""
        version_dir = self.root / session.session_id / f"v{session.current_version}"
        pass_dir = version_dir / f"repair_pass_{pass_num}"
        pass_dir.mkdir(parents=True, exist_ok=True)
        if docs.cover_letter:
            (pass_dir / "cover_letter.md").write_text(docs.cover_letter, encoding="utf-8")
        if docs.resume:
            (pass_dir / "resume.md").write_text(docs.resume, encoding="utf-8")
        if docs.interview_guide:
            (pass_dir / "interview_guide.md").write_text(docs.interview_guide, encoding="utf-8")
        if reviews:
            if reviews.truthfulness:
                (pass_dir / "truth_review.json").write_text(
                    reviews.truthfulness.model_dump_json(indent=2), encoding="utf-8"
                )
            if reviews.voice:
                (pass_dir / "voice_review.json").write_text(
                    reviews.voice.model_dump_json(indent=2), encoding="utf-8"
                )
            if reviews.ai_detection:
                (pass_dir / "ai_review.json").write_text(
                    reviews.ai_detection.model_dump_json(indent=2), encoding="utf-8"
                )

    def load_repair_pass(
        self, session: Session, pass_num: int, version: int | None = None
    ) -> tuple[DocumentSet | None, ReviewBundle | None]:
        """Load a repair-pass snapshot, or (None, None) if it doesn't exist."""
        v = version or session.current_version
        pass_dir = self.root / session.session_id / f"v{v}" / f"repair_pass_{pass_num}"
        if not pass_dir.exists():
            return None, None
        docs = DocumentSet(
            cover_letter=_read_opt(pass_dir / "cover_letter.md"),
            resume=_read_opt(pass_dir / "resume.md"),
            interview_guide=_read_opt(pass_dir / "interview_guide.md"),
        )
        truth = _load_model_opt(pass_dir / "truth_review.json", TruthfulnessResult)
        voice = _load_model_opt(pass_dir / "voice_review.json", VoiceReviewResult)
        ai = _load_model_opt(pass_dir / "ai_review.json", AIDetectionResult)
        if truth is None and voice is None and ai is None:
            return docs, None
        reviews = ReviewBundle(truthfulness=truth, voice=voice, ai_detection=ai)
        return docs, reviews

    def load_inputs(self, session: Session) -> tuple[CareerProfile, VoiceProfile]:
        """Reload the original career and voice profiles from the session."""
        inputs_dir = self.root / session.session_id / "inputs"
        from .models import CareerProfile as CP
        from .models import VoiceProfile as VP
        from .parsers import load_career_profile, load_voice_profile

        career = load_career_profile(inputs_dir / "career_profile.md")
        voice = load_voice_profile(inputs_dir / "voice_profile.md")
        return career, voice

    # --- List & retrieve ---------------------------------------------------

    def list_sessions(self) -> list[Session]:
        sessions = []
        for path in sorted(self.root.iterdir()):
            meta_file = path / "session.json"
            if path.is_dir() and meta_file.exists():
                try:
                    sessions.append(self._read_metadata(path.name))
                except Exception:
                    pass
        return sessions

    def get(self, session_id: str) -> Session:
        meta_file = self.root / session_id / "session.json"
        if not meta_file.exists():
            raise FileNotFoundError(f"Session not found: {session_id}")
        return self._read_metadata(session_id)

    def session_dir(self, session_id: str) -> Path:
        return self.root / session_id

    # --- Internal ----------------------------------------------------------

    def _write_metadata(self, session: Session) -> None:
        path = self.root / session.session_id / "session.json"
        path.write_text(session.model_dump_json(indent=2), encoding="utf-8")

    def _read_metadata(self, session_id: str) -> Session:
        path = self.root / session_id / "session.json"
        return Session(**json.loads(path.read_text(encoding="utf-8")))


_ALL_DOC_KEYS: tuple[DocumentKey, ...] = ("cover_letter", "resume", "interview_guide")


def _read_opt(path: Path) -> str | None:
    return path.read_text(encoding="utf-8") if path.exists() else None


def _load_model_opt(path: Path, model_cls):
    if not path.exists():
        return None
    try:
        return model_cls(**json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        return None
