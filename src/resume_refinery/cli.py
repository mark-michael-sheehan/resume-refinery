"""CLI entry point — resume-refinery commands."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from .agent import ResumeRefineryAgent
from .exporters import export_document_set
from .models import DocumentKey
from .parsers import load_career_profile, load_job_description, load_voice_profile
from .reviewers import DocumentReviewer
from .session import SessionStore

app = typer.Typer(
    help="Resume Refinery — AI-powered career document generation",
    no_args_is_help=True,
)
console = Console()
store = SessionStore()


# ---------------------------------------------------------------------------
# new
# ---------------------------------------------------------------------------


@app.command()
def new(
    career_profile: Annotated[Path, typer.Argument(help="Career profile markdown file")],
    voice_profile: Annotated[Path, typer.Argument(help="Voice profile markdown file")],
    job_description: Annotated[Path, typer.Argument(help="Job description markdown/text file")],
    skip_review: bool = typer.Option(False, "--skip-review", help="Skip auto-review after generation"),
    allow_unverified: bool = typer.Option(
        False,
        "--allow-unverified",
        help="Allow outputs even if strict truth review still finds unsupported claims",
    ),
):
    """Start a new session: generate all documents then run truth + quality reviews."""
    career = load_career_profile(career_profile)
    voice = load_voice_profile(voice_profile)
    job = load_job_description(job_description)

    session = store.create(job, career, voice)
    console.print(f"\n[bold green]Session created:[/bold green] {session.session_id}")

    agent = ResumeRefineryAgent()
    from .models import DocumentSet
    docs = DocumentSet()

    _generate_with_progress(agent, career, voice, job, docs)

    truth = _enforce_truth(agent, docs, career, voice, job)

    session = store.save_documents(session, docs)

    if truth and not truth.all_supported and not allow_unverified:
        from .models import ReviewBundle
        store.save_reviews(session, ReviewBundle(truthfulness=truth))
        _print_truth_summary(truth)
        console.print(
            "\n[red]Strict truth check failed. Re-run with --allow-unverified if you want to keep this version anyway.[/red]"
        )
        raise typer.Exit(2)

    _export_and_report(session, docs)

    if not skip_review:
        _run_reviews(session, docs, career, voice, truth)


# ---------------------------------------------------------------------------
# refine
# ---------------------------------------------------------------------------


@app.command()
def refine(
    session_id: Annotated[str, typer.Argument(help="Session ID to refine")],
    feedback: Annotated[str, typer.Option("--feedback", "-f", help="Feedback for the agent")],
    doc: Annotated[
        Optional[str],
        typer.Option("--doc", "-d", help="cover_letter | resume | interview_guide (default: all)"),
    ] = None,
    skip_review: bool = typer.Option(False, "--skip-review", help="Skip auto-review after generation"),
    allow_unverified: bool = typer.Option(
        False,
        "--allow-unverified",
        help="Allow outputs even if strict truth review still finds unsupported claims",
    ),
):
    """Regenerate one or all documents in a session with feedback."""
    session = store.get(session_id)
    career, voice = store.load_inputs(session)
    job = session.job_description
    agent = ResumeRefineryAgent()

    # Load existing documents to keep whichever ones aren't being regenerated
    current_docs = store.load_documents(session)

    keys_to_regen: list[DocumentKey]
    if doc:
        if doc not in ("cover_letter", "resume", "interview_guide"):
            console.print(f"[red]Unknown document: {doc}[/red]")
            raise typer.Exit(1)
        keys_to_regen = [doc]  # type: ignore[list-item]
    else:
        keys_to_regen = ["cover_letter", "resume", "interview_guide"]

    for key in keys_to_regen:
        console.print(f"\n[bold]Regenerating {key.replace('_', ' ')}...[/bold]")
        previous = current_docs.get(key)
        chunks = []
        for chunk in agent.stream_document(key, career, voice, job, feedback=feedback, previous_version=previous):
            console.print(chunk, end="")
            chunks.append(chunk)
        current_docs.set(key, "".join(chunks).strip())

    truth = _enforce_truth(agent, current_docs, career, voice, job, feedback=feedback)

    session = store.save_documents(
        session,
        current_docs,
        feedback=feedback,
        docs_regenerated=keys_to_regen,
    )

    if truth and not truth.all_supported and not allow_unverified:
        from .models import ReviewBundle
        store.save_reviews(session, ReviewBundle(truthfulness=truth))
        _print_truth_summary(truth)
        console.print(
            "\n[red]Strict truth check failed. Re-run with --allow-unverified if you want to keep this version anyway.[/red]"
        )
        raise typer.Exit(2)

    _export_and_report(session, current_docs)

    if not skip_review:
        _run_reviews(session, current_docs, career, voice, truth)


# ---------------------------------------------------------------------------
# review
# ---------------------------------------------------------------------------


@app.command()
def review(
    session_id: Annotated[str, typer.Argument(help="Session ID to review")],
    version: Annotated[Optional[int], typer.Option("--version", "-v")] = None,
):
    """Re-run voice-match and AI-detection reviews on a session's documents."""
    session = store.get(session_id)
    career, voice = store.load_inputs(session)
    docs = store.load_documents(session, version=version)

    _run_reviews(session, docs, career, voice)


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


@app.command(name="list")
def list_sessions():
    """List all sessions."""
    sessions = store.list_sessions()
    if not sessions:
        console.print("[dim]No sessions found.[/dim]")
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("Session ID")
    table.add_column("Job Title")
    table.add_column("Company")
    table.add_column("Created")
    table.add_column("Versions", justify="right")

    for s in sessions:
        table.add_row(
            s.session_id,
            s.job_description.title or "—",
            s.job_description.company or "—",
            s.created_at[:10],
            str(s.current_version),
        )
    console.print(table)


# ---------------------------------------------------------------------------
# show
# ---------------------------------------------------------------------------


@app.command()
def show(
    session_id: Annotated[str, typer.Argument(help="Session ID to show")],
    version: Annotated[Optional[int], typer.Option("--version", "-v")] = None,
    open_dir: bool = typer.Option(False, "--open", "-o", help="Open the session folder"),
):
    """Show session details and optionally open its folder."""
    session = store.get(session_id)
    v = version or session.current_version

    console.print(Panel(f"[bold]{session_id}[/bold]", title="Session"))
    console.print(f"  Job:       {session.job_description.title or '—'} @ {session.job_description.company or '—'}")
    console.print(f"  Created:   {session.created_at[:10]}")
    console.print(f"  Versions:  {session.current_version}")
    console.print()

    for vi in session.versions:
        marker = " [bold green]← current[/bold green]" if vi.version == session.current_version else ""
        reviews_indicator = " [dim](reviewed)[/dim]" if vi.has_reviews else ""
        console.print(
            f"  v{vi.version}  {vi.created_at[:16]}{reviews_indicator}{marker}"
        )
        if vi.feedback:
            console.print(f"       feedback: [italic]{vi.feedback}[/italic]")
        if vi.docs_regenerated and len(vi.docs_regenerated) < 3:
            console.print(f"       regenerated: {', '.join(vi.docs_regenerated)}")

    # Show review summary for chosen version
    reviews = store.load_reviews(session, version=v)
    if reviews.voice or reviews.ai_detection or reviews.truthfulness:
        console.print()
        _print_review_summary(reviews)

    session_dir = store.session_dir(session_id)
    console.print(f"\n[dim]Location:[/dim] {session_dir / f'v{v}'}")

    if open_dir:
        _open_folder(session_dir / f"v{v}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _generate_with_progress(agent, career, voice, job, docs) -> None:
    doc_labels = {
        "cover_letter": "Cover Letter",
        "resume": "Resume",
        "interview_guide": "Interview Guide",
    }
    for key, label in doc_labels.items():
        console.print(f"\n[bold cyan]Generating {label}...[/bold cyan]")
        chunks = []
        for chunk in agent.stream_document(key, career, voice, job):
            console.print(chunk, end="")
            chunks.append(chunk)
        docs.set(key, "".join(chunks))
        console.print()


def _export_and_report(session, docs) -> None:
    v = session.current_version
    version_dir = store.session_dir(session.session_id) / f"v{v}"
    exported = export_document_set(docs, version_dir)
    console.print(f"\n[bold green]v{v} saved:[/bold green] {version_dir}")
    for key, path in exported.items():
        console.print(f"  {path.name}")


def _run_reviews(session, docs, career, voice, truth=None) -> None:
    reviewer = DocumentReviewer()

    if truth is None:
        console.print("\n[bold cyan]Running strict truthfulness review...[/bold cyan]")
        with console.status(""):
            truth = reviewer.review_truthfulness(docs, career)

    console.print("\n[bold cyan]Running voice-match review...[/bold cyan]")
    with console.status(""):
        voice_result = reviewer.review_voice(docs, voice)

    console.print("[bold cyan]Running AI-detection review...[/bold cyan]")
    with console.status(""):
        ai_result = reviewer.review_ai_detection(docs)

    from .models import ReviewBundle
    reviews = ReviewBundle(voice=voice_result, ai_detection=ai_result, truthfulness=truth)
    store.save_reviews(session, reviews)
    _print_review_summary(reviews)


def _enforce_truth(agent, docs, career, voice, job, feedback: str | None = None, max_passes: int = 2):
    reviewer = DocumentReviewer()
    truth = None
    for _ in range(max_passes):
        console.print("\n[bold cyan]Running strict truthfulness review...[/bold cyan]")
        with console.status(""):
            truth = reviewer.review_truthfulness(docs, career)

        if truth.all_supported:
            _print_truth_summary(truth)
            return truth

        _print_truth_summary(truth)
        docs_to_fix = []
        if not truth.cover_letter.pass_strict:
            docs_to_fix.append("cover_letter")
        if not truth.resume.pass_strict:
            docs_to_fix.append("resume")
        if not truth.interview_guide.pass_strict:
            docs_to_fix.append("interview_guide")

        for key in docs_to_fix:
            claim_feedback = _claim_feedback_for_doc(key, truth)
            combined_feedback = "\n\n".join(
                p for p in [feedback, claim_feedback, "Rewrite strictly using only evidence from the career profile."] if p
            )
            previous = docs.get(key)
            regenerated = agent.generate_document(
                key,
                career,
                voice,
                job,
                feedback=combined_feedback,
                previous_version=previous,
            )
            docs.set(key, regenerated)

    return truth


def _claim_feedback_for_doc(key, truth) -> str:
    if key == "cover_letter":
        claims = truth.cover_letter.unsupported_claims
    elif key == "resume":
        claims = truth.resume.unsupported_claims
    else:
        claims = truth.interview_guide.unsupported_claims

    if not claims:
        return ""
    bullets = "\n".join(f"- Remove or evidence this claim: {c}" for c in claims[:8])
    return f"Unsupported claims to fix for {key}:\n{bullets}"


def _print_review_summary(reviews) -> None:
    if reviews.truthfulness:
        _print_truth_summary(reviews.truthfulness)

    if reviews.voice:
        r = reviews.voice
        match_color = {"strong": "green", "moderate": "yellow", "weak": "red"}[r.overall_match]
        console.print(
            Panel(
                f"Overall match: [{match_color}]{r.overall_match.upper()}[/{match_color}]\n\n"
                + (f"Issues:\n" + "\n".join(f"  • {i}" for i in r.specific_issues) if r.specific_issues else "")
                + ("\n\nSuggestions:\n" + "\n".join(f"  • {s}" for s in r.suggestions) if r.suggestions else ""),
                title="[bold]Voice Match Review[/bold]",
            )
        )

    if reviews.ai_detection:
        r = reviews.ai_detection
        risk_color = {"low": "green", "medium": "yellow", "high": "red"}[r.risk_level]
        all_flags = r.cover_letter_flags + r.resume_flags + r.interview_guide_flags
        console.print(
            Panel(
                f"AI-content risk: [{risk_color}]{r.risk_level.upper()}[/{risk_color}]\n\n"
                + (f"Flagged phrases:\n" + "\n".join(f'  • "{f}"' for f in all_flags[:6]) if all_flags else "No flags.")
                + ("\n\nSuggestions:\n" + "\n".join(f"  • {s}" for s in r.suggestions) if r.suggestions else ""),
                title="[bold]AI-Detection Review[/bold]",
            )
        )


def _print_truth_summary(truth) -> None:
    color = "green" if truth.all_supported else "red"
    lines = [
        f"All claims supported: [{color}]{str(truth.all_supported).upper()}[/{color}]",
        f"Cover Letter unsupported: {len(truth.cover_letter.unsupported_claims)}",
        f"Resume unsupported: {len(truth.resume.unsupported_claims)}",
        f"Interview Guide unsupported: {len(truth.interview_guide.unsupported_claims)}",
    ]
    if truth.suggestions:
        lines.append("\nSuggestions:\n" + "\n".join(f"  • {s}" for s in truth.suggestions[:5]))
    console.print(Panel("\n".join(lines), title="[bold]Truthfulness Review[/bold]"))


def _open_folder(path: Path) -> None:
    if sys.platform == "win32":
        subprocess.run(["explorer", str(path)], check=False)
    elif sys.platform == "darwin":
        subprocess.run(["open", str(path)], check=False)
    else:
        subprocess.run(["xdg-open", str(path)], check=False)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
