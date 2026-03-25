"""CLI entry point — resume-refinery commands."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .models import DocumentKey
from .orchestrator import ResumeRefineryOrchestrator
from .parsers import load_career_profile, load_job_description, load_voice_profile
from .session import SessionStore

app = typer.Typer(
    help="Resume Refinery — AI-powered career document generation",
    no_args_is_help=True,
)
console = Console()
store = SessionStore()
orchestrator = ResumeRefineryOrchestrator(store=store)


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

    result = orchestrator.create_session_run(
        career,
        voice,
        job,
        skip_review=skip_review,
        allow_unverified=allow_unverified,
        progress=_progress,
        stream_callback=_stream_chunk,
    )
    _report_result(result, show_quality_reviews=not skip_review)
    if result.strict_truth_failed:
        console.print(
            "\n[red]Strict truth check failed. Re-run with --allow-unverified if you want to keep this version anyway.[/red]"
        )
        raise typer.Exit(2)


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
    if doc:
        if doc not in ("cover_letter", "resume", "interview_guide"):
            console.print(f"[red]Unknown document: {doc}[/red]")
            raise typer.Exit(1)
    key = doc if doc else None
    result = orchestrator.refine_session_run(
        session_id,
        feedback,
        doc=key,  # type: ignore[arg-type]
        skip_review=skip_review,
        allow_unverified=allow_unverified,
        progress=_progress,
        stream_callback=_stream_chunk,
    )
    _report_result(result, show_quality_reviews=not skip_review)
    if result.strict_truth_failed:
        console.print(
            "\n[red]Strict truth check failed. Re-run with --allow-unverified if you want to keep this version anyway.[/red]"
        )
        raise typer.Exit(2)


# ---------------------------------------------------------------------------
# review
# ---------------------------------------------------------------------------


@app.command()
def review(
    session_id: Annotated[str, typer.Argument(help="Session ID to review")],
    version: Annotated[Optional[int], typer.Option("--version", "-v")] = None,
):
    """Re-run voice-match and AI-detection reviews on a session's documents."""
    result = orchestrator.review_session_run(session_id, version=version, progress=_progress)
    _print_review_summary(result.reviews, show_quality_reviews=True)


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


def _progress(message: str) -> None:
    console.print(f"\n[bold cyan]{message}[/bold cyan]")


def _stream_chunk(chunk: str) -> None:
    sys.stdout.write(chunk)
    sys.stdout.flush()


def _report_result(result, show_quality_reviews: bool) -> None:
    console.print(f"\n[bold green]v{result.session.current_version} saved:[/bold green] {store.session_dir(result.session.session_id) / f'v{result.session.current_version}'}")
    for path in result.exported_paths.values():
        console.print(f"  {Path(path).name}")
    _print_review_summary(result.reviews, show_quality_reviews=show_quality_reviews)


def _print_review_summary(reviews, show_quality_reviews: bool = True) -> None:
    if reviews.truthfulness:
        _print_truth_summary(reviews.truthfulness)

    if show_quality_reviews and reviews.voice:
        r = reviews.voice
        match_color = {"strong": "green", "moderate": "yellow", "weak": "red"}[r.overall_match]
        console.print(
            Panel(
                f"Overall match: [{match_color}]{r.overall_match.upper()}[/{match_color}]\n\n"
                + (f"Issues:\n" + "\n".join(f"  • {i}" for i in r.specific_issues) if r.specific_issues else ""),
                title="[bold]Voice Match Review[/bold]",
            )
        )

    if show_quality_reviews and reviews.ai_detection:
        r = reviews.ai_detection
        risk_color = {"low": "green", "medium": "yellow", "high": "red"}[r.risk_level]
        all_flags = r.cover_letter_flags + r.resume_flags + r.interview_guide_flags
        console.print(
            Panel(
                f"AI-content risk: [{risk_color}]{r.risk_level.upper()}[/{risk_color}]\n\n"
                + (f"Flagged phrases:\n" + "\n".join(f'  • "{f}"' for f in all_flags[:6]) if all_flags else "No flags."),
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
