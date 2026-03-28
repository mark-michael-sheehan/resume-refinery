"""Local web app for Resume Refinery."""

from __future__ import annotations

import html
from typing import Optional

import uvicorn
from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse

from .models import OrchestrationResult
from .orchestrator import ResumeRefineryOrchestrator
from .parsers import (
    parse_career_profile_content,
    parse_job_description_content,
    parse_voice_profile_content,
)
from .session import SessionStore
from .career_wizard import router as career_router
from .career_repo import CareerRepoStore

app = FastAPI(title="Resume Refinery", version="0.1.0")
store = SessionStore()
orchestrator = ResumeRefineryOrchestrator(store=store)
career_store = CareerRepoStore()

app.include_router(career_router)


def _page(title: str, body: str) -> HTMLResponse:
    return HTMLResponse(
        f"""
<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --bg: #f7f5ef;
      --card: #fffdf8;
      --ink: #1f2421;
      --muted: #55605a;
      --accent: #0f7b6c;
      --line: #d8d4c8;
    }}
    body {{
      margin: 0;
      font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
      background: radial-gradient(circle at 0% 0%, #fffaf0 0%, var(--bg) 40%, #efeae0 100%);
      color: var(--ink);
    }}
    .wrap {{ max-width: 1100px; margin: 2rem auto; padding: 0 1rem; }}
    .card {{ background: var(--card); border: 1px solid var(--line); border-radius: 14px; padding: 1rem 1.25rem; box-shadow: 0 8px 30px rgba(20, 35, 30, 0.06); margin-bottom: 1rem; }}
    h1, h2 {{ margin: 0.2rem 0 0.8rem; }}
    a {{ color: var(--accent); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }}
    @media (max-width: 900px) {{ .grid {{ grid-template-columns: 1fr; }} }}
    label {{ display: block; margin-top: 0.7rem; font-weight: 600; }}
    input[type=file], textarea, select {{ width: 100%; margin-top: 0.4rem; padding: 0.55rem; border-radius: 8px; border: 1px solid var(--line); background: #fff; }}
    button {{ margin-top: 0.9rem; background: var(--accent); color: #fff; border: none; border-radius: 10px; padding: 0.65rem 1rem; cursor: pointer; font-weight: 700; }}
    button:hover {{ opacity: 0.9; }}
    pre {{ white-space: pre-wrap; word-break: break-word; background: #f3f1eb; border: 1px solid var(--line); border-radius: 8px; padding: 0.75rem; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ border-bottom: 1px solid var(--line); text-align: left; padding: 0.5rem; }}
    .muted {{ color: var(--muted); }}
    .ok {{ color: #1d8f52; font-weight: 700; }}
    .bad {{ color: #b00020; font-weight: 700; }}
  </style>
</head>
<body>
  <div class=\"wrap\">{body}</div>
</body>
</html>
"""
    )


def _truth_failed(truth) -> bool:
    return bool(truth and not truth.all_supported)


def _truth_summary(truth) -> str:
    if not truth:
        return "<p class='muted'>No truth review available.</p>"
    status = "PASS" if truth.all_supported else "FAIL"
    klass = "ok" if truth.all_supported else "bad"
    return (
        f"<p>Strict truth check: <span class='{klass}'>{status}</span></p>"
        f"<ul>"
        f"<li>Cover letter unsupported claims: {len(truth.cover_letter.unsupported_claims)}</li>"
        f"<li>Resume unsupported claims: {len(truth.resume.unsupported_claims)}</li>"
        f"<li>Interview guide unsupported claims: {len(truth.interview_guide.unsupported_claims)}</li>"
        f"</ul>"
    )


def _artifact_summary(result: OrchestrationResult) -> str:
    evidence = result.evidence_pack
    style = result.voice_style_guide
    if not evidence and not style:
        return "<p class='muted'>No orchestration artifacts available.</p>"

    parts = ["<div class='grid'>"]
    if evidence:
        evidence_items = "".join(
            f"<li>{html.escape(item.requirement)} -> {html.escape(item.evidence)}</li>"
            for item in evidence.matched_evidence[:6]
        ) or "<li>No matched evidence extracted.</li>"
        gap_items = "".join(f"<li>{html.escape(gap)}</li>" for gap in evidence.gaps[:5]) or "<li>No obvious gaps.</li>"
        parts.append(
            "<div class='card'><h2>Evidence Pack</h2>"
            f"<p>Requirements: {len(evidence.job_requirements)} | Matches: {len(evidence.matched_evidence)}</p>"
            f"<h3>Top Matches</h3><ul>{evidence_items}</ul>"
            f"<h3>Potential Gaps</h3><ul>{gap_items}</ul></div>"
        )
    if style:
        style_items = "".join(f"<li>{html.escape(item)}</li>" for item in style.style_rules[:6]) or "<li>No style rules extracted.</li>"
        adjective_items = "".join(f"<li>{html.escape(item)}</li>" for item in style.core_adjectives[:6]) or "<li>No adjectives extracted.</li>"
        parts.append(
            "<div class='card'><h2>Voice Style Guide</h2>"
            f"<h3>Core Adjectives</h3><ul>{adjective_items}</ul>"
            f"<h3>Style Rules</h3><ul>{style_items}</ul></div>"
        )
    parts.append("</div>")
    return "".join(parts)


@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
    # Build career repo dropdown options
    repos = career_store.list_repos()
    repo_options = '<option value="">— Upload files instead —</option>'
    for r in repos:
        name = html.escape(r.identity.name or r.repo_id)
        repo_options += f'<option value="{html.escape(r.repo_id)}">{name}</option>'

    body = f"""
<div class=\"card\">
  <h1>Resume Refinery</h1>
  <p class=\"muted\">Local-only web app for tailored resume, cover letter, and interview focus points.</p>
  <p><a href=\"/sessions\">Browse sessions</a> &middot; <a href=\"/career\">Career Builder</a></p>
</div>
<div class=\"card\">
  <h2>New Session</h2>
  <form method=\"post\" action=\"/sessions/new\" enctype=\"multipart/form-data\">
    <label>Career Source</label>
    <select name=\"career_repo_id\">
      {repo_options}
    </select>

    <label>Career Profile (.md or .txt) — used when no career repo is selected</label>
    <input type=\"file\" name=\"career_profile\" />

    <label>Voice Profile (.md or .txt) — used when no career repo is selected</label>
    <input type=\"file\" name=\"voice_profile\" />

    <label>Job Description (.md or .txt)</label>
    <input type=\"file\" name=\"job_description\" required />

    <label><input type=\"checkbox\" name=\"skip_review\" value=\"true\" /> Skip voice and AI style reviews</label>
    <label><input type=\"checkbox\" name=\"allow_unverified\" value=\"true\" /> Allow saving when strict truth check fails</label>

    <button type=\"submit\">Generate</button>
  </form>
</div>
"""
    return _page("Resume Refinery", body)


@app.post("/sessions/new", response_class=HTMLResponse)
async def create_session(
    job_description: UploadFile,
    career_profile: Optional[UploadFile] = None,
    voice_profile: Optional[UploadFile] = None,
    career_repo_id: Optional[str] = Form(None),
    skip_review: Optional[str] = Form(None),
    allow_unverified: Optional[str] = Form(None),
) -> HTMLResponse:
    try:
        job_text = (await job_description.read()).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Input files must be UTF-8 text: {exc}")

    # Career + voice: from repo or from uploaded files
    if career_repo_id:
        try:
            repo = career_store.get(career_repo_id)
        except FileNotFoundError:
            raise HTTPException(status_code=400, detail=f"Career repo not found: {career_repo_id}")
        career = repo.to_career_profile()
        voice = parse_voice_profile_content(repo.voice_raw) if repo.voice_raw.strip() else None
        # Fall back to uploaded voice file if repo has no voice
        if voice is None and voice_profile is not None:
            try:
                voice_text = (await voice_profile.read()).decode("utf-8")
                voice = parse_voice_profile_content(voice_text)
            except Exception:
                pass
        if voice is None:
            voice = parse_voice_profile_content("")
    else:
        if career_profile is None or voice_profile is None:
            raise HTTPException(
                status_code=400,
                detail="Upload career and voice profile files, or select a career repository.",
            )
        try:
            career_text = (await career_profile.read()).decode("utf-8")
            voice_text = (await voice_profile.read()).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"Input files must be UTF-8 text: {exc}")
        career = parse_career_profile_content(career_text)
        voice = parse_voice_profile_content(voice_text)

    job = parse_job_description_content(job_text)

    result = orchestrator.create_session_run(
        career,
        voice,
        job,
        skip_review=bool(skip_review),
        allow_unverified=bool(allow_unverified),
    )

    strict_failed = result.strict_truth_failed
    body = """
<div class=\"card\">
  <h1>Generation Complete</h1>
  <p>Session: <a href=\"/sessions/{sid}\">{sid}</a></p>
  {truth_summary}
  <p><a href=\"/sessions\">Back to sessions</a></p>
</div>
{artifact_summary}
""".format(
        sid=html.escape(result.session.session_id),
        truth_summary=_truth_summary(result.reviews.truthfulness),
        artifact_summary=_artifact_summary(result),
    )

    if strict_failed:
        body += """
<div class=\"card\">
  <h2>Strict truth check failed</h2>
  <p>Outputs were saved for review, but strict mode did not pass. Refine with tighter evidence-based feedback or enable allow-unverified in the form.</p>
</div>
"""

    return _page("Generation Complete", body)


@app.get("/sessions", response_class=HTMLResponse)
def list_sessions() -> HTMLResponse:
    sessions = store.list_sessions()
    rows = []
    for s in sessions:
        rows.append(
            f"<tr><td><a href='/sessions/{html.escape(s.session_id)}'>{html.escape(s.session_id)}</a></td>"
            f"<td>{html.escape(s.job_description.title or '—')}</td>"
            f"<td>{html.escape(s.job_description.company or '—')}</td>"
            f"<td>{html.escape(s.created_at[:10])}</td><td>{s.current_version}</td></tr>"
        )

    body = (
        "<div class='card'><h1>Sessions</h1><p><a href='/'>New session</a></p>"
        "<table><thead><tr><th>ID</th><th>Title</th><th>Company</th><th>Created</th><th>Versions</th></tr></thead>"
        f"<tbody>{''.join(rows) if rows else '<tr><td colspan=5>No sessions found.</td></tr>'}</tbody></table></div>"
    )
    return _page("Sessions", body)


@app.get("/sessions/{session_id}", response_class=HTMLResponse)
def show_session(session_id: str) -> HTMLResponse:
    result = orchestrator.review_session_run(session_id)
    session = result.session
    docs = result.documents
    reviews = result.reviews

    def esc(text: str | None) -> str:
        return html.escape(text or "")

    body = f"""
<div class=\"card\">
  <h1>{html.escape(session.session_id)}</h1>
  <p class=\"muted\">{html.escape(session.job_description.title or '—')} @ {html.escape(session.job_description.company or '—')}</p>
  {_truth_summary(reviews.truthfulness)}
  <p><a href=\"/sessions\">Back to sessions</a></p>
</div>
{_artifact_summary(result)}
<div class=\"card\">
  <h2>Refine</h2>
  <form method=\"post\" action=\"/sessions/{html.escape(session.session_id)}/refine\">
    <label>Document</label>
    <select name=\"doc\">
      <option value=\"\">All documents</option>
      <option value=\"cover_letter\">Cover Letter</option>
      <option value=\"resume\">Resume</option>
      <option value=\"interview_guide\">Interview Guide</option>
    </select>
    <label>Feedback (free-form)</label>
    <textarea name=\"feedback\" rows=\"5\" required></textarea>
    <label><input type=\"checkbox\" name=\"skip_review\" value=\"true\" /> Skip voice and AI style reviews</label>
    <label><input type=\"checkbox\" name=\"allow_unverified\" value=\"true\" /> Allow saving when strict truth check fails</label>
    <button type=\"submit\">Refine</button>
  </form>
</div>
<div class=\"grid\">
  <div class=\"card\"><h2>Cover Letter</h2><pre>{esc(docs.cover_letter)}</pre></div>
  <div class=\"card\"><h2>Resume</h2><pre>{esc(docs.resume)}</pre></div>
</div>
<div class=\"card\"><h2>Interview Guide</h2><pre>{esc(docs.interview_guide)}</pre></div>
"""
    return _page(session.session_id, body)


@app.post("/sessions/{session_id}/refine")
def refine_session(
    session_id: str,
    feedback: str = Form(...),
    doc: str = Form(""),
    skip_review: Optional[str] = Form(None),
    allow_unverified: Optional[str] = Form(None),
):
    if doc and doc not in ("cover_letter", "resume", "interview_guide"):
        raise HTTPException(status_code=400, detail=f"Unknown document: {doc}")

    result = orchestrator.refine_session_run(
        session_id,
        feedback,
        doc=doc or None,
        skip_review=bool(skip_review),
        allow_unverified=bool(allow_unverified),
    )

    if result.strict_truth_failed:
        return _page(
            "Strict truth check failed",
            f"""
<div class=\"card\">
  <h1>Strict truth check failed</h1>
  {_truth_summary(result.reviews.truthfulness)}
  <p>Version v{result.session.current_version} was saved for review. Unsupported claims remain.</p>
  <p><a href=\"/sessions/{html.escape(result.session.session_id)}\">Back to session</a></p>
</div>
{_artifact_summary(result)}
""",
        )

    return RedirectResponse(url=f"/sessions/{result.session.session_id}", status_code=303)


def main() -> None:
    uvicorn.run(
        "resume_refinery.webapp:app",
        host="127.0.0.1",
        port=8765,
        reload=False,
    )
