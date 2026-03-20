"""Local web app for Resume Refinery."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse

from .agent import ResumeRefineryAgent
from .exporters import export_document_set
from .models import DocumentSet, ReviewBundle
from .parsers import (
    parse_career_profile_content,
    parse_job_description_content,
    parse_voice_profile_content,
)
from .reviewers import DocumentReviewer
from .session import SessionStore

app = FastAPI(title="Resume Refinery", version="0.1.0")
store = SessionStore()


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


def _claim_feedback_for_doc(key: str, truth) -> str:
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


def _enforce_truth(agent, docs, career, voice, job, feedback: str | None = None, max_passes: int = 2):
    reviewer = DocumentReviewer()
    truth = None
    for _ in range(max_passes):
        truth = reviewer.review_truthfulness(docs, career)
        if truth.all_supported:
            return truth

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


@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
    body = """
<div class=\"card\">
  <h1>Resume Refinery</h1>
  <p class=\"muted\">Local-only web app for tailored resume, cover letter, and interview focus points.</p>
  <p><a href=\"/sessions\">Browse sessions</a></p>
</div>
<div class=\"card\">
  <h2>New Session</h2>
  <form method=\"post\" action=\"/sessions/new\" enctype=\"multipart/form-data\">
    <label>Career Profile (.md or .txt)</label>
    <input type=\"file\" name=\"career_profile\" required />

    <label>Voice Profile (.md or .txt)</label>
    <input type=\"file\" name=\"voice_profile\" required />

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
    career_profile: UploadFile,
    voice_profile: UploadFile,
    job_description: UploadFile,
    skip_review: Optional[str] = Form(None),
    allow_unverified: Optional[str] = Form(None),
) -> HTMLResponse:
    try:
        career_text = (await career_profile.read()).decode("utf-8")
        voice_text = (await voice_profile.read()).decode("utf-8")
        job_text = (await job_description.read()).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Input files must be UTF-8 text: {exc}")

    career = parse_career_profile_content(career_text)
    voice = parse_voice_profile_content(voice_text)
    job = parse_job_description_content(job_text)

    session = store.create(job, career, voice)
    agent = ResumeRefineryAgent()

    docs = agent.generate_all(career, voice, job)
    truth = _enforce_truth(agent, docs, career, voice, job)

    session = store.save_documents(session, docs)
    version_dir = store.session_dir(session.session_id) / f"v{session.current_version}"
    export_document_set(docs, version_dir)

    reviewer = DocumentReviewer()
    if skip_review:
        store.save_reviews(session, ReviewBundle(truthfulness=truth))
    else:
        reviews = ReviewBundle(
            truthfulness=truth,
            voice=reviewer.review_voice(docs, voice),
            ai_detection=reviewer.review_ai_detection(docs),
        )
        store.save_reviews(session, reviews)

    strict_failed = _truth_failed(truth) and not allow_unverified
    body = """
<div class=\"card\">
  <h1>Generation Complete</h1>
  <p>Session: <a href=\"/sessions/{sid}\">{sid}</a></p>
  {truth_summary}
  <p><a href=\"/sessions\">Back to sessions</a></p>
</div>
""".format(sid=html.escape(session.session_id), truth_summary=_truth_summary(truth))

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
    session = store.get(session_id)
    docs = store.load_documents(session)
    reviews = store.load_reviews(session)

    def esc(text: str | None) -> str:
        return html.escape(text or "")

    body = f"""
<div class=\"card\">
  <h1>{html.escape(session.session_id)}</h1>
  <p class=\"muted\">{html.escape(session.job_description.title or '—')} @ {html.escape(session.job_description.company or '—')}</p>
  {_truth_summary(reviews.truthfulness)}
  <p><a href=\"/sessions\">Back to sessions</a></p>
</div>
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
    session = store.get(session_id)
    career, voice = store.load_inputs(session)
    job = session.job_description
    agent = ResumeRefineryAgent()
    docs = store.load_documents(session)

    keys_to_regen = ["cover_letter", "resume", "interview_guide"] if not doc else [doc]
    for key in keys_to_regen:
        if key not in ("cover_letter", "resume", "interview_guide"):
            raise HTTPException(status_code=400, detail=f"Unknown document: {key}")
        previous = docs.get(key)
        regenerated = agent.generate_document(
            key, career, voice, job, feedback=feedback, previous_version=previous
        )
        docs.set(key, regenerated)

    truth = _enforce_truth(agent, docs, career, voice, job, feedback=feedback)

    session = store.save_documents(
        session,
        docs,
        feedback=feedback,
        docs_regenerated=keys_to_regen,
    )
    version_dir = store.session_dir(session.session_id) / f"v{session.current_version}"
    export_document_set(docs, version_dir)

    reviewer = DocumentReviewer()
    if skip_review:
        store.save_reviews(session, ReviewBundle(truthfulness=truth))
    else:
        reviews = ReviewBundle(
            truthfulness=truth,
            voice=reviewer.review_voice(docs, voice),
            ai_detection=reviewer.review_ai_detection(docs),
        )
        store.save_reviews(session, reviews)

    if _truth_failed(truth) and not allow_unverified:
        return _page(
            "Strict truth check failed",
            f"""
<div class=\"card\">
  <h1>Strict truth check failed</h1>
  {_truth_summary(truth)}
  <p>Version v{session.current_version} was saved for review. Unsupported claims remain.</p>
  <p><a href=\"/sessions/{html.escape(session.session_id)}\">Back to session</a></p>
</div>
""",
        )

    return RedirectResponse(url=f"/sessions/{session.session_id}", status_code=303)


def main() -> None:
    uvicorn.run(
        "resume_refinery.webapp:app",
        host="127.0.0.1",
        port=8765,
        reload=False,
    )
