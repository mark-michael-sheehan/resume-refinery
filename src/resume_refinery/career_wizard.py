"""Career repository guided elicitation wizard — web routes.

All routes are mounted under ``/career`` by the main webapp.  HTMX is used for
partial-page updates so the wizard feels conversational without a JS build step.
"""

from __future__ import annotations

import html
import logging
import re
from typing import Iterator, Optional

from fastapi import APIRouter, Form, HTTPException, Request, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse

from .career_repo import CareerRepoStore
from .elicitation import ElicitationAgent
from .ingest_agent import IngestAgent, build_repo_from_parsed, consolidate_roles, consolidate_skills_meta, parse_ingest_response
from .models import (
    CareerIdentity,
    CareerMeta,
    CareerRepository,
    RoleEntry,
    SkillEntry,
    StoryEntry,
    WizardPhase,
)

router = APIRouter(prefix="/career", tags=["career"])
career_store = CareerRepoStore()
elicitation_agent = ElicitationAgent()
ingest_agent = IngestAgent()

_HTMX_CDN = "https://unpkg.com/htmx.org@2.0.4"

# ---------------------------------------------------------------------------
# Shared HTML helpers
# ---------------------------------------------------------------------------

_PHASE_LABELS: dict[WizardPhase, str] = {
    "identity": "Identity & Basics",
    "roles": "Role Timeline",
    "role_deepdive": "Role Deep Dive",
    "skills": "Skills Inventory",
    "stories": "Behavioral Stories",
    "meta": "Career Strategy",
    "voice": "Voice Profile",
    "complete": "Review & Finish",
}


def _esc(text: str | None) -> str:
    if not text:
        return ""
    return html.escape(text)


def _wizard_page(title: str, body: str, repo: CareerRepository | None = None) -> HTMLResponse:
    """Full page wrapper with HTMX loaded and progress bar."""
    progress = ""
    if repo:
        progress = _progress_bar(repo)
    safe_title = html.escape(title)
    return HTMLResponse(f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{safe_title} — Career Builder</title>
  <script src="{_HTMX_CDN}"></script>
  <style>
    :root {{
      --bg: #f7f5ef; --card: #fffdf8; --ink: #1f2421;
      --muted: #55605a; --accent: #0f7b6c; --line: #d8d4c8;
      --accent-light: #e8f5f1;
    }}
    body {{ margin:0; font-family:"Segoe UI",Tahoma,Geneva,Verdana,sans-serif;
            background:radial-gradient(circle at 0% 0%,#fffaf0 0%,var(--bg) 40%,#efeae0 100%);
            color:var(--ink); }}
    .wrap {{ max-width:820px; margin:2rem auto; padding:0 1rem; }}
    .card {{ background:var(--card); border:1px solid var(--line);
             border-radius:14px; padding:1.25rem 1.5rem;
             box-shadow:0 8px 30px rgba(20,35,30,.06); margin-bottom:1rem; }}
    h1,h2,h3 {{ margin:0.2rem 0 0.6rem; }}
    a {{ color:var(--accent); text-decoration:none; }}
    a:hover {{ text-decoration:underline; }}
    label {{ display:block; margin-top:0.7rem; font-weight:600; }}
    .hint {{ color:var(--muted); font-size:0.9rem; margin-top:0.15rem; }}
    input[type=text],input[type=email],input[type=tel],textarea,select {{
      width:100%; margin-top:0.35rem; padding:0.55rem; border-radius:8px;
      border:1px solid var(--line); background:#fff; box-sizing:border-box;
      font-family:inherit; font-size:0.95rem;
    }}
    textarea {{ resize:vertical; }}
    button,.btn {{ margin-top:0.9rem; background:var(--accent); color:#fff;
            border:none; border-radius:10px; padding:0.65rem 1.2rem;
            cursor:pointer; font-weight:700; font-size:0.95rem;
            display:inline-block; text-decoration:none; }}
    button:hover,.btn:hover {{ opacity:0.9; }}
    .btn-secondary {{ background:var(--line); color:var(--ink); }}
    .btn-sm {{ padding:0.4rem 0.8rem; font-size:0.85rem; margin-top:0.4rem; }}
    .btn-danger {{ background:#b00020; }}
    /* Progress bar */
    .progress {{ display:flex; gap:0; margin-bottom:1.2rem; border-radius:10px;
                 overflow:hidden; border:1px solid var(--line); }}
    .progress .step {{ flex:1; text-align:center; padding:0.45rem 0.2rem;
                       font-size:0.78rem; font-weight:600; background:var(--card);
                       border-right:1px solid var(--line); color:var(--muted);
                       text-overflow:ellipsis; overflow:hidden; white-space:nowrap; }}
    .progress .step:last-child {{ border-right:none; }}
    .progress .step.done {{ background:var(--accent); color:#fff; }}
    .progress .step.active {{ background:var(--accent-light); color:var(--accent); }}
    .muted {{ color:var(--muted); }}
    .grid {{ display:grid; grid-template-columns:1fr 1fr; gap:0.8rem; }}
    @media (max-width:700px) {{ .grid {{ grid-template-columns:1fr; }} }}
    table {{ width:100%; border-collapse:collapse; margin-top:0.5rem; }}
    th,td {{ border-bottom:1px solid var(--line); text-align:left; padding:0.45rem 0.5rem; }}
    .probe {{ background:var(--accent-light); border-left:3px solid var(--accent);
              padding:0.8rem 1rem; border-radius:0 8px 8px 0; margin:0.8rem 0; }}
    .warning {{ background:#fff3cd; border-left:3px solid #ffc107;
               padding:0.6rem 1rem; border-radius:0 8px 8px 0; margin:0.5rem 0;
               color:#664d03; font-size:0.9rem; }}
    .htmx-indicator {{ display:none; }}
    .htmx-request .htmx-indicator,
    .htmx-request.htmx-indicator {{ display:inline-block; }}
    .spinner {{ display:inline-block; width:1em; height:1em;
               border:2px solid var(--line); border-top-color:var(--accent);
               border-radius:50%; animation:spin .6s linear infinite; }}
    @keyframes spin {{ to {{ transform:rotate(360deg); }} }}
    .actions {{ display:flex; gap:0.5rem; align-items:center; margin-top:1rem; }}
    .role-card {{ border:1px solid var(--line); border-radius:10px; padding:0.8rem 1rem;
                  margin-bottom:0.6rem; background:#fff; }}
  </style>
</head>
<body>
  <div class="wrap">
    <p style="margin-bottom:0.3rem"><a href="/career">&larr; Career Builder</a></p>
    {progress}
    {body}
  </div>
</body>
</html>
""")


def _progress_bar(repo: CareerRepository) -> str:
    """Render the horizontal phase progress bar with clickable completed/active phases."""
    phases: list[WizardPhase] = [
        "identity", "roles", "role_deepdive", "skills", "stories", "meta", "voice", "complete",
    ]
    # Map phase name to its direct URL path segment
    _phase_url: dict[WizardPhase, str] = {
        "identity": "identity",
        "roles": "roles",
        "role_deepdive": "role_deepdive",
        "skills": "skills",
        "stories": "stories",
        "meta": "meta",
        "voice": "voice",
        "complete": "review",
    }
    phase_order = {p: i for i, p in enumerate(phases)}
    current_idx = phase_order.get(repo.current_phase, 0)

    steps: list[str] = []
    for i, phase in enumerate(phases):
        label = _PHASE_LABELS[phase]
        if i < current_idx:
            cls = "done"
            url = f"/career/{_esc(repo.repo_id)}/{_phase_url[phase]}"
            steps.append(f'<a href="{url}" class="step {cls}" style="text-decoration:none;color:inherit">{_esc(label)}</a>')
        elif i == current_idx:
            cls = "active"
            url = f"/career/{_esc(repo.repo_id)}/{_phase_url[phase]}"
            steps.append(f'<a href="{url}" class="step {cls}" style="text-decoration:none;color:inherit">{_esc(label)}</a>')
        else:
            steps.append(f'<div class="step">{_esc(label)}</div>')
    return f'<div class="progress">{"".join(steps)}</div>'


# ---------------------------------------------------------------------------
# Career list / create
# ---------------------------------------------------------------------------


@router.get("", response_class=HTMLResponse)
def career_index() -> HTMLResponse:
    repos = career_store.list_repos()
    rows: list[str] = []
    for r in repos:
        name = _esc(r.identity.name) or _esc(r.repo_id)
        phase = _esc(_PHASE_LABELS.get(r.current_phase, r.current_phase))
        roles = len(r.roles)
        rows.append(
            f"<tr><td><a href='/career/{_esc(r.repo_id)}'>{name}</a></td>"
            f"<td>{phase}</td><td>{roles}</td>"
            f"<td>{_esc(r.updated_at[:10])}</td></tr>"
        )

    table = (
        "<table><thead><tr><th>Name</th><th>Phase</th><th>Roles</th><th>Updated</th></tr></thead>"
        f"<tbody>{''.join(rows) if rows else '<tr><td colspan=4>No career profiles yet.</td></tr>'}</tbody></table>"
    )

    body = f"""
<div class="card">
  <h1>Career Builder</h1>
  <p class="muted">Build a structured career repository through guided questions.
     Use it to generate tailored resumes, cover letters, and interview prep.</p>
  {table}
</div>
<div class="card">
  <h2>Start New</h2>
  <form method="post" action="/career/new">
    <label>Your full name</label>
    <input type="text" name="name" required placeholder="e.g. Jordan Lee" />
    <button type="submit">Begin</button>
  </form>
</div>
<div class="card">
  <h2>Import from Documents</h2>
  <p class="muted">Upload resumes, CVs, or LinkedIn exports (PDF, DOCX, TXT, MD).
     An AI agent will extract your career data as a starting point.</p>
  <form method="post" action="/career/ingest" enctype="multipart/form-data">
    <label>Your full name</label>
    <input type="text" name="name" required placeholder="e.g. Jordan Lee" />
    <label>Documents</label>
    <p class="hint">Select one or more files. All content will be combined for extraction.</p>
    <input type="file" name="files" multiple required
           accept=".pdf,.docx,.doc,.txt,.md" />
    <button type="submit">Import &amp; Build Profile</button>
  </form>
</div>
<p class="muted" style="margin-top:1rem"><a href="/">Back to Resume Refinery</a></p>
"""
    return _wizard_page("Career Builder", body)


@router.post("/new")
def career_create(name: str = Form(...)) -> RedirectResponse:
    repo = career_store.create(name.strip())
    return RedirectResponse(url=f"/career/{repo.repo_id}", status_code=303)


_INGEST_PROGRESS_HEAD = (
    '<!doctype html><html lang="en"><head><meta charset="utf-8"/>'
    '<meta name="viewport" content="width=device-width,initial-scale=1"/>'
    "<title>Importing\u2026 \u2014 Career Builder</title>"
    "<style>"
    ":root{--bg:#f7f5ef;--card:#fffdf8;--ink:#1f2421;"
    "--muted:#55605a;--accent:#0f7b6c;--line:#d8d4c8;--accent-light:#e8f5f1}"
    "body{margin:0;font-family:'Segoe UI',Tahoma,Geneva,Verdana,sans-serif;"
    "background:radial-gradient(circle at 0% 0%,#fffaf0 0%,var(--bg) 40%,#efeae0 100%);"
    "color:var(--ink)}"
    ".wrap{max-width:820px;margin:2rem auto;padding:0 1rem}"
    ".card{background:var(--card);border:1px solid var(--line);border-radius:14px;"
    "padding:1.25rem 1.5rem;box-shadow:0 8px 30px rgba(20,35,30,.06);margin-bottom:1rem}"
    "h2{margin:.2rem 0 .6rem}"
    "a{color:var(--accent);text-decoration:none}"
    ".muted{color:var(--muted)}"
    "#progress-log p{margin:.4rem 0;padding:.3rem 0}"
    "#progress-log p:last-child::after{"
    "content:'';display:inline-block;width:.85em;height:.85em;"
    "border:2px solid var(--line);border-top-color:var(--accent);"
    "border-radius:50%;animation:spin .6s linear infinite;"
    "margin-left:.6em;vertical-align:middle}"
    "#progress-log.done p:last-child::after{display:none}"
    "@keyframes spin{to{transform:rotate(360deg)}}"
    ".step-ok{color:var(--accent)}.step-ok::before{content:'\u2713 ';font-weight:bold}"
    ".step-fail{color:#b00020}.step-fail::before{content:'\u2717 ';font-weight:bold}"
    "</style></head><body><div class='wrap'>"
    "<p style='margin-bottom:.3rem'><a href='/career'>&larr; Career Builder</a></p>"
    "<div class='card'><h2>Importing Career Data</h2>"
    "<div id='progress-log'>"
)

# Padding to ensure browsers start rendering before the first LLM call
_BROWSER_FLUSH_PAD = "<!-- " + " " * 1024 + " -->\n"


@router.post("/ingest")
async def career_ingest(
    name: str = Form(...),
    files: list[UploadFile] = File(...),
) -> StreamingResponse:
    """Upload documents and create a pre-filled career repository via LLM extraction.

    Returns a streamed HTML page that shows real-time progress as each LLM step
    completes, then auto-redirects to the finished career profile.
    """
    from .parsers import _read_file_content
    from pathlib import Path
    import tempfile

    if not files or all(not f.filename for f in files):
        raise HTTPException(status_code=400, detail="No files uploaded")

    # Read text from all uploaded files individually (async part, before streaming)
    document_texts: list[tuple[str, str]] = []  # (filename, text)
    allowed_extensions = {".pdf", ".docx", ".doc", ".txt", ".md"}

    for upload in files:
        if not upload.filename:
            continue
        ext = Path(upload.filename).suffix.lower()
        if ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {ext}. Allowed: {', '.join(sorted(allowed_extensions))}",
            )

        content = await upload.read()
        # Write to a temp file so _read_file_content can handle format detection
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)
        try:
            text = _read_file_content(tmp_path)
            if text.strip():
                document_texts.append((upload.filename, text))
        finally:
            tmp_path.unlink(missing_ok=True)

    if not document_texts:
        raise HTTPException(status_code=400, detail="No readable text found in uploaded files")

    # Create the repo first
    repo = career_store.create(name.strip())

    def _progress() -> Iterator[str]:
        """Sync generator: yields HTML chunks as each pipeline step completes."""
        nonlocal repo
        yield _INGEST_PROGRESS_HEAD + "\n" + _BROWSER_FLUSH_PAD

        total = len(document_texts)

        # Per-document extraction — one LLM call per file
        extraction_succeeded = False
        for i, (filename, text) in enumerate(document_texts, 1):
            yield f"<p>Extracting from <strong>{_esc(filename)}</strong> ({i}/{total})\u2026</p>\n"
            try:
                ingest_agent.ingest_to_repo(text, repo)
                extraction_succeeded = True
            except Exception:
                log.warning("Ingest failed for %s, skipping", filename, exc_info=True)
                yield f"<p class='step-fail'>Failed to extract {_esc(filename)}</p>\n"

        if extraction_succeeded:
            # Consolidate duplicate roles across documents via LLM (Pass 1)
            # so the user reviews a clean timeline, not one entry per source file
            yield "<p>Consolidating roles\u2026</p>\n"
            try:
                repo = consolidate_roles(repo, client=ingest_agent.client)
                yield "<p class='step-ok'>Roles consolidated</p>\n"
            except Exception:
                log.warning("Role consolidation failed, continuing with unmerged roles", exc_info=True)
                yield "<p class='step-fail'>Role consolidation failed, continuing with extracted data</p>\n"
            # Override name from form (user-provided takes priority)
            repo.identity.name = name.strip()
            # Land on roles phase so the user can verify extracted data
            # before skills/meta consolidation and story composition run
            repo.current_phase = "roles"
            repo.needs_consolidation = True
        else:
            log.warning("All document extractions failed, continuing with empty repo")
            repo.identity.name = name.strip()

        career_store.save(repo)

        # Mark progress complete and auto-redirect
        redirect_url = f"/career/{repo.repo_id}"
        yield (
            "</div>"  # close #progress-log
            f"<p class='step-ok' style='margin-top:.8rem;font-weight:600'>Done! Redirecting\u2026</p>"
            f"<script>setTimeout(function(){{window.location.href='{redirect_url}';}},1200);</script>"
            "</div></div></body></html>"
        )

    return StreamingResponse(_progress(), media_type="text/html")


# ---------------------------------------------------------------------------
# Phase router — dispatches to the current phase
# ---------------------------------------------------------------------------


@router.get("/{repo_id}", response_class=HTMLResponse)
def career_show(repo_id: str) -> HTMLResponse:
    repo = _load_repo(repo_id)
    phase = repo.current_phase
    if phase == "identity":
        return _render_identity(repo)
    elif phase == "roles":
        return _render_roles(repo)
    elif phase == "role_deepdive":
        return _render_role_deepdive(repo)
    elif phase == "skills":
        return _render_skills(repo)
    elif phase == "stories":
        return _render_stories(repo)
    elif phase == "meta":
        return _render_meta(repo)
    elif phase == "voice":
        return _render_voice(repo)
    else:  # "complete"
        return _render_review(repo)


# ---------------------------------------------------------------------------
# Phase 1: Identity
# ---------------------------------------------------------------------------


@router.get("/{repo_id}/identity", response_class=HTMLResponse)
def career_identity(repo_id: str) -> HTMLResponse:
    return _render_identity(_load_repo(repo_id))


def _render_identity(repo: CareerRepository) -> HTMLResponse:
    ident = repo.identity
    # Detect pre-filled data from ingestion
    _has_extracted = any(
        r.confidence_notes or r.extraction_confidence != "medium"
        for r in repo.roles
    )
    ingest_hint = ""
    if _has_extracted:
        ingest_hint = (
            '<div class="warning" style="border-left-color:var(--accent)">'
            'These fields were pre-filled from your uploaded documents. '
            'Please verify and correct anything the AI got wrong.'
            '</div>'
        )
    body = f"""
<div class="card">
  <h2>Phase 1: Identity &amp; Basics</h2>
  <p class="muted">Let's start with contact info and a quick headline about you.</p>
  {ingest_hint}
  <form method="post" action="/career/{_esc(repo.repo_id)}/identity">
    <div class="grid">
      <div>
        <label>Full Name</label>
        <input type="text" name="name" value="{_esc(ident.name)}" required />
      </div>
      <div>
        <label>Email</label>
        <input type="email" name="email" value="{_esc(ident.email)}" />
      </div>
      <div>
        <label>Phone</label>
        <input type="tel" name="phone" value="{_esc(ident.phone)}" />
      </div>
      <div>
        <label>Location</label>
        <input type="text" name="location" value="{_esc(ident.location)}" placeholder="e.g. San Francisco, CA" />
      </div>
      <div>
        <label>LinkedIn URL</label>
        <input type="text" name="linkedin" value="{_esc(ident.linkedin)}" />
      </div>
      <div>
        <label>GitHub URL</label>
        <input type="text" name="github" value="{_esc(ident.github)}" />
      </div>
    </div>
    <label>Professional Headline</label>
    <p class="hint">One sentence that captures who you are professionally.</p>
    <textarea name="headline" rows="2">{_esc(ident.headline)}</textarea>

    <label>Target Role Types</label>
    <p class="hint">Comma-separated. e.g. "Staff Engineer, Engineering Manager"</p>
    <input type="text" name="target_roles" value="{_esc(', '.join(ident.target_roles))}" />

    <label>Education</label>
    <p class="hint">Degrees, schools, graduation years, honors. Free-form.</p>
    <textarea name="education" rows="4">{_esc(repo.education)}</textarea>

    <label>Certifications</label>
    <p class="hint">Professional certifications, courses, training.</p>
    <textarea name="certifications" rows="3">{_esc(repo.certifications)}</textarea>

    <div class="actions">
      <button type="submit">Save &amp; Continue</button>
    </div>
  </form>
</div>
"""
    return _wizard_page("Identity", body, repo)


@router.post("/{repo_id}/identity")
def save_identity(
    repo_id: str,
    name: str = Form(""),
    email: str = Form(""),
    phone: str = Form(""),
    location: str = Form(""),
    linkedin: str = Form(""),
    github: str = Form(""),
    headline: str = Form(""),
    target_roles: str = Form(""),
    education: str = Form(""),
    certifications: str = Form(""),
) -> RedirectResponse:
    repo = _load_repo(repo_id)
    repo.identity = CareerIdentity(
        name=name.strip(),
        email=email.strip(),
        phone=phone.strip(),
        location=location.strip(),
        linkedin=linkedin.strip(),
        github=github.strip(),
        headline=headline.strip(),
        target_roles=[r.strip() for r in target_roles.split(",") if r.strip()],
    )
    repo.education = education.strip()
    repo.certifications = certifications.strip()
    repo.current_phase = "roles"
    career_store.save(repo)
    return RedirectResponse(url=f"/career/{repo.repo_id}", status_code=303)


# ---------------------------------------------------------------------------
# Phase 2: Role Timeline
# ---------------------------------------------------------------------------


@router.get("/{repo_id}/roles", response_class=HTMLResponse)
def career_roles(repo_id: str) -> HTMLResponse:
    return _render_roles(_load_repo(repo_id))


def _render_roles(repo: CareerRepository) -> HTMLResponse:
    role_cards: list[str] = []
    # Detect pre-filled data from ingestion
    _has_extracted = any(
        r.confidence_notes or r.extraction_confidence != "medium"
        for r in repo.roles
    )
    for i, role in enumerate(repo.roles):
        # Show per-role confidence badge when data came from ingestion
        conf_badge = ""
        if role.extraction_confidence != "medium" or role.confidence_notes:
            conf_color = {"high": "#1d8f52", "medium": "#b8860b", "low": "#b00020"}[role.extraction_confidence]
            conf_badge = f' <span style="font-size:0.8rem;color:{conf_color};font-weight:600">[{role.extraction_confidence.capitalize()}]</span>'
        role_cards.append(f"""
<div class="role-card">
  <strong>{_esc(role.title)}</strong> @ {_esc(role.company)}{conf_badge}
  <span class="muted">({_esc(role.start_date)} – {_esc(role.end_date)})</span>
  <div style="margin-top:0.4rem">
    <a href="/career/{_esc(repo.repo_id)}/roles/{i}/edit" class="btn btn-sm btn-secondary">Edit</a>
    <form method="post" action="/career/{_esc(repo.repo_id)}/roles/{i}/delete"
          style="display:inline"
          onsubmit="return confirm('Remove this role?')">
      <button type="submit" class="btn btn-sm btn-danger">Remove</button>
    </form>
  </div>
</div>
""")

    existing = "".join(role_cards) if role_cards else '<p class="muted">No roles added yet.</p>'

    ingest_hint = ""
    if _has_extracted and role_cards:
        ingest_hint = (
            '<div class="warning" style="border-left-color:var(--accent)">'
            'These roles were extracted from your uploaded documents. '
            'Verify titles, dates, and company names — then add any roles the AI missed.'
            '</div>'
        )

    # Choose appropriate continue action based on whether we're post-ingestion
    if repo.needs_consolidation:
        continue_btn = f"""
    <form method="post" action="/career/{_esc(repo.repo_id)}/finalize" style="display:inline">
      <button type="submit" {"disabled" if not repo.roles else ""}>Finalize &amp; Build Stories</button>
    </form>"""
    else:
        continue_btn = f"""
    <form method="post" action="/career/{_esc(repo.repo_id)}/advance/role_deepdive" style="display:inline">
      <button type="submit" {"disabled" if not repo.roles else ""}>Continue to Deep Dive</button>
    </form>"""

    body = f"""
<div class="card">
  <h2>Phase 2: Role Timeline</h2>
  <p class="muted">Add each job you've held. Start with the most recent.
     We'll dig into accomplishments in the next phase.</p>
  {ingest_hint}
  {existing}
</div>
<div class="card">
  <h3>Add a Role</h3>
  <form method="post" action="/career/{_esc(repo.repo_id)}/roles">
    <div class="grid">
      <div>
        <label>Company Name</label>
        <input type="text" name="company" required />
      </div>
      <div>
        <label>Your Title</label>
        <input type="text" name="title" required />
      </div>
      <div>
        <label>Start Date</label>
        <input type="text" name="start_date" required placeholder="e.g. Mar 2021" />
      </div>
      <div>
        <label>End Date</label>
        <input type="text" name="end_date" value="Present" placeholder="e.g. Feb 2023 or Present" />
      </div>
    </div>
    <button type="submit">Add Role</button>
  </form>
</div>
<div class="card">
  <div class="actions">
    <a href="/career/{_esc(repo.repo_id)}/identity" class="btn btn-secondary">Back</a>
    {continue_btn}
  </div>
</div>
"""
    return _wizard_page("Roles", body, repo)


@router.post("/{repo_id}/roles")
def add_role(
    repo_id: str,
    company: str = Form(...),
    title: str = Form(...),
    start_date: str = Form(...),
    end_date: str = Form("Present"),
) -> RedirectResponse:
    repo = _load_repo(repo_id)
    repo.roles.append(RoleEntry(
        company=company.strip(),
        title=title.strip(),
        start_date=start_date.strip(),
        end_date=end_date.strip(),
    ))
    career_store.save(repo)
    return RedirectResponse(url=f"/career/{repo.repo_id}", status_code=303)


@router.get("/{repo_id}/roles/{role_idx}/edit", response_class=HTMLResponse)
def edit_role_form(repo_id: str, role_idx: int) -> HTMLResponse:
    repo = _load_repo(repo_id)
    if role_idx < 0 or role_idx >= len(repo.roles):
        raise HTTPException(status_code=404, detail="Role not found")
    role = repo.roles[role_idx]
    body = f"""
<div class="card">
  <h2>Edit Role</h2>
  <form method="post" action="/career/{_esc(repo.repo_id)}/roles/{role_idx}/edit">
    <div class="grid">
      <div>
        <label>Company</label>
        <input type="text" name="company" value="{_esc(role.company)}" required />
      </div>
      <div>
        <label>Title</label>
        <input type="text" name="title" value="{_esc(role.title)}" required />
      </div>
      <div>
        <label>Start Date</label>
        <input type="text" name="start_date" value="{_esc(role.start_date)}" required />
      </div>
      <div>
        <label>End Date</label>
        <input type="text" name="end_date" value="{_esc(role.end_date)}" />
      </div>
    </div>
    <div class="actions">
      <button type="submit">Save</button>
      <a href="/career/{_esc(repo.repo_id)}" class="btn btn-secondary">Cancel</a>
    </div>
  </form>
</div>
"""
    return _wizard_page("Edit Role", body, repo)


@router.post("/{repo_id}/roles/{role_idx}/edit")
def save_role_edit(
    repo_id: str,
    role_idx: int,
    company: str = Form(...),
    title: str = Form(...),
    start_date: str = Form(...),
    end_date: str = Form("Present"),
) -> RedirectResponse:
    repo = _load_repo(repo_id)
    if role_idx < 0 or role_idx >= len(repo.roles):
        raise HTTPException(status_code=404, detail="Role not found")
    role = repo.roles[role_idx]
    role.company = company.strip()
    role.title = title.strip()
    role.start_date = start_date.strip()
    role.end_date = end_date.strip()
    career_store.save(repo)
    return RedirectResponse(url=f"/career/{repo.repo_id}", status_code=303)


@router.post("/{repo_id}/roles/{role_idx}/delete")
def delete_role(repo_id: str, role_idx: int) -> RedirectResponse:
    repo = _load_repo(repo_id)
    if 0 <= role_idx < len(repo.roles):
        repo.roles.pop(role_idx)
        career_store.save(repo)
    return RedirectResponse(url=f"/career/{repo.repo_id}", status_code=303)


# ---------------------------------------------------------------------------
# Post-ingestion finalize — consolidation + STAR stories after user review
# ---------------------------------------------------------------------------

_FINALIZE_PROGRESS_HEAD = (
    '<!doctype html><html lang="en"><head><meta charset="utf-8"/>'
    '<meta name="viewport" content="width=device-width,initial-scale=1"/>'
    "<title>Finalizing\u2026 \u2014 Career Builder</title>"
    "<style>"
    ":root{--bg:#f7f5ef;--card:#fffdf8;--ink:#1f2421;"
    "--muted:#55605a;--accent:#0f7b6c;--line:#d8d4c8;--accent-light:#e8f5f1}"
    "body{margin:0;font-family:'Segoe UI',Tahoma,Geneva,Verdana,sans-serif;"
    "background:radial-gradient(circle at 0% 0%,#fffaf0 0%,var(--bg) 40%,#efeae0 100%);"
    "color:var(--ink)}"
    ".wrap{max-width:820px;margin:2rem auto;padding:0 1rem}"
    ".card{background:var(--card);border:1px solid var(--line);border-radius:14px;"
    "padding:1.25rem 1.5rem;box-shadow:0 8px 30px rgba(20,35,30,.06);margin-bottom:1rem}"
    "h2{margin:.2rem 0 .6rem}"
    "a{color:var(--accent);text-decoration:none}"
    ".muted{color:var(--muted)}"
    "#progress-log p{margin:.4rem 0;padding:.3rem 0}"
    "#progress-log p:last-child::after{"
    "content:'';display:inline-block;width:.85em;height:.85em;"
    "border:2px solid var(--line);border-top-color:var(--accent);"
    "border-radius:50%;animation:spin .6s linear infinite;"
    "margin-left:.6em;vertical-align:middle}"
    "#progress-log.done p:last-child::after{display:none}"
    "@keyframes spin{to{transform:rotate(360deg)}}"
    ".step-ok{color:var(--accent)}.step-ok::before{content:'\u2713 ';font-weight:bold}"
    ".step-fail{color:#b00020}.step-fail::before{content:'\u2717 ';font-weight:bold}"
    "</style></head><body><div class='wrap'>"
    "<p style='margin-bottom:.3rem'><a href='/career'>&larr; Career Builder</a></p>"
    "<div class='card'><h2>Finalizing Career Data</h2>"
    "<div id='progress-log'>"
)


@router.post("/{repo_id}/finalize")
def career_finalize(repo_id: str) -> StreamingResponse:
    """Consolidate skills/meta and compose STAR stories after user review of roles.

    This runs the deferred LLM steps (skills/meta consolidation + story
    composition) that were skipped during document ingestion so the user
    could verify extracted roles first.  Role consolidation (Pass 1)
    already ran during ingest.
    """
    repo = _load_repo(repo_id)

    def _progress() -> Iterator[str]:
        nonlocal repo
        yield _FINALIZE_PROGRESS_HEAD + "\n" + _BROWSER_FLUSH_PAD

        # Consolidate skills, education, certifications, domain knowledge, meta (Pass 2)
        yield "<p>Consolidating skills &amp; metadata\u2026</p>\n"
        try:
            repo = consolidate_skills_meta(repo, client=ingest_agent.client)
            yield "<p class='step-ok'>Skills &amp; metadata consolidated</p>\n"
        except Exception:
            log.warning("Skills/meta consolidation failed, continuing with extracted data", exc_info=True)
            yield "<p class='step-fail'>Skills consolidation failed, continuing with extracted data</p>\n"

        # Compose STAR stories from merged accomplishments (per-role)
        yield "<p>Composing behavioral stories\u2026</p>\n"
        try:
            ingest_agent.compose_stories(repo)
            yield f"<p class='step-ok'>Composed {len(repo.stories)} stories across {len(repo.roles)} roles</p>\n"
        except Exception:
            log.warning("Story composition failed, continuing without stories", exc_info=True)
            yield "<p class='step-fail'>Story composition failed</p>\n"

        # Clear the flag and advance to deep-dive
        repo.needs_consolidation = False
        repo.current_phase = "role_deepdive"
        career_store.save(repo)

        redirect_url = f"/career/{repo.repo_id}"
        yield (
            "</div>"
            f"<p class='step-ok' style='margin-top:.8rem;font-weight:600'>Done! Redirecting\u2026</p>"
            f"<script>setTimeout(function(){{window.location.href='{redirect_url}';}},1200);</script>"
            "</div></div></body></html>"
        )

    return StreamingResponse(_progress(), media_type="text/html")


# ---------------------------------------------------------------------------
# Phase 3: Role Deep Dive
# ---------------------------------------------------------------------------


@router.get("/{repo_id}/role_deepdive", response_class=HTMLResponse)
def career_role_deepdive(repo_id: str) -> HTMLResponse:
    return _render_role_deepdive(_load_repo(repo_id))


def _render_role_deepdive(repo: CareerRepository) -> HTMLResponse:
    if not repo.roles:
        return _wizard_page("Deep Dive", '<div class="card"><p>No roles to deep-dive. '
                            f'<a href="/career/{_esc(repo.repo_id)}/roles">Add roles first.</a></p></div>', repo)

    # Detect whether data came from document ingestion
    _ingested = any(
        r.confidence_notes or r.extraction_confidence != "medium"
        for r in repo.roles
    )

    # Show a one-time summary banner if this looks like an ingested repo
    ingestion_banner = ""
    if _ingested:
        filled_roles = sum(1 for r in repo.roles if r.accomplishments.strip())
        low_conf = sum(1 for r in repo.roles if r.extraction_confidence == "low")
        ingestion_banner = (
            '<div class="card" style="background:var(--accent-light);border-left:3px solid var(--accent)">'
            f'<strong>Imported from documents</strong> — '
            f'{len(repo.roles)} role{"s" if len(repo.roles) != 1 else ""} extracted'
            f', {len(repo.skills)} skill{"s" if len(repo.skills) != 1 else ""}'
            f', {len(repo.stories)} stor{"ies" if len(repo.stories) != 1 else "y"}'
        )
        if low_conf:
            ingestion_banner += (
                f'<br><span style="color:#b00020;font-weight:600">{low_conf} role{"s" if low_conf != 1 else ""}'
                f' with low confidence</span> — these need the most attention'
            )
        ingestion_banner += (
            '<br><span class="muted" style="font-size:0.88rem">'
            'Tip: Use the progress bar above to review '
            '<a href="/career/' + _esc(repo.repo_id) + '/identity">Identity</a> and '
            '<a href="/career/' + _esc(repo.repo_id) + '/roles">Roles</a> that were pre-filled.'
            '</span></div>'
        )

    idx = repo.deepdive_role_index
    if idx >= len(repo.roles):
        idx = len(repo.roles) - 1

    role = repo.roles[idx]
    nav_links: list[str] = []
    for i, r in enumerate(repo.roles):
        style = "font-weight:700" if i == idx else ""
        nav_links.append(
            f'<a href="/career/{_esc(repo.repo_id)}/role_deepdive/{i}" style="{style}">'
            f'{_esc(r.company)}</a>'
        )
    nav = " &middot; ".join(nav_links)
    progress_text = f"Role {idx + 1} of {len(repo.roles)}"

    # Show extraction confidence when data came from ingestion
    confidence_html = ""
    if role.extraction_confidence != "medium" or role.confidence_notes:
        conf_color = {"high": "#1d8f52", "medium": "#b8860b", "low": "#b00020"}[role.extraction_confidence]
        conf_label = role.extraction_confidence.capitalize()
        notes = f" — {_esc(role.confidence_notes)}" if role.confidence_notes else ""
        confidence_html = (
            f'<div class="warning" style="border-left-color:{conf_color}">'
            f'Extraction confidence: <strong style="color:{conf_color}">{conf_label}</strong>{notes}'
            f'<br><span class="muted" style="font-size:0.85rem">Review and enrich the fields below — especially where the AI was uncertain.</span>'
            f'</div>'
        )

    body = f"""
{ingestion_banner}
<div class="card">
  <h2>Phase 3: Deep Dive — {_esc(role.title)} @ {_esc(role.company)}</h2>
  <p class="muted">{progress_text} &nbsp;|&nbsp; {nav}</p>
  {confidence_html}
  <form method="post" action="/career/{_esc(repo.repo_id)}/role_deepdive/{idx}">
    <label>Company Context</label>
    <p class="hint">What did this company do? How big was it? Stage? Industry?</p>
    <textarea name="company_context" rows="3">{_esc(role.company_context)}</textarea>

    <label>Team Context</label>
    <p class="hint">Team size, who you reported to, scope of team.</p>
    <textarea name="team_context" rows="2">{_esc(role.team_context)}</textarea>

    <label>What You Owned</label>
    <p class="hint">What were you specifically responsible for?</p>
    <textarea name="ownership" rows="3">{_esc(role.ownership)}</textarea>

    <label>Key Accomplishments</label>
    <p class="hint">Walk through your top 2-4 wins. Be specific — numbers, timelines,
       outcomes. What did you do and what happened?</p>
    <textarea name="accomplishments" rows="8">{_esc(role.accomplishments)}</textarea>

    <label>Technologies Used</label>
    <p class="hint">Languages, frameworks, tools, infrastructure.</p>
    <textarea name="technologies" rows="2">{_esc(role.technologies)}</textarea>

    <label>What You Learned</label>
    <p class="hint">What did this role teach you that you couldn't have learned elsewhere?</p>
    <textarea name="learnings" rows="3">{_esc(role.learnings)}</textarea>

    <label>Anti-Claims</label>
    <p class="hint">Anything from this role that should NOT appear on a resume?</p>
    <textarea name="anti_claims" rows="2">{_esc(role.anti_claims)}</textarea>

    <div class="actions">
      <button type="submit">Save This Role</button>
    </div>
  </form>
</div>

<div id="probe-area"></div>
<button class="btn btn-secondary btn-sm"
        hx-post="/career/{_esc(repo.repo_id)}/role_deepdive/{idx}/probe"
        hx-target="#probe-area"
        hx-swap="innerHTML"
        hx-indicator="#probe-loading">Get Follow-Up Questions</button>
<span id="probe-loading" class="htmx-indicator">
  <span class="spinner"></span> Generating questions&hellip;
</span>

<div class="card">
  <div class="actions">
    <a href="/career/{_esc(repo.repo_id)}/roles" class="btn btn-secondary">Back to Roles</a>
    {"" if idx >= len(repo.roles) - 1 else
     f'<a href="/career/{_esc(repo.repo_id)}/role_deepdive/{idx + 1}" class="btn btn-secondary">Next Role</a>'}
    <form method="post" action="/career/{_esc(repo.repo_id)}/advance/skills" style="display:inline">
      <button type="submit">Done with Deep Dives &rarr; Skills</button>
    </form>
  </div>
</div>
"""
    return _wizard_page("Deep Dive", body, repo)


@router.get("/{repo_id}/role_deepdive/{role_idx}", response_class=HTMLResponse)
def career_role_deepdive_specific(repo_id: str, role_idx: int) -> HTMLResponse:
    repo = _load_repo(repo_id)
    if role_idx < 0 or role_idx >= len(repo.roles):
        raise HTTPException(status_code=404, detail="Role not found")
    repo.deepdive_role_index = role_idx
    career_store.save(repo)
    return _render_role_deepdive(repo)


@router.post("/{repo_id}/role_deepdive/{role_idx}")
def save_role_deepdive(
    repo_id: str,
    role_idx: int,
    company_context: str = Form(""),
    team_context: str = Form(""),
    ownership: str = Form(""),
    accomplishments: str = Form(""),
    technologies: str = Form(""),
    learnings: str = Form(""),
    anti_claims: str = Form(""),
) -> RedirectResponse:
    repo = _load_repo(repo_id)
    if role_idx < 0 or role_idx >= len(repo.roles):
        raise HTTPException(status_code=404, detail="Role not found")
    role = repo.roles[role_idx]
    role.company_context = company_context.strip()
    role.team_context = team_context.strip()
    role.ownership = ownership.strip()
    role.accomplishments = accomplishments.strip()
    role.technologies = technologies.strip()
    role.learnings = learnings.strip()
    role.anti_claims = anti_claims.strip()
    repo.deepdive_role_index = role_idx
    career_store.save(repo)
    # Stay on the same page after save so user can request probes or continue
    return RedirectResponse(url=f"/career/{repo.repo_id}/role_deepdive/{repo.deepdive_role_index}", status_code=303)


# ---------------------------------------------------------------------------
# Phase 4: Skills Inventory
# ---------------------------------------------------------------------------


@router.get("/{repo_id}/skills", response_class=HTMLResponse)
def career_skills(repo_id: str) -> HTMLResponse:
    return _render_skills(_load_repo(repo_id))


def _render_skills(repo: CareerRepository, edit_idx: int | None = None) -> HTMLResponse:
    skill_rows: list[str] = []
    for i, s in enumerate(repo.skills):
        if i == edit_idx:
            # Inline edit form for this skill
            cat_opts = "".join(
                f'<option value="{c}"{" selected" if c == s.category else ""}>{c}</option>'
                for c in ("language", "framework", "infrastructure", "tool", "non_technical", "other")
            )
            prof_opts = "".join(
                f'<option value="{p}"{" selected" if p == s.proficiency else ""}>{p}</option>'
                for p in ("familiar", "working", "strong", "expert")
            )
            skill_rows.append(
                f'<tr><td colspan="6">'
                f'<form method="post" action="/career/{_esc(repo.repo_id)}/skills/{i}/edit" '
                f'style="display:flex;flex-wrap:wrap;gap:0.5rem;align-items:flex-end">'
                f'<div><label style="font-size:0.85rem">Name</label>'
                f'<input type="text" name="name" value="{_esc(s.name)}" required style="width:130px" /></div>'
                f'<div><label style="font-size:0.85rem">Category</label>'
                f'<select name="category">{cat_opts}</select></div>'
                f'<div><label style="font-size:0.85rem">Level</label>'
                f'<select name="proficiency">{prof_opts}</select></div>'
                f'<div><label style="font-size:0.85rem">Years</label>'
                f'<input type="text" name="years" value="{_esc(s.years or "")}" style="width:60px" /></div>'
                f'<div style="flex:1 1 100%"><label style="font-size:0.85rem">Evidence</label>'
                f'<textarea name="evidence" rows="2" style="width:100%">{_esc(s.evidence)}</textarea></div>'
                f'<div style="display:flex;gap:0.3rem">'
                f'<button type="submit" class="btn btn-sm">Save</button>'
                f'<a href="/career/{_esc(repo.repo_id)}/skills" class="btn btn-sm btn-secondary">Cancel</a>'
                f'</div></form></td></tr>'
            )
            continue

        if s.evidence.strip():
            evidence_cell = (
                f'<details><summary style="cursor:pointer;font-size:0.9rem">Show evidence</summary>'
                f'<p style="margin:0.4rem 0 0;white-space:pre-wrap;font-size:0.88rem">{_esc(s.evidence)}</p>'
                f'</details>'
            )
        else:
            evidence_cell = ""
        skill_rows.append(
            f"<tr><td>{_esc(s.name)}</td><td>{_esc(s.category)}</td>"
            f"<td>{_esc(s.proficiency)}</td><td>{_esc(s.years or '-')}</td>"
            f"<td>{evidence_cell}</td>"
            f'<td style="white-space:nowrap">'
            f'<a href="/career/{_esc(repo.repo_id)}/skills/{i}/edit" class="btn btn-sm btn-secondary" '
            f'style="margin-right:0.3rem">Edit</a>'
            f'<form method="post" action="/career/{_esc(repo.repo_id)}/skills/{i}/delete" style="display:inline">'
            f'<button type="submit" class="btn btn-sm btn-danger">X</button></form></td></tr>'
        )

    table = (
        '<table><thead><tr><th>Skill</th><th>Category</th><th>Level</th>'
        '<th>Years</th><th>Evidence</th><th></th></tr></thead>'
        f'<tbody>{"\n".join(skill_rows) if skill_rows else "<tr><td colspan=6>No skills added yet.</td></tr>"}'
        '</tbody></table>'
    )

    body = f"""
<div class="card">
  <h2>Phase 4: Skills Inventory</h2>
  <p class="muted">List your technical and non-technical skills. Be honest about proficiency levels.</p>
  {table}
</div>
<div class="card">
  <h3>Add a Skill</h3>
  <form method="post" action="/career/{_esc(repo.repo_id)}/skills">
    <div class="grid">
      <div>
        <label>Skill Name</label>
        <input type="text" name="name" required />
      </div>
      <div>
        <label>Category</label>
        <select name="category">
          <option value="language">Language</option>
          <option value="framework">Framework</option>
          <option value="infrastructure">Infrastructure</option>
          <option value="tool">Tool</option>
          <option value="non_technical">Non-Technical</option>
          <option value="other" selected>Other</option>
        </select>
      </div>
      <div>
        <label>Proficiency</label>
        <select name="proficiency">
          <option value="familiar">Familiar</option>
          <option value="working" selected>Working</option>
          <option value="strong">Strong</option>
          <option value="expert">Expert</option>
        </select>
      </div>
      <div>
        <label>Years</label>
        <input type="text" name="years" placeholder="e.g. 6+" />
      </div>
    </div>
    <label>Evidence</label>
    <p class="hint">Concrete evidence — "Primary language at DataFlow, built all backend services"</p>
    <textarea name="evidence" rows="2"></textarea>
    <button type="submit">Add Skill</button>
  </form>
</div>
<div class="card">
  <div class="actions">
    <a href="/career/{_esc(repo.repo_id)}/role_deepdive" class="btn btn-secondary">Back</a>
    <form method="post" action="/career/{_esc(repo.repo_id)}/advance/stories" style="display:inline">
      <button type="submit">Continue to Stories</button>
    </form>
  </div>
</div>
"""
    return _wizard_page("Skills", body, repo)


@router.post("/{repo_id}/skills")
def add_skill(
    repo_id: str,
    name: str = Form(...),
    category: str = Form("other"),
    proficiency: str = Form("working"),
    years: str = Form(""),
    evidence: str = Form(""),
) -> RedirectResponse:
    repo = _load_repo(repo_id)
    repo.skills.append(SkillEntry(
        name=name.strip(),
        category=category,  # type: ignore[arg-type]
        proficiency=proficiency,  # type: ignore[arg-type]
        years=years.strip() or None,
        evidence=evidence.strip(),
    ))
    career_store.save(repo)
    return RedirectResponse(url=f"/career/{repo.repo_id}/skills", status_code=303)


@router.get("/{repo_id}/skills/{skill_idx}/edit", response_class=HTMLResponse)
def edit_skill_form(repo_id: str, skill_idx: int) -> HTMLResponse:
    repo = _load_repo(repo_id)
    if not (0 <= skill_idx < len(repo.skills)):
        raise HTTPException(status_code=404, detail="Skill not found")
    return _render_skills(repo, edit_idx=skill_idx)


@router.post("/{repo_id}/skills/{skill_idx}/edit")
def edit_skill(
    repo_id: str,
    skill_idx: int,
    name: str = Form(...),
    category: str = Form("other"),
    proficiency: str = Form("working"),
    years: str = Form(""),
    evidence: str = Form(""),
) -> RedirectResponse:
    repo = _load_repo(repo_id)
    if not (0 <= skill_idx < len(repo.skills)):
        raise HTTPException(status_code=404, detail="Skill not found")
    repo.skills[skill_idx] = SkillEntry(
        name=name.strip(),
        category=category,  # type: ignore[arg-type]
        proficiency=proficiency,  # type: ignore[arg-type]
        years=years.strip() or None,
        evidence=evidence.strip(),
    )
    career_store.save(repo)
    return RedirectResponse(url=f"/career/{repo.repo_id}/skills", status_code=303)


@router.post("/{repo_id}/skills/{skill_idx}/delete")
def delete_skill(repo_id: str, skill_idx: int) -> RedirectResponse:
    repo = _load_repo(repo_id)
    if 0 <= skill_idx < len(repo.skills):
        repo.skills.pop(skill_idx)
        career_store.save(repo)
    return RedirectResponse(url=f"/career/{repo.repo_id}/skills", status_code=303)


# ---------------------------------------------------------------------------
# Phase 5: Behavioral Stories (STAR format)
# ---------------------------------------------------------------------------


@router.get("/{repo_id}/stories", response_class=HTMLResponse)
def career_stories(repo_id: str) -> HTMLResponse:
    return _render_stories(_load_repo(repo_id))


def _render_stories(repo: CareerRepository) -> HTMLResponse:
    story_cards: list[str] = []
    for i, s in enumerate(repo.stories):
        tags = ", ".join(s.tags) if s.tags else "no tags"
        # Show extraction confidence badge when data came from ingestion
        conf_badge = ""
        if s.extraction_confidence != "medium" or s.confidence_notes:
            conf_color = {"high": "#1d8f52", "medium": "#b8860b", "low": "#b00020"}[s.extraction_confidence]
            conf_label = s.extraction_confidence.capitalize()
            conf_badge = f' <span style="font-size:0.8rem;color:{conf_color};font-weight:600">[{conf_label} confidence]</span>'
            if s.confidence_notes:
                conf_badge += f' <span class="muted" style="font-size:0.8rem">— {_esc(s.confidence_notes)}</span>'
        star_parts: list[str] = []
        if s.situation:
            star_parts.append(f'<div><strong style="color:var(--accent)">Situation:</strong> {_esc(s.situation)}</div>')
        if s.task:
            star_parts.append(f'<div><strong style="color:var(--accent)">Task:</strong> {_esc(s.task)}</div>')
        if s.action:
            star_parts.append(f'<div><strong style="color:var(--accent)">Action:</strong> {_esc(s.action)}</div>')
        if s.result:
            star_parts.append(f'<div><strong style="color:var(--accent)">Result:</strong> {_esc(s.result)}</div>')
        if s.what_it_shows:
            star_parts.append(f'<div><strong style="color:var(--accent)">Shows:</strong> {_esc(s.what_it_shows)}</div>')
        star_html = (
            f'<div style="margin-top:0.5rem;display:flex;flex-direction:column;gap:0.3rem;'
            f'font-size:0.92rem;line-height:1.45">{"\n".join(star_parts)}</div>'
            if star_parts else '<p class="muted" style="font-size:0.9rem">No details entered yet.</p>'
        )
        story_cards.append(f"""
<div class="role-card">
  <strong>{_esc(s.title)}</strong> <span class="muted">({_esc(tags)})</span>{conf_badge}
  {star_html}
  <div style="margin-top:0.5rem">
    <a href="/career/{_esc(repo.repo_id)}/stories/{i}/edit" class="btn btn-sm btn-secondary">Edit</a>
    <form method="post" action="/career/{_esc(repo.repo_id)}/stories/{i}/delete"
          style="display:inline" onsubmit="return confirm('Remove this story?')">
      <button type="submit" class="btn btn-sm btn-danger">Remove</button>
    </form>
  </div>
</div>
""")

    existing = "".join(story_cards) if story_cards else '<p class="muted">No stories yet.</p>'

    prompts_html = """
<div class="probe">
  <strong>Story prompts — pick any that resonate:</strong>
  <ul style="margin:0.4rem 0 0">
    <li>A time you influenced a decision <em>without</em> authority</li>
    <li>A time something went wrong and you handled it well</li>
    <li>A time you saved significant money, time, or risk</li>
    <li>A time you mentored someone and it worked</li>
    <li>A time you had to learn something new under pressure</li>
    <li>A project where you drove the architecture or design</li>
  </ul>
</div>
"""

    body = f"""
<div class="card">
  <h2>Phase 5: Behavioral Stories</h2>
  <p class="muted">STAR-format stories are interview gold. Add 3-5 stories you can tell fluently.</p>
  {prompts_html}
  {existing}
</div>
<div class="card">
  <h3>Add a Story</h3>
  <form method="post" action="/career/{_esc(repo.repo_id)}/stories">
    <label>Story Title</label>
    <input type="text" name="title" required placeholder="e.g. Redis Caching Cost Savings" />

    <label>Tags</label>
    <p class="hint">Comma-separated. e.g. "cost-optimization, initiative, architecture"</p>
    <input type="text" name="tags" />

    <label>Situation</label>
    <p class="hint">What was the context? What was happening?</p>
    <textarea name="situation" rows="3"></textarea>

    <label>Task</label>
    <p class="hint">What was your responsibility or what did you decide to take on?</p>
    <textarea name="task" rows="2"></textarea>

    <label>Action</label>
    <p class="hint">What specifically did YOU do? Be concrete.</p>
    <textarea name="action" rows="4"></textarea>

    <label>Result</label>
    <p class="hint">What happened? Quantify if possible.</p>
    <textarea name="result" rows="3"></textarea>

    <label>What This Shows About You</label>
    <p class="hint">What trait or capability does this story demonstrate?</p>
    <textarea name="what_it_shows" rows="2"></textarea>

    <button type="submit">Add Story</button>
  </form>
</div>
<div class="card">
  <div class="actions">
    <a href="/career/{_esc(repo.repo_id)}/skills" class="btn btn-secondary">Back</a>
    <form method="post" action="/career/{_esc(repo.repo_id)}/advance/meta" style="display:inline">
      <button type="submit">Continue to Strategy</button>
    </form>
  </div>
</div>
"""
    return _wizard_page("Stories", body, repo)


@router.post("/{repo_id}/stories")
def add_story(
    repo_id: str,
    title: str = Form(...),
    tags: str = Form(""),
    situation: str = Form(""),
    task: str = Form(""),
    action: str = Form(""),
    result: str = Form(""),
    what_it_shows: str = Form(""),
) -> RedirectResponse:
    repo = _load_repo(repo_id)
    repo.stories.append(StoryEntry(
        title=title.strip(),
        tags=[t.strip() for t in tags.split(",") if t.strip()],
        situation=situation.strip(),
        task=task.strip(),
        action=action.strip(),
        result=result.strip(),
        what_it_shows=what_it_shows.strip(),
    ))
    career_store.save(repo)
    return RedirectResponse(url=f"/career/{repo.repo_id}/stories", status_code=303)


@router.get("/{repo_id}/stories/{story_idx}/edit", response_class=HTMLResponse)
def edit_story_form(repo_id: str, story_idx: int) -> HTMLResponse:
    repo = _load_repo(repo_id)
    if story_idx < 0 or story_idx >= len(repo.stories):
        raise HTTPException(status_code=404, detail="Story not found")
    s = repo.stories[story_idx]
    body = f"""
<div class="card">
  <h2>Edit Story: {_esc(s.title)}</h2>
  <form method="post" action="/career/{_esc(repo.repo_id)}/stories/{story_idx}/edit">
    <label>Title</label>
    <input type="text" name="title" value="{_esc(s.title)}" required />
    <label>Tags</label>
    <input type="text" name="tags" value="{_esc(', '.join(s.tags))}" />
    <label>Situation</label>
    <textarea name="situation" rows="3">{_esc(s.situation)}</textarea>
    <label>Task</label>
    <textarea name="task" rows="2">{_esc(s.task)}</textarea>
    <label>Action</label>
    <textarea name="action" rows="4">{_esc(s.action)}</textarea>
    <label>Result</label>
    <textarea name="result" rows="3">{_esc(s.result)}</textarea>
    <label>What This Shows About You</label>
    <textarea name="what_it_shows" rows="2">{_esc(s.what_it_shows)}</textarea>
    <div class="actions">
      <button type="submit">Save</button>
      <a href="/career/{_esc(repo.repo_id)}/stories" class="btn btn-secondary">Cancel</a>
    </div>
  </form>
</div>
"""
    return _wizard_page("Edit Story", body, repo)


@router.post("/{repo_id}/stories/{story_idx}/edit")
def save_story_edit(
    repo_id: str,
    story_idx: int,
    title: str = Form(...),
    tags: str = Form(""),
    situation: str = Form(""),
    task: str = Form(""),
    action: str = Form(""),
    result: str = Form(""),
    what_it_shows: str = Form(""),
) -> RedirectResponse:
    repo = _load_repo(repo_id)
    if story_idx < 0 or story_idx >= len(repo.stories):
        raise HTTPException(status_code=404, detail="Story not found")
    s = repo.stories[story_idx]
    s.title = title.strip()
    s.tags = [t.strip() for t in tags.split(",") if t.strip()]
    s.situation = situation.strip()
    s.task = task.strip()
    s.action = action.strip()
    s.result = result.strip()
    s.what_it_shows = what_it_shows.strip()
    career_store.save(repo)
    return RedirectResponse(url=f"/career/{repo.repo_id}/stories", status_code=303)


@router.post("/{repo_id}/stories/{story_idx}/delete")
def delete_story(repo_id: str, story_idx: int) -> RedirectResponse:
    repo = _load_repo(repo_id)
    if 0 <= story_idx < len(repo.stories):
        repo.stories.pop(story_idx)
        career_store.save(repo)
    return RedirectResponse(url=f"/career/{repo.repo_id}/stories", status_code=303)


# ---------------------------------------------------------------------------
# Phase 6: Career Strategy (Meta)
# ---------------------------------------------------------------------------


@router.get("/{repo_id}/meta", response_class=HTMLResponse)
def career_meta(repo_id: str) -> HTMLResponse:
    return _render_meta(_load_repo(repo_id))


def _render_meta(repo: CareerRepository) -> HTMLResponse:
    meta = repo.meta
    body = f"""
<div class="card">
  <h2>Phase 6: Career Strategy</h2>
  <p class="muted">Help the system understand what makes you different and what to avoid.</p>
  <form method="post" action="/career/{_esc(repo.repo_id)}/meta">
    <label>Career Arc</label>
    <p class="hint">Describe your career trajectory in 2-3 sentences. Where did you start, where are you going?</p>
    <textarea name="career_arc" rows="3">{_esc(meta.career_arc)}</textarea>

    <label>What Makes You Different</label>
    <p class="hint">If you could tell a hiring manager only 3 things, what would they be?</p>
    <textarea name="differentiators" rows="4">{_esc(meta.differentiators)}</textarea>

    <label>Themes to Emphasize</label>
    <p class="hint">One per line. e.g. "Ownership mentality", "Measurable impact"</p>
    <textarea name="themes" rows="4">{_esc(chr(10).join(meta.themes_to_emphasize))}</textarea>

    <label>Things to NEVER Claim</label>
    <p class="hint">One per line. Hard anti-claims — things a resume should never say about you.</p>
    <textarea name="anti_claims" rows="4">{_esc(chr(10).join(meta.anti_claims))}</textarea>

    <label>Known Gaps</label>
    <p class="hint">One per line. Areas where you're aware you're weaker.</p>
    <textarea name="known_gaps" rows="3">{_esc(chr(10).join(meta.known_gaps))}</textarea>

    <label>Domain Knowledge</label>
    <p class="hint">Industries, problem spaces, or subject areas you know deeply.</p>
    <textarea name="domain_knowledge" rows="4">{_esc(repo.domain_knowledge)}</textarea>

    <div class="actions">
      <a href="/career/{_esc(repo.repo_id)}/stories" class="btn btn-secondary">Back</a>
      <button type="submit">Save &amp; Continue to Voice</button>
    </div>
  </form>
</div>
"""
    return _wizard_page("Strategy", body, repo)


@router.post("/{repo_id}/meta")
def save_meta(
    repo_id: str,
    career_arc: str = Form(""),
    differentiators: str = Form(""),
    themes: str = Form(""),
    anti_claims: str = Form(""),
    known_gaps: str = Form(""),
    domain_knowledge: str = Form(""),
) -> RedirectResponse:
    repo = _load_repo(repo_id)
    repo.meta = CareerMeta(
        career_arc=career_arc.strip(),
        differentiators=differentiators.strip(),
        themes_to_emphasize=[t.strip() for t in themes.splitlines() if t.strip()],
        anti_claims=[c.strip() for c in anti_claims.splitlines() if c.strip()],
        known_gaps=[g.strip() for g in known_gaps.splitlines() if g.strip()],
    )
    repo.domain_knowledge = domain_knowledge.strip()
    repo.current_phase = "voice"
    career_store.save(repo)
    return RedirectResponse(url=f"/career/{repo.repo_id}", status_code=303)


# ---------------------------------------------------------------------------
# Phase 7: Voice Profile
# ---------------------------------------------------------------------------


@router.get("/{repo_id}/voice", response_class=HTMLResponse)
def career_voice(repo_id: str) -> HTMLResponse:
    return _render_voice(_load_repo(repo_id))


def _render_voice(repo: CareerRepository) -> HTMLResponse:
    voice_content = repo.voice_raw

    body = f"""
<div class="card">
  <h2>Phase 7: Voice Profile</h2>
  <p class="muted">Describe how you write and communicate. This shapes the tone of all generated documents.</p>

  <div class="probe">
    <strong>Think about:</strong>
    <ul style="margin:0.4rem 0 0">
      <li>Are you direct or diplomatic? Formal or casual?</li>
      <li>Do you use short punchy sentences or longer explanatory ones?</li>
      <li>What phrases do you actually use when talking about your work?</li>
      <li>What resume/cover letter cliches make you cringe?</li>
    </ul>
  </div>

  <form method="post" action="/career/{_esc(repo.repo_id)}/voice">
    <label>Core Adjectives</label>
    <p class="hint">5-7 words that describe your communication style. Comma-separated.</p>
    <input type="text" name="adjectives" value="{_esc(_voice_field(voice_content, 'Core Adjectives'))}" placeholder="e.g. Direct, analytical, grounded, warm" />

    <label>Style Notes</label>
    <p class="hint">How do you write? One rule per line.</p>
    <textarea name="style_notes" rows="5" placeholder="e.g. I use short declarative sentences for emphasis&#10;I state the outcome first, then explain how">{_esc(_voice_field(voice_content, 'Style Notes'))}</textarea>

    <label>Phrases You Actually Use</label>
    <p class="hint">One per line. Characteristic phrases from your real writing.</p>
    <textarea name="preferred_phrases" rows="4" placeholder="e.g. The key insight was...&#10;What that meant in practice...">{_esc(_voice_field(voice_content, 'Phrases I Actually Use'))}</textarea>

    <label>Phrases to Avoid</label>
    <p class="hint">One per line. Cliches or language you'd never use.</p>
    <textarea name="avoid_phrases" rows="4" placeholder="e.g. Passionate about&#10;Results-driven&#10;Proven track record">{_esc(_voice_field(voice_content, 'Phrases to Avoid'))}</textarea>

    <label>Writing Sample 1</label>
    <p class="hint">A paragraph in your natural voice — could be from an email, blog post, or Slack message.</p>
    <textarea name="sample_1" rows="5">{_esc(_voice_field(voice_content, 'Writing Sample 1'))}</textarea>

    <label>Writing Sample 2 (optional)</label>
    <textarea name="sample_2" rows="5">{_esc(_voice_field(voice_content, 'Writing Sample 2'))}</textarea>

    <div class="actions">
      <a href="/career/{_esc(repo.repo_id)}/meta" class="btn btn-secondary">Back</a>
      <button type="submit">Save &amp; Finish</button>
    </div>
  </form>
</div>
"""
    return _wizard_page("Voice", body, repo)


@router.post("/{repo_id}/voice")
def save_voice(
    repo_id: str,
    adjectives: str = Form(""),
    style_notes: str = Form(""),
    preferred_phrases: str = Form(""),
    avoid_phrases: str = Form(""),
    sample_1: str = Form(""),
    sample_2: str = Form(""),
) -> RedirectResponse:
    repo = _load_repo(repo_id)
    # Build the voice profile as structured markdown and store it as a
    # CareerRepository-level field so it can be extracted later.
    parts: list[str] = [f"# Voice Profile — {repo.identity.name}\n"]
    if adjectives.strip():
        parts.append("## Core Adjectives")
        for a in adjectives.split(","):
            a = a.strip()
            if a:
                parts.append(f"- {a}")
        parts.append("")
    if style_notes.strip():
        parts.append("## Style Notes")
        for line in style_notes.strip().splitlines():
            line = line.strip()
            if line:
                parts.append(f"- {line}")
        parts.append("")
    if preferred_phrases.strip():
        parts.append("## Phrases I Actually Use")
        for line in preferred_phrases.strip().splitlines():
            line = line.strip()
            if line:
                parts.append(f'- "{line}"')
        parts.append("")
    if avoid_phrases.strip():
        parts.append("## Phrases to Avoid")
        for line in avoid_phrases.strip().splitlines():
            line = line.strip()
            if line:
                parts.append(f'- "{line}"')
        parts.append("")
    if sample_1.strip():
        parts.append("## Writing Sample 1")
        parts.append(sample_1.strip())
        parts.append("")
    if sample_2.strip():
        parts.append("## Writing Sample 2")
        parts.append(sample_2.strip())
        parts.append("")

    # Store raw voice text on the repo — we'll persist this in career.json
    # via a dedicated field.
    repo.voice_raw = "\n".join(parts)
    repo.current_phase = "complete"
    career_store.save(repo)
    return RedirectResponse(url=f"/career/{repo.repo_id}", status_code=303)


# ---------------------------------------------------------------------------
# Phase 8: Review & Complete
# ---------------------------------------------------------------------------


@router.get("/{repo_id}/review", response_class=HTMLResponse)
def career_review(repo_id: str) -> HTMLResponse:
    return _render_review(_load_repo(repo_id))


def _render_review(repo: CareerRepository) -> HTMLResponse:
    ident = repo.identity

    # Summary stats
    n_roles = len(repo.roles)
    filled_roles = sum(1 for r in repo.roles if r.accomplishments.strip())
    n_skills = len(repo.skills)
    n_stories = len(repo.stories)
    has_meta = bool(repo.meta.career_arc or repo.meta.differentiators)
    has_voice = bool(repo.voice_raw.strip())

    completeness: list[str] = []
    if not ident.name:
        completeness.append("Missing name")
    if n_roles == 0:
        completeness.append("No roles added")
    if n_roles > 0 and filled_roles < n_roles:
        completeness.append(f"Only {filled_roles}/{n_roles} roles have accomplishments")
    if n_skills == 0:
        completeness.append("No skills added")
    if n_stories == 0:
        completeness.append("No behavioral stories — these help a lot for interview prep")
    if not has_meta:
        completeness.append("Career strategy section is empty")
    if not has_voice:
        completeness.append("No voice profile — generated docs will use a generic tone")

    if completeness:
        warnings = '<div class="probe"><strong>Gaps to consider filling:</strong><ul>' + \
            "".join(f"<li>{_esc(w)}</li>" for w in completeness) + "</ul></div>"
    else:
        warnings = '<p style="color:#1d8f52;font-weight:700">Profile looks complete!</p>'

    # Build section summaries
    roles_summary: list[str] = []
    for r in repo.roles:
        status = "detailed" if r.accomplishments.strip() else "skeleton only"
        roles_summary.append(f"<li>{_esc(r.title)} @ {_esc(r.company)} — <span class='muted'>{status}</span></li>")

    skills_summary = ", ".join(s.name for s in repo.skills[:15])
    if len(repo.skills) > 15:
        skills_summary += f" (+{len(repo.skills) - 15} more)"

    stories_summary: list[str] = []
    for s in repo.stories:
        stories_summary.append(f"<li>{_esc(s.title)}</li>")

    body = f"""
<div class="card">
  <h2>Career Repository: {_esc(ident.name)}</h2>
  <p class="muted">{_esc(ident.headline)}</p>
  {warnings}
</div>

<div class="grid">
  <div class="card">
    <h3>Roles ({n_roles})</h3>
    <ul>{"".join(roles_summary) if roles_summary else "<li>None</li>"}</ul>
    <a href="/career/{_esc(repo.repo_id)}/roles" class="btn btn-sm btn-secondary">Edit Roles</a>
  </div>
  <div class="card">
    <h3>Skills ({n_skills})</h3>
    <p>{_esc(skills_summary) if skills_summary else "None"}</p>
    <a href="/career/{_esc(repo.repo_id)}/skills" class="btn btn-sm btn-secondary">Edit Skills</a>
  </div>
  <div class="card">
    <h3>Stories ({n_stories})</h3>
    <ul>{"".join(stories_summary) if stories_summary else "<li>None</li>"}</ul>
    <a href="/career/{_esc(repo.repo_id)}/stories" class="btn btn-sm btn-secondary">Edit Stories</a>
  </div>
  <div class="card">
    <h3>Strategy</h3>
    <p>{"Filled" if has_meta else "Empty"}</p>
    <a href="/career/{_esc(repo.repo_id)}/meta" class="btn btn-sm btn-secondary">Edit Strategy</a>
  </div>
</div>

<div class="card">
  <h3>Use This Profile</h3>
  <p>Your career repository is ready to use for generating tailored documents.
     Go to <a href="/">New Session</a> and select this profile.</p>
  <div class="actions">
    <a href="/" class="btn">Go to Resume Refinery</a>
    <a href="/career/{_esc(repo.repo_id)}/identity" class="btn btn-secondary">Edit from Start</a>
  </div>
</div>
"""
    return _wizard_page("Review", body, repo)


# ---------------------------------------------------------------------------
# Phase advance helper
# ---------------------------------------------------------------------------


@router.post("/{repo_id}/advance/{phase}")
def advance_phase(repo_id: str, phase: str) -> RedirectResponse:
    valid_phases: set[WizardPhase] = {
        "identity", "roles", "role_deepdive", "skills", "stories", "meta", "voice", "complete",
    }
    if phase not in valid_phases:
        raise HTTPException(status_code=400, detail=f"Invalid phase: {phase}")
    repo = _load_repo(repo_id)
    repo.current_phase = phase  # type: ignore[assignment]
    career_store.save(repo)
    return RedirectResponse(url=f"/career/{repo.repo_id}", status_code=303)


# ---------------------------------------------------------------------------
# HTMX probe endpoint (ElicitationAgent hook — returns partial HTML)
# ---------------------------------------------------------------------------


@router.post("/{repo_id}/role_deepdive/{role_idx}/probe", response_class=HTMLResponse)
def probe_role_deepdive(repo_id: str, role_idx: int) -> HTMLResponse:
    """Ask the ElicitationAgent for follow-up probes on a role's answers.

    Returns a partial HTML fragment to be swapped into #probe-area via HTMX.
    Uses the LLM-backed ElicitationAgent, with static fallback on failure.
    """
    repo = _load_repo(repo_id)
    if role_idx < 0 or role_idx >= len(repo.roles):
        raise HTTPException(status_code=404, detail="Role not found")

    role = repo.roles[role_idx]
    result = elicitation_agent.probe_role(role)

    llm_notice = ""
    if not result.llm_used:
        llm_notice = (
            '<p class="warning">⚠ LLM is unavailable — showing basic suggestions. '
            "Start your Ollama server for smarter follow-up questions.</p>"
        )

    if not result.probes:
        return HTMLResponse(
            f'<div class="probe">{llm_notice}'
            "<p>Looks good — your answers are detailed enough.</p></div>"
        )

    items = "".join(f"<li>{_esc(p)}</li>" for p in result.probes)
    return HTMLResponse(f"""
<div class="probe">
  {llm_notice}
  <strong>Consider strengthening your answers:</strong>
  <ul>{items}</ul>
</div>
""")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _voice_field(voice_raw: str, heading: str) -> str:
    """Extract the content under a markdown ## heading from voice_raw.

    Returns the text between the given heading and the next heading (or EOF),
    with leading list markers (``- ``, ``"`` wrapping) stripped.
    """
    import re
    pattern = rf"^## {re.escape(heading)}\s*\n(.*?)(?=\n## |\Z)"
    m = re.search(pattern, voice_raw, re.DOTALL | re.MULTILINE)
    if not m:
        return ""
    block = m.group(1).strip()
    # Strip leading "- " and surrounding quotes from each line
    lines: list[str] = []
    for line in block.splitlines():
        line = line.strip()
        line = re.sub(r'^- "?(.*?)"?$', r"\1", line)
        if line:
            lines.append(line)
    return "\n".join(lines)


_SAFE_REPO_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,80}$")


def _load_repo(repo_id: str) -> CareerRepository:
    if not _SAFE_REPO_ID_RE.match(repo_id):
        raise HTTPException(status_code=400, detail="Invalid repository ID")
    try:
        return career_store.get(repo_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Career repo not found")
