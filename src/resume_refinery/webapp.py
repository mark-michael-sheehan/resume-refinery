"""Local web app for Resume Refinery."""

from __future__ import annotations

import html
import os
import queue
import re
import threading
import time
from pathlib import Path
from typing import Iterator, Optional

import json as _json

import markdown as md
import uvicorn
from fastapi import FastAPI, Form, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse

from .models import CandidacyNarrative, DocumentSet, DraftingContext, OrchestrationResult, ReviewBundle
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


@app.get("/api/browse", response_class=JSONResponse)
def browse_directories(path: str = Query("")) -> JSONResponse:
    """Return child directories of *path* for the directory picker.

    When *path* is empty the response lists filesystem roots (drive letters
    on Windows, ``/`` on POSIX).
    """
    if not path:
        # List filesystem roots
        if os.name == "nt":
            import string
            roots = [
                f"{d}:\\" for d in string.ascii_uppercase
                if os.path.isdir(f"{d}:\\")
            ]
        else:
            roots = ["/"]
        return JSONResponse({"path": "", "dirs": roots})

    try:
        resolved = Path(path).resolve(strict=True)
    except (OSError, ValueError):
        return JSONResponse({"path": path, "dirs": [], "error": "Invalid path"}, status_code=400)

    if not resolved.is_dir():
        return JSONResponse({"path": path, "dirs": [], "error": "Not a directory"}, status_code=400)

    dirs: list[str] = []
    try:
        for entry in sorted(resolved.iterdir()):
            if entry.is_dir() and not entry.name.startswith("."):
                dirs.append(entry.name)
    except PermissionError:
        pass

    parent = str(resolved.parent) if resolved.parent != resolved else ""
    return JSONResponse({"path": str(resolved), "parent": parent, "dirs": dirs})


def _validate_output_dir(raw: str) -> Path:
    """Validate an output directory path from user input.

    Raises HTTPException(400) if the path is invalid or unusable.
    """
    stripped = raw.strip()
    if not stripped:
        raise HTTPException(status_code=400, detail="Output directory is required.")
    try:
        p = Path(stripped).resolve()
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid output directory path: {exc}")
    if p.exists() and not p.is_dir():
        raise HTTPException(
            status_code=400,
            detail=f"Output path exists but is not a directory: {p}",
        )
    if not p.exists():
        parent = p.parent
        if not parent.exists():
            raise HTTPException(
                status_code=400,
                detail=f"Parent directory does not exist: {parent}",
            )
        if not parent.is_dir():
            raise HTTPException(
                status_code=400,
                detail=f"Parent path is not a directory: {parent}",
            )
    return p


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
    .rendered-md {{ line-height: 1.6; }}
    .rendered-md h1,.rendered-md h2,.rendered-md h3 {{ margin: 0.6rem 0 0.3rem; }}
    .rendered-md ul,.rendered-md ol {{ padding-left: 1.4rem; }}
    .rendered-md li {{ margin: 0.15rem 0; }}
    .rendered-md hr {{ border: none; border-top: 1px solid var(--line); margin: 0.8rem 0; }}
    .view-toggle {{ font-size: .85em; float: right; cursor: pointer; color: var(--accent); border: none; background: none; font-weight: 600; padding: 0; }}
    .view-toggle:hover {{ text-decoration: underline; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ border-bottom: 1px solid var(--line); text-align: left; padding: 0.5rem; }}
    .muted {{ color: var(--muted); }}
    .ok {{ color: #1d8f52; font-weight: 700; }}
    .bad {{ color: #b00020; font-weight: 700; }}
    /* Directory picker */
    .dir-picker-row {{ display: flex; gap: .5rem; align-items: center; margin-top: .4rem; }}
    .dir-picker-row input[type=text] {{ flex: 1; padding: .55rem; border-radius: 8px; border: 1px solid var(--line); background: #fff; }}
    .dir-picker-row button {{ margin-top: 0; padding: .55rem .85rem; font-size: .92em; }}
    .dir-modal-overlay {{ display:none; position:fixed; inset:0; background:rgba(0,0,0,.35); z-index:1000; justify-content:center; align-items:center; }}
    .dir-modal-overlay.open {{ display:flex; }}
    .dir-modal {{ background:var(--card); border-radius:14px; padding:1.25rem 1.5rem; width:500px; max-width:92vw; max-height:70vh; display:flex; flex-direction:column; box-shadow:0 12px 40px rgba(0,0,0,.18); }}
    .dir-modal h3 {{ margin:0 0 .4rem; }}
    .dir-modal .dir-path {{ font-size:.85em; color:var(--muted); word-break:break-all; margin-bottom:.5rem; min-height:1.2em; }}
    .dir-modal .dir-list {{ flex:1; overflow-y:auto; border:1px solid var(--line); border-radius:8px; background:#fff; max-height:45vh; }}
    .dir-modal .dir-list .dir-entry {{ padding:.45rem .7rem; cursor:pointer; border-bottom:1px solid #f0ede6; display:flex; align-items:center; gap:.4rem; }}
    .dir-modal .dir-list .dir-entry:hover {{ background:#f3f1eb; }}
    .dir-modal .dir-list .dir-entry.up {{ font-weight:600; color:var(--accent); }}
    .dir-modal .dir-btns {{ display:flex; gap:.5rem; justify-content:flex-end; margin-top:.7rem; }}
    .dir-modal .dir-btns button {{ margin-top:0; }}
  </style>
</head>
<body>
  <div class="wrap">{body}</div>
  <!-- Directory picker modal (shared by all pickers on the page) -->
  <div class="dir-modal-overlay" id="dirModal">
    <div class="dir-modal">
      <h3>Choose Output Directory</h3>
      <div class="dir-path" id="dirModalPath"></div>
      <div class="dir-list" id="dirModalList"></div>
      <div class="dir-btns">
        <button type="button" onclick="dirPickerCancel()">Cancel</button>
        <button type="button" onclick="dirPickerSelect()">Select This Folder</button>
      </div>
    </div>
  </div>
  <script>
  (function(){{
    let _target = null;
    let _currentPath = '';

    window.openDirPicker = function(inputId) {{
      _target = document.getElementById(inputId);
      _currentPath = _target.value || '';
      loadDir(_currentPath);
      document.getElementById('dirModal').classList.add('open');
    }};

    window.dirPickerCancel = function() {{
      document.getElementById('dirModal').classList.remove('open');
    }};

    window.dirPickerSelect = function() {{
      if (_target && _currentPath) _target.value = _currentPath;
      document.getElementById('dirModal').classList.remove('open');
    }};

    function loadDir(path) {{
      var url = '/api/browse?path=' + encodeURIComponent(path);
      fetch(url).then(function(r){{ return r.json(); }}).then(function(data){{
        if (data.error) {{ alert(data.error); return; }}
        _currentPath = data.path || '';
        document.getElementById('dirModalPath').textContent = _currentPath || '(select a drive)';
        var list = document.getElementById('dirModalList');
        list.innerHTML = '';
        if (data.parent !== undefined && data.parent !== null) {{
          var up = document.createElement('div');
          up.className = 'dir-entry up';
          up.textContent = '\u2191 Up';
          up.onclick = function(){{ loadDir(data.parent); }};
          list.appendChild(up);
        }}
        (data.dirs || []).forEach(function(d){{
          var el = document.createElement('div');
          el.className = 'dir-entry';
          el.textContent = '\U0001F4C1 ' + d;
          el.onclick = function(){{
            var sep = _currentPath.indexOf('/') !== -1 ? '/' : '\\\\';
            var child = _currentPath ? (_currentPath.replace(/[\\\\/]$/, '') + sep + d) : d;
            loadDir(child);
          }};
          list.appendChild(el);
        }});
      }}).catch(function(e){{ alert('Failed to browse: ' + e); }});
    }}
  }})();

  function toggleView(btn) {{
    var card = btn.closest('.card');
    var rendered = card.querySelector('.rendered-md');
    var raw = card.querySelector('pre');
    if (rendered.style.display === 'none') {{
      rendered.style.display = '';
      raw.style.display = 'none';
      btn.textContent = 'Show raw';
    }} else {{
      rendered.style.display = 'none';
      raw.style.display = '';
      btn.textContent = 'Show rendered';
    }}
  }}
  </script>
</body>
</html>
"""
    )


def _render_md(text: str | None) -> str:
    """Convert markdown text to HTML.  Returns empty string for None/empty."""
    if not text:
        return ""
    return md.markdown(text, extensions=["tables", "fenced_code", "nl2br"])


def _truth_failed(truth) -> bool:
    return bool(truth and not truth.all_supported)


_DOC_LABELS: dict[str, str] = {
    "resume": "Resume",
}


def _doc_label(key: str) -> str:
    """Human-readable label for a document key."""
    return _DOC_LABELS.get(key, key)


def _doc_cards(docs, esc) -> str:
    """Build HTML card for the resume document."""
    content = docs.get("resume")
    if not content:
        return "<div class='card'><p class='muted'>No resume generated yet.</p></div>"
    return (
        '<div class="card"><h2>Resume '
        '<button class="view-toggle" onclick="toggleView(this)">Show raw</button></h2>'
        f'<div class="rendered-md">{_render_md(content)}</div>'
        f'<pre style="display:none">{esc(content)}</pre></div>'
    )


def _truth_summary(truth) -> str:
    if not truth:
        return "<p class='muted'>No truth review available.</p>"
    status = "PASS" if truth.all_supported else "FAIL"
    klass = "ok" if truth.all_supported else "bad"
    parts = [f"<p>Strict truth check: <span class='{klass}'>{status}</span></p>"]
    claims = truth.resume.unsupported_claims
    if claims:
        parts.append(f"<h3>Resume — {len(claims)} unsupported claim(s)</h3><ul>")
        for claim in claims:
            parts.append(f"<li>&ldquo;{html.escape(claim)}&rdquo;</li>")
        parts.append("</ul>")
    else:
        parts.append("<p><strong>Resume</strong>: <span class='ok'>all claims supported</span></p>")
    return "".join(parts)


def _voice_summary(voice) -> str:
    if not voice:
        return "<p class='muted'>No voice review available.</p>"
    match = voice.overall_match
    klass = "ok" if match == "strong" else "muted" if match == "moderate" else "bad"
    parts = [
        f"<p>Voice match: <span class='{klass}' style='font-size:1.1em'>{html.escape(match)}</span></p>",
    ]
    badge = ""
    per_match = getattr(voice, "resume_match", None)
    if per_match:
        m_cls = "ok" if per_match == "strong" else "muted" if per_match == "moderate" else "bad"
        badge = f" <span class='{m_cls}'>[{html.escape(per_match)}]</span>"
    parts.append(f"<p><strong>Resume</strong>{badge}: {html.escape(voice.resume_assessment or '—')}</p>")
    if voice.specific_issues:
        parts.append("<h3>Issues</h3><ul>")
        for issue in voice.specific_issues:
            parts.append(f"<li>{html.escape(issue)}</li>")
        parts.append("</ul>")
    return "".join(parts)


def _ai_detection_summary(ai) -> str:
    if not ai:
        return "<p class='muted'>No AI detection review available.</p>"
    risk = ai.risk_level
    klass = "ok" if risk == "low" else "muted" if risk == "medium" else "bad"
    parts = [
        f"<p>AI-detection risk: <span class='{klass}' style='font-size:1.1em'>{html.escape(risk)}</span></p>",
    ]
    if ai.resume_flags:
        parts.append("<h3>Resume Flags</h3><ul>")
        for flag in ai.resume_flags:
            parts.append(f"<li>&ldquo;{html.escape(flag)}&rdquo;</li>")
        parts.append("</ul>")
    return "".join(parts)


def _hiring_manager_summary(hm) -> str:
    if not hm:
        return "<p class='muted'>No hiring-manager review available.</p>"
    pct = hm.advance_likelihood
    klass = "ok" if pct >= 70 else "bad" if pct < 40 else "muted"
    parts = [
        f"<p>Advance likelihood: <span class='{klass}' style='font-size:1.3em'>{pct}%</span></p>",
    ]
    if hm.summary:
        parts.append(f"<p>{html.escape(hm.summary)}</p>")
    if hm.strengths:
        parts.append("<h3>Strengths</h3><ul>")
        parts.extend(f"<li>{html.escape(s)}</li>" for s in hm.strengths)
        parts.append("</ul>")
    if hm.concerns:
        parts.append("<h3>Concerns</h3><ul>")
        parts.extend(f"<li>{html.escape(c)}</li>" for c in hm.concerns)
        parts.append("</ul>")
    if hm.improvements:
        parts.append("<h3>Suggested Improvements</h3><ul>")
        for imp in hm.improvements:
            impact_badge = {"high": "bad", "medium": "muted", "low": "muted"}[imp.impact]
            parts.append(
                f"<li><span class='{impact_badge}'>[{html.escape(imp.impact.upper())}]</span> "
                f"<strong>{html.escape(imp.area)}</strong>: {html.escape(imp.suggestion)}</li>"
            )
        parts.append("</ul>")
    return "".join(parts)


def _relevance_pruning_summary(pruning) -> str:
    if not pruning:
        return "<p class='muted'>No relevance-pruning review available.</p>"
    density = pruning.overall_density
    klass = "ok" if density == "lean" else "muted" if density == "balanced" else "bad"
    total = len(pruning.resume_issues)
    parts = [
        f"<p>Overall density: <span class='{klass}' style='font-size:1.1em'>{html.escape(density)}</span>"
        f" ({total} removal candidate{'s' if total != 1 else ''})</p>",
    ]
    if pruning.resume_issues:
        parts.append("<h3>Resume</h3><ul>")
        for issue in pruning.resume_issues:
            impact_badge = {"high": "bad", "medium": "muted", "low": "muted"}[issue.severity]
            parts.append(
                f"<li><span class='{impact_badge}'>[{html.escape(issue.severity.upper())}]</span> "
                f"<strong>{html.escape(issue.category)}</strong>: "
                f"&ldquo;{html.escape(issue.phrase[:120])}&rdquo; &mdash; {html.escape(issue.reason)}</li>"
            )
        parts.append("</ul>")
    return "".join(parts)


def _ats_keyword_summary(ats) -> str:
    if not ats:
        return "<p class='muted'>No ATS keyword review available.</p>"
    score = ats.alignment_score
    klass = "ok" if score == "strong" else "muted" if score == "moderate" else "bad"
    total_missing = len(ats.missing_keywords)
    total_stuffing = len(ats.stuffing_keywords)
    parts = [
        f"<p>Alignment: <span class='{klass}' style='font-size:1.1em'>{html.escape(score)}</span>"
        f" ({total_missing} missing, {total_stuffing} stuffing)</p>",
    ]
    if ats.missing_keywords:
        parts.append("<h3>Missing Keywords</h3><ul>")
        for kw in ats.missing_keywords:
            badge = "bad" if kw.priority == "high" else "muted"
            parts.append(
                f"<li><span class='{badge}'>[{html.escape(kw.priority.upper())}]</span> "
                f"<strong>{html.escape(kw.keyword)}</strong> &mdash; {html.escape(kw.suggestion)}"
                f" <em>({html.escape(kw.section)})</em></li>"
            )
        parts.append("</ul>")
    if ats.stuffing_keywords:
        parts.append("<h3>Keyword Stuffing</h3><ul>")
        for kw in ats.stuffing_keywords:
            parts.append(
                f"<li><strong>{html.escape(kw.keyword)}</strong> in {html.escape(kw.section)}"
                f" &mdash; {html.escape(kw.suggestion)}</li>"
            )
        parts.append("</ul>")
    return "".join(parts)


def _grammar_summary(grammar) -> str:
    if not grammar:
        return "<p class='muted'>No grammar review available.</p>"
    if grammar.clean:
        return "<p class='ok'>No grammar or mechanics issues found.</p>"
    total = len(grammar.resume_issues)
    parts = [f"<p class='bad'>{total} issue(s) found.</p>"]
    if grammar.resume_issues:
        parts.append("<h3>Resume</h3><ul>")
        for issue in grammar.resume_issues:
            badge = "bad" if issue.severity == "high" else "muted"
            parts.append(
                f"<li><span class='{badge}'>[{html.escape(issue.severity.upper())}]</span> "
                f"<strong>{html.escape(issue.category)}</strong>: "
                f"&ldquo;{html.escape(issue.phrase[:100])}&rdquo; &mdash; {html.escape(issue.issue)}"
                f" <em>Suggestion: {html.escape(issue.suggestion[:100])}</em></li>"
            )
        parts.append("</ul>")
    return "".join(parts)


def _narrative_coherence_summary(nc) -> str:
    if not nc:
        return "<p class='muted'>No narrative coherence review available.</p>"
    color = {"strong": "ok", "moderate": "muted", "weak": "bad"}[nc.alignment]
    total = len(nc.resume_issues)
    parts = [f"<p class='{color}'>Alignment: <strong>{html.escape(nc.alignment.upper())}</strong> ({total} issue(s))</p>"]
    if nc.resume_issues:
        parts.append("<ul>")
        for issue in nc.resume_issues:
            badge = "bad" if issue.severity == "high" else "muted"
            sug = f" <em>Suggestion: {html.escape(issue.suggestion[:100])}</em>" if issue.suggestion else ""
            parts.append(
                f"<li><span class='{badge}'>[{html.escape(issue.severity.upper())}]</span> "
                f"&ldquo;{html.escape(issue.phrase[:100])}&rdquo; &mdash; {html.escape(issue.issue)}"
                f"{sug}</li>"
            )
        parts.append("</ul>")
    return "".join(parts)


def _narrative_summary(narrative: CandidacyNarrative | None) -> str:
    """Render the candidacy narrative as an HTML card."""
    if not narrative:
        return "<p class='muted'>No candidacy narrative available.</p>"

    parts = ["<div class='card'><h2>Candidacy Narrative</h2>"]

    # Thesis
    if narrative.thesis:
        parts.append(f"<h3>Thesis</h3><p>{html.escape(narrative.thesis)}</p>")

    # Pillars
    if narrative.pillars:
        parts.append(f"<h3>Pillars ({len(narrative.pillars)})</h3>")
        for pillar in narrative.pillars:
            parts.append(f"<h4>{html.escape(pillar.theme)}</h4>")
            parts.append(f"<p>{html.escape(pillar.argument)}</p>")
            if pillar.career_evidence:
                parts.append("<ul>")
                for ev in pillar.career_evidence:
                    parts.append(f"<li>{html.escape(ev)}</li>")
                parts.append("</ul>")

    # Gap framing
    if narrative.gap_framing:
        parts.append("<h3>Gap Framing</h3><ul>")
        for gap in narrative.gap_framing:
            parts.append(f"<li>{html.escape(gap)}</li>")
        parts.append("</ul>")

    parts.append("</div>")
    return "".join(parts)


def _artifact_summary(context: DraftingContext | None) -> str:
    narrative = context.narrative if context else None
    style = context.voice_style_guide if context else None
    if not narrative and not style:
        return "<p class='muted'>No orchestration artifacts available.</p>"

    parts = ["<div class='grid'>"]
    if narrative:
        parts.append(_narrative_summary(narrative))
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


# ------------------------------------------------------------------
# Streaming progress page shared by generate / refine
# ------------------------------------------------------------------

_PROGRESS_PAGE_HEAD = (
    '<!doctype html><html lang="en"><head><meta charset="utf-8"/>'
    '<meta name="viewport" content="width=device-width,initial-scale=1"/>'
    "<title>Working\u2026 \u2014 Resume Refinery</title>"
    "<style>"
    ":root{--bg:#f7f5ef;--card:#fffdf8;--ink:#1f2421;"
    "--muted:#55605a;--accent:#0f7b6c;--line:#d8d4c8}"
    "body{margin:0;font-family:'Segoe UI',Tahoma,Geneva,Verdana,sans-serif;"
    "background:radial-gradient(circle at 0% 0%,#fffaf0 0%,var(--bg) 40%,#efeae0 100%);"
    "color:var(--ink)}"
    ".wrap{max-width:900px;margin:2rem auto;padding:0 1rem}"
    ".card{background:var(--card);border:1px solid var(--line);border-radius:14px;"
    "padding:1.25rem 1.5rem;box-shadow:0 8px 30px rgba(20,35,30,.06);margin-bottom:1rem}"
    "h2{margin:.2rem 0 .6rem}"
    "a{color:var(--accent);text-decoration:none}"
    ".muted{color:var(--muted)}"
    "#progress-log p{margin:.35rem 0;padding:.2rem 0}"
    "#progress-log p:last-child::after{"
    "content:'';display:inline-block;width:.85em;height:.85em;"
    "border:2px solid var(--line);border-top-color:var(--accent);"
    "border-radius:50%;animation:spin .6s linear infinite;"
    "margin-left:.6em;vertical-align:middle}"
    "#progress-log.done p:last-child::after{display:none}"
    "@keyframes spin{to{transform:rotate(360deg)}}"
    ".step-ok{color:var(--accent)}.step-ok::before{content:'\u2713 ';font-weight:bold}"
    ".step-fail{color:#b00020}.step-fail::before{content:'\u2717 ';font-weight:bold}"
    ".step-time{font-size:.82em;color:var(--muted);margin-left:.6em;font-weight:normal}"
    "#elapsed-clock{font-size:.92em;color:var(--muted);float:right;font-variant-numeric:tabular-nums}"
    "details{margin:.3rem 0 .5rem .8rem;border:1px solid var(--line);border-radius:8px;"
    "padding:.4rem .7rem;background:#f9f7f2}"
    "details summary{cursor:pointer;font-weight:600;color:var(--muted);font-size:.92em}"
    "details pre{white-space:pre-wrap;word-break:break-word;margin:.4rem 0 0;font-size:.88em;"
    "background:transparent;border:none;padding:0}"
    "</style></head><body><div class='wrap'>"
    "<p style='margin-bottom:.3rem'><a href='/'>&larr; Resume Refinery</a></p>"
    "<div class='card'><h2>Working\u2026 <span id='elapsed-clock'>0:00</span></h2>"
    "<div id='progress-log'>"
    "<script>(function(){var t0=Date.now(),el=document.getElementById('elapsed-clock');"
    "setInterval(function(){var s=Math.floor((Date.now()-t0)/1000);"
    "var m=Math.floor(s/60);s=s%60;"
    "el.textContent=m+':'+(s<10?'0':'')+s;},500);})();</script>"
)

# Padding so browsers flush the initial shell before the first LLM call blocks.
_BROWSER_FLUSH_PAD = "<!-- " + " " * 1024 + " -->\n"

# Regex to strip Rich markup tags like [green], [/green], [bold], etc.
_RICH_TAG_RE = re.compile(r"\[/?[a-z ]+\]")


def _strip_rich(text: str) -> str:
    """Remove Rich console markup tags, returning plain text."""
    return _RICH_TAG_RE.sub("", text)


def _is_detail_message(msg: str) -> bool:
    """Return True if this progress message is a multi-line detail block
    (review summary, repair edits, acceptances) that should be collapsed."""
    return "\n" in msg


def _progress_chunk(msg: str) -> str:
    """Convert an orchestrator progress message into an HTML chunk."""
    plain = _strip_rich(msg)

    if _is_detail_message(plain):
        # Multi-line detail: wrap in a collapsible <details> block
        lines = plain.split("\n")
        summary_text = html.escape(lines[0].strip())
        body_text = html.escape("\n".join(lines[1:]))
        return (
            f"<details><summary>{summary_text}</summary>"
            f"<pre>{body_text}</pre></details>\n"
        )

    # Heading-style lines (review pass separators)
    stripped = plain.strip()
    if stripped.startswith("───") or stripped.startswith("---"):
        return f"<p><strong>{html.escape(stripped)}</strong></p>\n"

    return f"<p>{html.escape(stripped)}</p>\n"


def _stream_orchestration(
    run_fn,
    redirect_url_fn,
    error_redirect: str = "/",
) -> StreamingResponse:
    """Run *run_fn* in a background thread, streaming progress HTML chunks.

    *run_fn* receives a ``progress`` callback and must return an
    ``OrchestrationResult``.  *redirect_url_fn* is called with the result
    to determine the auto-redirect URL.
    """
    q: queue.Queue[str | None] = queue.Queue()

    def _run() -> None:
        step_start = time.monotonic()

        def _timed_progress(msg: str) -> None:
            nonlocal step_start
            now = time.monotonic()
            elapsed = now - step_start
            step_start = now
            chunk = _progress_chunk(msg)
            badge = f"<span class='step-time'>({elapsed:.1f}s)</span>"
            # Inject badge before the closing tag of the first element
            if chunk.startswith("<details>"):
                chunk = chunk.replace("</summary>", f" {badge}</summary>", 1)
            elif chunk.startswith("<p>"):
                chunk = chunk.replace("</p>", f" {badge}</p>", 1)
            q.put(chunk)

        try:
            result = run_fn(progress=_timed_progress)
            sid = html.escape(result.session.session_id)
            url = redirect_url_fn(result)
            # Emit narrative summary inline in the streaming output
            if result.narrative:
                q.put(
                    "</div></div>"  # close progress-log and its card
                    + _narrative_summary(result.narrative)
                    + "<div class='card'><div id='progress-log'>"  # re-open for final messages
                )
            # Final success message + total elapsed time
            total = time.monotonic() - run_start
            total_min = int(total // 60)
            total_sec = total % 60
            total_str = f"{total_min}:{total_sec:04.1f}" if total_min else f"{total_sec:.1f}s"
            q.put(
                "<p class='step-ok' style='margin-top:.8rem;font-weight:600'>"
                f"Done in {html.escape(total_str)}!  Redirecting to <a href='{url}'>{sid}</a>\u2026</p>\n"
                "<script>document.getElementById('elapsed-clock').style.color='var(--accent)';</script>\n"
            )
            if result.strict_truth_failed:
                q.put(
                    "<p class='step-fail'>"
                    "Strict truth check failed \u2014 outputs saved for review.</p>\n"
                )
            q.put(
                f"<script>setTimeout(function(){{window.location.href='{url}';}},1500);</script>"
            )
        except Exception as exc:
            q.put(
                f"<p class='step-fail'>Error: {html.escape(str(exc))}</p>\n"
                f"<p><a href='{error_redirect}'>Back</a></p>\n"
            )
        finally:
            q.put(None)  # sentinel

    run_start = time.monotonic()

    def _generate() -> Iterator[str]:
        yield _PROGRESS_PAGE_HEAD + "\n" + _BROWSER_FLUSH_PAD
        threading.Thread(target=_run, daemon=True).start()
        while True:
            chunk = q.get()
            if chunk is None:
                break
            yield chunk
        yield "</div></div></div></body></html>"

    return StreamingResponse(_generate(), media_type="text/html")


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
  <p class=\"muted\">Local-only web app for tailored resume generation.</p>
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
    <label>Company Name (optional — overrides auto-extraction)</label>
    <input type=\"text\" name=\"company\" placeholder=\"e.g. Acme Corp\" />
    <label>Job Title (optional — overrides auto-extraction)</label>
    <input type=\"text\" name=\"title\" placeholder=\"e.g. Staff Engineer\" />
    <label>Output Directory</label>
    <div class="dir-picker-row">
      <input type="text" name="output_dir" id="output_dir_new" readonly required />
      <button type="button" onclick="openDirPicker('output_dir_new')">Browse…</button>
    </div>
    <label><input type=\"checkbox\" name=\"skip_review\" value=\"true\" /> Skip voice and AI style reviews</label>
    <label><input type=\"checkbox\" name=\"allow_unverified\" value=\"true\" /> Allow saving when strict truth check fails</label>

    <button type=\"submit\">Generate</button>
  </form>
</div>
"""
    return _page("Resume Refinery", body)


@app.post("/sessions/new")
async def create_session(
    job_description: UploadFile,
    career_profile: Optional[UploadFile] = None,
    voice_profile: Optional[UploadFile] = None,
    career_repo_id: Optional[str] = Form(None),
    company: Optional[str] = Form(None),
    title: Optional[str] = Form(None),
    output_dir: str = Form(...),
    skip_review: Optional[str] = Form(None),
    allow_unverified: Optional[str] = Form(None),
) -> StreamingResponse:
    # Validate output directory
    output_path = _validate_output_dir(output_dir)

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
        voice = parse_voice_profile_content(repo.voice_raw) if repo.voice.has_content() else None
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

    job = parse_job_description_content(
        job_text,
        company=company.strip() if company and company.strip() else None,
        title=title.strip() if title and title.strip() else None,
    )

    _skip = bool(skip_review)
    _allow = bool(allow_unverified)

    def _run(progress):
        session, context = orchestrator.extract_context(
            career, voice, job,
            progress=progress,
        )
        # Persist generation options for the generate route.
        opts_path = store.session_dir(session.session_id) / "staging_options.json"
        opts_path.write_text(_json.dumps({
            "output_dir": str(output_path),
            "skip_review": _skip,
            "allow_unverified": _allow,
        }), encoding="utf-8")
        return OrchestrationResult(
            session=session,
            documents=DocumentSet(),
            reviews=ReviewBundle(),
            narrative=context.narrative,
            voice_style_guide=context.voice_style_guide,
        )

    return _stream_orchestration(
        run_fn=_run,
        redirect_url_fn=lambda r: f"/sessions/{r.session.session_id}/curate",
        error_redirect="/",
    )


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


@app.get("/sessions/{session_id}/curate", response_class=HTMLResponse)
def review_narrative(session_id: str) -> HTMLResponse:
    """Render the narrative review page where users can review the candidacy narrative before generation."""
    session = store.get(session_id)
    context = store.load_staging_context(session)
    if context is None:
        # No staged context — session already generated; redirect to detail.
        return RedirectResponse(f"/sessions/{session_id}", status_code=303)

    narrative = context.narrative

    # Build narrative display
    thesis_html = f"<p>{html.escape(narrative.thesis)}</p>" if narrative.thesis else "<p class='muted'>No thesis generated.</p>"

    pillars_html = ""
    if narrative.pillars:
        for pillar in narrative.pillars:
            evidence_items = "".join(f"<li>{html.escape(ev)}</li>" for ev in pillar.career_evidence)
            pillars_html += (
                f"<div style='margin-bottom:.8rem'>"
                f"<h3>{html.escape(pillar.theme)}</h3>"
                f"<p>{html.escape(pillar.argument)}</p>"
                f"{'<ul>' + evidence_items + '</ul>' if evidence_items else ''}"
                f"</div>"
            )
    else:
        pillars_html = "<p class='muted'>No pillars generated.</p>"

    gap_items = ""
    if narrative.gap_framing:
        for gap in narrative.gap_framing:
            gap_items += f"<li>{html.escape(gap)}</li>"

    body = f"""
<div class="card">
  <h1>Review Narrative</h1>
  <p class="muted">{html.escape(session.job_description.title or '—')} @ {html.escape(session.job_description.company or '—')}</p>
  <p>Review the candidacy narrative below. This narrative will guide how your resume is framed.</p>
</div>
<div class="card">
  <h2>Thesis</h2>
  {thesis_html}
</div>
<div class="card">
  <h2>Pillars ({len(narrative.pillars)})</h2>
  {pillars_html}
</div>
{"<div class='card'><h2>Gap Framing</h2><ul>" + gap_items + "</ul></div>" if gap_items else ""}
<form method="post" action="/sessions/{html.escape(session_id)}/generate">
<div class="card" style="display:flex;gap:.8rem;justify-content:flex-end">
  <a href="/" style="padding:.65rem 1rem;color:var(--muted);text-decoration:none;font-weight:600">Cancel</a>
  <button type="submit">Generate Resume</button>
</div>
</form>
"""
    return _page("Review Narrative", body)


@app.post("/sessions/{session_id}/generate")
def generate_session(session_id: str):
    """Run document generation using the staged narrative context."""
    session = store.get(session_id)
    context = store.load_staging_context(session)
    if context is None:
        raise HTTPException(status_code=400, detail="No staged context — session may already be generated.")

    # Load generation options saved during extraction.
    opts_path = store.session_dir(session_id) / "staging_options.json"
    if not opts_path.exists():
        raise HTTPException(status_code=400, detail="Missing generation options. Please start a new session.")
    opts = _json.loads(opts_path.read_text(encoding="utf-8"))
    output_path = _validate_output_dir(opts["output_dir"])
    _skip = opts.get("skip_review", False)
    _allow = opts.get("allow_unverified", False)

    # NOTE: staging_options.json is cleaned up by orchestrator.clear_staging_context()
    # after successful generation, not here — so the narrative review page still works on error.

    return _stream_orchestration(
        run_fn=lambda progress: orchestrator.generate_session_run(
            session_id,
            context=context,
            output_dir=output_path,
            skip_review=_skip,
            allow_unverified=_allow,
            progress=progress,
        ),
        redirect_url_fn=lambda r: f"/sessions/{r.session.session_id}",
        error_redirect=f"/sessions/{html.escape(session_id)}/curate",
    )


@app.get("/sessions/{session_id}", response_class=HTMLResponse)
def show_session(session_id: str) -> HTMLResponse:
    session = store.get(session_id)
    docs = store.load_documents(session)
    reviews = store.load_reviews(session)
    context = store.load_context(session)

    def esc(text: str | None) -> str:
        return html.escape(text or "")

    body = f"""
<div class=\"card\">
  <h1>{html.escape(session.session_id)}</h1>
  <p class=\"muted\">{html.escape(session.job_description.title or '—')} @ {html.escape(session.job_description.company or '—')}</p>
  <p><a href=\"/sessions\">Back to sessions</a></p>
</div>
<div class=\"card\">
  <h2>Truthfulness</h2>
  {_truth_summary(reviews.truthfulness)}
</div>
<div class=\"card\">
  <h2>Voice Match</h2>
  {_voice_summary(reviews.voice)}
</div>
<div class=\"card\">
  <h2>AI Detection</h2>
  {_ai_detection_summary(reviews.ai_detection)}
</div>
<div class=\"card\">
  <h2>Hiring Manager Review</h2>
  {_hiring_manager_summary(reviews.hiring_manager)}
</div>
<div class=\"card\">
  <h2>Relevance Pruning</h2>
  {_relevance_pruning_summary(reviews.relevance_pruning)}
</div>
<div class=\"card\">
  <h2>ATS Keyword Alignment</h2>
  {_ats_keyword_summary(reviews.ats_keyword)}
</div>
<div class=\"card\">
  <h2>Grammar &amp; Mechanics</h2>
  {_grammar_summary(reviews.grammar)}
</div>
<div class=\"card\">
  <h2>Narrative Coherence</h2>
  {_narrative_coherence_summary(reviews.narrative_coherence)}
</div>
{_artifact_summary(context)}
<div class=\"card\">
  <h2>Refine</h2>
  <form method=\"post\" action=\"/sessions/{html.escape(session.session_id)}/refine\">
    <label>Feedback (free-form)</label>
    <textarea name=\"feedback\" rows=\"5\" required></textarea>
    <label>Output Directory</label>
    <div class="dir-picker-row">
      <input type="text" name="output_dir" id="output_dir_refine" readonly required />
      <button type="button" onclick="openDirPicker('output_dir_refine')">Browse…</button>
    </div>
    <label><input type=\"checkbox\" name=\"allow_unverified\" value=\"true\" /> Allow saving when strict truth check fails</label>
    <button type=\"submit\">Refine</button>
  </form>
</div>
{_doc_cards(docs, esc)}
"""
    return _page(session.session_id, body)


@app.post("/sessions/{session_id}/refine")
def refine_session(
    session_id: str,
    feedback: str = Form(...),
    output_dir: str = Form(...),
    allow_unverified: Optional[str] = Form(None),
):
    output_path = _validate_output_dir(output_dir)

    _allow = bool(allow_unverified)

    return _stream_orchestration(
        run_fn=lambda progress: orchestrator.refine_session_run(
            session_id,
            feedback,
            output_dir=output_path,
            allow_unverified=_allow,
            progress=progress,
        ),
        redirect_url_fn=lambda r: f"/sessions/{r.session.session_id}",
        error_redirect=f"/sessions/{html.escape(session_id)}",
    )


def main() -> None:
    uvicorn.run(
        "resume_refinery.webapp:app",
        host="127.0.0.1",
        port=8765,
        reload=False,
    )
