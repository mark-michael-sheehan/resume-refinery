"""Tests for Markdown → DOCX export."""

import pytest
from docx import Document

from resume_refinery.exporters import export_document_set, markdown_to_docx


def test_markdown_to_docx_creates_file(tmp_path):
    md = "# John Smith\n\nSome cover letter text.\n"
    out = tmp_path / "test.docx"
    markdown_to_docx(md, out)
    assert out.exists()


def test_docx_has_heading_for_h1(tmp_path):
    md = "# Jordan Lee\n\nSome text.\n"
    out = tmp_path / "resume.docx"
    markdown_to_docx(md, out)
    doc = Document(str(out))
    headings = [p for p in doc.paragraphs if p.style.name.startswith("Heading")]
    assert any("Jordan Lee" in p.text for p in headings)


def test_docx_has_bullet_list(tmp_path):
    md = "## Experience\n\n- Built something\n- Did another thing\n"
    out = tmp_path / "resume.docx"
    markdown_to_docx(md, out)
    doc = Document(str(out))
    bullet_paras = [p for p in doc.paragraphs if "List" in p.style.name]
    assert len(bullet_paras) >= 2


def test_docx_inline_bold(tmp_path):
    md = "I am a **bold statement** in text.\n"
    out = tmp_path / "test.docx"
    markdown_to_docx(md, out)
    doc = Document(str(out))
    # Find a run that is bold
    bold_runs = [
        run
        for para in doc.paragraphs
        for run in para.runs
        if run.bold and run.text.strip()
    ]
    assert any("bold statement" in r.text for r in bold_runs)


def test_export_document_set_writes_docx_files(tmp_path, document_set):
    written = export_document_set(document_set, tmp_path)
    assert "cover_letter" in written
    assert "resume" in written
    assert "interview_guide" in written
    assert written["cover_letter"].suffix == ".docx"
    assert written["cover_letter"].exists()


def test_export_document_set_partial(tmp_path):
    from resume_refinery.models import DocumentSet
    docs = DocumentSet(cover_letter="# Cover\n\nText.", resume=None, interview_guide=None)
    written = export_document_set(docs, tmp_path)
    assert "cover_letter" in written
    assert "resume" not in written


def test_markdown_to_docx_handles_horizontal_rule(tmp_path):
    md = "## Section\n\n---\n\nSome text after.\n"
    out = tmp_path / "test.docx"
    markdown_to_docx(md, out)  # Should not raise
    assert out.exists()
