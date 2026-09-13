"""
Rebuild talk decks WITHOUT breaking diagrams/arrows.

Root cause of blank pages / shifted arrows: earlier OPC zip merge remapped
prior slides onto Sep blank layouts.

Root cause of Sep2026 not opening in PowerPoint COM: slide 6 has float EMUs
(e.g. y="2788920.0") which desktop PowerPoint rejects.

This script:
1) Builds Wendy_Prior_Updates_styled.pptx via COM (native copy; text boxes only)
2) Writes a COM-openable COPY of Sep2026 with integer EMUs (original untouched)
3) Builds Wendy_Talk_Sep2026_full_trail.pptx via InsertFromFile (native copy)

NEVER modifies Wendy_Progress_Update_Sep2026.pptx.
"""

from __future__ import annotations

import re
import shutil
import zipfile
from pathlib import Path

import win32com.client

SLIDES = Path(r'c:\Users\Wendy\Desktop\RLHC\docs\Slides')
SEP = SLIDES / 'Wendy_Progress_Update_Sep2026.pptx'
MIDTERM = SLIDES / 'Wendy_Midterm_update.pptx'
PROGRESS = SLIDES / 'progress_update_deck (1).pptx'
PRIOR_OUT = SLIDES / 'Wendy_Prior_Updates_styled.pptx'
SEP_READY = SLIDES / '_sep2026_com_ready.pptx'
FULL_OUT = SLIDES / 'Wendy_Talk_Sep2026_full_trail.pptx'

COL_TITLE = (0x33, 0x1B, 0x0D)
COL_BODY = (0x5E, 0x46, 0x30)
COL_MUTED = (0x6B, 0x5A, 0x3D)
COL_ACCENT = (0x8C, 0x77, 0x53)
FONT_MAIN = 'Plus Jakarta Sans'


def rgb(r, g, b):
    return r + (g << 8) + (b << 16)


def restyle_text_boxes_only(slide) -> None:
    """Only free text boxes — never connectors, groups, pictures, tables."""
    for shape in slide.Shapes:
        try:
            if shape.Type != 17:  # msoTextBox
                continue
            if not (shape.HasTextFrame and shape.TextFrame.HasText):
                continue
            tr = shape.TextFrame.TextRange
            try:
                tr.Font.Name = FONT_MAIN
            except Exception:
                pass
            try:
                size = float(tr.Font.Size) if tr.Font.Size else 14
                if size >= 24:
                    tr.Font.Color.RGB = rgb(*COL_TITLE)
                elif size <= 12:
                    tr.Font.Color.RGB = rgb(*COL_MUTED)
                else:
                    tr.Font.Color.RGB = rgb(*COL_BODY)
            except Exception:
                pass
        except Exception:
            continue


def add_divider(pres, after_index: int, kicker: str, title: str, subtitle: str) -> None:
    master = pres.SlideMaster
    layout = None
    for i in range(1, master.CustomLayouts.Count + 1):
        cl = master.CustomLayouts(i)
        if 'Blank' in str(cl.Name):
            layout = cl
            break
    if layout is None:
        layout = master.CustomLayouts(1)
    slide = pres.Slides.AddSlide(after_index + 1, layout)
    for i in range(slide.Shapes.Count, 0, -1):
        try:
            if slide.Shapes(i).Type == 14:
                slide.Shapes(i).Delete()
        except Exception:
            pass
    bar = slide.Shapes.AddShape(5, 40, 80, 72, 10)
    bar.Fill.Solid()
    bar.Fill.ForeColor.RGB = rgb(*COL_ACCENT)
    bar.Line.Visible = 0
    for text, top, size, bold, color in [
        (kicker, 100, 12, True, COL_MUTED),
        (title, 140, 36, True, COL_TITLE),
        (subtitle, 230, 14, False, COL_BODY),
    ]:
        tb = slide.Shapes.AddTextbox(1, 40, top, 880, 80 if size >= 30 else 40)
        tb.TextFrame.TextRange.Text = text
        tb.TextFrame.TextRange.Font.Name = FONT_MAIN
        tb.TextFrame.TextRange.Font.Size = size
        tb.TextFrame.TextRange.Font.Bold = bold
        tb.TextFrame.TextRange.Font.Color.RGB = rgb(*color)


def build_prior(app) -> Path:
    if PRIOR_OUT.exists():
        PRIOR_OUT.unlink()
    shutil.copy2(PROGRESS, PRIOR_OUT)
    out = app.Presentations.Open(str(PRIOR_OUT.resolve()), False, False, True)
    try:
        mid = app.Presentations.Open(str(MIDTERM.resolve()), True, False, False)
        n_mid = mid.Slides.Count
        mid.Close()
        out.Slides.InsertFromFile(str(MIDTERM.resolve()), 0, 1, n_mid)
        print(f'Prior: {out.Slides.Count} slides after midterm insert')

        for i in range(1, out.Slides.Count + 1):
            restyle_text_boxes_only(out.Slides(i))

        add_divider(
            out, n_mid,
            'PRIOR UPDATE  ·  13 JUNE 2026', 'Progress update',
            'GPT-2 runs and the blocked path into the real Safe RLHF codebase.',
        )
        add_divider(
            out, 0,
            'PRIOR UPDATE  ·  APRIL 2026', 'Midterm',
            'Proposal framing and the first pilot number — kept for the record.',
        )
        out.Save()
        print(f'Saved {PRIOR_OUT.name} ({out.Slides.Count} slides)')
    finally:
        out.Close()
    return PRIOR_OUT


def make_sep_com_ready(src: Path, dest: Path) -> Path:
    """Copy of Sep2026 that desktop PowerPoint can open (float EMUs fixed)."""
    with zipfile.ZipFile(src) as zin:
        parts = {n: zin.read(n) for n in zin.namelist()}

    for name, data in list(parts.items()):
        if not name.endswith('.xml'):
            continue
        text = data.decode('utf-8')
        new = text.replace(
            "<?xml version='1.0' encoding='UTF-8' standalone='yes'?>",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        )
        if name == 'ppt/presentation.xml':
            new = new.replace('type="screen4x3"', 'type="screen16x9"')
        if re.fullmatch(r'ppt/slides/slide\d+\.xml', name):
            new, n = re.subn(
                r'( (?:x|y|cx|cy)=")(\d+\.\d+)(")',
                lambda m: f'{m.group(1)}{int(round(float(m.group(2))))}{m.group(3)}',
                new,
            )
            if n:
                print(f'Fixed {n} float EMUs in {name}')
        if new != text:
            parts[name] = new.encode('utf-8')

    if dest.exists():
        dest.unlink()
    with zipfile.ZipFile(dest, 'w', compression=zipfile.ZIP_DEFLATED) as zout:
        for name, data in parts.items():
            zout.writestr(name, data)
    print(f'COM-ready Sep copy -> {dest.name}')
    return dest


def build_full(app, prior: Path, sep_ready: Path) -> None:
    if FULL_OUT.exists():
        try:
            FULL_OUT.unlink()
        except PermissionError as exc:
            raise SystemExit(f'Close {FULL_OUT.name} in PowerPoint and retry') from exc
    shutil.copy2(sep_ready, FULL_OUT)

    out = app.Presentations.Open(str(FULL_OUT.resolve()), False, False, True)
    try:
        n_sep = out.Slides.Count
        print(f'Full base (Sep COM-ready) opens with {n_sep} slides')
        prior_pres = app.Presentations.Open(str(prior.resolve()), True, False, False)
        n_prior = prior_pres.Slides.Count
        prior_pres.Close()
        # Insert prior at front — native PowerPoint copy keeps arrows/layouts
        out.Slides.InsertFromFile(str(prior.resolve()), 0, 1, n_prior)
        print(f'Inserted {n_prior} prior slides; total={out.Slides.Count}')
        add_divider(
            out, n_prior,
            'TODAY  ·  8 SEPTEMBER 2026', 'This report',
            'Your Sep 2026 progress update — content unchanged.',
        )
        out.Save()
        print(f'Saved {FULL_OUT.name} ({out.Slides.Count} slides)')
    finally:
        out.Close()


def main():
    for req in (SEP, MIDTERM, PROGRESS):
        if not req.exists():
            raise SystemExit(f'Missing {req}')

    app = win32com.client.Dispatch('PowerPoint.Application')
    app.Visible = 1
    app.DisplayAlerts = 0
    try:
        prior = build_prior(app)
        sep_ready = make_sep_com_ready(SEP, SEP_READY)
        test = app.Presentations.Open(str(sep_ready.resolve()), True, False, True)
        print(f'Sep COM-ready opens: {test.Slides.Count} slides')
        # quick sanity: slide 6 should exist and have shapes
        s6 = test.Slides(6)
        print(f'Slide 6 shapes={s6.Shapes.Count}')
        test.Close()
        build_full(app, prior, sep_ready)
        print(f'Original Sep untouched: {SEP.name} size={SEP.stat().st_size}')
    finally:
        try:
            app.Quit()
        except Exception:
            pass


if __name__ == '__main__':
    main()
