"""Repair a COPY of Sep2026 for PowerPoint COM, then build the full trail deck.

NEVER modifies Wendy_Progress_Update_Sep2026.pptx.
"""

from __future__ import annotations

import io
import shutil
import zipfile
from pathlib import Path

import win32com.client

SLIDES = Path(r'c:\Users\Wendy\Desktop\RLHC\docs\Slides')
SEP2026 = SLIDES / 'Wendy_Progress_Update_Sep2026.pptx'
MIDTERM = SLIDES / 'Wendy_Midterm_update.pptx'
PROGRESS = SLIDES / 'progress_update_deck (1).pptx'
SEP_REPAIRED = SLIDES / '_sep2026_for_combine.pptx'
OUT = SLIDES / 'Wendy_Talk_Sep2026_full_trail.pptx'

COL_TITLE = (0x33, 0x1B, 0x0D)
COL_BODY = (0x5E, 0x46, 0x30)
COL_MUTED = (0x6B, 0x5A, 0x3D)
COL_FAINT = (0x9C, 0x8A, 0x72)
COL_ACCENT = (0x8C, 0x77, 0x53)
FONT_MAIN = 'Plus Jakarta Sans'
FONT_META = 'Consolas'


def rgb(r: int, g: int, b: int) -> int:
    return r + (g << 8) + (b << 16)


def repair_sep2026_copy() -> Path:
    """Add missing sldSz/notesSz so desktop PowerPoint will open a working copy."""
    if SEP_REPAIRED.exists():
        SEP_REPAIRED.unlink()

    buf = io.BytesIO()
    with zipfile.ZipFile(SEP2026, 'r') as zin, zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == 'ppt/presentation.xml':
                text = data.decode('utf-8')
                if '<p:sldSz' not in text:
                    insert = (
                        '<p:sldSz cx="12191695" cy="6858000"/>'
                        '<p:notesSz cx="6858000" cy="9144000"/>'
                    )
                    # Place sizes before defaultTextStyle (or before closing tag)
                    if '<p:defaultTextStyle' in text:
                        text = text.replace('<p:defaultTextStyle', insert + '<p:defaultTextStyle', 1)
                    else:
                        text = text.replace('</p:presentation>', insert + '</p:presentation>', 1)
                    # Normalize XML declaration quotes (some hosts dislike single quotes)
                    text = text.replace("<?xml version='1.0' encoding='UTF-8' standalone='yes'?>",
                                        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>')
                    data = text.encode('utf-8')
            zout.writestr(item, data)
    SEP_REPAIRED.write_bytes(buf.getvalue())
    print(f'Repaired copy written: {SEP_REPAIRED.name} (original untouched)')
    return SEP_REPAIRED


def restyle_slide(slide) -> None:
    for shape in slide.Shapes:
        try:
            if shape.HasTextFrame and shape.TextFrame.HasText:
                tr = shape.TextFrame.TextRange
                try:
                    tr.Font.Name = FONT_MAIN
                except Exception:
                    pass
                try:
                    size = float(tr.Font.Size) if tr.Font.Size else 14
                    if size >= 24:
                        tr.Font.Color.RGB = rgb(*COL_TITLE)
                    elif size <= 11:
                        tr.Font.Color.RGB = rgb(*COL_FAINT)
                        tr.Font.Name = FONT_META
                    else:
                        tr.Font.Color.RGB = rgb(*COL_BODY)
                except Exception:
                    pass
        except Exception:
            continue


def add_section_divider(pres, after_index: int, kicker: str, title: str, subtitle: str) -> None:
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

    tb_k = slide.Shapes.AddTextbox(1, 40, 100, 800, 30)
    tb_k.TextFrame.TextRange.Text = kicker
    tb_k.TextFrame.TextRange.Font.Name = FONT_MAIN
    tb_k.TextFrame.TextRange.Font.Size = 12
    tb_k.TextFrame.TextRange.Font.Bold = True
    tb_k.TextFrame.TextRange.Font.Color.RGB = rgb(*COL_MUTED)

    tb_t = slide.Shapes.AddTextbox(1, 40, 140, 860, 80)
    tb_t.TextFrame.TextRange.Text = title
    tb_t.TextFrame.TextRange.Font.Name = FONT_MAIN
    tb_t.TextFrame.TextRange.Font.Size = 36
    tb_t.TextFrame.TextRange.Font.Bold = True
    tb_t.TextFrame.TextRange.Font.Color.RGB = rgb(*COL_TITLE)

    tb_s = slide.Shapes.AddTextbox(1, 40, 230, 860, 60)
    tb_s.TextFrame.TextRange.Text = subtitle
    tb_s.TextFrame.TextRange.Font.Name = FONT_MAIN
    tb_s.TextFrame.TextRange.Font.Size = 14
    tb_s.TextFrame.TextRange.Font.Color.RGB = rgb(*COL_BODY)


def main() -> None:
    repaired = repair_sep2026_copy()

    # Verify COM can open repaired copy
    app = win32com.client.Dispatch('PowerPoint.Application')
    app.Visible = 1
    app.DisplayAlerts = 0
    try:
        test = app.Presentations.Open(str(repaired.resolve()), True, False, True)
        print(f'Repaired opens in PowerPoint: {test.Slides.Count} slides')
        test.Close()
    except Exception as e:
        app.Quit()
        raise SystemExit(f'Repaired Sep2026 still will not open in PowerPoint: {e}')

    if OUT.exists():
        try:
            OUT.unlink()
        except PermissionError:
            pass
    shutil.copy2(repaired, OUT)

    out = app.Presentations.Open(str(OUT.resolve()), False, False, True)
    try:
        n_sep = out.Slides.Count
        print(f'Base deck has {n_sep} Sep2026 slides')

        mid = app.Presentations.Open(str(MIDTERM.resolve()), True, False, False)
        n_mid = mid.Slides.Count
        mid.Close()
        out.Slides.InsertFromFile(str(MIDTERM.resolve()), 0, 1, n_mid)
        print(f'Inserted midterm ({n_mid})')

        prog = app.Presentations.Open(str(PROGRESS.resolve()), True, False, False)
        n_prog = prog.Slides.Count
        prog.Close()
        out.Slides.InsertFromFile(str(PROGRESS.resolve()), n_mid, 1, n_prog)
        print(f'Inserted June progress ({n_prog})')

        n_prior = n_mid + n_prog
        for i in range(1, n_prior + 1):
            restyle_slide(out.Slides(i))
        print(f'Restyled prior slides 1–{n_prior} only')

        # Dividers from back to front so indices stay valid
        add_section_divider(
            out, n_prior,
            'TODAY  ·  8 SEPTEMBER 2026',
            'This report',
            'Sep 2026 progress update — content unchanged from your edited deck.',
        )
        add_section_divider(
            out, n_mid,
            'PRIOR UPDATE  ·  13 JUNE 2026',
            'Progress update',
            'GPT-2 runs and the blocked path into the real Safe RLHF codebase.',
        )
        add_section_divider(
            out, 0,
            'PRIOR UPDATE  ·  APRIL 2026',
            'Midterm',
            'Proposal framing and the first pilot number — kept for the record.',
        )

        out.Save()
        total = out.Slides.Count
        print(f'Saved {OUT.name} with {total} slides')
        print(f'Original unchanged: {SEP2026.name}')
    finally:
        try:
            out.Close()
        except Exception:
            pass
        try:
            app.Quit()
        except Exception:
            pass


if __name__ == '__main__':
    main()
