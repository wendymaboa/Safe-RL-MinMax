"""Create styled prior-updates deck (midterm + June) via PowerPoint COM."""

from pathlib import Path
import win32com.client

SLIDES = Path(r'c:\Users\Wendy\Desktop\RLHC\docs\Slides')
MIDTERM = SLIDES / 'Wendy_Midterm_update.pptx'
PROGRESS = SLIDES / 'progress_update_deck (1).pptx'
OUT = SLIDES / 'Wendy_Prior_Updates_styled.pptx'

COL_TITLE = (0x33, 0x1B, 0x0D)
COL_BODY = (0x5E, 0x46, 0x30)
COL_MUTED = (0x6B, 0x5A, 0x3D)
COL_FAINT = (0x9C, 0x8A, 0x72)
COL_ACCENT = (0x8C, 0x77, 0x53)
FONT_MAIN = 'Plus Jakarta Sans'
FONT_META = 'Consolas'


def rgb(r, g, b):
    return r + (g << 8) + (b << 16)


def restyle_slide(slide):
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


def add_divider(pres, after_index, kicker, title, subtitle):
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
        tb = slide.Shapes.AddTextbox(1, 40, top, 860, 80 if size >= 30 else 40)
        tb.TextFrame.TextRange.Text = text
        tb.TextFrame.TextRange.Font.Name = FONT_MAIN
        tb.TextFrame.TextRange.Font.Size = size
        tb.TextFrame.TextRange.Font.Bold = bold
        tb.TextFrame.TextRange.Font.Color.RGB = rgb(*color)


def main():
    if OUT.exists():
        OUT.unlink()

    app = win32com.client.Dispatch('PowerPoint.Application')
    app.Visible = 1
    app.DisplayAlerts = 0

    # Start from June progress (same 16:9-ish size as Sep2026)
    shutil_copy = __import__('shutil').copy2
    shutil_copy(PROGRESS, OUT)

    out = app.Presentations.Open(str(OUT.resolve()), False, False, True)
    try:
        # Clear progress slides temporarily? Better: insert midterm at 0, keep progress
        # Current OUT is progress copy — insert midterm at front
        mid = app.Presentations.Open(str(MIDTERM.resolve()), True, False, False)
        n_mid = mid.Slides.Count
        mid.Close()
        out.Slides.InsertFromFile(str(MIDTERM.resolve()), 0, 1, n_mid)
        n_prog = out.Slides.Count - n_mid
        print(f'Prior deck: {n_mid} midterm + {n_prog} progress = {out.Slides.Count}')

        for i in range(1, out.Slides.Count + 1):
            restyle_slide(out.Slides(i))

        add_divider(out, n_mid, 'PRIOR UPDATE  ·  13 JUNE 2026', 'Progress update',
                    'GPT-2 runs and the blocked path into the real Safe RLHF codebase.')
        add_divider(out, 0, 'PRIOR UPDATE  ·  APRIL 2026', 'Midterm',
                    'Proposal framing and the first pilot number — kept for the record.')
        add_divider(out, out.Slides.Count,
                    'NEXT  ·  8 SEPTEMBER 2026', 'This report continues →',
                    'Open Wendy_Progress_Update_Sep2026.pptx next (or use the full-trail file if present).')

        out.Save()
        print(f'Saved {OUT} ({out.Slides.Count} slides)')
    finally:
        out.Close()
        app.Quit()


if __name__ == '__main__':
    main()
