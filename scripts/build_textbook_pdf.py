"""Build a printable PDF of the Docsify textbook chapters.

Usage:
  python scripts/build_textbook_pdf.py

Outputs:
  docs/worklog/textbook/Safe-RL-MinMax-Textbook.pdf
  docs/worklog/textbook/Safe-RL-MinMax-Textbook.html  (browser → Print → PDF fallback)
"""

from __future__ import annotations

import html
import re
from pathlib import Path

import markdown
from fpdf import FPDF
from fpdf.enums import XPos, YPos

ROOT = Path(__file__).resolve().parents[1]
TEXTBOOK = ROOT / 'docs' / 'worklog' / 'textbook'
OUT_PDF = TEXTBOOK / 'Safe-RL-MinMax-Textbook.pdf'
OUT_HTML = TEXTBOOK / 'Safe-RL-MinMax-Textbook.html'

CHAPTERS = [
    'README.md',
    '01-transformers.md',
    '02-preferences-and-scores.md',
    '03-reward-vs-cost.md',
    '04-rlhf-ppo.md',
    '05-safe-rlhf.md',
    '06-minmax.md',
    '07-abc-design.md',
    '08-failure-modes.md',
    '09-reading-results.md',
    '10-papers.md',
]


def load_chapters() -> str:
    parts = []
    for name in CHAPTERS:
        path = TEXTBOOK / name
        if not path.exists():
            raise SystemExit(f'Missing {path}')
        text = path.read_text(encoding='utf-8')
        # Drop Docsify-relative nav footers clutter
        text = re.sub(r'\n\*\*Next:\*\*.*$', '', text, flags=re.M)
        text = re.sub(r'\n\*\*Back to:\*\*.*$', '', text, flags=re.M)
        # Mermaid → note
        text = re.sub(
            r'```mermaid\n.*?```',
            '\n*[Diagram omitted in PDF — see Docsify site.]*\n',
            text,
            flags=re.S,
        )
        # Docsify /worklog/ links → plain text labels
        text = re.sub(r'\[([^\]]+)\]\(/worklog/[^)]+\)', r'\1', text)
        text = re.sub(r'\[([^\]]+)\]\((?:\.\./)+assets/[^)]+\)', r'\1', text)
        # Callout divs → blockquotes
        text = re.sub(
            r'<div class="finding[^"]*">\s*<span class="label">([^<]+)</span>\s*',
            r'> **\1.** ',
            text,
        )
        text = text.replace('</div>', '\n')
        parts.append(text.strip())
    return '\n\n\\newpage\n\n'.join(parts)


def md_to_html_doc(body_md: str) -> str:
    body_md = body_md.replace('\\newpage', '<hr class="pagebreak" />')
    body = markdown.markdown(
        body_md,
        extensions=['tables', 'fenced_code', 'nl2br', 'sane_lists'],
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Safe RL + MinMax — Textbook</title>
<style>
  @page {{ margin: 18mm; }}
  body {{ font-family: Georgia, 'Times New Roman', serif; font-size: 11pt; line-height: 1.45;
         color: #331B0D; max-width: 720px; margin: 24px auto; padding: 0 16px; }}
  h1 {{ font-size: 20pt; border-bottom: 2px solid #8C7753; padding-bottom: 6px; page-break-before: auto; }}
  h2 {{ font-size: 14pt; color: #5E4630; margin-top: 1.4em; }}
  h3 {{ font-size: 12pt; color: #6B5A3D; }}
  code, pre {{ font-family: Consolas, monospace; font-size: 9.5pt; }}
  pre {{ background: #F7F3EC; padding: 10px; overflow-x: auto; }}
  table {{ border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 10pt; }}
  th, td {{ border: 1px solid #D2C2A0; padding: 6px 8px; vertical-align: top; }}
  th {{ background: #F0E8DA; }}
  blockquote {{ border-left: 3px solid #8C7753; margin: 12px 0; padding: 6px 12px; color: #5E4630; }}
  hr.pagebreak {{ border: none; border-top: 1px dashed #C4B49A; margin: 28px 0; page-break-after: always; }}
  a {{ color: #5E4630; }}
  em {{ color: #6B5A3D; }}
</style>
</head>
<body>
<h1>Safe Reinforcement Learning with a MinMax Penalty</h1>
<p><strong>Textbook</strong> — concepts behind the Phase 1 / Phase 2 worklogs.<br/>
Wendy Maboa · RLHC · generated from <code>docs/worklog/textbook/</code></p>
{body}
</body>
</html>
"""


class TextbookPDF(FPDF):
    def header(self):
        self.set_font('Body', 'I', 8)
        self.set_text_color(108, 90, 61)
        self.cell(0, 8, 'Safe RL + MinMax — Textbook', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(2)

    def footer(self):
        self.set_y(-12)
        self.set_font('Body', 'I', 8)
        self.set_text_color(156, 138, 114)
        self.cell(0, 8, f'{self.page_no()}', align='C')


WIN_FONTS = Path(r'C:\Windows\Fonts')


def _strip_md_inline(s: str) -> str:
    s = re.sub(r'\*\*([^*]+)\*\*', r'\1', s)
    s = re.sub(r'\*([^*]+)\*', r'\1', s)
    s = re.sub(r'`([^`]+)`', r'\1', s)
    s = re.sub(r'\$\$[^$]+\$\$', '[equation]', s)
    s = re.sub(r'\$([^$]+)\$', r'\1', s)
    s = s.replace('\\(', '').replace('\\)', '')
    s = s.replace('\\[', '').replace('\\]', '')
    s = re.sub(
        r'\\(?:text|mathrm|mid|le|ge|approx|to|leftarrow|rightarrow|big|cases|'
        r'begin|end|sigma|pi|theta|phi|beta|tilde|hat|quad)\{?([^}]*)\}?',
        r'\1',
        s,
    )
    repl = {
        '—': '-', '–': '-', '→': '->', '←': '<-', '≈': '~=', '≥': '>=', '≤': '<=',
        '×': 'x', '·': '-', '☐': '[ ]', '✓': '[x]', '≃': '~', '≫': '>>',
        '“': '"', '”': '"', '‘': "'", '’': "'", ' ': ' ',
    }
    for a, b in repl.items():
        s = s.replace(a, b)
    # drop remaining non-latin1 for core safety with Arial subset issues
    return s.encode('latin-1', errors='replace').decode('latin-1')


def render_pdf(md: str, path: Path) -> None:
    pdf = TextbookPDF(format='A4')
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.add_font('Body', '', str(WIN_FONTS / 'arial.ttf'))
    pdf.add_font('Body', 'B', str(WIN_FONTS / 'arialbd.ttf'))
    pdf.add_font('Body', 'I', str(WIN_FONTS / 'ariali.ttf'))
    pdf.add_font('Body', 'BI', str(WIN_FONTS / 'arialbi.ttf'))
    pdf.add_page()

    def write(text: str, size: int = 10, style: str = '', color=(51, 27, 13), lh: float | None = None):
        pdf.set_font('Body', style, size)
        pdf.set_text_color(*color)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(
            pdf.epw,
            lh or (size * 0.55),
            text,
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT,
        )

    write('Safe RL + MinMax', size=18, style='B', lh=10)
    write('Textbook - concepts behind the Phase 1 / Phase 2 worklogs', size=12, lh=7)
    pdf.ln(4)

    in_code = False
    for raw_line in md.splitlines():
        line = raw_line.rstrip()
        if line.strip().startswith('```'):
            in_code = not in_code
            continue
        if in_code:
            write(_strip_md_inline(line), size=9, color=(70, 70, 70), lh=5)
            continue
        if line.strip() == '\\newpage':
            pdf.add_page()
            continue
        if not line.strip():
            pdf.ln(3)
            continue
        if line.startswith('# '):
            pdf.ln(4)
            write(_strip_md_inline(line[2:]), size=16, style='B', lh=9)
            pdf.ln(2)
            continue
        if line.startswith('## '):
            pdf.ln(3)
            write(_strip_md_inline(line[3:]), size=13, style='B', color=(94, 70, 48), lh=8)
            pdf.ln(1)
            continue
        if line.startswith('### '):
            write(_strip_md_inline(line[4:]), size=11, style='B', color=(107, 90, 61), lh=7)
            continue
        if line.startswith('> '):
            write(_strip_md_inline(line[2:]), size=10, style='I', color=(94, 70, 48), lh=6)
            continue
        if line.startswith('|') and '|' in line[1:]:
            cells = [c.strip() for c in line.strip('|').split('|')]
            if all(set(c) <= set('-: ') for c in cells):
                continue
            write(' | '.join(_strip_md_inline(c) for c in cells), size=9, lh=5)
            continue
        if line.startswith('- [ ]') or line.startswith('- [x]'):
            mark = '[x]' if line.startswith('- [x]') else '[ ]'
            write(mark + ' ' + _strip_md_inline(line[5:].strip()), size=10, lh=6)
            continue
        if line.startswith('- ') or line.startswith('* '):
            write('- ' + _strip_md_inline(line[2:]), size=10, lh=6)
            continue
        if re.match(r'^\d+\.\s', line):
            write(_strip_md_inline(line), size=10, lh=6)
            continue
        write(_strip_md_inline(line), size=10, lh=6)

    pdf.output(str(path))


def main() -> None:
    md = load_chapters()
    html_doc = md_to_html_doc(md)
    OUT_HTML.write_text(html_doc, encoding='utf-8')
    print(f'Wrote {OUT_HTML.relative_to(ROOT)}')
    try:
        render_pdf(md, OUT_PDF)
        print(f'Wrote {OUT_PDF.relative_to(ROOT)}')
    except Exception as exc:
        print(f'PDF engine failed ({exc}). Open the HTML and use Print -> Save as PDF:')
        print(f'  {OUT_HTML}')


if __name__ == '__main__':
    main()
