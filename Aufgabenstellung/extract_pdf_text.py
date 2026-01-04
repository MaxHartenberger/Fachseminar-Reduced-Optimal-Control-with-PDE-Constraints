from pathlib import Path
import sys

try:
    from PyPDF2 import PdfReader
except Exception as e:
    print('PyPDF2 import failed:', e)
    raise

root = Path(__file__).parent
pdfs = [root / 'Ex1.pdf', root / 'Ex2.pdf', root / 'Ex3.pdf']

for pdf in pdfs:
    if not pdf.exists():
        print(f'PDF not found: {pdf}')
        continue
    print(f'Reading {pdf} ...')
    try:
        reader = PdfReader(str(pdf))
    except Exception as e:
        print(f'Failed to open {pdf}:', e)
        continue
    texts = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text()
            if text is None:
                text = ''
            texts.append(text)
        except Exception as e:
            print(f'Failed to extract page {i} of {pdf}:', e)
            texts.append('')
    out = root / (pdf.stem + '.txt')
    try:
        out.write_text('\n\n'.join(texts), encoding='utf-8')
        print(f'Wrote {out} (length {out.stat().st_size} bytes)')
    except Exception as e:
        print(f'Failed to write {out}:', e)

print('done')
