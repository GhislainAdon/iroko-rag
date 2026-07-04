#!/usr/bin/env python3
"""Universal document ingestion for PageIndex.

Routes any document to the right pipeline before building the tree:

    .pdf  (text layer)   -> native PDF pipeline (page_index_main)
    .pdf  (scanned)      -> OCR fallback: ocrmypdf if available (keeps page
                            numbers), else Tesseract via pytesseract -> Markdown
    .md / .markdown      -> native Markdown pipeline (md_to_tree)
    .docx .odt .rtf .epub .html .tex .rst .org
                         -> Pandoc -> Markdown (heading styles become #/##)
    .pptx .xlsx .csv .msg ... and anything else
                         -> MarkItDown -> Markdown

Usage:
    python ingest.py --input report.docx
    python ingest.py --input slides.pptx --convert-only
    python ingest.py --input scanned.pdf --ocr-lang fra
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

# Formats where Pandoc gives the best structure (real heading levels).
PANDOC_EXTS = {'.docx', '.odt', '.rtf', '.epub', '.html', '.htm',
               '.tex', '.rst', '.org', '.ipynb'}
# Formats MarkItDown handles better (or Pandoc not at all).
MARKITDOWN_EXTS = {'.pptx', '.ppt', '.xlsx', '.xls', '.csv', '.tsv',
                   '.json', '.xml', '.msg', '.eml', '.zip', '.wav', '.mp3'}
MARKDOWN_EXTS = {'.md', '.markdown'}

# Below this average of extracted characters per page, a PDF is
# considered scanned (image-only) and goes through OCR.
SCANNED_CHARS_PER_PAGE = 25


def which_or_die(tool, hint):
    if shutil.which(tool) is None:
        sys.exit(f"error: '{tool}' not found on PATH. {hint}")
    return tool


def convert_with_pandoc(input_path, md_path):
    which_or_die('pandoc', 'Install it from https://pandoc.org/installing.html')
    subprocess.run(
        ['pandoc', input_path, '-t', 'gfm', '--wrap=none', '-o', md_path],
        check=True)


def convert_with_markitdown(input_path, md_path):
    which_or_die('markitdown', "Install it with 'pip install markitdown[all]'")
    with open(md_path, 'w', encoding='utf-8') as out:
        subprocess.run(['markitdown', input_path], check=True, stdout=out)


def pdf_is_scanned(pdf_path):
    """True when the PDF has (almost) no extractable text layer."""
    import pymupdf
    with pymupdf.open(pdf_path) as doc:
        if doc.page_count == 0:
            return False
        chars = sum(len(page.get_text()) for page in doc)
        return chars / doc.page_count < SCANNED_CHARS_PER_PAGE


def ocr_pdf_to_searchable(pdf_path, out_pdf_path, lang):
    """OCR with ocrmypdf: output is a PDF with a text layer, so the native
    page-based pipeline (and its page numbers) still applies."""
    subprocess.run(
        ['ocrmypdf', '--language', lang, '--skip-text', pdf_path, out_pdf_path],
        check=True)


def ocr_pdf_to_markdown(pdf_path, md_path, lang):
    """Fallback OCR with Tesseract only: renders each page and emits one
    Markdown file. Page numbers are lost; the Markdown pipeline is used."""
    which_or_die('tesseract',
                 'Install it (winget/apt/brew install tesseract) '
                 "or 'pip install ocrmypdf' for the better path.")
    import pymupdf
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        sys.exit("error: OCR fallback needs 'pip install pytesseract pillow'.")

    import io
    title = os.path.splitext(os.path.basename(pdf_path))[0]
    parts = [f'# {title}\n']
    with pymupdf.open(pdf_path) as doc:
        for i, page in enumerate(doc, start=1):
            pix = page.get_pixmap(matrix=pymupdf.Matrix(3, 3))
            img = Image.open(io.BytesIO(pix.tobytes('png')))
            text = pytesseract.image_to_string(img, lang=lang)
            parts.append(f'<!-- page {i} -->\n{text.strip()}\n')
            print(f'  OCR page {i}/{doc.page_count}', file=sys.stderr)
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(parts))


def ensure_headers(md_path, fallback_title):
    """md_to_tree builds the tree from '#' headers. If the conversion produced
    none (e.g. a docx without heading styles), wrap the document under a
    single root header so the pipeline still yields a usable node.

    Header detection mirrors pageindex.page_index_md.extract_nodes_from_markdown
    but stays dependency-free so --convert-only works without the package's
    requirements installed."""
    import re
    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()
    has_header = False
    in_code_block = False
    for line in content.split('\n'):
        stripped = line.strip()
        if stripped.startswith('```'):
            in_code_block = not in_code_block
            continue
        if not in_code_block and re.match(r'^#{1,6}\s+.+$', stripped):
            has_header = True
            break
    if not has_header:
        print('warning: no headings found in converted Markdown; '
              'wrapping content under a single root node', file=sys.stderr)
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(f'# {fallback_title}\n\n{content}')


def run_pdf_pipeline(pdf_path, opt_overrides):
    from pageindex import page_index_main
    from pageindex.utils import ConfigLoader
    opt = ConfigLoader().load({k: v for k, v in opt_overrides.items()
                               if v is not None})
    return page_index_main(pdf_path, opt)


def run_md_pipeline(md_path, opt_overrides):
    import asyncio
    from pageindex.page_index_md import md_to_tree
    from pageindex.utils import ConfigLoader
    opt = ConfigLoader().load({k: v for k, v in opt_overrides.items()
                               if v is not None})
    return asyncio.run(md_to_tree(
        md_path=md_path,
        if_add_node_summary=opt.if_add_node_summary,
        model=opt.model,
        if_add_doc_description=opt.if_add_doc_description,
        if_add_node_text=opt.if_add_node_text,
        if_add_node_id=opt.if_add_node_id,
    ))


def save_result(tree, input_path, output_dir):
    name = os.path.splitext(os.path.basename(input_path))[0]
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'{name}_structure.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(tree, f, indent=2, ensure_ascii=False)
    print(f'Tree structure saved to: {output_file}')


def main():
    parser = argparse.ArgumentParser(
        description='Ingest any document format into a PageIndex tree')
    parser.add_argument('--input', required=True, help='Path to the document')
    parser.add_argument('--model', type=str, default=None,
                        help='LLM model (overrides config.yaml)')
    parser.add_argument('--ocr-lang', type=str, default='eng',
                        help="Tesseract language(s), e.g. 'fra' or 'fra+eng'")
    parser.add_argument('--convert-only', action='store_true',
                        help='Stop after conversion; print the Markdown path '
                             'instead of building the tree (no LLM calls)')
    parser.add_argument('--output-dir', type=str, default='./results')
    parser.add_argument('--keep-md', action='store_true',
                        help='Keep the intermediate Markdown next to the input')
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    if not os.path.isfile(input_path):
        sys.exit(f'error: file not found: {input_path}')
    ext = os.path.splitext(input_path)[1].lower()
    base = os.path.splitext(os.path.basename(input_path))[0]
    opt_overrides = {'model': args.model}

    tmp_dir = tempfile.mkdtemp(prefix='pageindex_ingest_')
    md_dir = os.path.dirname(input_path) if args.keep_md or args.convert_only \
        else tmp_dir
    md_path = os.path.join(md_dir, f'{base}.md')

    try:
        # --- native PDF, or scanned PDF via OCR ---------------------------
        if ext == '.pdf':
            if not pdf_is_scanned(input_path):
                print('PDF with text layer: native PageIndex pipeline')
                if args.convert_only:
                    sys.exit('error: --convert-only does not apply to '
                             'text-layer PDFs (no conversion involved)')
                save_result(run_pdf_pipeline(input_path, opt_overrides),
                            input_path, args.output_dir)
                return
            print('Scanned PDF detected (no text layer): OCR required')
            if shutil.which('ocrmypdf'):
                searchable = os.path.join(tmp_dir, f'{base}_ocr.pdf')
                print('Using ocrmypdf (page numbers preserved)')
                ocr_pdf_to_searchable(input_path, searchable, args.ocr_lang)
                if args.convert_only:
                    kept = os.path.join(os.path.dirname(input_path),
                                        f'{base}_ocr.pdf')
                    shutil.copy(searchable, kept)
                    print(f'Searchable PDF written to: {kept}')
                    return
                save_result(run_pdf_pipeline(searchable, opt_overrides),
                            input_path, args.output_dir)
                return
            print('ocrmypdf not found: falling back to Tesseract -> Markdown '
                  '(page numbers are lost)')
            ocr_pdf_to_markdown(input_path, md_path, args.ocr_lang)

        # --- already Markdown ---------------------------------------------
        elif ext in MARKDOWN_EXTS:
            md_path = input_path

        # --- office & text formats -> Markdown -----------------------------
        elif ext in PANDOC_EXTS:
            print(f'Converting {ext} with Pandoc...')
            convert_with_pandoc(input_path, md_path)
        elif ext in MARKITDOWN_EXTS:
            print(f'Converting {ext} with MarkItDown...')
            convert_with_markitdown(input_path, md_path)
        else:
            # Unknown extension: MarkItDown casts the widest net, Pandoc as
            # a second chance.
            print(f"Unknown extension '{ext}': trying MarkItDown, "
                  'then Pandoc...')
            try:
                convert_with_markitdown(input_path, md_path)
            except (subprocess.CalledProcessError, SystemExit):
                convert_with_pandoc(input_path, md_path)

        if md_path != input_path:
            ensure_headers(md_path, base)
        if args.convert_only:
            print(f'Markdown written to: {md_path}')
            return
        save_result(run_md_pipeline(md_path, opt_overrides),
                    input_path, args.output_dir)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == '__main__':
    main()
