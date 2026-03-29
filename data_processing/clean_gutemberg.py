import os
import re
import csv
import json
import unicodedata
from openai import OpenAI


def strip_gutenberg_header(text):
    """Remove everything before the *** START OF ... *** marker."""
    match = re.search(r'\*\*\* START OF (?:THE |THIS )?PROJECT GUTENBERG .+?\*\*\*', text)
    if match:
        return text[match.end():]
    return text


def strip_gutenberg_footer(text):
    """Remove everything from the *** END OF ... *** marker onward."""
    match = re.search(r'\*\*\* END OF (?:THE |THIS )?PROJECT GUTENBERG .+?\*\*\*', text)
    if match:
        return text[:match.start()]
    return text


def remove_illustration_tags(text):
    """Remove [Illustration] and [Illustration: description] tags."""
    return re.sub(r'\[Illustration[^\]]*\]', '', text)


def collapse_blank_lines(text):
    """Collapse 3+ consecutive blank lines into a single paragraph break."""
    return re.sub(r'\n{3,}', '\n\n', text)


def normalize_unicode(text):
    """Replace common Unicode variants with ASCII equivalents."""
    replacements = {
        '\u2018': "'",   # left single quote
        '\u2019': "'",   # right single quote
        '\u201c': '"',   # left double quote
        '\u201d': '"',   # right double quote
        '\u2014': '--',  # em dash
        '\u2013': '-',   # en dash
        '\u2026': '...', # ellipsis
        '\u00a0': ' ',   # non-breaking space
        '\u200b': '',    # zero-width space
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    # NFC normalize anything remaining
    text = unicodedata.normalize('NFC', text)
    return text


def strip_line_whitespace(text):
    """Remove leading/trailing whitespace from each line."""
    lines = text.split('\n')
    lines = [line.strip() for line in lines]
    return '\n'.join(lines)


def unwrap_paragraphs(text):
    """
    Unwrap lines within paragraphs (replace single newlines with spaces)
    while preserving double newlines as paragraph breaks.
    """
    paragraphs = text.split('\n\n')
    cleaned = []
    for p in paragraphs:
        clean_p = p.replace('\n', ' ').strip()
        if clean_p:
            cleaned.append(clean_p)
    return '\n\n'.join(cleaned)


def clean_gutenberg_content(text):
    """Full cleaning pipeline for a Gutenberg text."""
    # Normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    # Strip header and footer
    text = strip_gutenberg_header(text)
    text = strip_gutenberg_footer(text)
    # Remove illustration tags
    text = remove_illustration_tags(text)
    # Normalize unicode characters
    text = normalize_unicode(text)
    # Strip per-line whitespace
    text = strip_line_whitespace(text)
    # Collapse excessive blank lines
    text = collapse_blank_lines(text)
    # Unwrap paragraphs
    text = unwrap_paragraphs(text)
    # Final trim
    text = text.strip()
    return text


def extract_first_page(filepath, char_limit=2000):
    """Extract the first ~2000 chars of raw text for classification."""
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read(char_limit)


def classify_books_with_openai(file_snippets, api_key=None):
    """
    Send book snippets to OpenAI in batches and get back structured metadata.
    Returns a list of dicts with: filename, title, author, language, year, genre, topics.
    """
    client = OpenAI(api_key=api_key)  # uses OPENAI_API_KEY env var if None

    results = []
    # Process in batches of 10 to stay within token limits
    batch_size = 10
    total_batches = (len(file_snippets) + batch_size - 1) // batch_size
    for i in range(0, len(file_snippets), batch_size):
        batch = file_snippets[i:i + batch_size]
        batch_num = i // batch_size + 1
        print(f"  [Batch {batch_num}/{total_batches}] Classifying {len(batch)} books...")

        books_text = ""
        for idx, (filename, snippet) in enumerate(batch):
            books_text += f"\n--- BOOK {idx + 1} (filename: {filename}) ---\n{snippet}\n"

        prompt = (
            "You are a librarian cataloging books. For each book below, extract or infer:\n"
            "- title: the book's title\n"
            "- author: the author's name\n"
            "- language: the language it's written in\n"
            "- year: original publication year (approximate if needed)\n"
            "- genre: e.g. Novel, Poetry, Philosophy, Science, Drama, etc.\n"
            "- topics: 2-4 short topic tags (e.g. 'adventure, ocean, revenge')\n\n"
            "Return a JSON array with one object per book, each having keys: "
            "filename, title, author, language, year, genre, topics.\n"
            "Return ONLY valid JSON, no markdown fences.\n\n"
            f"Books to classify:\n{books_text}"
        )

        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )

        raw = response.choices[0].message.content.strip()
        # Strip markdown fences if the model adds them anyway
        raw = re.sub(r'^```(?:json)?\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)

        try:
            batch_results = json.loads(raw)
            results.extend(batch_results)
        except json.JSONDecodeError:
            print(f"  Warning: failed to parse OpenAI response for batch {i // batch_size + 1}")
            # Add placeholder entries so we don't lose track
            for filename, _ in batch:
                results.append({
                    "filename": filename, "title": "PARSE_ERROR", "author": "",
                    "language": "", "year": "", "genre": "", "topics": ""
                })

    return results


def write_catalog_csv(catalog, output_path):
    """Write the catalog to a CSV file."""
    fieldnames = ['filename', 'title', 'author', 'language', 'year', 'genre', 'topics']
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for entry in catalog:
            row = {k: entry.get(k, '') for k in fieldnames}
            # topics might be a list, join it
            if isinstance(row['topics'], list):
                row['topics'] = ', '.join(row['topics'])
            writer.writerow(row)


def process_directory(input_dir, output_dir):
    """Clean all books and generate a catalog CSV."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    files = sorted([f for f in os.listdir(input_dir) if f.endswith(".txt")])

    if not files:
        print(f"No .txt files found in '{input_dir}'")
        return

    print(f"Found {len(files)} files. Starting cleanup...\n")

    # --- Phase 1: Clean all files ---
    total = len(files)
    print("=== Phase 1: Cleaning texts ===")
    for idx, filename in enumerate(files, 1):
        in_path = os.path.join(input_dir, filename)
        out_path = os.path.join(output_dir, filename)

        try:
            with open(in_path, 'r', encoding='utf-8', errors='ignore') as f:
                raw_text = f.read()

            clean_text = clean_gutenberg_content(raw_text)

            with open(out_path, 'w', encoding='utf-8') as f:
                f.write(clean_text)

            print(f"  [{idx}/{total}] Cleaned: {filename}")

        except Exception as e:
            print(f"  [{idx}/{total}] Error processing {filename}: {e}")

    # --- Phase 2: Classify books via OpenAI ---
    print("\n=== Phase 2: Classifying books with OpenAI ===")
    file_snippets = []
    for filename in files:
        in_path = os.path.join(input_dir, filename)
        snippet = extract_first_page(in_path)
        file_snippets.append((filename, snippet))

    catalog = classify_books_with_openai(file_snippets)

    catalog_path = os.path.join(output_dir, "catalog.csv")
    write_catalog_csv(catalog, catalog_path)
    print(f"\n  Catalog saved to: {catalog_path}")

    # Print summary table
    print(f"\n{'='*100}")
    print(f"{'Filename':<20} {'Title':<30} {'Author':<20} {'Genre':<15} {'Topics'}")
    print(f"{'='*100}")
    for entry in catalog:
        topics = entry.get('topics', '')
        if isinstance(topics, list):
            topics = ', '.join(topics)
        print(f"{entry.get('filename',''):<20} "
              f"{str(entry.get('title',''))[:28]:<30} "
              f"{str(entry.get('author',''))[:18]:<20} "
              f"{str(entry.get('genre',''))[:13]:<15} "
              f"{topics}")
    print(f"{'='*100}")
    print(f"\nTotal books: {len(catalog)}")
    print("Done!")


if __name__ == "__main__":
    INPUT_FOLDER = "gutemberg"
    OUTPUT_FOLDER = "gutemberg_clean"

    if os.path.exists(INPUT_FOLDER):
        process_directory(INPUT_FOLDER, OUTPUT_FOLDER)
    else:
        print(f"Input directory '{INPUT_FOLDER}' does not exist.")
