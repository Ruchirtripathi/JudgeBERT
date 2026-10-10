import csv
import json
import pathlib
from collections import Counter, defaultdict

def read_csv(path):
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

def report():
    csv_path = pathlib.Path(__file__).resolve().parents[2] / 'data' / 'processed' / 'lmp_pairs.csv'
    rows = read_csv(csv_path)
    total = len(rows)
    # Required columns check
    required = ['pair_id', 'language', 'legal_domain', 'original_text', 'simplified_text', 'source', 'split', 'source_group', 'verbatim_quote']
    missing_fields = []
    for i, row in enumerate(rows, start=1):
        for col in required:
            if col not in row or row[col] == '' :
                missing_fields.append((i, col, row.get('pair_id','')))
    # Language distribution
    lang_counts = Counter(r['language'] for r in rows)
    # Domain distribution overall
    domain_counts = Counter(r['legal_domain'] for r in rows)
    # Split distribution per language and domain
    split_counts = defaultdict(lambda: Counter())
    for r in rows:
        key = f"{r['language']}_{r['split']}"
        split_counts[key][r['legal_domain']] += 1
    # Duplicate pair_id
    dup_pair_ids = [pid for pid, cnt in Counter(r['pair_id'] for r in rows).items() if cnt > 1]
    # Duplicate original_text
    dup_original = [txt for txt, cnt in Counter(r['original_text'] for r in rows).items() if cnt > 1]
    # Duplicate simplified_text
    dup_simplified = [txt for txt, cnt in Counter(r['simplified_text'] for r in rows).items() if cnt > 1]
    # Check source groups spanning splits
    group_splits = defaultdict(set)
    for r in rows:
        group_splits[r['source_group']].add(r['split'])
    groups_spanning = {g: splits for g, splits in group_splits.items() if len(splits) > 1}
    # Verbose quote consistency
    verbatim_mismatch = []
    for r in rows:
        if r['language'] == 'en' and r['verbatim_quote'].lower() != 'true':
            verbatim_mismatch.append(r['pair_id'])
        if r['language'] == 'hi' and r['verbatim_quote'].lower() != 'false':
            verbatim_mismatch.append(r['pair_id'])
    # Simplification identical check
    identical_simpl = [r['pair_id'] for r in rows if r['original_text'].strip() == r['simplified_text'].strip()]

    report_data = {
        'total_rows': total,
        'missing_fields': missing_fields,
        'language_counts': dict(lang_counts),
        'domain_counts': dict(domain_counts),
        'split_counts': {k: dict(v) for k, v in split_counts.items()},
        'duplicate_pair_ids': dup_pair_ids,
        'duplicate_original_texts': dup_original[:5],
        'duplicate_simplified_texts': dup_simplified[:5],
        'source_groups_spanning_splits': groups_spanning,
        'verbatim_quote_mismatches': verbatim_mismatch,
        'identical_original_simplified': identical_simpl,
    }
    out_path = pathlib.Path(__file__).resolve().parents[2] / 'reports' / 'final_corpus_audit.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)
    print('Audit completed. JSON saved to', out_path)

if __name__ == '__main__':
    report()
