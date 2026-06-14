import json

with open('AmazonMLChallenge (1).ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Extract all code cells with meaningful content
print("=" * 80)
print("NOTEBOOK CODE EXTRACTION")
print("=" * 80)

for i, cell in enumerate(nb['cells']):
    if cell.get('cell_type') == 'code':
        source = ''.join(cell.get('source', []))
        if source.strip() and len(source) > 50:  # Only meaningful code
            print(f"\n{'='*80}")
            print(f"CELL {i}")
            print(f"{'='*80}")
            print(source[:3000])  # Print first 3000 chars
            if len(source) > 3000:
                print(f"\n... [truncated, total {len(source)} chars]")
