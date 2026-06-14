import json

with open('AmazonMLChallenge (1).ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Print cell summaries
for i, cell in enumerate(nb['cells'][:30]):
    cell_type = cell.get('cell_type', 'unknown')
    if cell_type == 'code':
        source = ''.join(cell.get('source', [])).strip()
        preview = source[:80].replace('\n', ' ')
        print(f"Cell {i}: CODE - {preview}...")
    elif cell_type == 'markdown':
        source = ''.join(cell.get('source', [])).strip()
        preview = source[:60].replace('\n', ' ')
        print(f"Cell {i}: MD - {preview}...")
