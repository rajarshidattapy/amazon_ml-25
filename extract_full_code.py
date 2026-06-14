import json

with open('AmazonMLChallenge (1).ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Extract all code cells
code_content = []
for i, cell in enumerate(nb['cells']):
    if cell.get('cell_type') == 'code':
        source = ''.join(cell.get('source', []))
        if source.strip():
            code_content.append(f"# Cell {i}\n{source}\n")

# Write to file
with open('extracted_notebook_code.py', 'w', encoding='utf-8') as f:
    f.write('\n'.join(code_content))

print(f"Extracted {len(code_content)} code cells to extracted_notebook_code.py")
print(f"Total size: {sum(len(c) for c in code_content)} characters")
