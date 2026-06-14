import json
import re

with open('AmazonMLChallenge (1).ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find key functions and classes
for i, cell in enumerate(nb['cells']):
    if cell.get('cell_type') == 'code':
        source = ''.join(cell.get('source', []))

        # Look for class definitions, function definitions, and important imports
        if re.search(r'\bclass\s+\w+|def\s+\w+|from\s+sklearn|import\s+\w+', source):
            lines = source.split('\n')
            # Show lines with classes, functions, or imports
            for j, line in enumerate(lines[:30]):
                if re.search(r'\bclass\s+\w+|def\s+\w+|from\s+sklearn|import\s+\w+', line):
                    print(f"Cell {i}, Line {j}: {line.strip()}")
