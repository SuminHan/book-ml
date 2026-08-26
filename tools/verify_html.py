import sys, glob, re

def check(path):
    with open(path, encoding='utf-8') as f:
        html = f.read()
    issues = []
    for op, cl in [(r'\(', r'\)'), (r'\[', r'\]')]:
        # count \( vs \) etc as literal substrings "\(" "\)"
        pass
    n_open_inline = html.count('\\(')
    n_close_inline = html.count('\\)')
    # LaTeX "\[Npt]" row-spacing inside cases/array environments (e.g. "\\[6pt]")
    # is valid LaTeX (a bare "]" with no preceding backslash), not a block-math
    # delimiter pair -- exclude its "\[" from the open count so it doesn't
    # produce a false-positive mismatch (it never contributes a "\]" close).
    spacing_directives = len(re.findall(r'\\\[\d+(?:pt|em|ex|mm|cm|in)\]', html))
    n_open_block = html.count('\\[') - spacing_directives
    n_close_block = html.count('\\]')
    if n_open_inline != n_close_inline:
        issues.append(f"inline delimiter mismatch: \\( ={n_open_inline} \\)={n_close_inline}")
    if n_open_block != n_close_block:
        issues.append(f"block delimiter mismatch: \\[ ={n_open_block} \\]={n_close_block}")
    for bad in ['<em>{\\text', '}<em>', '</em>{\\text', '}</em>{', '<strong>{\\text', '}<strong>']:
        if bad in html:
            issues.append(f"possible emphasis corruption: contains {bad!r}")
    return issues

paths = sys.argv[1:]
total_issues = 0
for p in paths:
    for f in sorted(glob.glob(p)):
        issues = check(f)
        if issues:
            total_issues += len(issues)
            print(f"{f}:")
            for i in issues:
                print(f"  - {i}")
print(f"\nTotal files checked: {sum(len(glob.glob(p)) for p in paths)}, issues found: {total_issues}")
