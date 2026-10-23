"""Generate an installed-distribution license inventory, not a legal approval."""
import argparse
from importlib.metadata import distributions
from pathlib import Path


def inventory():
    rows = []
    for distribution in distributions():
        metadata = distribution.metadata
        license_text = metadata.get('License-Expression') or metadata.get('License') or ''
        if not license_text:
            license_text = '; '.join(c.removeprefix('License :: ') for c in metadata.get_all('Classifier', [])
                                     if c.startswith('License :: ')) or 'UNDECLARED'
        summary = ' '.join(license_text.split()).replace('|', '\\|')
        if len(summary) > 180:
            summary = summary[:177]+'...'
        rows.append((metadata.get('Name', 'UNKNOWN'), distribution.version, summary))
    return sorted(rows, key=lambda row: row[0].casefold())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args()
    lines = ['# Installed dependency license inventory', '',
        'Generated from the separate verification environment. Includes optional audit tools.',
        'Declarations are package metadata, not an approval of licensing or redistribution.',
        'Read the full distribution license/NOTICE files before redistribution. See the generated SBOM.', '',
        '| Distribution | Installed version | Declared license metadata |', '|---|---|---|']
    lines.extend(f'| {name} | {version} | {license_text} |' for name, version, license_text in inventory())
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text('\n'.join(lines)+'\n', encoding='utf-8')
