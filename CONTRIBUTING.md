# Contributing

Thank you for improving MIRAGE.

## Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
ruff check .
pytest
```

Changes to numerical thresholds or signal-processing behavior must include:

1. the reason for the change;
2. a synthetic or redistributable regression case;
3. before-and-after measurements;
4. an explanation of whether paper results remain comparable.

Do not update golden values merely to make a dependency upgrade pass. First
identify the numerical behavior that changed.

## Data and privacy

Only contribute images that you have the right to redistribute. Do not submit
private user images, credentials, internal URLs, account identifiers, or
production traces.

By submitting a contribution, you represent that you have the right to license
it under this repository's applicable license.
