# Optional regeneration

The deterministic local route requires no API key or network access.

When diagnosed artifacts overlap legitimate content, MIRAGE stops for human
review by default. Passing `--allow-regen` permits a bounded attempt using a
cleaned `reference_only` image:

```bash
export MIYANG_API_KEY=...
mirage input.png output/ --allow-regen --max-regen 1
```

Optional environment variables:

- `MIYANG_BASE_URL`: API base URL; defaults to `https://miyang.cn/api/v1`.
- `MIYANG_PROXY`: explicit fallback proxy used only after a connection-establishment
  failure. Environment proxy variables are otherwise ignored.
- `MIRAGE_WORKERS`: worker count for deterministic spectral median filtering.

The `RegenClient` protocol can be implemented by another provider without
changing diagnosis or treatment logic:

```python
class MyClient:
    def edit(self, image_bytes, filename, prompt, model, size, quality):
        # Return encoded PNG/JPEG bytes.
        ...
```

Pass the client to `mirage.pipeline.run(..., allow_regen=True, client=client)`.

## Safety boundaries

- A prompt sidecar is written before the billable request.
- Credentials are never written to sidecars.
- Read timeouts and interrupted responses are not automatically retried because
  billing state may be unknown.
- Regeneration output can change semantics, dimensions, framing, and color.
- Every regenerated image is a candidate pending human acceptance.
