# World Labs API Python Client

A small, readable Python client for the [World Labs API](https://platform.worldlabs.ai/), plus helpers to
load [SPZ](https://github.com/nianticlabs/spz) splats and render videos with [gsplat](https://github.com/nerfstudio-project/gsplat).

## Requirements

- Python 3.12+
- CUDA-capable GPU for gsplat rendering (optional if you only need API + SPZ load)

## Install (uv)

```bash
uv sync
source .venv/bin/activate
```

## Quickstart

```bash
export WORLDLABS_API_KEY="..."
python examples/generate_world.py
```

## Client Usage

### Sync Client

```python
from worldlabs_api.client import WorldLabsClient
from worldlabs_api.models import WorldTextPrompt, WorldsGenerateRequest

with WorldLabsClient(api_key="...") as client:
    request = WorldsGenerateRequest(
        display_name="Mystical Forest",
        world_prompt=WorldTextPrompt(text_prompt="A mystical forest"),
    )
    op = client.generate_world(request)
    done = client.poll_operation(op.operation_id)
    world = done.response
```

### Async Client

```python
from worldlabs_api.client import AsyncWorldLabsClient
from worldlabs_api.models import WorldTextPrompt, WorldsGenerateRequest

async with AsyncWorldLabsClient(api_key="...") as client:
    request = WorldsGenerateRequest(
        display_name="Mystical Forest",
        world_prompt=WorldTextPrompt(text_prompt="A mystical forest"),
    )
    op = await client.generate_world(request)
    done = await client.poll_operation(op.operation_id)
    world = done.response
```

## Examples

### Generate a World

```bash
python examples/generate_world.py
```

### List All Worlds

```bash
python examples/list_worlds.py
```

### Load SPZ into Gaussian3D

```bash
python examples/load_splat.py <world_id>
```

### Render a Turntable Video

```bash
python examples/render_video.py <world_id>
```

### Export to PLY

```bash
python examples/export_ply.py <world_id>
```

## Notes

- The API key is passed via the `WLT-Api-Key` header.
- World generation is asynchronous; poll operations until `done` is true.
