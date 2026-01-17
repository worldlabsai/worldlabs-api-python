"""List all worlds and their metadata."""

import os

import structlog

from worldlabs_api.client import WorldLabsClient
from worldlabs_api.models import ListWorldsRequest

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ],
)
logger = structlog.get_logger(__name__)


def main() -> None:
    api_key = os.environ.get("WORLDLABS_API_KEY")
    if not api_key:
        raise RuntimeError("Set WORLDLABS_API_KEY")

    with WorldLabsClient(api_key=api_key) as client:
        # Fetch all worlds, paginating if necessary
        all_worlds = []
        page_token = None

        while True:
            request = ListWorldsRequest(
                page_size=100,
                page_token=page_token,
                sort_by="created_at",
            )
            response = client.list_worlds(request)
            all_worlds.extend(response.worlds)

            if not response.next_page_token:
                break
            page_token = response.next_page_token

    print(f"\n{'='*80}")
    print(f"Found {len(all_worlds)} worlds")
    print(f"{'='*80}\n")

    for world in all_worlds:
        print(f"ID:           {world.id}")
        print(f"Name:         {world.display_name or '(unnamed)'}")
        print(f"Model:        {world.model or '(unknown)'}")
        print(f"Created:      {world.created_at}")

        # Show prompt type and content
        if world.world_prompt:
            prompt_type = world.world_prompt.type
            text = getattr(world.world_prompt, "text_prompt", None)
            print(f"Prompt type:  {prompt_type}")
            if text:
                # Truncate long prompts
                text_display = text[:60] + "..." if len(text) > 60 else text
                print(f"Text prompt:  {text_display}")

        # Show tags
        if world.tags:
            print(f"Tags:         {', '.join(world.tags)}")

        # Show visibility
        if world.permission:
            visibility = "public" if world.permission.public else "private"
            print(f"Visibility:   {visibility}")

        # Show available assets
        if world.assets:
            assets = []
            if world.assets.splats and world.assets.splats.spz_urls:
                assets.append(f"splats({', '.join(world.assets.splats.spz_urls.keys())})")
            if world.assets.imagery and world.assets.imagery.pano_url:
                assets.append("pano")
            if world.assets.mesh and world.assets.mesh.collider_mesh_url:
                assets.append("mesh")
            if world.assets.thumbnail_url:
                assets.append("thumbnail")
            if assets:
                print(f"Assets:       {', '.join(assets)}")

        # Show marble URL
        if world.world_marble_url:
            print(f"Marble URL:   {world.world_marble_url}")

        print(f"{'-'*80}")


if __name__ == "__main__":
    main()
