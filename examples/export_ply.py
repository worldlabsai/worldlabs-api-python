"""Export a splat to PLY using gsplat utilities."""

import argparse
import os
import pathlib

import structlog

from worldlabs_api.client import WorldLabsClient
from worldlabs_api.helpers.export import save_ply
from worldlabs_api.helpers.spz import download_spz, load_spz

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ],
)
logger = structlog.get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a world's splat to PLY")
    parser.add_argument("world_id", help="World ID to fetch")
    args = parser.parse_args()

    api_key = os.environ.get("WORLDLABS_API_KEY")
    if not api_key:
        raise RuntimeError("Set WORLDLABS_API_KEY")

    with WorldLabsClient(api_key=api_key) as client:
        world = client.get_world(args.world_id)

    spz_urls = world.assets.splats.spz_urls if world.assets else None
    if not spz_urls:
        raise RuntimeError("World does not include SPZ assets")

    spz_url = spz_urls["full_res"]
    spz_path = pathlib.Path("outputs") / f"{args.world_id}.spz"
    download_spz(spz_url, spz_path)
    gaussians = load_spz(spz_path)

    output_path = pathlib.Path("outputs") / f"{args.world_id}.ply"
    save_ply(gaussians, output_path)
    logger.info("saved_ply", path=str(output_path))


if __name__ == "__main__":
    main()
