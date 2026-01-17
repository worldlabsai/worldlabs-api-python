"""Fetch a world and load its SPZ splat into a Gaussian3D."""

import argparse
import os
import pathlib

import structlog

from worldlabs_api.client import WorldLabsClient
from worldlabs_api.helpers.spz import download_spz, load_spz

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ],
)
logger = structlog.get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Load a world's SPZ splat")
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

    spz_url = spz_urls.get("full_res") or next(iter(spz_urls.values()))
    output_path = pathlib.Path("outputs") / f"{args.world_id}.spz"
    download_spz(spz_url, output_path)
    gaussians = load_spz(output_path)
    logger.info("loaded_gaussians", num_gaussians=gaussians.mean.shape[0])


if __name__ == "__main__":
    main()
