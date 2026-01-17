"""Generate a world from a text prompt and poll until done."""

import os
import time

import structlog

from worldlabs_api.client import WorldLabsClient
from worldlabs_api.models import WorldTextPrompt, WorldsGenerateRequest

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
        raise RuntimeError("Set WORLDLABS_API_KEY in your environment")

    request = WorldsGenerateRequest(
        display_name="Medieval Kitchen",
        model="Marble 0.1-mini",
        world_prompt=WorldTextPrompt(
            text_prompt=(
                "A cartoon style medieval kitchen with stone walls and a roaring hearth"
            )
        ),
    )

    with WorldLabsClient(api_key=api_key) as client:
        logger.info("generating_world", display_name=request.display_name)
        start_time = time.time()
        operation = client.generate_world(request)
        logger.info("operation_started", operation_id=operation.operation_id)

        done = client.poll_operation(operation.operation_id)
        elapsed = time.time() - start_time

        logger.info("operation_completed", done=done.done, elapsed_seconds=round(elapsed, 2))
        if done.response:
            logger.info(
                "world_created",
                world_id=done.response.id,
                world_url=done.response.world_marble_url,
            )


if __name__ == "__main__":
    main()
