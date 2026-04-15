# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import logging

from pathlib import Path

import click
import hydra

from vipe import get_config_path
from vipe.pipeline import make_pipeline
from vipe.streams.base import ProcessedVideoStream
from vipe.streams.frame_dir_stream import FrameDirStream
from vipe.streams.raw_mp4_stream import RawMp4Stream
from vipe.utils.viser import run_viser


def configure_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    return logging.getLogger("vipe")


@click.command()
@click.option("--video", "-v", type=click.Path(exists=True, path_type=Path), help="Path to the video file")
@click.option("--image-dir", "-i", type=click.Path(exists=True, path_type=Path), help="Path to the directory of images")
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=Path.cwd() / "vipe_results",
    help="Output directory (default: current directory)",
)
@click.option("--pipeline", "-p", default="default", help="Pipeline configuration to use")
@click.option("--visualize", "-vis", is_flag=True, help="Enable visualization of intermediate results")
@click.argument("overrides", nargs=-1)
def infer(video: Path, image_dir: Path, output: Path, pipeline: str, visualize: bool, overrides: tuple[str]):
    """Run inference on a video file or directory of images."""

    logger = configure_logging()

    # Validate that exactly one input source is provided
    if not video and not image_dir:
        click.echo("Error: Must provide either a video file or --image-dir", err=True)
        raise click.Abort()
    
    if video and image_dir:
        click.echo("Error: Cannot provide both video file and --image-dir", err=True)
        raise click.Abort()

    hydra_overrides = [f"pipeline={pipeline}", f"pipeline.output.path={output}", "pipeline.output.save_artifacts=true"]
    if visualize:
        hydra_overrides.append("pipeline.output.save_viz=true")
        hydra_overrides.append("pipeline.slam.visualize=true")
    else:
        hydra_overrides.append("pipeline.output.save_viz=false")

    # Add user overrides
    hydra_overrides.extend(list(overrides))

    # Set up stream configuration based on input type
    if image_dir:
        hydra_overrides.extend([
            "streams=frame_dir_stream",
            f"streams.base_path={image_dir}"
        ])
        input_path = image_dir
        input_desc = f"image directory {image_dir}"
    else:
        input_path = video
        input_desc = f"video {video}"

    with hydra.initialize_config_dir(config_dir=str(get_config_path()), version_base=None):
        args = hydra.compose("default", overrides=hydra_overrides)

    logger.info(f"Processing {input_desc}...")
    vipe_pipeline = make_pipeline(args.pipeline)

    if image_dir:
        # Use frame directory stream
        video_stream = ProcessedVideoStream(FrameDirStream(image_dir), []).cache(
            desc="Reading image frames",
            online=True,
        )
    else:
        # Some input videos can be malformed, so we need to cache the videos to obtain correct number of frames.
        video_stream = ProcessedVideoStream(RawMp4Stream(video), []).cache(desc="Reading video stream")

    vipe_pipeline.run(video_stream)
    logger.info("Finished")


@click.command()
@click.argument("data_path", type=click.Path(exists=True, path_type=Path), default=Path.cwd() / "vipe_results")
@click.option("--port", "-p", default=20540, type=int, help="Port for the visualization server (default: 20540)")
def visualize(data_path: Path, port: int):
    run_viser(data_path, port)


@click.group()
@click.version_option()
def main():
    """NVIDIA Video Pose Engine (ViPE) CLI"""
    pass


# Add subcommands
main.add_command(infer)
main.add_command(visualize)


if __name__ == "__main__":
    main()
