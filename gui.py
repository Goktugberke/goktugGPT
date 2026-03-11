"""
gui.py — Launch the goktugGPT Gradio web interface.

Usage:
    python gui.py                    # Open browser at localhost:7860
    python gui.py --config medium    # Use medium model config
    python gui.py --share            # Create public Gradio link
    python gui.py --port 8080        # Use a different port
"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Launch goktugGPT GUI")
    parser.add_argument(
        "--config", choices=["tiny", "default", "medium"], default="tiny"
    )
    parser.add_argument("--share", action="store_true", help="Create public share link")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument(
        "--checkpoint-dir", type=str, default="checkpoints"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    from config import ModelConfig, TinyConfig, MediumConfig
    from src.gui import launch_gui

    if args.config == "tiny":
        config = TinyConfig()
    elif args.config == "medium":
        config = MediumConfig()
    else:
        config = ModelConfig()

    if args.checkpoint_dir:
        config.checkpoint_dir = args.checkpoint_dir

    launch_gui(
        checkpoint_dir=config.checkpoint_dir,
        config=config,
        share=args.share,
        port=args.port,
    )


if __name__ == "__main__":
    main()
