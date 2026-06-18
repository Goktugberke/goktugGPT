"""
goktugGPT — Gradio Chat GUI

Features:
  • Clean chat interface with message bubbles
  • Collapsible "Thinking" panel showing the model's reasoning
  • Temperature / Top-K / Top-P sliders
  • Toggle to enable/disable the thinking stage
  • Model info panel (parameters, device, vocab size)
  • Clear conversation button
"""

import os
import sys
import time
from typing import Optional

import gradio as gr
import torch

# Allow running this file directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from config import ModelConfig, TinyConfig
from src.tokenizer import BPETokenizer
from src.model import GoktugGPT
from src.thinking import ThinkingEngine


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

_engine: Optional[ThinkingEngine] = None
_config: Optional[ModelConfig] = None


def _find_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Find best or most recent checkpoint in the checkpoint directory."""
    if not os.path.isdir(checkpoint_dir):
        return None
    # Priority: best_model > final_model > latest checkpoint
    for name in ["best_model.pt", "final_model.pt"]:
        p = os.path.join(checkpoint_dir, name)
        if os.path.exists(p):
            return p
    # Look for numbered checkpoints
    pts = sorted(
        [f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")],
        reverse=True,
    )
    if pts:
        return os.path.join(checkpoint_dir, pts[0])
    return None


def load_model(
    checkpoint_dir: str = "checkpoints",
    config: Optional[ModelConfig] = None,
    device: str = "auto",
) -> tuple[Optional[ThinkingEngine], str]:
    """
    Load the model from a checkpoint and return (engine, status_message).
    Returns (None, error_msg) if loading fails.
    """
    global _engine, _config

    if config is None:
        config = TinyConfig()
    _config = config

    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    # Check tokenizer
    tok_path = config.tokenizer_path
    if not os.path.exists(tok_path):
        return None, (
            f"Tokenizer not found at '{tok_path}'.\n"
            "Please run train.py first to train the model."
        )

    tokenizer = BPETokenizer.load(tok_path)

    # Check checkpoint
    ckpt_path = _find_checkpoint(checkpoint_dir)
    if ckpt_path is None:
        return None, (
            f"No checkpoint found in '{checkpoint_dir}'.\n"
            "Please run train.py first to train the model."
        )

    try:
        model, extra = GoktugGPT.load_checkpoint(
            ckpt_path,
            vocab_size=config.vocab_size,
            n_embed=config.n_embed,
            n_head=config.n_head,
            n_layer=config.n_layer,
            dropout=0.0,
            max_seq_len=config.max_seq_len,
            device=device,
        )
    except Exception as e:
        return None, f"Failed to load checkpoint: {e}"

    _engine = ThinkingEngine(model, tokenizer, config)
    n_params = model.num_parameters()
    status = (
        f"Model loaded from: {ckpt_path}\n"
        f"Parameters: {n_params/1e6:.2f}M\n"
        f"Vocab size: {len(tokenizer)}\n"
        f"Device: {device}\n"
        f"Layers: {config.n_layer}  |  Heads: {config.n_head}  |  Embed dim: {config.n_embed}"
    )
    return _engine, status


# ---------------------------------------------------------------------------
# Chat logic
# ---------------------------------------------------------------------------

def chat(
    message: str,
    history: list,
    use_thinking: bool,
    temperature: float,
    top_k: int,
    top_p: float,
    max_new_tokens: int,
    max_think_tokens: int,
):
    """Main chat function called by Gradio."""
    if _engine is None:
        return history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "Model not loaded. Please run train.py first."},
        ], ""

    if not message.strip():
        return history, ""

    # Format history for the engine
    formatted_history = [
        {"role": h["role"], "content": h["content"]}
        for h in history
    ]

    thinking, answer = _engine.generate_with_thinking(
        user_message=message,
        history=formatted_history,
        use_thinking=use_thinking,
        temperature=temperature,
        top_k=int(top_k),
        top_p=top_p,
        max_new_tokens=int(max_new_tokens),
        max_think_tokens=int(max_think_tokens),
    )

    # Build updated history
    new_history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer},
    ]

    # Format thinking text for display
    thinking_display = ""
    if thinking:
        thinking_display = f"**Thinking process:**\n\n{thinking}"

    return new_history, thinking_display


def clear_chat():
    return [], ""


# ---------------------------------------------------------------------------
# Build Gradio UI
# ---------------------------------------------------------------------------

def build_ui(engine: Optional[ThinkingEngine], model_info: str) -> gr.Blocks:
    with gr.Blocks(
        title="goktugGPT",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="slate",
            neutral_hue="slate",
        ),
        css="""
        .thinking-box {
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 8px;
            padding: 12px;
            font-family: monospace;
            font-size: 13px;
            color: #94a3b8;
            min-height: 60px;
        }
        .model-info {
            background: #0f172a;
            border: 1px solid #1e293b;
            border-radius: 8px;
            padding: 10px;
            font-family: monospace;
            font-size: 12px;
            color: #64748b;
        }
        .title-header {
            text-align: center;
            padding: 10px 0;
        }
        """,
    ) as demo:

        # ----------------------------------------------------------------
        # Header
        # ----------------------------------------------------------------
        with gr.Row(elem_classes="title-header"):
            gr.Markdown(
                """
                # goktugGPT
                *A locally-trained language model built from scratch*
                *Transformer · Multi-head Attention · BPE Tokenizer · Embedded Thinking*
                """
            )

        # ----------------------------------------------------------------
        # Main layout: chat + sidebar
        # ----------------------------------------------------------------
        with gr.Row():

            # --- Left: Chat ---
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=480,
                    type="messages",
                    bubble_full_width=False,
                    show_copy_button=True,
                )

                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Type your message here...",
                        label="",
                        scale=5,
                        container=False,
                        lines=1,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)
                    clear_btn = gr.Button("Clear", variant="secondary", scale=1)

                # Thinking display
                with gr.Accordion("Thinking Stage", open=False):
                    thinking_out = gr.Markdown(
                        value="*No thinking generated yet.*",
                        elem_classes="thinking-box",
                    )

            # --- Right: Settings + Info ---
            with gr.Column(scale=1):

                gr.Markdown("### Generation Settings")

                use_thinking = gr.Checkbox(
                    label="Enable Thinking Stage",
                    value=True,
                    info="Model reasons before answering",
                )

                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=0.8,
                    step=0.05,
                    label="Temperature",
                    info="Higher = more creative",
                )

                top_k = gr.Slider(
                    minimum=1,
                    maximum=200,
                    value=50,
                    step=1,
                    label="Top-K",
                    info="Keep top K tokens",
                )

                top_p = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.9,
                    step=0.05,
                    label="Top-P (Nucleus)",
                    info="Nucleus sampling threshold",
                )

                max_new_tokens = gr.Slider(
                    minimum=10,
                    maximum=512,
                    value=128,
                    step=10,
                    label="Max Response Tokens",
                )

                max_think_tokens = gr.Slider(
                    minimum=10,
                    maximum=256,
                    value=64,
                    step=10,
                    label="Max Thinking Tokens",
                )

                gr.Markdown("### Model Info")
                gr.Textbox(
                    value=model_info,
                    label="",
                    interactive=False,
                    lines=7,
                    elem_classes="model-info",
                )

        # ----------------------------------------------------------------
        # Event wiring
        # ----------------------------------------------------------------

        def _submit(message, history, thinking, temp, k, p, max_tok, max_think):
            updated_history, thinking_text = chat(
                message, history, thinking, temp, k, p, max_tok, max_think
            )
            return updated_history, thinking_text, ""

        send_btn.click(
            fn=_submit,
            inputs=[msg_input, chatbot, use_thinking, temperature, top_k, top_p,
                    max_new_tokens, max_think_tokens],
            outputs=[chatbot, thinking_out, msg_input],
        )

        msg_input.submit(
            fn=_submit,
            inputs=[msg_input, chatbot, use_thinking, temperature, top_k, top_p,
                    max_new_tokens, max_think_tokens],
            outputs=[chatbot, thinking_out, msg_input],
        )

        clear_btn.click(
            fn=clear_chat,
            inputs=[],
            outputs=[chatbot, thinking_out],
        )

    return demo


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def launch_gui(
    checkpoint_dir: str = "checkpoints",
    config: Optional[ModelConfig] = None,
    share: bool = False,
    port: int = 7860,
):
    """
    Load the model and launch the Gradio interface.

    Args:
        checkpoint_dir: Directory containing model checkpoints.
        config:         Model config (defaults to TinyConfig).
        share:          If True, create a public Gradio share link.
        port:           Local port to serve on.
    """
    global _engine

    print("Loading goktugGPT...")
    engine, model_info = load_model(checkpoint_dir, config)
    _engine = engine

    if engine is None:
        print(f"\nWarning: {model_info}")
        print("Launching in demo mode (no model loaded).\n")

    demo = build_ui(engine, model_info)
    print(f"\nLaunching GUI at http://localhost:{port}\n")
    demo.launch(server_port=port, share=share, inbrowser=True)
