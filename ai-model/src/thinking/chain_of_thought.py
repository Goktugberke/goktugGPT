"""
Embedded Thinking Stage — Chain-of-Thought reasoning built into goktugGPT.

Inspired by DeepSeek-R1 / OpenAI o1:

  The model is trained to produce:
    <think>
      [internal reasoning / scratchpad]
    </think>
    [actual answer]

  During inference:
    1. We prompt the model with <think>
    2. We let it generate until </think>
    3. We capture that as the "thinking" trace
    4. We then let it generate the real answer

  Why does this help?
    - Allocates extra "compute tokens" for complex problems
    - The model can work through steps, correct itself, and then summarise
    - The thinking is visible to the user (or can be hidden)
    - No architectural change needed — purely a prompting / training trick

  Training:
    - Some training examples include <think>...</think> segments
    - The model learns to associate complex questions with thinking first
    - Loss is computed over ALL tokens including thinking tokens
"""

import re
from typing import Optional, Tuple

import torch

from ..tokenizer import BPETokenizer
from ..model import GoktugGPT


class ThinkingEngine:
    """
    Wraps GoktugGPT with a thinking-stage generation loop.

    Usage:
        engine = ThinkingEngine(model, tokenizer, config)
        thinking, answer = engine.generate_with_thinking("What is 15 * 23?")
    """

    THINK_START = "<think>"
    THINK_END = "</think>"
    USER_TOK = "<user>"
    ASST_TOK = "<assistant>"
    EOS_TOK = "<eos>"

    def __init__(self, model: GoktugGPT, tokenizer: BPETokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = next(model.parameters()).device

        # Cache token IDs for special tokens
        self._think_start_id = tokenizer.token_to_id(self.THINK_START)
        self._think_end_id = tokenizer.token_to_id(self.THINK_END)
        self._eos_id = tokenizer.token_to_id(self.EOS_TOK)
        self._asst_id = tokenizer.token_to_id(self.ASST_TOK)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_prompt(self, user_message: str, history: Optional[list] = None) -> str:
        """
        Format conversation history + current message into a prompt string.

        Format:
            <user> message1 <assistant> response1 <user> message2 <assistant>
        """
        prompt = ""
        if history:
            for turn in history:
                role, text = turn["role"], turn["content"]
                if role == "user":
                    prompt += f"{self.USER_TOK} {text.strip()} "
                else:
                    prompt += f"{self.ASST_TOK} {text.strip()} "
        prompt += f"{self.USER_TOK} {user_message.strip()} {self.ASST_TOK}"
        return prompt

    @torch.no_grad()
    def generate_with_thinking(
        self,
        user_message: str,
        history: Optional[list] = None,
        use_thinking: bool = True,
        temperature: float = 0.7,
        top_k: int = 30,
        top_p: float = 0.9,
        repetition_penalty: float = 1.3,
        max_new_tokens: int = 200,
        max_think_tokens: int = 60,
    ) -> Tuple[str, str]:
        """
        Generate a response, optionally preceded by a thinking phase.

        Returns:
            (thinking_text, answer_text)
            thinking_text is empty string if use_thinking=False
        """
        self.model.eval()
        prompt = self.build_prompt(user_message, history)

        if use_thinking:
            return self._generate_think_then_answer(
                prompt,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                max_think_tokens=max_think_tokens,
                max_answer_tokens=max_new_tokens,
            )
        else:
            answer = self._generate_plain(
                prompt,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                max_new_tokens=max_new_tokens,
            )
            return "", answer

    # ------------------------------------------------------------------
    # Internal generation helpers
    # ------------------------------------------------------------------

    def _encode_prompt(self, prompt: str) -> torch.Tensor:
        """Encode a prompt string and return a (1, T) tensor."""
        ids = self.tokenizer.encode(prompt)
        return torch.tensor([ids], dtype=torch.long, device=self.device)

    def _generate_think_then_answer(
        self,
        prompt: str,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
        max_think_tokens: int,
        max_answer_tokens: int,
    ) -> Tuple[str, str]:
        """
        Two-phase generation:
          Phase 1: Generate thinking tokens until </think> or limit.
          Phase 2: Generate the actual answer until <eos>.
        """
        # Phase 1: seed with <think>
        think_prompt = prompt + f" {self.THINK_START} "
        input_ids = self._encode_prompt(think_prompt)

        # Generate thinking (stop at </think>)
        after_think = self.model.generate(
            input_ids,
            max_new_tokens=max_think_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            eos_token_id=self._think_end_id,
            stop_token_ids=[self._think_end_id],
        )

        # Extract thinking text
        think_ids = after_think[0, input_ids.shape[1]:].tolist()

        think_concluded = self._think_end_id in think_ids

        if think_concluded:
            # Clean thinking text: everything before </think>
            end_pos = think_ids.index(self._think_end_id)
            thinking_text = self.tokenizer.decode(
                think_ids[:end_pos], skip_special_tokens=True
            ).strip()
            # Use the naturally concluded sequence as answer context
            answer_input = after_think
        else:
            # Model didn't conclude thinking — do NOT include garbage tokens in answer context.
            # Fall back to the original prompt + </think> so the answer phase starts clean.
            thinking_text = self.tokenizer.decode(think_ids, skip_special_tokens=True).strip()
            end_tensor = torch.tensor([[self._think_end_id]], device=self.device)
            answer_input = torch.cat([input_ids, end_tensor], dim=1)

        final_ids = self.model.generate(
            answer_input,
            max_new_tokens=max_answer_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            eos_token_id=self._eos_id,
            stop_token_ids=[self._eos_id, self._asst_id],
        )

        # Extract only the new answer tokens
        answer_ids = final_ids[0, answer_input.shape[1]:].tolist()
        answer_text = self.tokenizer.decode(answer_ids, skip_special_tokens=True).strip()

        return thinking_text, answer_text

    def _generate_plain(
        self,
        prompt: str,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
        max_new_tokens: int,
    ) -> str:
        """Generate without a thinking stage."""
        input_ids = self._encode_prompt(prompt)
        output_ids = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            eos_token_id=self._eos_id,
            stop_token_ids=[self._eos_id, self._asst_id],
        )
        new_ids = output_ids[0, input_ids.shape[1]:].tolist()
        return self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    # ------------------------------------------------------------------
    # Training data utilities
    # ------------------------------------------------------------------

    @staticmethod
    def format_training_example(
        user: str,
        thinking: str,
        answer: str,
        include_thinking: bool = True,
    ) -> str:
        """
        Format a single training example with optional thinking stage.

        Output format:
            <user> {user} <assistant> <think> {thinking} </think> {answer} <eos>
            or
            <user> {user} <assistant> {answer} <eos>
        """
        if include_thinking and thinking.strip():
            return (
                f"<user> {user.strip()} "
                f"<assistant> <think> {thinking.strip()} </think> "
                f"{answer.strip()} <eos>"
            )
        return f"<user> {user.strip()} <assistant> {answer.strip()} <eos>"

    @staticmethod
    def extract_thinking_and_answer(generated_text: str) -> Tuple[str, str]:
        """
        Parse a generated string to extract thinking and answer parts.

        Returns:
            (thinking, answer)
        """
        think_match = re.search(r"<think>(.*?)</think>", generated_text, re.DOTALL)
        if think_match:
            thinking = think_match.group(1).strip()
            # Answer is everything after </think>
            answer = generated_text[think_match.end():].strip()
        else:
            thinking = ""
            answer = generated_text.strip()

        # Strip leftover special tokens from answer
        for tok in ["<eos>", "<user>", "<assistant>", "<think>", "</think>"]:
            answer = answer.replace(tok, "").strip()

        return thinking, answer
