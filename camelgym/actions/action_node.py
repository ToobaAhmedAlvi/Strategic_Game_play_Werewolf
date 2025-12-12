import json
from typing import Any, Optional
from camelgym.schema import Message
from camelgym.logs import logger


class ActionNode:
    """
    ActionNode generates candidate natural-language actions, embeds them,
    scores them using the RL policy, and selects the best one.
    """

    def __init__(self, key: str, instruction: str = "", example: str = "", expected_type=str):
        self.key = key
        self.instruction = instruction
        self.example = example
        self.expected_type = expected_type

        self.context = None
        self.llm = None

        self.content: Optional[str] = None
        self.raw_response: Optional[str] = None

    # --------------------------------------------------------------

    def set_context(self, context):
        self.context = context

    def set_llm(self, llm):
        self.llm = llm

    # --------------------------------------------------------------

    def build_prompt(self, memory: str = "") -> str:
        """
        Build the LLM prompt that generates candidate actions.
        """
        return f"""
You are generating the next natural-language action for the Werewolf game.

Context:
{memory}

Instruction:
{self.instruction}

Example:
{self.example}

Respond with only the action text.
"""

    # --------------------------------------------------------------

    async def generate_candidate(self, prompt: str) -> str:
        """
        Query the LLM for a single candidate action.
        """
        rsp = await self.llm.aask(prompt)
        return rsp.strip()

    # --------------------------------------------------------------

    async def simple_fill(self, memory: str = "", K: int = 3):
        """
        1. Generate K actions
        2. Embed them
        3. Score using RL policy
        4. Select the best
        5. Store RL trajectory
        6. Track action for innovation metrics
        """
        if self.context is None:
            raise RuntimeError("Context not set in ActionNode.")
        if self.llm is None:
            raise RuntimeError("LLM not set in ActionNode.")

        # === Build LLM Prompt ===
        prompt = self.build_prompt(memory)

        # === Generate Candidate Actions ===
        candidates = []
        for _ in range(K):
            candidates.append(await self.generate_candidate(prompt))

        logger.info(f"[ActionNode] Candidates for {self.key}: {candidates}")

        # === Embed Candidates ===
        embeds = self.context.embedder.embed(candidates)

        import torch
        embeds_tensor = torch.tensor(embeds, dtype=torch.float)

        # === Score Candidates Using RL Policy ===
        scores = self.context.policy(embeds_tensor).squeeze()
        best_idx = torch.argmax(scores).item()
        best_action = candidates[best_idx]

        # === Store RL Trajectory ===
        if self.context.buffer is not None:
            self.context.buffer.add(embeds, best_idx, reward=0)

        # === Save Action ===
        self.content = best_action
        self.raw_response = best_action

        # === NEW: Track action diversity (Innovation Feature) ===
        if hasattr(self.context, "action_history") and self.context.action_history is not None:
            self.context.action_history.append(best_action)

        logger.info(f"[ActionNode] Selected action for {self.key}: {self.content}")

        return self
