import os
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from camelgym.call_config import Config
from camelgym.configs.llm_config import LLMConfig
from camelgym.provider.base_llm import BaseLLM
from camelgym.provider.llm_provider_registry import create_llm_instance
from camelgym.utils.cost_manager import CostManager
from camelgym.utils.git_repository import GitRepository
from camelgym.utils.project_repo import ProjectRepo

# RL modules
from camelgym.rl.embedder import LocalEmbedder
from camelgym.rl.policy import RLPolicy
from camelgym.rl.buffer import ExperienceBuffer
from camelgym.rl.trainer import RLTrainer


class AttrDict(BaseModel):
    model_config = ConfigDict(
        extra="allow",
        arbitrary_types_allowed=True
    )

    def set(self, key, val: Any):
        setattr(self, key, val)

    def get(self, key, default: Any = None):
        return getattr(self, key, default)

    def remove(self, key):
        if hasattr(self, key):
            delattr(self, key)


class Context(BaseModel):
    """Full context for the environment: LLM + RL modules."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    kwargs: AttrDict = AttrDict()
    config: Config = Config.default()

    repo: Optional[ProjectRepo] = None
    git_repo: Optional[GitRepository] = None
    src_workspace: Optional[Path] = None
    cost_manager: CostManager = CostManager()

    _llm: Optional[BaseLLM] = None

    # RL components (initialized safely later)
    embedder: Optional[LocalEmbedder] = None
    policy: Optional[RLPolicy] = None
    buffer: Optional[ExperienceBuffer] = None
    trainer: Any = Field(default=None, exclude=True)
    action_history: list[str] = []

    # -----------------------------------------------------------
    def new_environ(self):
        return os.environ.copy()

    # -----------------------------------------------------------
    def llm(self) -> BaseLLM:
        self._llm = create_llm_instance(self.config.llm)
        if self._llm.cost_manager is None:
            self._llm.cost_manager = self.cost_manager
        return self._llm

    def llm_with_cost_manager_from_llm_config(self, llm_config: LLMConfig) -> BaseLLM:
        llm = create_llm_instance(llm_config)
        if llm.cost_manager is None:
            llm.cost_manager = self.cost_manager
        return llm

    # -----------------------------------------------------------
    def model_post_init(self, __context=None):
        """Initialize RL modules after model creation."""
        try:
            self.embedder = LocalEmbedder()          # Your real embedder
            self.policy = RLPolicy()                 # Scoring network
            self.buffer = ExperienceBuffer()         # Stores (embeds, idx, reward)
            self.trainer = RLTrainer(self.policy)    # Reinforce trainer

        except Exception as e:
            print("[WARNING] RL modules not initialized:", e)
            self.embedder = None
            self.policy = None
            self.buffer = None
            self.trainer = None
