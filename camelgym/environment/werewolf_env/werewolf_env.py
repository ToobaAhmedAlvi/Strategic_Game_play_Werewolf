#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc   : MG Werewolf Env

from typing import List, Optional

from pydantic import Field

from camelgym.environment.base_env import Environment
from camelgym.environment.werewolf_env.werewolf_ext_env import WerewolfExtEnv
from camelgym.logs import logger
from camelgym.schema import Message


class WerewolfEnv(Environment, WerewolfExtEnv):
    # timestamp used to prefix messages so that identical content is not deduplicated
    timestamp: int = Field(default=0)

    # list of all messages published via pub_mes; used later for analysis (Fig. 4)
    log_messages: List[Message] = Field(default_factory=list)

    # winner can be set by Moderator when the game finishes
    winner: Optional[str] = Field(default=None)

    def pub_mes(self, message: Message, add_timestamp: bool = True):
        """Post information to the environment and also record it for analysis."""
        logger.debug(f"publish_message: {message.dump()}")

        if add_timestamp:
            # prefix with timestamp to avoid automatic deduplication
            message.content = f"{self.timestamp} | " + message.content

        # send into the underlying environment / memory system
        self.publish_message(message=message)

        # record in our own list for later evaluation
        self.log_messages.append(message)

    async def run(self, k: int = 1):
        """Process all roles' runs in order, for k ticks."""
        for _ in range(k):
            for role in self.roles.values():
                await role.run()
            self.timestamp += 1
