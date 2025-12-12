import re
from collections import Counter
from datetime import datetime
import sys

sys.path.append("..")

from camelgym.const import DEFAULT_WORKSPACE_ROOT, MESSAGE_ROUTE_TO_ALL
from camelgym.roles import Role
from camelgym.schema import Message
from camelgym.logs import logger
from actions.moderator_actions import (
    InstructSpeak,
    ParseSpeak,
    AnnounceGameResult,
    STEP_INSTRUCTIONS,
)
from actions import Hunt, Protect, Verify, Save, Poison
from camelgym.actions import UserRequirement


class Moderator(Role):
    def __init__(
        self,
        name: str = "Moderator",
        profile: str = "Moderator",
        **kwargs,
    ):
        super().__init__(name=name, profile=profile, **kwargs)
        self._watch([UserRequirement, InstructSpeak, ParseSpeak])
        self.set_actions([InstructSpeak, ParseSpeak, AnnounceGameResult])
        self.step_idx = 0
        self.eval_step_idx = []

        # game states
        self.game_setup = ""
        self.living_players: list[str] = []
        self.werewolf_players: list[str] = []
        self.villager_players: list[str] = []
        self.special_role_players: list[str] = []
        self.winner: str | None = None
        self.win_reason: str | None = None
        self.witch_poison_left = 1
        self.witch_antidote_left = 1

        # player states of current night
        self.player_hunted: str | None = None
        self.player_protected: str | None = None
        self.is_hunted_player_saved: bool = False
        self.player_poisoned: str | None = None
        self.player_current_dead: list[str] = []

        # track which night we are in (0 = first night)
        self.night_index: int = 0

    async def _observe(self) -> int:
        await super()._observe()
        # Only messages sent to all ("") or to oneself (self.profile) need to go through
        self.rc.news = [
            msg
            for msg in self.rc.news
            if any(element in [MESSAGE_ROUTE_TO_ALL, self.profile] for element in msg.send_to)
            and any(element == 1 for element in msg.send_to)
        ]
        if not len(self.rc.news):
            self.rc.news = [
                Message(
                    content="all the players are waiting",
                    role=self.profile,
                    sent_from=self.name,
                    cause_by=InstructSpeak,
                    send_to="Moderator",
                )
            ]
        return len(self.rc.news)

    def _parse_game_setup(self, game_setup: str):
        self.game_setup = game_setup
        self.living_players = re.findall(r"Player[0-9]+", game_setup)

        self.werewolf_players = re.findall(r"Player[0-9]+: Werewolf", game_setup)
        self.werewolf_players = [p.replace(": Werewolf", "") for p in self.werewolf_players]

        self.villager_players = re.findall(r"Player[0-9]+: Villager", game_setup)
        self.villager_players = [p.replace(": Villager", "") for p in self.villager_players]

        self.special_role_players = [
            p
            for p in self.living_players
            if p not in self.werewolf_players + self.villager_players
        ]

    def update_player_status(self, player_names: list[str]):
        if not player_names:
            return
        roles_in_env = self.rc.env.get_roles()
        for role_setting, role in roles_in_env.items():
            for player_name in player_names:
                if player_name in role_setting:
                    role.set_status(new_status=1)

    def _record_all_experiences(self):
        roles_in_env = self.rc.env.get_roles()
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        for _, role in roles_in_env.items():
            if role == self:
                continue
            if self.winner == "werewolf":
                outcome = "won" if role.name in self.werewolf_players else "lost"
            else:
                outcome = "won" if role.name not in self.werewolf_players else "lost"
            role.record_experiences(round_id=timestamp, outcome=outcome, game_setup=self.game_setup)

    async def _instruct_speak(self):
        step_idx = self.step_idx % len(STEP_INSTRUCTIONS)
        self.step_idx += 1
        return await InstructSpeak().run(
            step_idx,
            living_players=self.living_players,
            werewolf_players=self.werewolf_players,
            player_hunted=self.player_hunted,
            player_current_dead=self.player_current_dead,
        )

    async def _parse_speak(self, memories):
        logger.info(self.step_idx)

        latest_msg = memories[-1]
        latest_msg_content = latest_msg.content

        match = re.search(r"Player[0-9]+", latest_msg_content[-10:])
        target = match.group(0) if match else ""

        # default return
        msg_content = "Understood"
        send_to = MESSAGE_ROUTE_TO_ALL

        msg_cause_by = latest_msg.cause_by
        if msg_cause_by == Hunt:
            self.player_hunted = target
        elif msg_cause_by == Protect:
            self.player_protected = target
        elif msg_cause_by == Verify:
            if target in self.werewolf_players:
                msg_content = f"{target} is a werewolf"
            else:
                msg_content = f"{target} is a good guy"
            send_to = "Seer"
        elif msg_cause_by == Save:
            if "pass" in latest_msg_content.lower():
                pass
            elif not self.witch_antidote_left:
                msg_content = "You have no antidote left and thus can not save the player"
                send_to = "Witch"
            else:
                self.witch_antidote_left -= 1
                self.is_hunted_player_saved = True
        elif msg_cause_by == Poison:
            if "pass" in latest_msg_content.lower():
                pass
            elif not self.witch_poison_left:
                msg_content = "You have no poison left and thus can not poison the player"
                send_to = "Witch"
            else:
                self.witch_poison_left -= 1
                self.player_poisoned = target

        return msg_content, send_to

    def _update_game_states(self, memories):
        step_idx = self.step_idx % len(STEP_INSTRUCTIONS)
        if step_idx not in [15, 18] or self.step_idx in self.eval_step_idx:
            return
        else:
            # record evaluation, avoid repetitive evaluation at the same step
            self.eval_step_idx.append(self.step_idx)

        # NIGHT ENDS
        if step_idx == 15:
            from camelgym.actions import UserRequirement

            # log first-night werewolf kill + guard save for Fig. 4
            if self.night_index == 0:
                if self.player_hunted:
                    self.rc.env.pub_mes(
                        Message(
                            role=self.profile,
                            sent_from=self.name,
                            content=f"METRIC_WEREWOLF_KILL {self.player_hunted}",
                            cause_by=UserRequirement,
                            send_to="Moderator",
                        )
                    )

                if self.player_protected:
                    self.rc.env.pub_mes(
                        Message(
                            role=self.profile,
                            sent_from=self.name,
                            content=f"METRIC_GUARD_SAVE {self.player_protected}",
                            cause_by=UserRequirement,
                            send_to="Moderator",
                        )
                    )

                # only the first night is used for Fig.4
                self.night_index += 1

            # night ends: after all special roles acted, process the whole night
            self.player_current_dead = []

            if self.player_hunted != self.player_protected and not self.is_hunted_player_saved:
                self.player_current_dead.append(self.player_hunted)
            if self.player_poisoned:
                self.player_current_dead.append(self.player_poisoned)

            self.living_players = [
                p for p in self.living_players if p not in self.player_current_dead
            ]
            self.update_player_status(self.player_current_dead)
            # reset
            self.player_hunted = None
            self.player_protected = None
            self.is_hunted_player_saved = False
            self.player_poisoned = None

        # DAY ENDS
        elif step_idx == 18:
            # day ends: after all roles voted, process all votings
            voting_msgs = memories[-len(self.living_players):]
            voted_all: list[str] = []

            from camelgym.actions import UserRequirement

            for msg in voting_msgs:
                # who this player voted for (if anyone)
                voted = re.search(r"Player[0-9]+", msg.content[-10:])
                if not voted:
                    target_name = "NONE"
                else:
                    target_name = voted.group(0)

                voter_name = msg.sent_from  # e.g. "Player4"

                # METRIC: one vote log per player
                self.rc.env.pub_mes(
                    Message(
                        role=self.profile,
                        sent_from=self.name,
                        content=f"METRIC_VOTE {voter_name} {target_name}",
                        cause_by=UserRequirement,
                        send_to="Moderator",
                    )
                )

                if not voted:
                    continue
                voted_all.append(voted.group(0))

            # majority-vote logic
            if voted_all:
                self.player_current_dead = [
                    Counter(voted_all).most_common(1)[0][0]
                ]
                self.living_players = [
                    p for p in self.living_players if p not in self.player_current_dead
                ]
                self.update_player_status(self.player_current_dead)

        # game's termination condition
        living_werewolf = [p for p in self.werewolf_players if p in self.living_players]
        living_villagers = [p for p in self.villager_players if p in self.living_players]
        living_special_roles = [
            p for p in self.special_role_players if p in self.living_players
        ]

        if not living_werewolf:
            self.winner = "good guys"
            self.win_reason = "werewolves all dead"
        elif not living_villagers or not living_special_roles:
            self.winner = "werewolf"
            self.win_reason = (
                "villagers all dead" if not living_villagers else "special roles all dead"
            )

        if self.winner is not None:
            self._record_all_experiences()

    def _record_game_history(self):
        if self.step_idx % len(STEP_INSTRUCTIONS) == 0 or self.winner is not None:
            logger.info("a night and day cycle completed, examine all history")
            print(self.get_all_memories())
            with open(DEFAULT_WORKSPACE_ROOT / "werewolf_transcript.txt", "w") as f:
                f.write(self.get_all_memories())

    async def _think(self):
        if self.winner is not None:
            self.rc.todo = AnnounceGameResult()
            return

        latest_msg = self.rc.memory.get()[-1]
        if latest_msg.role in ["User"]:
            game_setup = latest_msg.content
            self._parse_game_setup(game_setup)
            self.rc.todo = InstructSpeak()

        elif latest_msg.role in [self.profile]:
            self.rc.todo = InstructSpeak()

        else:
            self.rc.todo = ParseSpeak()

    async def _act(self):
        todo = self.rc.todo
        logger.info(f"{self._setting} ready to {todo}")

        memories = self.get_all_memories(mode="msg")

        self._record_game_history()
        self._update_game_states(memories)

        if isinstance(todo, InstructSpeak):
            msg_content, need_res, msg_to_send_to = await self._instruct_speak()
            if need_res == "yes":
                msg_to_send_to = [msg_to_send_to, need_res]
            msg = Message(
                content=msg_content,
                role=self.profile,
                sent_from=self.name,
                cause_by=InstructSpeak,
                send_to=msg_to_send_to,
            )

        elif isinstance(todo, ParseSpeak):
            msg_content, msg_to_send_to = await self._parse_speak(memories)
            msg = Message(
                content=msg_content,
                role=self.profile,
                sent_from=self.name,
                cause_by=ParseSpeak,
                send_to=msg_to_send_to,
            )

        elif isinstance(todo, AnnounceGameResult):
            msg_content = await AnnounceGameResult().run(
                winner=self.winner, win_reason=self.win_reason
            )
            msg = Message(
                content=msg_content,
                role=self.profile,
                sent_from=self.name,
                cause_by=AnnounceGameResult,
            )

        logger.info(f"{self._setting}: {msg_content}")

        return msg

    def get_all_memories(self, mode: str = "str"):
        memories = self.rc.memory.get()
        if mode == "str":
            memories = [f"{m.sent_from}({m.role}): {m.content}" for m in memories]
            memories = "\n".join(memories)
        return memories
