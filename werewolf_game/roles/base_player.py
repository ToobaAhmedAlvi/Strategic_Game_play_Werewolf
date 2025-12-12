import re
import sys
sys.path.append("..")

from camelgym.roles import Role
from camelgym.schema import Message
from camelgym.logs import logger
from actions import ACTIONS, InstructSpeak, Speak, Reflect, NighttimeWhispers
from actions.experience_operation import AddNewExperiences, RetrieveExperiences
from schema import RoleExperience
from camelgym.const import MESSAGE_ROUTE_TO_ALL


class BasePlayer(Role):
    def __init__(
        self,
        name="PlayerXYZ",
        profile="BasePlayer",
        special_action_names=None,
        use_reflection=True,
        use_experience=False,
        use_memory_selection=False,
        new_experience_version="",
        **kwargs,
    ):
        super().__init__(name=name, profile=profile, **kwargs)

        self.status = 0
        self._watch([InstructSpeak])

        special_action_names = special_action_names or []
        self.special_actions = [ACTIONS[n] for n in special_action_names]

        self.set_actions([Speak] + self.special_actions)

        self.use_reflection = use_reflection
        self.use_experience = use_reflection and use_experience
        self.new_experience_version = new_experience_version
        self.use_memory_selection = use_memory_selection

        self.experiences: list[RoleExperience] = []

        self.addresses = {name, profile}

    # -----------------------------------------------------------
    async def _observe(self) -> int:
        if self.status == 1:
            return 0

        await super()._observe()

        self.rc.news = [
            m for m in self.rc.news
            if any(x in [MESSAGE_ROUTE_TO_ALL, self.profile] for x in m.send_to)
        ]

        self.rc.news = [m for m in self.rc.news if "yes" in m.send_to]
        return len(self.rc.news)

    # -----------------------------------------------------------
    async def _think(self):
        news = self.rc.news[0]
        assert news.cause_by == InstructSpeak or \
               news.cause_by == "actions.moderator_actions.InstructSpeak"

        if MESSAGE_ROUTE_TO_ALL in news.send_to:
            self.rc.todo = Speak()
        else:
            self.rc.todo = self.special_actions[0]()

    # -----------------------------------------------------------
    async def _act(self):
        from camelgym.actions.action_node import ActionNode

        todo = self.rc.todo
        logger.info(f"{self._setting}: ready to {todo}")

        memories = self.get_all_memories()
        latest_instruction = self.get_latest_instruction()

        reflection = await Reflect().run(
            profile=self.profile,
            name=self.name,
            context=memories,
            latest_instruction=latest_instruction,
        ) if self.use_reflection else ""

        experiences = RetrieveExperiences().run(
            query=reflection,
            profile=self.profile,
            excluded_version=self.new_experience_version,
        ) if self.use_experience else ""

        memory_block = f"{memories}\n\nReflection:{reflection}\nExperiences:{experiences}"

        # ---------- RL ActionNode ----------
        node = ActionNode(
            key=f"{self.name}_action",
            instruction="Generate the most appropriate Werewolf action.",
            example="I vote Player3.",
            expected_type=str,
        )

        node.set_context(self.rc.env.context)     # RL context must exist
        node.set_llm(self.rc.env.context.llm())   # LLM always required

        await node.simple_fill(memory=memory_block, K=3)

        rsp = node.content
        # ------------------------------------

        send_to = MESSAGE_ROUTE_TO_ALL if isinstance(todo, Speak) else "Moderator"

        msg = Message(
            content=rsp,
            role=self.profile,
            sent_from=self.name,
            cause_by=type(todo),
            send_to=send_to,
        )

        self.experiences.append(
            RoleExperience(
                name=self.name,
                profile=self.profile,
                reflection=reflection,
                instruction=latest_instruction,
                response=rsp,
                version=self.new_experience_version,
            )
        )

        logger.info(f"{self._setting}: {rsp}")
        return msg

    # -----------------------------------------------------------
    def get_all_memories(self) -> str:
        memories = self.rc.memory.get()

        cleaned = []
        pattern = r"[0-9]+ \| "

        for m in memories:
            clean_msg = re.sub(pattern, "", m.content)
            cleaned.append(f"{m.sent_from}: {clean_msg}")

        return "\n".join(cleaned)


    def get_latest_instruction(self) -> str:
        return self.rc.important_memory[-1].content

    def set_status(self, new_status):
        self.status = new_status

    def record_experiences(self, round_id, outcome, game_setup):
        valid = [e for e in self.experiences if len(e.reflection) > 2]
        for e in valid:
            e.round_id = round_id
            e.outcome = outcome
            e.game_setup = game_setup
        AddNewExperiences().run(valid)
