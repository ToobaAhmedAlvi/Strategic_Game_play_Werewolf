from camelgym.actions import Action
import json
import re
from camelgym.const import DEFAULT_WORKSPACE_ROOT
from tenacity import retry, stop_after_attempt, wait_fixed
from pydantic import Field

def rindex(s, sub):
    # Reverse the original string and the substring
    s_reversed = s[::-1]
    sub_reversed = sub[::-1]
    
    # Find the index of the reversed substring in the reversed string
    index_reversed = s_reversed.index(sub_reversed)
    
    # Calculate the original index from the end of the string
    index_original = len(s) - index_reversed - len(sub)
    
    return index_original

def robust_json_loads(input_str):
    input_str = str(input_str) 
    # Attempt to find the starting index of the actual JSON part
    try:
        # Find the first occurrence of '{' which marks the beginning of JSON object
        start_index = input_str.index('{')
        end_index = rindex(input_str, '}')
    except ValueError:
        # If '{' is not found, the input is likely not a valid JSON string
        print("Error: Input string does not contain a valid JSON object. the string is", input_str)
        return "Error: Input string does not contain a valid JSON object."

    # Extract the substring from the first '{' to the end, trimming any leading/trailing whitespace
    json_str = input_str[start_index:end_index+1].strip()
    print("in robust json is", json_str)
    # Attempt to parse the extracted JSON string
    try:
        data = json.loads(json_str)
        return data  # Return the parsed dictionary
    except json.JSONDecodeError as e:
        return f"Failed to decode JSON: {e}"

class Speak(Action):
    """Action: Any speak action in a game"""

    PROMPT_TEMPLATE : str = """
    {
    "BACKGROUND": "It's a Werewolf game, in this game, we have 2 werewolves, 2 villagers, 1 guard, 1 witch, 1 seer. You are __profile__. Note that villager, seer, guard and witch are all in villager side, they have the same objective. Werewolves can collectively hunt ONE player at night."
    ,"HISTORY": "You have knowledge to the following conversation: __context__"
    ,"ATTENTION": "You can NOT VOTE a player who is NOT ALIVE now!"
    ,"REFLECTION": "__reflection__"
    ,"STRATEGY": __strategy__
    ,"PAST_EXPERIENCES": "__experiences__"
    ,"MODERATOR_INSTRUCTION": __latest_instruction__,
    ,"RULE": "Please follow the moderator's latest instruction, figure out if you need to speak your opinion or directly to vote:
              1. If the instruction is to SPEAK, speak in 200 words. Remember the goal of your role and try to achieve it using your speech;
              2. If the instruction is to VOTE, you MUST vote and ONLY say 'I vote to eliminate PlayerX', replace PlayerX with the actual player name, DO NOT include any other words."
    ,"OUTPUT_FORMAT":
        {
        "ROLE": "Your role, in this case, __profile__"
        ,"PLAYER_NAME": "Your name, in this case, __name__"
        ,"LIVING_PLAYERS": "List living players based on MODERATOR_INSTRUCTION. Return a json LIST datatype."
        ,"THOUGHTS": "Based on `MODERATOR_INSTRUCTION` and `RULE`, carefully think about what to say or vote so that your chance of win as __profile__ maximizes.
                      If you find similar situation in `PAST_EXPERIENCES`, you may draw lessons from them to refine your strategy, take better vote action, or improve your speech.
                      Give your step-by-step thought process, you should think no more than 3 steps. For example: My step-by-step thought process:..."
        ,"RESPONSE": "Based on `MODERATOR_INSTRUCTION`, `RULE`, and the 'THOUGHTS' you had, express your opinion or cast a vote."
        }
    }
    """
    STRATEGY : str = """
    Decide whether to reveal your identity based on benefits vs. risks, provide useful information, and vote to eliminate the most suspicious.
    If you have special abilities, pay attention to those who falsely claims your role, for they are probably werewolves.
    """

    def __init__(self, name="Speak", context=None, llm=None):
        super().__init__(name = name, context = context, llm = llm)

    @retry(stop=stop_after_attempt(2), wait=wait_fixed(1))
    async def run(self, profile: str, name: str, context: str, latest_instruction: str, reflection: str = "", experiences: str = ""):

        prompt = (
            self.PROMPT_TEMPLATE.replace("__context__", context).replace("__profile__", profile)
            .replace("__name__", name).replace("__latest_instruction__", latest_instruction)
            .replace("__strategy__", self.STRATEGY).replace("__reflection__", reflection)
            .replace("__experiences__", experiences)
        )

        rsp = await self._aask(prompt)
        rsp = rsp.replace("\n", " ")
        # rsp = rsp[3:-4]
        # rsp = rsp.replace("json", " ").strip()
        rsp_json = robust_json_loads(rsp)
        # rsp_json = json.loads(rsp)
        try:
            res = rsp_json['RESPONSE']
        except:
            res = str(rsp)

        return res

class NighttimeWhispers(Action):
    """

    Action: nighttime whispers with thinking processes

    Usage Example:

        class Hunt(NighttimeWhispers):
            def __init__(self, name="Hunt", context=None, llm=None):
                super().__init__(name, context, llm)

        class Protect(NighttimeWhispers):
            def __init__(self, name="Protect", context=None, llm=None):
                super().__init__(name, context, llm)

        class Verify(NighttimeWhispers):
            def __init__(self, name="Verify", context=None, llm=None):
                super().__init__(name, context, llm)

        class Save(NighttimeWhispers):
            def __init__(self, name="Save", context=None, llm=None):
                super().__init__(name, context, llm)

            def _update_prompt_json(self, prompt_json: dict, profile: str, name: str, context: str, **kwargs):
                del prompt_json['ACTION']
                del prompt_json['ATTENTION']
                prompt_json["OUTPUT_FORMAT"]["THOUGHTS"] = "It is night time. Return the thinking steps of your decision of whether to save the player JUST be killed at this night."
                prompt_json["OUTPUT_FORMAT"]["RESPONSE"] = "Follow the Moderator's instruction, decide whether you want to save that person or not. Return SAVE or PASS."
                return prompt_json

        class Poison(NighttimeWhispers):
            def __init__(self, name="Poison", context=None, llm=None):
                super().__init__(name, context, llm)

            def _update_prompt_json(self, prompt_json: dict, profile: str, name: str, context: str, **kwargs):
                prompt_json["OUTPUT_FORMAT"]["RESPONSE"] += "Or if you want to PASS, return PASS."
                return prompt_json
    """

    PROMPT_TEMPLATE : str = """
    {
    "BACKGROUND": "It's a Werewolf game, in this game, we have 2 werewolves, 2 villagers, 1 guard, 1 witch, 1 seer. You are __profile__. Note that villager, seer, guard and witch are all in villager side, they have the same objective. Werewolves can collectively hunt ONE player at night."
    ,"HISTORY": "You have knowledge to the following conversation: __context__"
    ,"ACTION": "Choose one living player to __action__."
    ,"ATTENTION": "1. You can only __action__ a player who is alive this night! And you can not __action__ a player who is dead this night!  2. `HISTORY` is all the information you observed, DONT hallucinate other player actions!"
    ,"REFLECTION": "__reflection__"
    ,"STRATEGY": "__strategy__"
    ,"PAST_EXPERIENCES": "__experiences__"
    ,"OUTPUT_FORMAT":
        {
        "ROLE": "Your role, in this case, __profile__"
        ,"PLAYER_NAME": "Your name, in this case, __name__"
        ,"LIVING_PLAYERS": "List the players who is alive based on moderator's latest instruction. Return a json LIST datatype."
        ,"THOUGHTS": "Choose one living player from `LIVING_PLAYERS` to __action__ this night. Return the reason why you choose to __action__ this player. If you observe nothing at first night, DONT imagine unexisting player actions! If you find similar situation in `PAST_EXPERIENCES`, you may draw lessons from them to refine your strategy and take better actions. Give your step-by-step thought process, you should think no more than 3 steps. For example: My step-by-step thought process:..."
        ,"RESPONSE": "As a __profile__, you should choose one living player from `LIVING_PLAYERS` to __action__ this night according to the THOUGHTS you have just now. Return the player name ONLY."
        }
    }
    """
    STRATEGY : str =  """
    Decide which player is most threatening to you or most needs your support, take your action correspondingly.
    """

    def __init__(self, name="NightTimeWhispers", context=None, llm=None):
        super().__init__(name = name, context = context, llm = llm)

    def _construct_prompt_json(self, role_profile: str, role_name: str, context: str, reflection: str, experiences: str, **kwargs):
        prompt_template = self.PROMPT_TEMPLATE

        def replace_string(prompt_json: dict):
            k: str
            for k in prompt_json.keys():
                if isinstance(prompt_json[k], dict):
                    prompt_json[k] = replace_string(prompt_json[k])
                    continue
                prompt_json[k] = prompt_json[k].replace("__profile__", role_profile)
                prompt_json[k] = prompt_json[k].replace("__name__", role_name)
                prompt_json[k] = prompt_json[k].replace("__context__", context)
                prompt_json[k] = prompt_json[k].replace("__action__", self.name)
                prompt_json[k] = prompt_json[k].replace("__strategy__", self.STRATEGY)
                prompt_json[k] = prompt_json[k].replace("__reflection__", reflection)
                prompt_json[k] = prompt_json[k].replace("__experiences__", experiences)

            return prompt_json
        
        prompt_json: dict = json.loads(prompt_template)

        prompt_json = replace_string(prompt_json)

        prompt_json: dict = self._update_prompt_json(prompt_json, role_profile, role_name, context, reflection, experiences, **kwargs)
        assert isinstance(prompt_json, dict)

        prompt: str = json.dumps(prompt_json, indent=4, ensure_ascii=False)
        
        return prompt

    def _update_prompt_json(
        self, prompt_json: dict, role_profile: str, role_name: str, context: str, reflection: str, experiences: str
    ) -> dict:
        # one can modify the prompt_json dictionary here
        return prompt_json

    @retry(stop=stop_after_attempt(2), wait=wait_fixed(1))
    async def run(self, context: str, profile: str, name: str, reflection: str = "", experiences: str = ""):

        prompt = self._construct_prompt_json(
            role_profile=profile, role_name=name, context=context, reflection=reflection, experiences=experiences
        )

        rsp = await self._aask(prompt)
        rsp = rsp.replace("\n", " ")

        # Try to parse JSON first
        rsp_json = robust_json_loads(rsp)

        choice = None

        # Case 1: proper dict JSON
        if isinstance(rsp_json, dict):
            if "RESPONSE" in rsp_json:
                choice = rsp_json["RESPONSE"]
            else:
                # maybe weird capitalization or spaces
                for k, v in rsp_json.items():
                    if k.strip().upper() == "RESPONSE":
                        choice = v
                        break

        # Case 2: list of dicts
        elif isinstance(rsp_json, list):
            for item in rsp_json:
                if isinstance(item, dict):
                    if "RESPONSE" in item:
                        choice = item["RESPONSE"]
                        break
                    for k, v in item.items():
                        if k.strip().upper() == "RESPONSE":
                            choice = v
                            break
                if choice is not None:
                    break

        # Case 3: parsing failed or returned a string → regex on raw text
        if choice is None:
            s = str(rsp_json)
            # Try to match 'RESPONSE: PlayerX'
            m = re.findall(r'RESPONSE\s*:\s*([A-Za-z0-9_"]+)', s)
            if m:
                choice = m[-1]  # last RESPONSE in the text
            else:
                # fallback: any PlayerN mention
                m2 = re.findall(r'(Player[1-9][0-9]*)', s)
                if m2:
                    choice = m2[-1]
                else:
                    # absolute last fallback: PASS (do nothing)
                    choice = "PASS"

        # Normalize choice to a clean string
        if isinstance(choice, (list, dict)):
            choice = json.dumps(choice, ensure_ascii=False)
        else:
            choice = str(choice).strip().strip('"').strip("'")

        return f"{self.name} {choice}"

class Reflect(Action):

    PROMPT_TEMPLATE: str = """
    {
    "BACKGROUND": "It's a Werewolf game, in this game, we have 2 werewolves, 2 villagers, 1 guard, 1 witch, 1 seer. You are __profile__. Note that villager, seer, guard and witch are all in villager side, they have the same objective. Werewolves can collectively hunt ONE player at night."
    ,"HISTORY": "You have knowledge to the following conversation: __context__"
    ,"MODERATOR_INSTRUCTION": __latest_instruction__,
    ,"OUTPUT_FORMAT" (a json):
        {
        "ROLE": "Your role, in this case, __profile__"
        ,"PLAYER_NAME": "Your name, in this case, __name__"
        "GAME_STATES": "You are about to follow `MODERATOR_INSTRUCTION`, but before taking any action, analyze each player, including the living and the dead, and summarize the game states.
                        For each player, your reflection should be a ONE-LINE json covering the following dimension, return a LIST of jsons (return an empty LIST for the first night):
                        [
                            {"TARGET": "the player you will analyze, if the player is yourself or your werewolf partner, indicate it" ,"STATUS": "living or dead, if dead, how was he/she possibly killed?", "CLAIMED_ROLE": "claims a role or not, if so, what role, any contradiction to others? If there is no claim, return 'None'", "SIDE_WITH": "sides with which players? If none, return 'None'", "ACCUSE": "accuses which players? If none, return 'None'"}
                            ,{...}
                            ,...
                        ]"
        ,"REFLECTION": "Based on the whole `GAME_STATES`, return a json (return an empty string for the first night):
                       {
                            "Player1": "the true role (werewolf / special role / villager, living or dead) you infer about him/her, and why is this role? If the player is yourself or your werewolf partner, indicate it."
                            ,...
                            ,"Player7": "the true role (werewolf / special role / villager, living or dead) you infer about him/her, and why is this role? If the player is yourself or your werewolf partner, indicate it."
                            ,"GAME_STATE_SUMMARIZATION": "summarize the current situation from your standpoint in one sentence, your summarization should catch the most important information from your reflection, such as conflicts, number of living werewolves, special roles, and villagers."
                       }"
        }
    }
    """

    def __init__(self, name: str = "Reflect", context=None, llm=None):
        super().__init__(name=name, context=context, llm=llm)

    @retry(stop=stop_after_attempt(2), wait=wait_fixed(1))
    async def run(self, profile: str, name: str, context: str, latest_instruction: str):

        prompt = (
            self.PROMPT_TEMPLATE.replace("__context__", context)
            .replace("__profile__", profile)
            .replace("__name__", name)
            .replace("__latest_instruction__", latest_instruction)
        )

        rsp = await self._aask(prompt)
        # flatten newlines so the JSON extractor has an easier time
        rsp = rsp.replace("\n", " ")

        # Try to parse with the helper
        rsp_json = robust_json_loads(rsp)

        # ---- Robust handling: handle dict / list / string safely ----

        # Case 1: already a dict, ideal path
        if isinstance(rsp_json, dict):
            if "REFLECTION" in rsp_json:
                return json.dumps(rsp_json["REFLECTION"], ensure_ascii=False, indent=2)
            # No REFLECTION key, just return the whole dict
            return json.dumps(rsp_json, ensure_ascii=False, indent=2)

        # Case 2: list of dicts, maybe one has REFLECTION
        if isinstance(rsp_json, list):
            for item in rsp_json:
                if isinstance(item, dict) and "REFLECTION" in item:
                    return json.dumps(item["REFLECTION"], ensure_ascii=False, indent=2)
            # No REFLECTION key anywhere, return entire list
            return json.dumps(rsp_json, ensure_ascii=False, indent=2)

        # Case 3: anything else (string, etc.) – don't crash, just return it
        try:
            return json.dumps(rsp_json, ensure_ascii=False, indent=2)
        except TypeError:
            # completely non-JSON, return plain text
            return str(rsp_json)
