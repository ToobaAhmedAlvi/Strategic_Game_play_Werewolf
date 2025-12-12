import asyncio
import random
import fire

from camelgym.logs import logger
from camelgym.team import Team
from camelgym.environment.werewolf_env.werewolf_env import WerewolfEnv

from roles import Moderator, Villager, Werewolf, Guard, Seer, Witch
from roles.human_player import prepare_human_player

from camelgym.actions import UserRequirement
from camelgym.schema import Message


def init_game_setup(
    shuffle=True,
    add_human=False,
    use_reflection=True,
    use_experience=False,
    use_memory_selection=False,
    new_experience_version=""
):
    roles = [Villager, Villager, Werewolf, Werewolf, Guard, Seer, Witch]

    if shuffle:
        random.shuffle(roles)

    if add_human:
        idx = random.randint(0, len(roles) - 1)
        assigned = roles[idx]
        roles[idx] = prepare_human_player(assigned)

    players = [
        role(
            name=f"Player{i+1}",
            use_reflection=use_reflection,
            use_experience=use_experience,
            use_memory_selection=use_memory_selection,
            new_experience_version=new_experience_version
        )
        for i, role in enumerate(roles)
    ]

    if add_human:
        logger.info(
            f"You are assigned {players[idx].name}({players[idx].profile})"
        )

    setup_lines = ["Game setup:"] + [
        f"{p.name}: {p.profile}," for p in players
    ]
    game_setup = "\n".join(setup_lines)

    return game_setup, players


# ----------------------------------------------------------------------
async def start_game(
    investment=3.0,
    n_round=1,
    shuffle=True,
    add_human=False,
    use_reflection=True,
    use_experience=False,
    use_memory_selection=False,
    new_experience_version=""
):
    env = WerewolfEnv(desc="werewolf game")

    game_setup, players = init_game_setup(
        shuffle=shuffle,
        add_human=add_human,
        use_reflection=use_reflection,
        use_experience=use_experience,
        use_memory_selection=use_memory_selection,
        new_experience_version=new_experience_version
    )

    players = [Moderator()] + players
    env.add_roles(players)

    for p in players:
        env.set_addresses(p, p.addresses)

    env.pub_mes(
        Message(
            role="User",
            content=game_setup,
            cause_by=UserRequirement,
            restricted_to="Moderator"
        )
    )

    game = Team(investment=investment, env=env, roles=players)
    await game.run(n_round=n_round)


# ----------------------------------------------------------------------
async def run_one_game_async(
    investment=3.0,
    n_round=1,
    shuffle=True,
    add_human=False,
    use_reflection=True,
    use_experience=False,
    use_memory_selection=False,
    new_experience_version=""
):
    env = WerewolfEnv(desc="werewolf game")

    # Track actions for diversity metric
    ctx = env.context
    ctx.action_history = []   # stores chosen action text

    game_setup, players = init_game_setup(
        shuffle=shuffle,
        add_human=add_human,
        use_reflection=use_reflection,
        use_experience=use_experience,
        use_memory_selection=use_memory_selection,
        new_experience_version=new_experience_version
    )

    players = [Moderator()] + players
    env.add_roles(players)

    for p in players:
        env.set_addresses(p, p.addresses)

    env.pub_mes(
        Message(
            role="User",
            content=game_setup,
            cause_by=UserRequirement,
            restricted_to="Moderator"
        )
    )

    game = Team(investment=investment, env=env, roles=players)
    await game.run(n_round=n_round)

    # ---------------------------------------------------------
    # RL TRAINING SECTION
    # ---------------------------------------------------------
    ctx = env.context

    # ------------- BASE REWARD -------------------
    if env.winner == "good guys":
        base_reward = +1
    else:
        base_reward = -1

    # ------------- REWARD SHAPING ----------------
    shaped_trajectories = []

    # Helper: check if player is alive at end
    def player_survived(name):
        for entry in env.players_status:
            if entry["name"] == name:
                return entry["is_alive"]
        return False

    for (embeds, action_index, _) in ctx.buffer.trajectories:

        shaped_reward = base_reward

        # -----------------------------------------------------
        # 1️ Reward survival of the acting player
        # Each ActionNode belongs to 1 player → index aligns
        # -----------------------------------------------------
        try:
            acting_player = players[action_index + 1]   # +1 because Moderator is index 0
            if player_survived(acting_player.name):
                shaped_reward += 0.20
        except:
            pass

        # -----------------------------------------------------
        # 2️ Action diversity penalty → avoid repeating actions
        # -----------------------------------------------------
        if len(ctx.action_history) > 1:
            last_action = ctx.action_history[-1]
            if last_action == ctx.action_history[-2]:
                shaped_reward -= 0.10   # penalize repetition
            else:
                shaped_reward += 0.05   # reward diversity

        shaped_trajectories.append((embeds, action_index, shaped_reward))

    # -----------------------------------------------------
    # TRAIN THE POLICY (entropy regularization already inside trainer)
    # -----------------------------------------------------
    loss = None
    if len(shaped_trajectories) > 0:
        loss = ctx.trainer.train(shaped_trajectories)
        print("[RL] Training Loss:", loss)

    ctx.buffer.clear()

    history = env.log_messages

    return {
        "winner": env.winner,
        "history": history,
        "players": players,
        "loss": loss,
        "actions": ctx.action_history,
    }


# ----------------------------------------------------------------------
def run_one_game(
    investment=3.0,
    n_round=1,
    shuffle=True,
    add_human=False,
    use_reflection=False,
    use_experience=False,
    use_memory_selection=False,
    new_experience_version=""
):
    return asyncio.run(
        run_one_game_async(
            investment=investment,
            n_round=n_round,
            shuffle=shuffle,
            add_human=add_human,
            use_reflection=use_reflection,
            use_experience=use_experience,
            use_memory_selection=use_memory_selection,
            new_experience_version=new_experience_version,
        )
    )


# ----------------------------------------------------------------------
def main(
    investment=20.0,
    n_round=1,
    shuffle=True,
    add_human=False,
    use_reflection=False,
    use_experience=False,
    use_memory_selection=False,
    new_experience_version=""
):
    asyncio.run(
        start_game(
            investment,
            n_round,
            shuffle,
            add_human,
            use_reflection,
            use_experience,
            use_memory_selection,
            new_experience_version
        )
    )


if __name__ == "__main__":
    fire.Fire(main)
