import re
from collections import Counter
import matplotlib.pyplot as plt

# >>>>>>>>>> CONFIG: change this to your real log file <<<<<<<<<<
LOG_PATH = "logs/strategic_log.txt"   # or "logggg.txt" etc.
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


def analyze_log(text: str):
    """
    Parse the combined log and return stats for:
      - 'llm'      : content BEFORE "Running experiments for Strategic Language Agent ..."
      - 'strategic': content AFTER that line

    For each agent we collect:
      stats[agent]['kills'][role]  = how many times werewolves intended to kill that role
      stats[agent]['saves'][role]  = how many times guard intended to protect that role
      stats[agent]['votes'][cat]   = vote categories: 'right', 'wrong', 'others'
    """
    # By your description: everything at the start is LLM
    agent_type = "llm"

    stats = {
        "llm": {
            "kills": Counter(),
            "saves": Counter(),
            "votes": Counter(),
        },
        "strategic": {
            "kills": Counter(),
            "saves": Counter(),
            "votes": Counter(),
        },
    }

    # Current game setup: PlayerX -> Role
    player_role = {}
    parsing_setup = False
    setup_remaining = 0

    # Track current actor
    last_player = None
    last_role = None
    last_action = None
    used_action = False

    for line in text.splitlines():

        # ---- switch to Strategic section when we see this line ----
        if "Running experiments for Strategic Language Agent" in line:
            agent_type = "strategic"

        # ---- game setup (Player1: Seer, ...) ----
        if "Game setup:" in line:
            player_role = {}
            parsing_setup = True
            setup_remaining = 7   # 7 players
            continue

        if parsing_setup and setup_remaining > 0:
            m = re.search(r"(Player\d+):\s*([A-Za-z]+)", line)
            if m:
                player_role[m.group(1)] = m.group(2)
                setup_remaining -= 1
                if setup_remaining == 0:
                    parsing_setup = False
            continue

        # ---- who is about to act? ----
        m = re.search(
            r"roles\.base_player:_act:90 - (Player\d+)\(([^)]+)\): ready to (\w+)",
            line,
        )
        if m:
            last_player = m.group(1)
            last_role = m.group(2)
            last_action = m.group(3)
            used_action = False
            continue

        # =========================================================
        # 1) Werewolf night kills (all nights)
        # =========================================================
        if last_role == "Werewolf" and last_action in ("Hunt", "NightTimeWhispers") and not used_action:
            km = re.search(r"\b(?:Kill|RESPONSE|My response)\s*:? *Player(\d+)", line)
            if km:
                target = f"Player{km.group(1)}"
                role = player_role.get(target, "Unknown")
                stats[agent_type]["kills"][role] += 1
                used_action = True
                continue

        # =========================================================
        # 2) Guard saves (all nights)
        # =========================================================
        if last_role == "Guard" and last_action == "Protect" and not used_action:
            gm = re.search(r"\b(?:Protect|RESPONSE|Save)\s*:? *Player(\d+)", line)
            if gm:
                target = f"Player{gm.group(1)}"
                role = player_role.get(target, "Unknown")
                stats[agent_type]["saves"][role] += 1
                used_action = True
                continue

        # =========================================================
        # 3) Daytime votes: "I vote to eliminate PlayerX"
        # =========================================================
        if last_action in ("Speak", "Impersonate") and not used_action:
            vm = re.search(r"I vote to eliminate Player(\d+)", line)
            if vm:
                target = f"Player{vm.group(1)}"
                target_role = player_role.get(target, "Unknown")

                if last_role == "Werewolf":
                    # all werewolf votes count as 'others'
                    cat = "others"
                else:
                    # good roles
                    if target_role == "Werewolf":
                        cat = "right"
                    else:
                        cat = "wrong"

                stats[agent_type]["votes"][cat] += 1
                used_action = True
                continue

    return stats


def normalize(counter: Counter, categories):
    total = sum(counter.values())
    if total == 0:
        return [0.0 for _ in categories]
    return [counter.get(c, 0) / total for c in categories]


def main():
    with open(LOG_PATH, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    stats = analyze_log(text)

    kill_roles = ["Villager", "Seer", "Guard", "Witch", "Werewolf", "Unknown"]
    save_roles = ["Villager", "Seer", "Guard", "Witch", "Werewolf", "Unknown"]
    vote_cats = ["right", "wrong", "others","non vote"]

    agents = ["llm", "strategic"]
    agent_titles = {
        "llm": "Pure LLM-Based Agent (before Strategic line)",
        "strategic": "Strategic Language Agent",
    }

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    for row, agent in enumerate(agents):
        # ---- kills ----
        ax_k = axes[row, 0]
        p_k = normalize(stats[agent]["kills"], kill_roles)
        ax_k.bar(kill_roles, p_k)
        ax_k.set_ylim(0, 1.0)
        if row == 0:
            ax_k.set_title("Werewolf kill targets (all nights)")
        ax_k.set_xticklabels(kill_roles, rotation=30, ha="right")
        for x, y in enumerate(p_k):
            if y > 0:
                ax_k.text(x, y + 0.01, f"{y:.2f}", ha="center", va="bottom", fontsize=8)

        # ---- saves ----
        ax_s = axes[row, 1]
        p_s = normalize(stats[agent]["saves"], save_roles)
        ax_s.bar(save_roles, p_s)
        ax_s.set_ylim(0, 1.0)
        if row == 0:
            ax_s.set_title("Guard protect targets (all nights)")
        ax_s.set_xticklabels(save_roles, rotation=30, ha="right")
        for x, y in enumerate(p_s):
            if y > 0:
                ax_s.text(x, y + 0.01, f"{y:.2f}", ha="center", va="bottom", fontsize=8)

        # ---- votes ----
        ax_v = axes[row, 2]
        p_v = normalize(stats[agent]["votes"], vote_cats)
        ax_v.bar(vote_cats, p_v)
        ax_v.set_ylim(0, 1.0)
        if row == 0:
            ax_v.set_title("Vote categories (all days)")
        ax_v.set_xticklabels(vote_cats, rotation=0)
        for x, y in enumerate(p_v):
            if y > 0:
                ax_v.text(x, y + 0.01, f"{y:.2f}", ha="center", va="bottom", fontsize=8)

        # y label with agent name
        axes[row, 0].set_ylabel(f"{agent_titles[agent]}\n\nProbability", fontsize=9)

    plt.tight_layout()
    plt.savefig("fig4_all_nights.png", dpi=300)
   

    print("=== Raw counts for checking ===")
    for a in agents:
        print(f"\n[{agent_titles[a]}]")
        print("Kills:", dict(stats[a]["kills"]))
        print("Saves:", dict(stats[a]["saves"]))
        print("Votes:", dict(stats[a]["votes"]))


if __name__ == "__main__":
    main()
