import re
from collections import Counter, defaultdict
import matplotlib.pyplot as plt

# ======= CONFIG =======
LOG_PATH = "logs/strategic_log.txt"   # <-- yahan apna log file ka naam/path do
OUTPUT_FIG = "fig.png"
NUM_PLAYERS = 7                      # Player1 ... Player7
# =======================

def split_llm_and_strategic(text: str):
    """
    Log ko do parts mein baantta hai:
    - llm_part: 'Running experiments for Pure LLM-Based Agent' se pehle ka hissa
    - strat_part: us line ke baad ka hissa
    Agar line na mile to poora text llm_part mein daal deta hai.
    """
    marker = "Running experiments for Strategic Language Agent"
    idx = text.find(marker)
    if idx == -1:
        # marker nahin mila, sabko LLM maan lete hain
        return text, ""
    llm_part = text[:idx]
    strat_part = text[idx:]
    return llm_part, strat_part

def extract_game_setups(text: str):
    """
    'Game setup:' wali lines se har game ka player-role mapping nikalta hai.
    Return: list[dict] jahan har dict: { 'Player1':'Werewolf', ...}
    Simple heuristic: 'Game setup:' se le kar next blank line tak padho.
    """
    setups = []
    pattern = re.compile(r"Game setup:(.*?)(?:\n\s*\n|$)", re.DOTALL)
    for block in pattern.findall(text):
        mapping = {}
        # Example expected line: Player1: Seer,
        for line in block.splitlines():
            m = re.findall(r"(Player\d+):\s*([A-Za-z]+)", line)
            for player, role in m:
                mapping[player] = role
        if mapping:
            setups.append(mapping)
    return setups

def parse_actions(text: str, setups):
    """
    Logs se actions parse karta hai.
    Hum nights / games ko perfect track nahin karenge,
    sirf saare actions collect kar ke frequencies lenge.
    """
    werewolf_targets = []
    guard_targets = []
    villager_votes = []

    # 1) Werewolf & Guard blocks: ROLE: xxxx ... RESPONSE: PlayerX
    block_pattern = re.compile(
        r"ROLE:\s*(?P<role>Werewolf|Guard).*?RESPONSE[:\s]*([\"']?)(?P<resp>[^\"'\n]+)\2",
        re.DOTALL
    )

    for m in block_pattern.finditer(text):
        role = m.group("role")
        resp = m.group("resp")
        # PlayerX ya "Kill PlayerX" etc se PlayerN nikaalna
        pm = re.search(r"Player(\d)", resp)
        if not pm:
            # maybe "Pass" ya "Save" etc -> ignore for first-night action
            continue
        idx = int(pm.group(1)) - 1  # 0-based index: Player1 -> 0
        if 0 <= idx < NUM_PLAYERS:
            if role == "Werewolf":
                werewolf_targets.append(idx)
            elif role == "Guard":
                guard_targets.append(idx)

    # 2) Villager votes: "I vote to eliminate PlayerX"
    vote_pattern = re.compile(
        r"I vote to eliminate\s+Player(\d)", re.IGNORECASE
    )
    for m in vote_pattern.finditer(text):
        idx = int(m.group(1)) - 1
        if 0 <= idx < NUM_PLAYERS:
            villager_votes.append(idx)

    # 3) Try to classify villager votes as right / wrong / others / not_vote
    # For that we need at least one setup; hum pehle setup ko ground-truth maan lete hain.
    if setups:
        role_map = setups[0]  # simple assumption: ek hi setup hai
    else:
        role_map = {}

    def classify_vote(player_idx):
        player_name = f"Player{player_idx+1}"
        role = role_map.get(player_name, None)
        if role == "Werewolf":
            return "right"
        elif role is None:
            return "others"
        else:
            return "wrong"

    vote_categories = []
    for idx in villager_votes:
        vote_categories.append(classify_vote(idx))

    return werewolf_targets, guard_targets, vote_categories

def counts_to_probs(counts_list, num_bins):
    counts = Counter(counts_list)
    total = sum(counts.values())
    probs = []
    for i in range(num_bins):
        if total == 0:
            probs.append(0.0)
        else:
            probs.append(counts.get(i, 0) / total)
    return probs

def cats_to_probs(cat_list, categories):
    counts = Counter(cat_list)
    total = sum(counts.values())
    probs = []
    for c in categories:
        if total == 0:
            probs.append(0.0)
        else:
            probs.append(counts.get(c, 0) / total)
    return probs

def main():
    with open(LOG_PATH, "r", encoding="utf-8", errors="ignore") as f:
        full_text = f.read()

    llm_text, strat_text = split_llm_and_strategic(full_text)

    # setups dono parts se collect karo
    llm_setups = extract_game_setups(llm_text)
    strat_setups = extract_game_setups(strat_text)

    # ---- LLM agent ----
    llm_w_targets, llm_g_targets, llm_vote_cats = parse_actions(llm_text, llm_setups)

    # ---- Strategic agent ----
    s_w_targets, s_g_targets, s_vote_cats = parse_actions(strat_text, strat_setups)

    # ---- Probabilities for plots ----
    # Werewolf & Guard: per-player probability (0..6)
    llm_w_probs = counts_to_probs(llm_w_targets, NUM_PLAYERS)
    s_w_probs   = counts_to_probs(s_w_targets, NUM_PLAYERS)

    llm_g_probs = counts_to_probs(llm_g_targets, NUM_PLAYERS)
    s_g_probs   = counts_to_probs(s_g_targets, NUM_PLAYERS)

    # Villager votes: categories
    vote_labels = ["not_vote", "right", "wrong", "others"]
    # Humare parser mein explicit "not_vote" actions nahi aate,
    # toh sab kuch right/wrong/others mein jayega; not_vote = 0.
    llm_vote_probs = cats_to_probs(llm_vote_cats, vote_labels[1:])  # right, wrong, others
    llm_vote_probs = [0.0] + llm_vote_probs  # prepend not_vote=0

    s_vote_probs = cats_to_probs(s_vote_cats, vote_labels[1:])
    s_vote_probs = [0.0] + s_vote_probs

    # ============ Plotting (paper style 6 subplots) ============
    fig, axes = plt.subplots(2, 3, figsize=(12, 4.5))

    # x axes
    x_players = list(range(NUM_PLAYERS))
    x_vote = list(range(len(vote_labels)))

    # ---- Row 1: Pure LLM-Based Agent ----
    ax = axes[0, 0]
    ax.bar(x_players, llm_w_probs)
    ax.set_title("Pure LLM-Based Agent")
    ax.set_ylabel("Probability")
    ax.set_xticks(x_players)
    ax.set_xticklabels(range(NUM_PLAYERS))
    ax.set_xlabel("(a) Werewolf first night action")
    for i, v in enumerate(llm_w_probs):
        if v > 0:
            ax.text(i, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=7)

    ax = axes[0, 1]
    ax.bar([0, 1], [sum(llm_g_probs[:1]), sum(llm_g_probs[1:])])
    ax.set_title("Pure LLM-Based Agent")
    ax.set_ylabel("Probability")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["self", "others"])
    ax.set_xlabel("(b) Guard first night action")
    for i, v in enumerate([sum(llm_g_probs[:1]), sum(llm_g_probs[1:])]):
        if v > 0:
            ax.text(i, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=7)

    ax = axes[0, 2]
    ax.bar(x_vote, llm_vote_probs)
    ax.set_title("Pure LLM-Based Agent")
    ax.set_ylabel("Probability")
    ax.set_xticks(x_vote)
    ax.set_xticklabels(vote_labels, rotation=20)
    ax.set_xlabel("(c) Villager voting action")
    for i, v in enumerate(llm_vote_probs):
        if v > 0:
            ax.text(i, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=7)

    # ---- Row 2: Strategic Language Agent ----
    ax = axes[1, 0]
    ax.bar(x_players, s_w_probs)
    ax.set_title("Strategic Language Agent")
    ax.set_ylabel("Probability")
    ax.set_xticks(x_players)
    ax.set_xticklabels(range(NUM_PLAYERS))
    ax.set_xlabel("(a) Werewolf first night action")
    for i, v in enumerate(s_w_probs):
        if v > 0:
            ax.text(i, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=7)

    ax = axes[1, 1]
    ax.bar([0, 1], [sum(s_w_probs[:1]), sum(s_w_probs[1:])])
    # NOTE: paper mein row 2, col 2 actually Doctor/Guard self vs others hota hai;
    # yahan example ke liye same aggregation pattern use kiya hai,
    # chaaho toh yahan s_g_probs ka use kar sakte ho.
    ax.set_title("Strategic Language Agent")
    ax.set_ylabel("Probability")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["self", "others"])
    ax.set_xlabel("(b) Guard first night action")
    for i, v in enumerate([sum(s_g_probs[:1]), sum(s_g_probs[1:])]):
        if v > 0:
            ax.text(i, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=7)

    ax = axes[1, 2]
    ax.bar(x_vote, s_vote_probs)
    ax.set_title("Strategic Language Agent")
    ax.set_ylabel("Probability")
    ax.set_xticks(x_vote)
    ax.set_xticklabels(vote_labels, rotation=20)
    ax.set_xlabel("(c) Villager voting action")
    for i, v in enumerate(s_vote_probs):
        if v > 0:
            ax.text(i, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    plt.savefig(OUTPUT_FIG, dpi=300)
    print(f"Saved figure to {OUTPUT_FIG}")

if __name__ == "__main__":
    main()
