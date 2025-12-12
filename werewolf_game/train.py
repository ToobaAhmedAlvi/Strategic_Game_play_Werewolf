import asyncio
import matplotlib.pyplot as plt
import numpy as np

from start_game import run_one_game_async

# Global trackers for innovation metrics
ACTION_HISTORY = []     # Stores all chosen actions across all games
LOSS_HISTORY = []       # RL loss per game


# -----------------------------------------------------------------------------
# TRAINING FUNCTION
# -----------------------------------------------------------------------------
async def train_self_play(n_games=50, log_every=5):

    wins = []   # 1 = villagers win, 0 = werewolves win

    for game_id in range(1, n_games + 1):
        print(f"\n=== Running Game {game_id}/{n_games} ===")

        # -----------------------------
        # RUN ONE GAME
        # -----------------------------
        result = await run_one_game_async(
            investment=3.0,
            n_round=1,
            shuffle=True,
            add_human=False,
            use_reflection=True,
            use_experience=False,
        )

        winner = result["winner"]

        # -----------------------------
        # WIN TRACKING
        # -----------------------------
        if winner == "good guys":
            wins.append(1)
            print(f"Game {game_id} Result: Villagers Win (+1 reward)")
        else:
            wins.append(0)
            print(f"Game {game_id} Result: Werewolves Win (-1 reward)")

        # -----------------------------
        # RL LOSS TRACKING
        # -----------------------------
        # Trainer already printed loss; run_one_game_async stored last loss
        if "loss" in result:
            LOSS_HISTORY.append(result["loss"])

        # -----------------------------
        # ACTION HISTORY TRACKING
        # -----------------------------
        if "actions" in result:
            ACTION_HISTORY.extend(result["actions"])

        if game_id % log_every == 0:
            print(f"Progress: {sum(wins)} wins / {game_id} games")

    return wins


# -----------------------------------------------------------------------------
# ANALYSIS & PLOTTING
# -----------------------------------------------------------------------------
def plot_all(wins):

    total_games = len(wins)
    games = range(total_games)

    # =====================================================================
    # 1. SCATTER PLOT OF RAW WINS
    # =====================================================================
    plt.figure(figsize=(8,5))
    plt.scatter(games, wins)
    plt.title("Raw Win Outcomes Per Game (1=Villagers, 0=Werewolves)")
    plt.xlabel("Game #")
    plt.ylabel("Win")
    plt.grid(True)
    plt.savefig("scatter_wins.png", dpi=300)
    plt.show()

    # =====================================================================
    # 2. MOVING AVERAGE WIN RATE (TREND OF LEARNING)
    # =====================================================================
    window = max(3, total_games // 10)
    moving_winrate = np.convolve(wins, np.ones(window)/window, mode="valid")

    plt.figure(figsize=(8,5))
    plt.plot(moving_winrate, linewidth=2)
    plt.title("Moving Average Win Rate (Learning Trend)")
    plt.xlabel("Game #")
    plt.ylabel("Win Rate")
    plt.grid(True)
    plt.savefig("moving_winrate.png", dpi=300)
    plt.show()

    # =====================================================================
    # 3. TRAINING LOSS CURVE
    # =====================================================================
    if len(LOSS_HISTORY) > 0:
        plt.figure(figsize=(8,5))
        plt.plot(LOSS_HISTORY, color="red", linewidth=2)
        plt.title("RL Policy Training Loss Across Games")
        plt.xlabel("Game #")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.savefig("loss_curve.png", dpi=300)
        plt.show()

    # =====================================================================
    # 4. ACTION DIVERSITY ANALYSIS
    # =====================================================================
    if len(ACTION_HISTORY) > 0:
        unique_actions = len(set(ACTION_HISTORY))

        plt.figure(figsize=(7,5))
        plt.bar(["Unique Actions", "Total Actions"],
                [unique_actions, len(ACTION_HISTORY)],
                color=["blue", "gray"])
        plt.title("Action Diversity (Higher = Better Exploration)")
        plt.ylabel("Count")
        plt.savefig("action_diversity.png", dpi=300)
        plt.show()

        print("\n=== ACTION DIVERSITY REPORT ===")
        print("Total actions generated:", len(ACTION_HISTORY))
        print("Unique actions:", unique_actions)
        print("Diversity ratio:", unique_actions / len(ACTION_HISTORY))


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    n_games = int(input("How many self-play games to train? (e.g., 50): "))

    wins = asyncio.run(train_self_play(n_games))

    # Final win rate
    win_rate = sum(wins) / len(wins)
    print(f"\nFinal Win Rate: {win_rate*100:.2f}%")

    # Generate all analysis plots
    plot_all(wins)


if __name__ == "__main__":
    main()
