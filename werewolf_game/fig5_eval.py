import re
import numpy as np
import matplotlib.pyplot as plt

# ----- 1. choose which log to use -----
LOG_PATH = "logs/20251208.txt"          # ← use the synthetic example
# LOG_PATH = "logggg.txt"         # ← later, switch to your real log

with open(LOG_PATH, "r", encoding="utf-8", errors="ignore") as f:
    text = f.read()

# ----- 2. split into LLM part and Strategic part -----
marker = "Running experiments for Strategic Language Agent"
idx = text.find(marker)

if idx == -1:
    raise ValueError("Could not find Strategic marker in log file.")

llm_text = text[:idx]
str_text = text[idx:]  # includes the marker and everything after

# ----- 3. helper to count villager vs werewolf wins -----
vill_pattern = r"Villagers win|Villager win"
wolf_pattern = r"Werewolves win|Werewolf win"

def win_rate(block: str):
    v = len(re.findall(vill_pattern, block))
    w = len(re.findall(wolf_pattern, block))
    total = v + w
    if total == 0:
        return 0.0, v, w
    return v / total, v, w

llm_rate, llm_v, llm_w = win_rate(llm_text)
str_rate, str_v, str_w = win_rate(str_text)

print(f"LLM villager win rate:       {llm_rate:.2f}  ({llm_v}/{llm_v+llm_w})")
print(f"Strategic villager win rate: {str_rate:.2f}  ({str_v}/{str_v+str_w})")

# ----- 4. build a 2×2 heatmap (Villagers strategy vs Werewolves strategy) -----
# For this toy example, we just assume: when Villagers use LLM vs
# Werewolves use LLM → same rate as LLM block, etc.
heat = np.array([
    [llm_rate, llm_rate],   # Villagers LLM vs Werewolves LLM / Strategic
    [str_rate, str_rate],   # Villagers Strategic vs Werewolves LLM / Strategic
])

labels = ["LLM", "Strategic"]

# ----- 5. plot heatmap like a mini Fig. 5 -----
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(heat, cmap="RdPu", vmin=0, vmax=1)

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)

# text inside each cell
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{heat[i, j]:.2f}",
                ha="center", va="center", color="black", fontsize=13)

ax.set_xlabel("Werewolves strategy")
ax.set_ylabel("Villagers strategy")
ax.set_title("Villager Win-Rate Heatmap ")

cbar = plt.colorbar(im)
cbar.set_label("Win rate")

plt.tight_layout()
plt.savefig("fig5_heatmap.png", dpi=300)