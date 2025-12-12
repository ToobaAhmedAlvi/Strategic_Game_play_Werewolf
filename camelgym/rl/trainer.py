# trainer.py
import torch
import torch.optim as optim
import torch.nn.functional as F

class RLTrainer:
    """
    REINFORCE training with entropy regularization for better exploration.
    """

    def __init__(self, policy, lr=1e-4, entropy_beta=0.01):
        self.policy = policy
        self.optimizer = optim.Adam(policy.parameters(), lr=lr)
        self.entropy_beta = entropy_beta  # Strength of entropy regularization

    def train(self, trajectories):
        """
        trajectories: list of (embeds, action_index, reward)
        """
        losses = []

        for embeds, action_index, reward in trajectories:
            embeds = torch.tensor(embeds, dtype=torch.float)   # (K, embed_dim)

            scores = self.policy(embeds).squeeze()             # (K)
            probs = F.softmax(scores, dim=0)
            log_probs = F.log_softmax(scores, dim=0)

            # REINFORCE loss
            log_prob = log_probs[action_index]
            reinforce_loss = -log_prob * reward

            # Entropy regularization
            entropy = -torch.sum(probs * log_probs)
            entropy_loss = -self.entropy_beta * entropy

            # Total loss
            loss = reinforce_loss + entropy_loss
            losses.append(loss)

        total_loss = torch.stack(losses).sum()

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item()
