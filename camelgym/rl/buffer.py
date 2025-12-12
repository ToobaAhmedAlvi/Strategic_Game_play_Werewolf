class ExperienceBuffer:
    """
    Stores (embeddings, chosen_index, reward) tuples.
    RLTrainer consumes these trajectories after each game.
    """

    def __init__(self):
        self.trajectories = []

    def add(self, embeds, action_index, reward=0):
        """
        embeds: list of vectors (K x 384)
        action_index: chosen action index
        reward: default 0, filled later
        """
        self.trajectories.append([embeds, action_index, reward])

    def apply_reward_to_all(self, reward):
        """Set reward for every action taken during the game."""
        for traj in self.trajectories:
            traj[2] = reward

    def get(self):
        return self.trajectories

    def clear(self):
        self.trajectories = []
