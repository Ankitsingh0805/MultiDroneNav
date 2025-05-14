class MetricTracker:
    def __init__(self):
        self.episode_rewards = []
        self.collision_counts = []
        self.coverage_percentages = []
        
    def update(self, episode_data):
        self.episode_rewards.append(
            sum(episode_data.values()) / len(episode_data)
        )
        
    def avg_reward(self, window=10):
        if len(self.episode_rewards) < window:
            return sum(self.episode_rewards) / len(self.episode_rewards)
        return sum(self.episode_rewards[-window:]) / window
    
    def reset(self):
        self.episode_rewards = []
        self.collision_counts = []
        self.coverage_percentages = []
