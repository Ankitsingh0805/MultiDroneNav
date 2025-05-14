import numpy as np

def normalize_position(pos, grid_size):
    return pos / grid_size

def calculate_coverage(mapped_targets, total_targets):
    return len(mapped_targets) / total_targets

def action_mask(valid_actions):
    return [1 if valid else 0 for valid in valid_actions]
