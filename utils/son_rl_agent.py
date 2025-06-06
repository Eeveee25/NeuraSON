import random

class SimpleRLSimulator:
    def __init__(self):
        self.actions = ['increase_power', 'adjust_tilt', 'handover_optimize', 'do_nothing']
    
    def get_state(self, row):
        state = (
            'low_rsrp' if row['RSRP'] < -95 else 'high_rsrp',
            'low_sinr' if row['SINR'] < 10 else 'high_sinr'
        )
        return state

    def choose_action(self, state):
        # Placeholder: random action for now (simulate RL policy)
        return random.choice(self.actions)

    def get_action_recommendation(self, row):
        state = self.get_state(row)
        action = self.choose_action(state)
        return action
