from learnrl import Agent, Playground


class OptionDQAgent(Agent):

    def __init__(self, env, options):
        self.option_id = None
        self.options = options
    
    def use_option(self, option_id, observation):
        option_index = self.option_id - self.n_actions
        action, option_done = self.options[option_index](observation, greedy)[0]
        if option_done:
            self.option_id = None
    
    def deliberate_option(self, observations):
        Q = self.action_value(observations)
        return self.control(Q, greedy)[0]

    def act(self, observation, greedy=False):
        observations = tf.expand_dims(observation, axis=0)

        if self.option_id is not None:
            self.use_option(self.option_id, observation)
        else:
            action = self.deliberate_option(observations)

        if action > self.n_actions:
            self.option_id = action
            self.use_option(self.option_id, observation)

        return action
