class Environment:
    def reset(self, desired_state=None):
        """
        Returns:
            state, goal
        """
        raise NotImplementedError()

    def step(self, action):
        """
        Returns:
            state, goal, done
        """
        raise NotImplementedError()
