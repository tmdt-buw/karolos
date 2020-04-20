import numpy as np

class Joint:

    def __init__(self, initial_position, limits, max_velocity, torque):

        self.initial_position = initial_position
        self.limits = limits
        self.max_velocity = max_velocity
        self.torque = torque

    def normalize_position(self, position):
        """Normalize position in interval limits to interval [-1,1]"""
        normalized_position = position - self.limits[0]
        normalized_position /= self.limits[1] - self.limits[0]
        normalized_position *= 2
        normalized_position -= 1

        return normalized_position
    
    def normalize_velocity(self, velocity):
        """Normalize position in interval limits to interval [-1,1]"""
        normalized_velocity = velocity + self.max_velocity
        normalized_velocity /= 2 * self.max_velocity
        normalized_velocity *= 2
        normalized_velocity -= 1

        return normalized_velocity

    def denormalize_position(self, normalized_position):
        """Denormalize position in interval [-1,1] to interval limits"""
        position = normalized_position + 1.
        position /= 2.
        position *= self.limits[1] - self.limits[0]
        position += self.limits[0]

        return position

    def get_random_position(self):
        return np.random.uniform(*self.limits)


if __name__ == "__main__":
    joint = Joint(0, (-2.8973, 2.8973), 2.1750, 87)

    normalized_position = joint.normalize_position(1.4)
    print(normalized_position)
    denormalized_position = joint.denormalize_position(normalized_position)

    print(denormalized_position)