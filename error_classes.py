import numpy as np
from core_classes import Position


class ErrorModel:
    """Base class for different error models (GPS, INS, etc.)."""

    def apply_error(self, position):
        """Apply error to a position. To be implemented by subclasses."""
        raise NotImplementedError


class GPSErrorModel(ErrorModel):
    """Implements GPS-specific error model."""

    def __init__(self, std_dev):
        self.std_dev = std_dev

    def apply_error(self, position_tuple):
        """Apply GPS error to a position."""

        # Add random Gaussian error directly in meters
        error_x = np.random.normal(0, self.std_dev)
        error_y = np.random.normal(0, self.std_dev)

        perturbed_x = position_tuple[0] + error_x
        perturbed_y = position_tuple[1] + error_y

        # Calculate the precision radius (magnitude of error vector)
        precision_radius = np.sqrt(error_x ** 2 + error_y ** 2)

        # Create and return a new Position object with the perturbed coordinates
        return Position(perturbed_x, perturbed_y, precision_radius)


class CommunicationDistanceErrorModel(ErrorModel):
    """Implements a communication-specific distance error model."""

    def __init__(self, std_dev, systematic_bias):
        self.std_dev = std_dev
        self.systematic_bias = systematic_bias

    def apply_error(self, original_distance):
        """Apply random Gaussian and systematic error to a communication distance."""
        random_error = np.random.normal(0, self.std_dev)
        perturbed_distance = original_distance + random_error + self.systematic_bias
        return perturbed_distance


class PositionEstimator:
    """Base class for different position estimation methods."""

    def estimate_position(self, target_vehicle, all_vehicles, rsus, step):
        """Estimate position. To be implemented by subclasses."""
        raise NotImplementedError
