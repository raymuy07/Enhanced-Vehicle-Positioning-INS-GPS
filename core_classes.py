import geopy


class Position:
    """Represents a geographical position with optional precision information."""

    def __init__(self, latitude, longitude, precision_radius=None):
        self.latitude = latitude
        self.longitude = longitude
        self.precision_radius = precision_radius

    def calculate_distance(self, other_position):
        """Calculate distance between this position and another."""
        return geopy.distance.geodesic((self.latitude, self.longitude),
                                  (other_position.latitude, other_position.longitude)).meters


class Vehicle:
    """Represents a vehicle in the simulation."""

    def __init__(self, vehicle_id):
        self.id = vehicle_id
        self.history = []  # Will store PositionRecord objects

    def update(self, real_position, speed, step, error_model):
        """Update vehicle with new position data."""
        # Generate position with error using the provided error model
        measured_position = error_model.apply_error(real_position)

        record = PositionRecord(
            step=step,
            real_position=real_position,
            measured_position=measured_position,
            speed=speed
        )
        self.history.append(record)

    @property
    def current_record(self):
        """Get the most recent position record."""
        return self.history[-1] if self.history else None


class PositionRecord:
    """Stores position data for a specific time step."""

    def __init__(self, step, real_position, measured_position, speed):
        self.step = step
        self.real_position = real_position  # Position without error
        self.measured_position = measured_position  # Position with error
        self.speed = speed
        self.estimated_positions = {}  # Different position estimates, keyed by method name


class RSU:
    """Roadside Unit - fixed infrastructure position."""

    def __init__(self, rsu_id, position):
        self.id = rsu_id
        self.position = position  # A Position object
        self.proximity_radius = 0.1  # Default value