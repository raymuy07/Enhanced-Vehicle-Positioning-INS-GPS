from geopy.distance import geodesic
from rsu_config import rsu_points_by_simulation


class Position:
    """Represents a geographical position with optional precision information."""

    def __init__(self, x, y, precision_radius=None):
        self.x = x
        self.y = y
        self.precision_radius = precision_radius

    def cartesian_distance_to(self, other):
        """Calculate Euclidian distance to another position or (x, y) tuple."""
        if isinstance(other, Position):
            x2, y2 = other.x, other.y
        elif isinstance(other, tuple) and len(other) == 2:
            x2, y2 = other
        else:
            raise ValueError("Unsupported input for cartesian_distance_to")

        distance = ((self.x - x2) ** 2 + (self.y - y2) ** 2) ** 0.5
        return distance

    def geodesic_distance_to(self, other):
        """Calculate geodesic distance to another position or (lat, lon) tuple."""
        coords1 = (self.x, self.y)
        if isinstance(other, Position):
            coords2 = (other.x, other.y)
        elif isinstance(other, tuple) and len(other) == 2:
            coords2 = other
        else:
            raise ValueError("Unsupported input type for geodesic_distance_to")
        distance = geodesic(coords1, coords2).meters
        return distance


class SimpleVehicle:
    """Represents a basic vehicle with minimal attributes."""

    def __init__(self, vehicle_id, position, speed=0):
        self.id = vehicle_id
        self.position = position
        self.speed = speed

    def update_position(self, new_position, new_speed=None):
        """Update the vehicle's position and optionally speed."""
        self.position = new_position
        if new_speed is not None:
            self.speed = new_speed


class Vehicle:
    """Represents the main vehicle in the simulation."""

    def __init__(self, vehicle_id):

        self.id = vehicle_id
        self.position_history = []  # Will store PositionRecord objects
        self.neighbors = {}

    def update_data(self, step, speed, acceleration, heading, real_position,
                    nearby_vehicles=None, nearby_rsus=None, measured_position=None):
        """Update vehicle with new position data."""

        #convert the positions to Position attribute
        real_position = Position(real_position[0], real_position[1])

        record = StepRecord(
            step=step,
            real_position=real_position,
            measured_position=measured_position,
            speed=speed,
            acceleration=acceleration,
            heading=heading,
            nearby_vehicles=nearby_vehicles,
            nearby_rsus=nearby_rsus
        )
        self.position_history.append(record)

    @property
    def current_record(self):
        """Get the most recent position record."""
        return self.position_history[-1] if self.position_history else None


class StepRecord:
    """Stores position data for a specific time step."""

    def __init__(self, step, real_position, measured_position, speed, acceleration, heading, nearby_vehicles=None, nearby_rsus=None):
        self.step = step
        self.real_position = real_position  # Position without error
        self.measured_position = measured_position  # Position with error
        self.speed = speed
        self.acceleration = acceleration
        self.heading = heading
        self.estimated_positions = {}
        self.nearby_vehicles = nearby_vehicles or []
        self.nearby_rsus = nearby_rsus or []


class RSU:
    def __init__(self, rsu_id, position):
        self.id = rsu_id
        self.position = position

    def __repr__(self):
        return f"RSU(id={self.id}, x={self.position.x}, y={self.position.y})"


class RSUManager:

    def __init__(self, simulation_type, rsu_flag, reception_radius):
        self.reception_radius = reception_radius
        self.rsu_positions = []

        rsu_points = rsu_points_by_simulation.get(simulation_type)

        if rsu_flag and rsu_points:
            self.generate_rsus_from_list(rsu_points)

        else:
            print("No RSUs generated — RSU flag is off or no points provided.")

    def generate_rsus_from_list(self, rsu_points):
        """
        Converts a list of (lat, lon) points to RSU objects.
        """
        for idx, (x, y) in enumerate(rsu_points):
            rsu = RSU(f"rsu_{idx}", Position(x, y))
            self.rsu_positions.append(rsu)
