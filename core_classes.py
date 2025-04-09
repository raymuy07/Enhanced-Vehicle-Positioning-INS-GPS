import geopy
from geopy.distance import geodesic

from rsu_config import rsu_points_by_simulation

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
        self.position_history = []  # Will store PositionRecord objects

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
        self.position_history.append(record)

    @property
    def current_record(self):
        """Get the most recent position record."""
        return self.position_history[-1] if self.position_history else None


class PositionRecord:
    """Stores position data for a specific time step."""

    def __init__(self, step, real_position, measured_position, speed):
        self.step = step
        self.real_position = real_position  # Position without error
        self.measured_position = measured_position  # Position with error
        self.speed = speed
        self.estimated_positions = {}  # Different position estimates, keyed by method name



"""

NOTE: This code wasn't written by Guy and Omri! 
it was taken from the previous project and modified to fit the current project

"""


class RSU:
    def __init__(self, simulation_type, rsu_flag, reception_radius):

        rsu_points = rsu_points_by_simulation.get(simulation_type)
        if rsu_flag:
            rsu_grid = self.generate_rsu_grid(*rsu_points)
        else:
            rsu_grid = []

        """Attributes of the RSUManager class"""
        self.rsu_locations = rsu_grid
        self.reception_radius = reception_radius
        self.vehicle_proximity = {step: {rsu_index: [] for rsu_index in range(len(self.rsu_locations))} for step in
                                  range(100)}  # Assuming 100 steps for simplicity



    def generate_rsu_grid(self, point1, point2, point3, point4, interval_km=1):
        """
        Generate a grid of RSUs within the boundaries defined by four points.

        :param point1: tuple, (latitude, longitude) of the first point.
        :param point2: tuple, (latitude, longitude) of the second point.
        :param point3: tuple, (latitude, longitude) of the third point.
        :param point4: tuple, (latitude, longitude) of the fourth point.
        :param interval_km: float, the interval distance in kilometers for each RSU.
        :return: list of tuples, RSU positions (latitude, longitude)
        """
        # Define the boundaries
        lat_min = min(point1[0], point2[0], point3[0], point4[0])
        lat_max = max(point1[0], point2[0], point3[0], point4[0])
        lon_min = min(point1[1], point2[1], point3[1], point4[1])
        lon_max = max(point1[1], point2[1], point3[1], point4[1])

        # Generate RSU grid
        rsu_positions = []
        current_lat = lat_min
        while current_lat <= lat_max:
            current_lon = lon_min
            while current_lon <= lon_max:
                rsu_positions.append((current_lat, current_lon))
                # Move 1 kilometer east
                current_lon = geodesic(kilometers=interval_km).destination((current_lat, current_lon), 90).longitude
            # Move 1 kilometer north
            current_lat = geodesic(kilometers=interval_km).destination((current_lat, lon_min), 0).latitude

        return rsu_positions
