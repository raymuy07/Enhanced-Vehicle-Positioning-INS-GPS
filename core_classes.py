import numpy as np
import traci
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
    """Represents a vehicle in the simulation."""

    def __init__(self, vehicle_id, error_model):
        self.id = vehicle_id
        self.error_model = error_model
        self.position_history = []  # Will store PositionRecord objects
        self.neighbors = {}

    def update(self, real_position, speed, step, nearby_vehicles=None, nearby_rsus=None):
        """Update vehicle with new position data."""

        # convert the positions to Position attribute
        real_position = Position(real_position[0], real_position[1])
        measured_position = self.error_model.apply_error(real_position)

        record = StepRecord(
            step=step,
            real_position=real_position,
            measured_position=measured_position,
            speed=speed,
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

    def __init__(self, step, real_position, measured_position, speed, nearby_vehicles=None, nearby_rsus=None):
        self.step = step
        self.real_position = real_position  # Position without error
        self.measured_position = measured_position  # Position with error
        self.speed = speed
        self.estimated_positions = {}
        self.nearby_vehicles = nearby_vehicles or []
        self.nearby_rsus = nearby_rsus or []


class RSU:
    def __init__(self, rsu_id, x, y):
        self.id = rsu_id
        self.x = x  ##  might need to change from x, y attributes to position attribute.
        self.y = y

    def __repr__(self):
        return f"RSU(id={self.id}, x={self.x}, y={self.y})"


class RSUManager:

    def __init__(self, simulation_type, rsu_flag, reception_radius):
        self.reception_radius = reception_radius
        self.rsu_locations = []

        rsu_points = rsu_points_by_simulation.get(simulation_type)

        if rsu_flag and rsu_points:
            self.generate_rsu_grid_cartesian(*rsu_points)
        else:
            print("No RSUs generated — RSU flag is off or no points provided.")

    def generate_rsu_grid_cartesian(self, point1, point2, point3, point4, interval_km=1):

        # Define the boundaries
        lat_min = min(point1[0], point2[0], point3[0], point4[0])
        lat_max = max(point1[0], point2[0], point3[0], point4[0])
        lon_min = min(point1[1], point2[1], point3[1], point4[1])
        lon_max = max(point1[1], point2[1], point3[1], point4[1])

        rsu_id = 0

        current_lat = lat_min
        while current_lat <= lat_max:
            current_lon = lon_min
            while current_lon <= lon_max:
                x, y = traci.simulation.convertGeo(current_lat, current_lon, fromGeo=True)
                rsu = RSU(f"rsu_{rsu_id}", x, y)
                self.rsu_locations.append(rsu)
                rsu_id += 1

                # Move 1 kilometer east
                current_lon = geodesic(kilometers=interval_km).destination((current_lat, current_lon), 90).longitude
            # Move 1 kilometer north
            current_lat = geodesic(kilometers=interval_km).destination((current_lat, lon_min), 0).latitude
