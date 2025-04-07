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


    def apply_error(self, position):
        """Apply GPS error to a position."""

        error_in_degrees = self.std_dev / 111320  # 1 degree ≈ 111.32 km at the equator
        # Add random error to latitude and longitude
        error_lat = np.random.normal(0, error_in_degrees)
        error_lon = np.random.normal(0, error_in_degrees)

        perturbed_lat = position.latitude + error_lat
        perturbed_lon = position.longitude + error_lon

        error_magnitude = np.sqrt(error_lat ** 2 + error_lon ** 2) * 111320  # Convert back to meters
        precision_radius = error_magnitude

        # Create and return a new Position object with the perturbed coordinates
        return Position(perturbed_lat, perturbed_lon, precision_radius)




class PositionEstimator:
    """Base class for different position estimation methods."""

    def estimate_position(self, target_vehicle, all_vehicles, rsus, step):
        """Estimate position. To be implemented by subclasses."""
        raise NotImplementedError


class TriangulationEstimator(PositionEstimator):
    """Implements triangulation-based position estimation."""

    def __init__(self, num_neighbors, proximity_radius, quality_filter=None):
        self.num_neighbors = num_neighbors
        self.proximity_radius = proximity_radius
        self.quality_filter = quality_filter

    def estimate_position(self, target_vehicle, all_vehicles, rsus, step):
        """Estimate position using triangulation with nearby vehicles and RSUs."""
        # Implementation details...
        positions, distances, precision_radii = self._gather_reference_points(
            target_vehicle, all_vehicles, rsus, step)

        selected_positions, selected_distances, is_better, selected_accuracies = \
            self._select_best_references(positions, distances, precision_radii,
                                         target_vehicle, step)

        # Perform triangulation with selected points
        estimated_position = self._triangulate(selected_positions, selected_distances)

        # Store results
        target_vehicle.current_record.estimated_positions['triangulation'] = estimated_position
        return estimated_position, is_better

    def _gather_reference_points(self, target_vehicle, all_vehicles, rsus, step):
        # Logic to gather reference points from nearby vehicles and RSUs
        positions, distances, precision_radii = [], [], []

        # Add RSUs
        for rsu in rsus:
            distance = target_vehicle.current_record.real_position.distance_to(rsu.position)
            if distance <= self.proximity_radius:
                positions.append(rsu.position)
                distances.append(distance)
                precision_radii.append(rsu.proximity_radius)

        # Add nearby vehicles
        for vehicle_id, vehicle in all_vehicles.items():
            if vehicle_id == target_vehicle.id:
                continue

            distance = target_vehicle.current_record.real_position.distance_to(
                vehicle.current_record.real_position)

            # Add communication error to distance
            perturbed_distance = add_communication_distance_error(distance)

            if perturbed_distance <= self.proximity_radius:
                positions.append(vehicle.current_record.measured_position)
                distances.append(perturbed_distance)
                precision_radii.append(vehicle.current_record.measured_position.precision_radius)

        return positions, distances, precision_radii

    def _select_best_references(self, positions, distances, precision_radii, target_vehicle, step):
        # Simplified version of select_positions_for_triangulation
        if len(positions) < self.num_neighbors:
            return positions, distances, False, precision_radii

        indices = np.argsort(precision_radii)[:self.num_neighbors]
        selected_positions = [positions[i] for i in indices]
        selected_distances = [distances[i] for i in indices]
        selected_accuracies = [precision_radii[i] for i in indices]

        current_precision = target_vehicle.current_record.measured_position.precision_radius
        is_better = not (selected_accuracies[2] > current_precision)

        if not self.quality_filter:
            is_better = True

        return selected_positions, selected_distances, is_better, selected_accuracies

    def _triangulate(self, positions, distances):
        # Implement triangulation algorithm
        # This would be the actual triangulation logic
        # Return a Position object
        pass


class SimulationManager:
    """Manages the overall simulation."""

    def __init__(self, simulation_config, error_model, estimator):
        self.config = simulation_config
        self.error_model = error_model
        self.estimator = estimator
        self.vehicles = {}
        self.rsus = []
        self.results = {
            'better_values': [],
            'not_better_values': [],
            'errors': []
        }

    def initialize_rsus(self, rsu_positions):
        """Initialize RSUs at specified positions."""
        for idx, position in enumerate(rsu_positions):
            rsu = RSU(f"rsu_{idx}", Position(position[0], position[1]))
            self.rsus.append(rsu)

    def update_vehicle(self, vehicle_id, geo_position, speed, step):
        """Update or create a vehicle with new data."""
        if vehicle_id not in self.vehicles:
            self.vehicles[vehicle_id] = Vehicle(vehicle_id)

        position = Position(geo_position[0], geo_position[1])
        self.vehicles[vehicle_id].update(position, speed, step, self.error_model)

    def estimate_positions(self, target_vehicle_id, step):
        """Estimate positions for a target vehicle."""
        if target_vehicle_id not in self.vehicles:
            return None

        target_vehicle = self.vehicles[target_vehicle_id]
        estimated_position, is_better = self.estimator.estimate_position(
            target_vehicle, self.vehicles, self.rsus, step)

        # Store result statistics
        if is_better:
            self.results['better_values'].append(estimated_position)
        else:
            self.results['not_better_values'].append(estimated_position)

        # Calculate error
        real_pos = target_vehicle.current_record.real_position
        measured_pos = target_vehicle.current_record.measured_position
        estimated_pos = estimated_position

        # Calculate errors (could be more sophisticated)
        gps_error = real_pos.distance_to(measured_pos)
        estimated_error = real_pos.distance_to(estimated_pos)

        self.results['errors'].append((gps_error, estimated_error))

        return estimated_position

    def run_simulation(self, simulation_path, specific_car_id, num_steps):
        """Run the full simulation."""
        # Start SUMO
        traci.start(["sumo", "-c", simulation_path])

        for step in range(num_steps):
            traci.simulationStep()
            vehicle_ids = traci.vehicle.getIDList()

            # Update data for all existing cars
            for vehicle_id in vehicle_ids:
                position = traci.vehicle.getPosition(vehicle_id)
                geo_position = traci.simulation.convertGeo(position[0], position[1])
                speed = traci.vehicle.getSpeed(vehicle_id)

                self.update_vehicle(vehicle_id, geo_position, speed, step)

            # Perform position estimation for the specific car
            if specific_car_id in vehicle_ids:
                self.estimate_positions(specific_car_id, step)

        # End simulation
        traci.close()
        return self.results