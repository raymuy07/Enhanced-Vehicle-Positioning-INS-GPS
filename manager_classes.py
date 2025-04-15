import geopy
import numpy as np
import traci

from core_classes import Position, RSU, Vehicle


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

        # Add random Gaussian error directly in meters
        error_x = np.random.normal(0, self.std_dev)
        error_y = np.random.normal(0, self.std_dev)

        perturbed_x = position.x + error_x
        perturbed_y = position.y + error_y

        # Calculate the precision radius (magnitude of error vector)
        precision_radius = np.sqrt(error_x ** 2 + error_y ** 2)

        # Create and return a new Position object with the perturbed coordinates
        return Position(perturbed_x, perturbed_y, precision_radius)

class CommunicationDistanceErrorModel(ErrorModel):
    """Implements a communication-specific distance error model."""

    def __init__(self, std_dev=2, systematic_bias=0.3):
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

    def __init__(self, simulation_params,simulation_type,gps_error_model,comm_error_model):

        ##temporary
        self.neighbor_comparison = {}

        self.simulation_type = simulation_type
        self.gps_error_model = gps_error_model
        self.comm_error_model = comm_error_model
        #self.estimator = estimator
        self.num_steps = simulation_params.get('number_of_steps', 0)
        self.num_of_neighbors = simulation_params.get('num_of_neighbors', 0)
        self.proximity_radius = simulation_params.get('proximity_radius', 0)
        rsu_flag = simulation_params.get('rsu_flag', False)
        self.vehicles = {}

        self.results = {
            'no_mod_values': [],
            'better_values': [],
            'errors': []
        }

        self.initialize_rsus(rsu_flag,self.rsu_proximity_radius)

    def initialize_rsus(self,rsu_flag,rsu_proximity_radius):
        """Initialize RSUs at specified positions."""
        self.rsu_object = RSU(self.simulation_type, rsu_flag, rsu_proximity_radius)

    def update_vehicle(self, vehicle_id, cartesian_position, speed, step):
        """Update or create a vehicle with new data."""

        # Create or update vehicle
        if vehicle_id not in self.vehicles:
            self.vehicles[vehicle_id] = Vehicle(vehicle_id)
        # Create Position object
        real_position = Position(cartesian_position[0], cartesian_position[1])
        # Update vehicle with new data
        self.vehicles[vehicle_id].update(real_position, speed, step, self.gps_error_model)


    def find_neighbours(self, specific_car_id,step):
        """Find nearby vehicles using both methods for comparison."""


        left_behind = traci.vehicle.getNeighbors(specific_car_id, 0)
        right_behind = traci.vehicle.getNeighbors(specific_car_id, 1)
        left_ahead = traci.vehicle.getNeighbors(specific_car_id, 2)
        right_ahead = traci.vehicle.getNeighbors(specific_car_id, 3)


        # Combine all neighbors with filtering before appending

        nearby_vehicles = []
        for neighbor_set in [left_behind, right_behind, left_ahead, right_ahead]:
            for neighbor_tuple in neighbor_set:
                vehicle_id, distance = neighbor_tuple
                real_world_distance = self.comm_error_model.apply_error(abs(distance))
                # Use absolute distance and filter immediately
                if real_world_distance <= self.proximity_radius:
                    nearby_vehicles.append((vehicle_id, abs(distance)))

        step = traci.simulation.getTime()

        # Using SUMO's getNeighbors
        nearby_sumo = []

        # Get neighbors in all directions
        left_behind = traci.vehicle.getNeighbors(specific_car_id, 0)
        right_behind = traci.vehicle.getNeighbors(specific_car_id, 1)
        left_ahead = traci.vehicle.getNeighbors(specific_car_id, 2)
        right_ahead = traci.vehicle.getNeighbors(specific_car_id, 3)

        # Combine all neighbors
        nearby_sumo = list(set(left_behind + right_behind + left_ahead + right_ahead))
        if nearby_sumo:
            print(nearby_sumo)
        # # Filter by distance if needed
        # if self.rsu_proximity_radius < 300:  # Only if we have a specific radius
        #     nearby_sumo_filtered = []
        #     for other_id in nearby_sumo:
        #         other_position = traci.vehicle.getPosition(other_id)
        #         other_geo = traci.simulation.convertGeo(other_position[0], other_position[1])
        #
        #         distance = self.calculate_distance(specific_car_geo, other_geo)
        #         if distance <= self.rsu_proximity_radius:
        #             nearby_sumo_filtered.append(other_id)
        #     nearby_sumo = nearby_sumo_filtered
        #
        # return nearby_sumo

    def print_neighbor_comparison_summary(self):
        """Print summary of neighbor detection comparison."""
        total_original = sum(data['original_only'] + data['overlap'] for data in self.neighbor_comparison.values())
        total_sumo = sum(data['sumo_only'] + data['overlap'] for data in self.neighbor_comparison.values())
        total_overlap = sum(data['overlap'] for data in self.neighbor_comparison.values())

        print("\nNeighbor Detection Comparison Summary:")
        print(f"Total neighbors found by Original method: {total_original}")
        print(f"Total neighbors found by SUMO method: {total_sumo}")
        print(f"Total overlap: {total_overlap}")
        # print(f"Efficiency: Original found {total_original / total_sumo * 100:.1f}% compared to SUMO")

    def run_simulation(self, simulation_path, specific_car_id):
        """Run the full simulation."""

        # Start SUMO
        traci.start(["sumo", "-c", simulation_path])

        for step in range(self.num_steps):

            traci.simulationStep()
            vehicle_ids = traci.vehicle.getIDList()

            # Update data for all existing cars
            for vehicle_id in vehicle_ids:

                cartesian_position = traci.vehicle.getPosition(vehicle_id)
                speed = traci.vehicle.getSpeed(vehicle_id)

                self.update_vehicle(vehicle_id, cartesian_position, speed, step)

                self.find_neighbours(specific_car_id,step)

        self.print_neighbor_comparison_summary()

        ## TODO Get triangulation estimates
        """ a replecement for the find_nearby_vehicles_and_check_rsus method should come here and is 
        partially  implemented in the TriangulationEstimator class
        """


        # End simulation
        traci.close()
        return self.results