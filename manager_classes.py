import numpy as np
import traci
import random
from core_classes import Position, RSU, Vehicle, RSUManager
from utility_functions import calculate_cartesian_distance
from core_classes import Position, RSU, Vehicle
from utility_functions import calculate_absolute_error, calculate_squared_error, mod_trilaterate_gps
from geopy.distance import geodesic



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


# class TriangulationEstimator(PositionEstimator):
#     """Implements triangulation-based position estimation."""
#
#     def __init__(self, num_neighbors, proximity_radius, quality_filter=None):
#         self.num_neighbors = num_neighbors
#         self.proximity_radius = proximity_radius
#         self.quality_filter = quality_filter
#
#     def estimate_position(self, target_vehicle, all_vehicles, rsus, step):
#         """Estimate position using triangulation with nearby vehicles and RSUs."""
#         # Implementation details...
#         positions, distances, precision_radii = self._gather_reference_points(
#             target_vehicle, all_vehicles, rsus, step)
#
#         selected_positions, selected_distances, is_better, selected_accuracies = \
#             self._select_best_references(positions, distances, precision_radii,
#                                          target_vehicle, step)
#
#         # Perform triangulation with selected points
#         estimated_position = self._triangulate(selected_positions, selected_distances)
#
#         # Store results
#         target_vehicle.current_record.estimated_positions['triangulation'] = estimated_position
#         return estimated_position, is_better
#
#     def _gather_reference_points(self, target_vehicle, all_vehicles, rsus, step):
#
#         # Logic to gather reference points from nearby vehicles and RSUs
#         positions, distances, precision_radii = [], [], []
#
#         # Add RSUs
#         for rsu in rsus:
#             distance = target_vehicle.current_record.real_position.distance_to(rsu.position)
#             if distance <= self.proximity_radius:
#                 positions.append(rsu.position)
#                 distances.append(distance)
#                 precision_radii.append(rsu.proximity_radius)
#
#         # Add nearby vehicles
#         for vehicle_id, vehicle in all_vehicles.items():
#             if vehicle_id == target_vehicle.id:
#                 continue
#
#             distance = target_vehicle.current_record.real_position.distance_to(
#                 vehicle.current_record.real_position)
#
#             # Add communication error to distance
#             perturbed_distance = add_communication_distance_error(distance)
#
#             if perturbed_distance <= self.proximity_radius:
#                 positions.append(vehicle.current_record.measured_position)
#                 distances.append(perturbed_distance)
#                 precision_radii.append(vehicle.current_record.measured_position.precision_radius)
#
#         return positions, distances, precision_radii
#
#     def _select_best_references(self, positions, distances, precision_radii, target_vehicle, step):
#         # Simplified version of select_positions_for_triangulation
#         if len(positions) < self.num_neighbors:
#             return positions, distances, False, precision_radii
#
#         indices = np.argsort(precision_radii)[:self.num_neighbors]
#         selected_positions = [positions[i] for i in indices]
#         selected_distances = [distances[i] for i in indices]
#         selected_accuracies = [precision_radii[i] for i in indices]
#
#         current_precision = target_vehicle.current_record.measured_position.precision_radius
#         is_better = not (selected_accuracies[2] > current_precision)
#
#         if not self.quality_filter:
#             is_better = True
#
#         return selected_positions, selected_distances, is_better, selected_accuracies
#
#     def _triangulate(self, positions, distances):
#         # Implement triangulation algorithm
#         # This would be the actual triangulation logic
#         # Return a Position object
#         pass


class SimulationManager:
    """Manages the overall simulation."""

    def __init__(self, simulation_params, simulation_type, gps_error_model, comm_error_model):

        self.rsu_manager = None
        self.main_vehicle_obj = None

        self.simulation_type = simulation_type
        self.gps_error_model = gps_error_model
        self.comm_error_model = comm_error_model

        self.num_steps = simulation_params.get('number_of_steps', 0)
        self.num_of_neighbors = simulation_params.get('num_of_neighbors', 0)
        self.proximity_radius = simulation_params.get('proximity_radius', 0)
        self.rsu_flag = simulation_params.get('rsu_flag', False)

        self.results = {
            'no_mod_values': [],
            'better_values': [],
            'errors': []
        }

    @staticmethod
    def get_random_main_vehicle(initial_steps):
        """This function is for making our simulation more relaistic
        instead it will focus on the same vehicle all the time, we want
        to select a random vehicle each simulation.

        arg: initial_steps: it's the amount of steps the simulation will run before
        choosing the main vehicle.
        """
        vehicle_ids = None
        random_vehicle = None

        for step in range(initial_steps):
            traci.simulationStep()
            vehicle_ids = traci.vehicle.getIDList()

        if vehicle_ids:
            random_vehicle = random.choice(vehicle_ids)

        if random_vehicle:
            return random_vehicle
        else:
            print("Error:couldn't choose random vehicle.")

    def create_snapshot(self, neighbor_id, real_world_distance):
        """Create a snapshot of the vehicle's current state."""

        map_tuple_position = traci.vehicle.getPosition(neighbor_id)
        map_position = Position(map_tuple_position[0], map_tuple_position[1])
        position_w_error = self.gps_error_model.apply_error(map_tuple_position)

        return {
            'id': neighbor_id,
            'true_position': map_position,
            'position_w_error': position_w_error,
            'real_world_distance': real_world_distance
        }

    def find_neighbours(self):
        """Find nearby vehicles using get neighbors."""

        specific_car_id = self.main_vehicle_obj.id

        left_behind = traci.vehicle.getNeighbors(specific_car_id, 0)
        right_behind = traci.vehicle.getNeighbors(specific_car_id, 1)
        left_ahead = traci.vehicle.getNeighbors(specific_car_id, 2)
        right_ahead = traci.vehicle.getNeighbors(specific_car_id, 3)

        # Combine all neighbors with filtering before appending
        """Here I want to create a list of Simple_Veicles class that are initialized to the following data:
        Position(x,y,percision_radius)
        real_world_distance i.e the distance after applying communication error,
    
        """
        nearby_vehicles = []
        for neighbor_set in [left_behind, right_behind, left_ahead, right_ahead]:
            for neighbor_tuple in neighbor_set:

                vehicle_id, distance = neighbor_tuple
                real_world_distance = self.comm_error_model.apply_error(abs(distance))

                if real_world_distance <= self.proximity_radius:
                    neighbor_vehicle_snapshot = self.create_snapshot(vehicle_id, real_world_distance)
                    nearby_vehicles.append(neighbor_vehicle_snapshot)

        return nearby_vehicles

    def find_nearby_rsu(self, vehicle_cartesian_position):

        nearby_rsus = []
        ##TODO: change this enumarate and give the Rsu's id's

        # for rsu ... in self.rsu_manager.rsu_locations:
        for rsu in self.rsu_manager.rsu_locations:

            distance_to_rsu = calculate_cartesian_distance(vehicle_cartesian_position, (rsu.x, rsu.y))
            real_world_distance_rsu = self.comm_error_model.apply_error(distance_to_rsu)

            if real_world_distance_rsu <= self.proximity_radius:
                nearby_rsus.append({
                    'rsu': rsu,
                    'distance_from_veh': real_world_distance_rsu
                })

        return nearby_rsus

    def run_simulation(self, simulation_path):
        """Run the full simulation."""

        initial_steps = 20

        # Start SUMO
        traci.start(["sumo", "-c", simulation_path])

        # initialize the RSU, (it must be here cause we need the simulation).
        self.rsu_manager = RSUManager(self.simulation_type, self.rsu_flag, self.proximity_radius)

        ##initilaize the random vehicle
        random_vehicle = self.get_random_main_vehicle(initial_steps)
        self.main_vehicle_obj = Vehicle(random_vehicle, self.gps_error_model)

        for step in range(initial_steps, self.num_steps):

            traci.simulationStep()

            if self.main_vehicle_obj.id in traci.vehicle.getIDList():

                ### Get the position and speed
                vehicle_cartesian_position = traci.vehicle.getPosition(self.main_vehicle_obj.id)
                speed = traci.vehicle.getSpeed(self.main_vehicle_obj.id)
                current_neighbours = self.find_neighbours()
                nearby_rsu = self.find_nearby_rsu(vehicle_cartesian_position)

                self.main_vehicle_obj.update(vehicle_cartesian_position, speed, step, current_neighbours)

            else:
                # Main car is not in the simulation.
                traci.close()
                break

        return self.main_vehicle_obj



class CalculationManager:
    def __init__(self, main_vehicle):
        self.main_vehicle = main_vehicle
        self.dsrc_errors = {"absolute": [], "mse": []}
        self.ins_errors = {"absolute": [], "mse": []}
        self.fused_errors = {"absolute": [], "mse": []}

    def get_dsrc_position(self, step_record, alpha=1.0):
        """
        Estimate position based on RSU and DSRC trilateration - 'better' algorithm.
        This is work done in past project and modified by us.

        Parameters:
        - step_record: StepRecord object
        - alpha: weight tuning parameter

        Returns:
        - Position object with estimated coordinates
        """
        surrounding_positions = []
        distances = []
        error_radii = []

        # Add RSUs
        for rsu_coords in step_record.nearby_rsus: # need to check whats is step_record.nearby_rsus
            rsu_pos = Position(rsu_coords[0], rsu_coords[1])
            surrounding_positions.append(rsu_pos)
            distances.append(geodesic((step_record.measured_position.x, step_record.measured_position.y),  # can be replaced with a function
                                      (rsu_pos.x, rsu_pos.y)).meters)
            error_radii.append(0.1)  # assume high confidence

        # Add vehicles
        for vehicle_id, perturbed_distance, vehicle_pos, _ in step_record.nearby_vehicles: # need to check whats is step_record.nearby_vehicles
            surrounding_positions.append(vehicle_pos)
            distances.append(perturbed_distance)
            error_radii.append(
                vehicle_pos.precision_radius if vehicle_pos.precision_radius else 8.0)  # fallback precision

        # Minimum 3+ points needed
        if len(surrounding_positions) < 3:
            return None   # not reliable, maybe we need to return step_record.measured_position

        # Estimate position
        estimated_pos = mod_trilaterate_gps(
            surrounding_positions,
            distances,
            error_radii,
            step_record.measured_position,
            alpha
        )

        return estimated_pos

    def get_ins_position(self, step_record):
        """
        Placeholder: use ML model to compute projected position from IMU/ECU data.
        """
        pass

    def get_fused_position(self, dsrc_pos, ins_pos):
        """
        Fuse DSRC and INS positions using Kalman filter or weighted average.
        """
        pass

    def calculate_all_errors(self):
        """
        Loops over all steps in the main vehicle's history and calculates
        absolute and squared error for each localization method:
        - DSRC-enhanced
        - INS-enhanced
        - Fused

        Errors are stored in class dictionaries.
        """
        for step_record in self.main_vehicle.position_history:
            real_pos = step_record.real_position

            dsrc_pos = self.get_dsrc_position(step_record)
            ins_pos = self.get_ins_position(step_record)
            fused_pos = self.get_fused_position(dsrc_pos, ins_pos)

            if dsrc_pos:
                abs_e = calculate_absolute_error(dsrc_pos, real_pos)
                sqr_e = calculate_squared_error(dsrc_pos, real_pos)
                if abs_e is not None and sqr_e is not None:
                    self.dsrc_errors["absolute"].append(abs_e)
                    self.dsrc_errors["mse"].append(sqr_e)

            if ins_pos:
                abs_e = calculate_absolute_error(ins_pos, real_pos)
                sqr_e = calculate_squared_error(ins_pos, real_pos)
                if abs_e is not None and sqr_e is not None:
                    self.ins_errors["absolute"].append(abs_e)
                    self.ins_errors["mse"].append(sqr_e)

            if fused_pos:
                abs_e = calculate_absolute_error(fused_pos, real_pos)
                sqr_e = calculate_squared_error(fused_pos, real_pos)
                if abs_e is not None and sqr_e is not None:
                    self.fused_errors["absolute"].append(abs_e)
                    self.fused_errors["mse"].append(sqr_e)

    def calculate_average_error(self, method, error_type='absolute'):
        """
        Calculates average error for the specified method and error type.

        Parameters:
        - method: 'dsrc', 'ins', or 'fused'   #### needs to be changed for our selected names
        - error_type: 'absolute' or 'mse'

        Returns:
        - Average error as float, or None if input is invalid or data is missing
        """
        error_dict = {
            "dsrc": self.dsrc_errors,
            "ins": self.ins_errors,
            "fused": self.fused_errors
        }

        if method not in error_dict:
            print(f"[Error] Unknown method '{method}'. Choose from 'dsrc', 'ins', or 'fused'.")
            return None

        if error_type not in error_dict[method]:
            print(f"[Error] Unknown error type '{error_type}'. Choose 'absolute' or 'mse'.")
            return None

        errors = error_dict[method][error_type]
        if not errors:
            print(f"[Warning] No errors recorded for {method} - {error_type}.")
            return None

        return sum(errors) / len(errors)

