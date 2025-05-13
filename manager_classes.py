import traci
import random
from geopy.distance import geodesic
from scipy.optimize import minimize
from core_classes import Position, RSU, Vehicle, RSUManager
from core_classes import Position, RSU, Vehicle
from utility_functions import calculate_absolute_error, calculate_squared_error, cartesian_distance
import numpy as np
from scipy.linalg import block_diag

class SimulationManager:
    """Manages the overall simulation."""

    def __init__(self, simulation_params, simulation_type, gps_error_model, comm_error_model):

        self.rsu_manager = None
        self.main_vehicle_obj = None

        self.simulation_type = simulation_type
        self.gps_error_model = gps_error_model
        self.comm_error_model = comm_error_model

        self.gps_refresh_rate = simulation_params.get('gps_refresh_rate', 10)
        self.dsrc_refresh_rate = simulation_params.get('dsrc_refresh_rate', 5)
        self.ins_refresh_rate = simulation_params.get('ins_refresh_rate', 1)

        self.num_steps = simulation_params.get('number_of_steps', 500)
        self.proximity_radius = simulation_params.get('proximity_radius', 300)
        self.rsu_flag = simulation_params.get('rsu_flag', False)

    ##TODO: Check if this function really necessary
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

            distance_to_rsu = cartesian_distance(vehicle_cartesian_position, (rsu.x, rsu.y))
            real_world_distance_rsu = self.comm_error_model.apply_error(distance_to_rsu)

            if real_world_distance_rsu <= self.proximity_radius:
                nearby_rsus.append({
                    'rsu': rsu,
                    'distance_from_veh': real_world_distance_rsu
                })

        return nearby_rsus

    def run_simulation(self, simulation_path):
        """Run the full simulation."""

        initial_steps = 10


        # Start SUMO
        traci.start(["sumo", "-c", simulation_path])

        # initialize the RSU, (it must be here cause we need the simulation).
        self.rsu_manager = RSUManager(self.simulation_type, self.rsu_flag, self.proximity_radius)

        # initialize the random vehicle
        random_vehicle = self.get_random_main_vehicle(initial_steps)
        self.main_vehicle_obj = Vehicle(random_vehicle)

        for step in range(initial_steps, self.num_steps):

            traci.simulationStep()

            if self.main_vehicle_obj.id in traci.vehicle.getIDList():

                # Get main vehicle state
                vehicle_cartesian_position = traci.vehicle.getPosition(self.main_vehicle_obj.id)
                speed = traci.vehicle.getSpeed(self.main_vehicle_obj.id)
                veh_acc = traci.vehicle.getAcceleration(self.main_vehicle_obj.id)
                veh_heading = traci.vehicle.getAngle(self.main_vehicle_obj.id) #Returns the angle of the named vehicle within the last step [°]

                # DSRC update
                if step % self.dsrc_refresh_rate == 0:
                    current_neighbours = self.find_neighbours()
                    nearby_rsus = self.find_nearby_rsu(vehicle_cartesian_position)
                else:
                    current_neighbours = nearby_rsus = None

                # GPS update
                measured_position = self.gps_error_model.apply_error(
                    vehicle_cartesian_position) if step % self.gps_refresh_rate == 0 else None

                # Build the data dictionary
                vehicle_data = {
                    'step': step,
                    'speed': speed,
                    'acceleration': veh_acc,
                    'heading': veh_heading,
                    'real_position': vehicle_cartesian_position,
                    'nearby_vehicles': current_neighbours,
                    'nearby_rsus': nearby_rsus,
                    'measured_position': measured_position
                }

                self.main_vehicle_obj.update_data(**vehicle_data)

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
        
        # Initialize Kalman filter state
        self.state = np.zeros(6)  # [x, y, vx, vy, ax, ay]
        self.P = np.eye(6) * 1000  # Initial covariance matrix
        self.dt = 1.0  # Time step (1 second)
        
        # Process noise covariance
        self.Q = block_diag(
            np.eye(2) * 0.1,  # Position process noise
            np.eye(2) * 1.0,  # Velocity process noise
            np.eye(2) * 10.0  # Acceleration process noise
        )
        
        # Measurement noise covariance for GPS
        self.R_gps = np.eye(2) * 10.0  # GPS measurement noise
        
        # Measurement matrix for GPS (only measures position)
        self.H_gps = np.zeros((2, 6))
        self.H_gps[0, 0] = 1
        self.H_gps[1, 1] = 1

    @staticmethod
    def _trilaterate_position(positions, distances, error_radii, initial_guess, alpha=1.0):
        """
        Weighted GPS trilateration using geodesic distances and error-based weighting.
        """
        # Define the objective function for optimization
        def objective(x):
            est_coords = (x[0], x[1])
            return sum(((geodesic(est_coords, (pos.x, pos.y)).meters - dist) / error_radius ** alpha) ** 2
                       for pos, dist, error_radius in zip(positions, distances, error_radii))

        initial_guess_arr = np.array([initial_guess.x, initial_guess.y])
        bounds = [(-90, 90), (-180, 180)]

        result = minimize(objective, initial_guess_arr, bounds=bounds)
        return Position(result.x[0], result.x[1])

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
        for rsu_coords in step_record.nearby_rsus:  # need to check whats is step_record.nearby_rsus
            rsu_pos = Position(rsu_coords[0], rsu_coords[1])
            surrounding_positions.append(rsu_pos)
            distances.append(geodesic((step_record.measured_position.x, step_record.measured_position.y),
                                      # can be replaced with a function
                                      (rsu_pos.x, rsu_pos.y)).meters)
            error_radii.append(0.1)  # assume high confidence

        # Add vehicles
        for vehicle_id, perturbed_distance, vehicle_pos, _ in step_record.nearby_vehicles:  # need to check whats is step_record.nearby_vehicles
            surrounding_positions.append(vehicle_pos)
            distances.append(perturbed_distance)
            error_radii.append(
                vehicle_pos.precision_radius if vehicle_pos.precision_radius else 8.0)  # fallback precision

        # Minimum 3+ points needed
        if len(surrounding_positions) < 3:
            return None  # not reliable, maybe we need to return step_record.measured_position

        # Estimate position
        estimated_pos = self._trilaterate_position(
            surrounding_positions,
            distances,
            error_radii,
            step_record.measured_position,
            alpha
        )

        return estimated_pos

    def _predict_step(self, acceleration, heading):
        """Perform the prediction step of the Kalman filter."""
        # Convert acceleration and heading to x,y components
        ax = acceleration * np.cos(np.radians(heading))
        ay = acceleration * np.sin(np.radians(heading))
        
        # State transition matrix
        F = np.eye(6)
        F[0, 2] = self.dt
        F[1, 3] = self.dt
        F[0, 4] = 0.5 * self.dt**2
        F[1, 5] = 0.5 * self.dt**2
        F[2, 4] = self.dt
        F[3, 5] = self.dt
        
        # Predict state
        self.state = F @ self.state
        self.state[4] = ax  # Update acceleration in state
        self.state[5] = ay
        
        # Predict covariance
        self.P = F @ self.P @ F.T + self.Q
        
        return Position(self.state[0], self.state[1])

    def _update_step(self, measurement):
        """Perform the update step of the Kalman filter."""
        # Kalman gain
        K = self.P @ self.H_gps.T @ np.linalg.inv(self.H_gps @ self.P @ self.H_gps.T + self.R_gps)
        
        # Update state
        measurement_vector = np.array([measurement.x, measurement.y])
        self.state = self.state + K @ (measurement_vector - self.H_gps @ self.state)
        
        # Update covariance
        self.P = (np.eye(6) - K @ self.H_gps) @ self.P
        
        return Position(self.state[0], self.state[1])

    def get_ins_position(self, step_record):
        """
        Estimate position using Kalman filter that fuses INS and GPS data.
        
        Parameters:
        - step_record: StepRecord object containing sensor data
        
        Returns:
        - Position object with estimated coordinates
        """
        # Predict step using acceleration and heading
        predicted_pos = self._predict_step(step_record.acceleration, step_record.heading)
        
        # Update step if GPS measurement is available
        if step_record.measured_position is not None:
            return self._update_step(step_record.measured_position)
        
        return predicted_pos

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


import numpy as np
from scipy.linalg import block_diag
import matplotlib.pyplot as plt


class VehicleEKF:
    def __init__(self, first_measurement):
        # State vector: [x, y, speed, heading, acceleration]
        self.state_dim = 5

        # Initial state estimate
        pos = first_measurement.measured_position

        # Convert SUMO heading (0° = North, clockwise) to standard math heading (0° = East, counterclockwise)
        heading_rad = np.radians(90 - first_measurement.heading)

        self.x = np.array([
            [pos.x],
            [pos.y],
            [first_measurement.speed],
            [heading_rad],
            [first_measurement.acceleration]
        ])

        # Initial covariance - lower for better stability
        self.P = np.eye(self.state_dim) * 1.0

        # Process noise - much lower to reduce jumpiness
        self.Q = np.diag([0.01, 0.01, 0.05, 0.005, 0.05])

        # Measurement noise - balanced to avoid jumps
        self.R_imu = np.diag([0.02, 0.05, 0.02])  # heading, acceleration, speed
        self.R_gps = np.diag([9.0, 9.0])  # GPS noise - tuned for smoother transitions

        # Time step (in seconds)
        self.dt = 0.1

        # Last GPS position and time for jump detection
        self.last_gps_pos = None
        self.last_gps_time = 0

        # Flag for stationary detection
        self.is_stationary = False
        self.stationary_threshold = 0.1  # m/s

        # Step counter and history
        self.step_count = 0
        self.history = {
            'true_position': [],
            'estimated_position': [],
            'gps_position': [],
            'step': []
        }

    def predict(self):
        """Improved prediction with stationary handling."""
        x, y, speed, heading, acc = self.x.flatten()

        # Calculate expected movement per step
        dx = speed * np.cos(heading) * self.dt
        dy = speed * np.sin(heading) * self.dt

        print(f"Predict: speed={speed:.2f} m/s, heading={np.degrees(heading):.1f}° (math), "
              f"dt={self.dt:.3f}s → movement: dx={dx:.3f}m, dy={dy:.3f}m")


        # Detect if vehicle is stationary
        self.is_stationary = abs(speed) < self.stationary_threshold

        if self.is_stationary:
            # When stopped, don't apply motion model
            new_x = x
            new_y = y
            new_speed = 0.0  # Force to zero to prevent drift
            new_heading = heading
            new_acc = 0.0  # Force to zero when stopped

            # Use simplified state transition matrix - no movement
            F = np.eye(self.state_dim)

            # Use small process noise for position when stopped
            Q_stationary = np.diag([0.001, 0.001, 0.01, 0.001, 0.01])
            self.P = F @ self.P @ F.T + Q_stationary
        else:
            # Normal motion when moving
            new_x = x + speed * np.cos(heading) * self.dt
            new_y = y + speed * np.sin(heading) * self.dt
            new_speed = speed + acc * self.dt
            new_heading = heading
            new_acc = acc

            # Standard Jacobian
            F = np.array([
                [1, 0, np.cos(heading) * self.dt, -speed * np.sin(heading) * self.dt, 0],
                [0, 1, np.sin(heading) * self.dt, speed * np.cos(heading) * self.dt, 0],
                [0, 0, 1, 0, self.dt],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1]
            ])

            self.P = F @ self.P @ F.T + self.Q

        self.x = np.array([[new_x], [new_y], [new_speed], [new_heading], [new_acc]])

    def update_imu(self, heading, acceleration, speed):
        """Update using IMU with stationary detection."""
        # Convert SUMO heading to math convention
        heading_rad = np.radians(90 - heading)

        # Detect if vehicle is stationary
        is_stationary_now = abs(speed) < self.stationary_threshold

        # If just stopped or just started moving, reset relevant state
        if is_stationary_now != self.is_stationary:
            if is_stationary_now:
                self.x[2, 0] = 0.0  # Reset speed to zero
                self.x[4, 0] = 0.0  # Reset acceleration to zero
            self.is_stationary = is_stationary_now

        # Adjust measurement noise based on motion state
        if self.is_stationary:
            R_imu_current = np.diag([0.01, 0.01, 0.01])  # Lower noise when stopped
        else:
            R_imu_current = self.R_imu

        z = np.array([[heading_rad], [acceleration], [speed]])
        h = np.array([[self.x[3, 0]], [self.x[4, 0]], [self.x[2, 0]]])

        # Handle angle wrapping
        y = z - h
        y[0, 0] = np.arctan2(np.sin(y[0, 0]), np.cos(y[0, 0]))

        H = np.zeros((3, self.state_dim))
        H[0, 3] = 1
        H[1, 4] = 1
        H[2, 2] = 1

        S = H @ self.P @ H.T + R_imu_current
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(self.state_dim) - K @ H) @ self.P

        # Normalize heading
        self.x[3, 0] = np.arctan2(np.sin(self.x[3, 0]), np.cos(self.x[3, 0]))

    def update_gps(self, gps_position):
        """Smoothed GPS update with outlier rejection."""
        z = np.array([gps_position.x, gps_position.y]).reshape(-1, 1)
        h = self.x[:2]

        # Calculate innovation
        y = z - h

        H = np.zeros((2, self.state_dim))
        H[0, 0] = 1
        H[1, 1] = 1

        S = H @ self.P @ H.T + self.R_gps

        # Outlier rejection using Mahalanobis distance
        mahalanobis = y.T @ np.linalg.inv(S) @ y
        if mahalanobis > 16.0:  # Chi-square 99.9% confidence for 2 DOF
            print(f"Warning: GPS outlier rejected at step {self.step_count}. Distance: {mahalanobis[0, 0]:.2f}")
            return

        # Progressive update for smoother transitions
        # Instead of applying the full update at once, apply it gradually
        alpha = 0.7  # Smoothing factor (1.0 = standard KF, smaller = smoother)

        K = alpha * self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(self.state_dim) - K @ H) @ self.P

        # Update last GPS info for next time
        self.last_gps_pos = z
        self.last_gps_time = self.step_count

    def process_step(self, step_record):
        """Process step with improved state handling."""
        self.step_count = step_record.step
        print(f"Step {self.step_count}: Raw SUMO values - "
              f"speed={step_record.speed:.2f} m/s, "
              f"heading={step_record.heading:.1f}°, "
              f"acc={step_record.acceleration:.2f} m/s²")

        # Prediction
        self.predict()

        # IMU update
        self.update_imu(
            step_record.heading,
            step_record.acceleration,
            step_record.speed
        )

        # GPS update every 10 steps
        if self.step_count % 10 == 0 and (step_record.measured_position is not None):
            self.update_gps(step_record.measured_position)

        # Save history
        self.history['true_position'].append([step_record.real_position.x, step_record.real_position.y])
        self.history['estimated_position'].append(self.x[:2, 0])

        if step_record.measured_position is not None and self.step_count % 10 == 0:
            pos = step_record.measured_position
            self.history['gps_position'].append([pos.x, pos.y])
        else:
            self.history['gps_position'].append(None)

        self.history['step'].append(self.step_count)

    def plot_results(self):
        true_pos = np.array(self.history['true_position'])
        est_pos = np.array(self.history['estimated_position'])
        gps_pos = np.array([p if p is not None else [np.nan, np.nan] for p in self.history['gps_position']])
        plt.figure(figsize=(10, 8))
        plt.plot(true_pos[:, 0], true_pos[:, 1], 'b-', label='True Position')
        plt.plot(est_pos[:, 0], est_pos[:, 1], 'r--', label='EKF Estimate')
        valid = ~np.isnan(gps_pos[:, 0])
        plt.scatter(gps_pos[valid, 0], gps_pos[valid, 1], marker='x', label='GPS')
        plt.legend();
        plt.grid(True)
        plt.xlabel('X');
        plt.ylabel('Y');
        plt.axis('equal')
        plt.show()
        # Error plot
        error = np.linalg.norm(true_pos - est_pos, axis=1)
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['step'], error)
        plt.xlabel('Step');
        plt.ylabel('Position Error');
        plt.grid(True)
        plt.show()
        print("stop")
# Usage remains:
# for step_record in main_vehicle.position_history:
#     ekf.process_step(step_record)
# ekf.plot_results()


