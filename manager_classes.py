import traci
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from core_classes import Position, Vehicle, RSUManager
from utility_functions import cartesian_distance
import traci.constants
import sumolib


class SimulationManager:
    """Manages the overall simulation."""

    def __init__(self, simulation_params, simulation_type, gps_error_model, comm_error_model, gps_outage):

        self.rsu_manager = None
        self.vehicle_objs = None
        self.tracked_ids = None

        self.simulation_type = simulation_type
        self.gps_error_model = gps_error_model
        self.comm_error_model = comm_error_model

        self.num_vehicles_to_track = simulation_params.get('num_vehicles_to_track', 1)
        self.gps_refresh_rate = simulation_params.get('gps_refresh_rate', 10)
        self.dsrc_refresh_rate = simulation_params.get('dsrc_refresh_rate', 2)
        self.ins_refresh_rate = simulation_params.get('ins_refresh_rate', 1)

        self.num_steps = simulation_params.get('number_of_steps', 500)
        self.proximity_radius = simulation_params.get('proximity_radius', 300)
        self.rsu_flag = simulation_params.get('rsu_flag', False)
        self.gps_outage = []
        if gps_outage:
            self.gps_outage = range(self.num_steps//2 - 300, self.num_steps//2)

    @staticmethod
    def get_random_main_vehicles(initial_steps, n):
        for _ in range(initial_steps):
            traci.simulationStep()
        ids = traci.vehicle.getIDList()
        if not ids:
            raise RuntimeError("No vehicles in the scenario")
        return random.sample(ids, min(n, len(ids)))

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

    def find_neighbours(self, veh_id):
        context = traci.vehicle.getContextSubscriptionResults(veh_id)
        if context is None:
            return []

        neighbors = []
        for veh_id, var_dict in context.items():
            if traci.vehicle.getTypeID(veh_id) != "DEFAULT_VEHTYPE":
                continue
            pos = var_dict.get(traci.constants.VAR_POSITION)
            if pos is None:
                continue
            x, y = pos
            distance = np.linalg.norm(np.array(pos) - np.array(traci.vehicle.getPosition(veh_id)))
            real_world_distance = self.comm_error_model.apply_error(distance)

            if real_world_distance <= self.proximity_radius:
                map_position = Position(x, y)
                position_w_error = self.gps_error_model.apply_error((x, y))

                snapshot = {
                    'id': veh_id,
                    'true_position': map_position,
                    'position_w_error': position_w_error,
                    'real_world_distance': real_world_distance
                }

                neighbors.append(snapshot)

        return neighbors

    def find_nearby_rsu(self, vehicle_cartesian_position):
        """
        Returns a list of nearby RSUs within communication range of the vehicle.
        Each entry contains the RSU object and it's noisy distance from the vehicle.
        """

        nearby_rsus = []

        for rsu in self.rsu_manager.rsu_positions:

            distance_to_rsu = rsu.position.cartesian_distance_to(vehicle_cartesian_position)
            measured_distance_to_rsu = self.comm_error_model.apply_error(distance_to_rsu)

            if measured_distance_to_rsu <= self.proximity_radius:
                nearby_rsus.append({
                    'rsu': rsu,
                    'distance_from_veh': measured_distance_to_rsu
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
        self.tracked_ids = self.get_random_main_vehicles(initial_steps, self.num_vehicles_to_track)
        self.vehicle_objs = {vid: Vehicle(vid) for vid in self.tracked_ids}
        subscribed = set()

        active_ids = set(self.tracked_ids)
        for step in range(initial_steps, self.num_steps):
            traci.simulationStep()
            current_ids = set(traci.vehicle.getIDList())

            vanished = active_ids - current_ids
            for veh_id in vanished:
                active_ids.remove(veh_id)
            if not active_ids:
                break

            # subscribe any newly appearing tracked car
            for veh_id in active_ids - subscribed:
                traci.vehicle.subscribeContext(
                    veh_id,
                    traci.constants.CMD_GET_VEHICLE_VARIABLE,
                    self.proximity_radius,
                    [traci.constants.VAR_POSITION]
                )
                subscribed.add(veh_id)

            # update every vehicle still in simulation
            for veh_id in list(active_ids):
                veh_obj = self.vehicle_objs[veh_id]
                pos = traci.vehicle.getPosition(veh_id)  # real position
                # gather IMU data
                speed = traci.vehicle.getSpeed(veh_id)
                acc = traci.vehicle.getAcceleration(veh_id)
                heading = traci.vehicle.getAngle(veh_id)
                # gather DSRC data
                if step % self.dsrc_refresh_rate == 0:
                    neighbours = self.find_neighbours(veh_id)
                    nearby_rsus = self.find_nearby_rsu(pos)
                else:
                    neighbours = nearby_rsus = None
                # gather GPS data
                measured_position = (self.gps_error_model.apply_error(pos) if
                                     (step % self.gps_refresh_rate == 0 and step not in self.gps_outage) else None)
                vehicle_data = {
                    'step': step,
                    'speed': speed,
                    'acceleration': acc,
                    'heading': heading,
                    'real_position': pos,
                    'nearby_vehicles': neighbours,
                    'nearby_rsus': nearby_rsus,
                    'measured_position': measured_position
                }
                veh_obj.update_data(**vehicle_data)
        traci.close()
        return list(self.vehicle_objs.values())


class DSRCPositionEstimator:

    @staticmethod
    def isBetter(positions, distances, precision_radii, precision_radius):
        """
        Selects the best sources for triangulation and decides if they are enough for triangulation.

        Returns:
            (better_flag, selected positions, selected distances, selected_precisions)
        """
        better = True
        index = np.argsort(precision_radii)[:]
        sorted_positions = [positions[i] for i in index]
        sorted_distances = [distances[i] for i in index]
        sorted_precisions = [precision_radii[i] for i in index]

        if sorted_precisions[2] > precision_radius:
            better = False

        return better, sorted_positions, sorted_distances, sorted_precisions

    @staticmethod
    def _trilaterate_position(positions, distances, error_radii, initial_guess, alpha=1.0):
        """
        Weighted GPS trilateration using Euclidian distances and error-based weighting.
        """

        # Define the objective function for optimization
        def objective(x):
            est_x, est_y = x[0], x[1]
            return sum(
                (
                    (((est_x - pos.x)**2 + (est_y - pos.y)**2)**0.5 - dist) / error_radius**alpha
                ) ** 2
                for pos, dist, error_radius in zip(positions, distances, error_radii)
            )

        initial_guess_arr = np.array([initial_guess.x, initial_guess.y])
        bounds = [(0, 10000), (0, 10000)]  # adapt to your simulation's x/y bounds

        result = minimize(objective, initial_guess_arr, bounds=bounds)

        if not result.success:
            print(f"[WARNING] Trilateration failed: {result.message}")
            return initial_guess

        return Position(result.x[0], result.x[1])

    def get_dsrc_position(self, step_record, last_pos=None, alpha=1.0):
        """
        Estimate position based on RSU and DSRC trilateration.
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
        if step_record.measured_position:
            pos = step_record.measured_position
            pr = pos.precision_radius / 3
        else:
            pos = last_pos
            pr = pos.precision_radius

        # Add RSUs
        for rsu in step_record.nearby_rsus:  # need to check whats is step_record.nearby_rsus
            rsu_pos = rsu.get("rsu").position
            surrounding_positions.append(rsu_pos)
            distances.append(rsu.get("distance_from_veh"))
            error_radii.append(0.1)  # assume high confidence

        # Add vehicles
        for neighbor in step_record.nearby_vehicles:  # need to check what's in step_record.nearby_vehicles
            surrounding_positions.append(neighbor.get("position_w_error"))
            distances.append(neighbor.get("real_world_distance"))
            precision = neighbor.get("position_w_error").precision_radius
            error_radii.append(precision if precision else 8.0)  # fallback precision

        # Minimum 3+ points needed
        if len(surrounding_positions) < 3:
            return pos if step_record.measured_position else None  # not reliable

        better, sorted_positions, sorted_distances, sorted_precisions = self.isBetter(
            surrounding_positions, distances, error_radii, pos.precision_radius)

        if not better:
            return pos if step_record.measured_position else None  # not reliable

        # Estimate position
        estimated_pos = self._trilaterate_position(
            sorted_positions,
            sorted_distances,
            sorted_precisions,
            pos,
            alpha
        )
        estimated_pos.precision_radius = pr

        return estimated_pos


class VehicleEKF:
    def __init__(self, vehicle_id, dsrc_pos_estimator, first_measurement, use_dsrc):
        self.vehicle_id = vehicle_id
        self.use_dsrc = use_dsrc
        self.dsrc_pos_estimator = dsrc_pos_estimator
        # State vector: [x, y, speed, heading, acceleration]
        self.state_dim = 5

        # Initial state estimate
        if self.use_dsrc:
            pos = self.dsrc_pos_estimator.get_dsrc_position(first_measurement, None)
        else:
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
        self.R_dsrc = np.diag([4.5, 4.5])  # tune to DSRC pos error

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
            'step': [],
            'true_position': [],
            'gps_position': [],
            'dsrc_position': [],
            'estimated_position': [],
            'gps_error': [],
            'dsrc_error': [],
            'ekf_error': []
        }

    def predict(self):
        """Improved prediction with stationary handling."""
        x, y, speed, heading, acc = self.x.flatten()

        # Calculate expected movement per step
        dx = speed * np.cos(heading) * self.dt
        dy = speed * np.sin(heading) * self.dt

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

    def update_position_measurement(self, position_measurement, use_dsrc=True):
        """
        Update using an external position measurement (GPS, DSRC).

        """
        z = np.array([position_measurement.x, position_measurement.y]).reshape(-1, 1)
        h = self.x[:2]

        y = z - h

        H = np.zeros((2, self.state_dim))
        H[0, 0] = 1
        H[1, 1] = 1

        if use_dsrc:
            R = self.R_dsrc
        else:
            R = self.R_gps

        S = H @ self.P @ H.T + R
        mahalanobis = y.T @ np.linalg.inv(S) @ y
        if mahalanobis > 16.0:
            print(
                f"WARNING: Position measurement outlier rejected at step {self.step_count}. Distance: {mahalanobis[0, 0]:.2f}")
            return
        alpha = 0.7
        K = alpha * self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(self.state_dim) - K @ H) @ self.P

        # Update last GPS info for next time
        self.last_gps_pos = z
        self.last_gps_time = self.step_count

    def process_step(self, step_record):
        """Process step with improved state handling."""
        self.step_count = step_record.step

        # Prediction
        self.predict()

        # IMU update
        self.update_imu(
            step_record.heading,
            step_record.acceleration,
            step_record.speed
        )

        # Position measurement update
        gps_pos = step_record.measured_position
        gps_xy = None
        dsrc_xy = None
        if gps_pos:
            gps_xy = [gps_pos.x, gps_pos.y]
            if self.use_dsrc:
                dsrc_pos = self.dsrc_pos_estimator.get_dsrc_position(step_record)
                dsrc_xy = [dsrc_pos.x, dsrc_pos.y]
                self.update_position_measurement(dsrc_pos)
            else:
                self.update_position_measurement(gps_pos, use_dsrc=False)
        else:
            if self.use_dsrc and (step_record.nearby_rsus or step_record.nearby_vehicles):
                est_x, est_y = self.x[0, 0], self.x[1, 0]
                prior_pos = Position(est_x, est_y, 1.5)
                dsrc_pos = self.dsrc_pos_estimator.get_dsrc_position(step_record, prior_pos)
                if dsrc_pos:
                    dsrc_xy = [dsrc_pos.x, dsrc_pos.y]
                    self.update_position_measurement(dsrc_pos)

        true_xy = [step_record.real_position.x, step_record.real_position.y]
        ekf_xy = [float(self.x[0, 0]), float(self.x[1, 0])]

        self.history['step'].append(self.step_count)
        self.history['true_position'].append(true_xy)
        self.history['gps_position'].append(gps_xy)
        self.history['dsrc_position'].append(dsrc_xy)
        self.history['estimated_position'].append(ekf_xy)

        if gps_xy:
            self.history['gps_error'].append(cartesian_distance(true_xy, gps_xy))
        else:
            self.history['gps_error'].append(None)
        if dsrc_xy:
            self.history['dsrc_error'].append(cartesian_distance(true_xy, dsrc_xy))
        else:
            self.history['dsrc_error'].append(None)
        self.history['ekf_error'].append(cartesian_distance(true_xy, ekf_xy))


class PlottingManager:
    def __init__(self, ekf=None, net_file=None, dsrc_flag=False, gps_outage=None):
        self.dsrc_flag = dsrc_flag
        self.gps_outage = gps_outage or []
        self.net_file = net_file
        if ekf is not None:
            self.true_pos = np.array(ekf.history['true_position'])
            self.est_pos = np.array(ekf.history['estimated_position'])
            self.dsrc_pos = np.array([p if p is not None else [np.nan, np.nan] for p in ekf.history['dsrc_position']])
            self.gps_pos = np.array([p if p is not None else [np.nan, np.nan] for p in ekf.history['gps_position']])
        else:
            self.true_pos = self.est_pos = self.gps_pos = None

        self.history = ekf.history
        self.ekf_error = np.linalg.norm(self.true_pos - self.est_pos, axis=1)
        # For GPS error, we need to handle the potential NaN values
        self.gps_error = np.full_like(self.ekf_error, np.nan)
        for i in range(len(self.true_pos)):
            if not np.isnan(self.gps_pos[i, 0]) and not np.isnan(self.gps_pos[i, 1]):
                self.gps_error[i] = np.linalg.norm(self.true_pos[i] - self.gps_pos[i])
        self.dsrc_error = np.full_like(self.ekf_error, np.nan)
        if self.dsrc_flag:
            for i in range(len(self.true_pos)):
                if not np.isnan(self.dsrc_pos[i, 0]) and not np.isnan(self.dsrc_pos[i, 1]):
                    self.dsrc_error[i] = np.linalg.norm(self.true_pos[i] - self.dsrc_pos[i])

    def draw_network_background(self):
        if not self.net_file:
            return
        net = sumolib.net.readNet(self.net_file)
        for edge in net.getEdges():
            shape = edge.getShape()
            if len(shape) >= 2:
                xs, ys = zip(*shape)
                plt.plot(xs, ys, color='lightgray', linewidth=0.5, zorder=0)

    def plot_trajectory_comparison(self):
        if self.true_pos is None:
            print("Error: No EKF history data provided.")
            return

        plt.figure(figsize=(10, 8))
        self.draw_network_background()
        plt.plot(self.true_pos[:, 0], self.true_pos[:, 1], 'b-', label='True Position', zorder=3)
        plt.plot(self.est_pos[:, 0], self.est_pos[:, 1], 'r--', label='EKF Estimate', zorder=3)
        valid = ~np.isnan(self.gps_pos[:, 0])
        plt.scatter(self.gps_pos[valid, 0], self.gps_pos[valid, 1], marker='x', label='GPS', zorder=4, s=30)
        plt.legend(fontsize=16)
        plt.grid(True)
        plt.xlabel('X position (m)', fontsize=18)
        plt.ylabel('Y position (m)', fontsize=18)
        plt.title('Vehicle Trajectory Comparison', fontsize=20)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.axis('equal')

        # Calculate percentiles to handle outliers
        all_x = np.concatenate([self.true_pos[:, 0], self.est_pos[:, 0], self.gps_pos[valid, 0]])
        all_y = np.concatenate([self.true_pos[:, 1], self.est_pos[:, 1], self.gps_pos[valid, 1]])

        # Use 1st and 99th percentiles to exclude extreme outliers
        x_min = np.percentile(all_x, 1)
        x_max = np.percentile(all_x, 99)
        y_min = np.percentile(all_y, 1)
        y_max = np.percentile(all_y, 99)

        # Add some margin (5% of range)
        x_margin = (x_max - x_min) * 0.05
        y_margin = (y_max - y_min) * 0.05

        plt.xlim(x_min - x_margin, x_max + x_margin)
        plt.ylim(y_min - y_margin, y_max + y_margin)

        plt.show()

    def plot_error_comparison(self):
        plt.figure(figsize=(10, 6))

        # Plot both errors
        plt.plot(self.history['step'], self.ekf_error, 'r-', label='EKF Error')
        steps = np.array(self.history['step'])
        valid = ~np.isnan(self.gps_error)
        plt.plot(steps[valid], np.array(self.gps_error)[valid], linestyle='-', color='gray', linewidth=0.5, alpha=0.3,
                 label='GPS Error')
        valid_dsrc = ~np.isnan(self.dsrc_error)
        plt.plot(steps[valid_dsrc], np.array(self.dsrc_error)[valid_dsrc], linestyle='-', color='orange', linewidth=0.5,
                 alpha=0.3, label='DSRC Error')

        # Calculate and display average errors & add horizontal lines for average errors
        avg_ekf_error = np.nanmean(self.ekf_error)
        plt.axhline(y=avg_ekf_error, color='r', linestyle=':', label=f'Avg EKF Error: {avg_ekf_error:.2f}m')
        avg_gps_error = np.nanmean(self.gps_error)
        plt.axhline(y=avg_gps_error, color='b', linestyle=':', label=f'Avg GPS Error: {avg_gps_error:.2f}m')
        if self.dsrc_flag:
            avg_dsrc_error = np.nanmean(self.dsrc_error)
            plt.axhline(y=avg_dsrc_error, color='darkorange', linestyle=':', label=f'Avg DSRC Error: {avg_dsrc_error:.2f}m')

        plt.xlabel('Simulation Step', fontsize=18)
        plt.ylabel('Position Error (m)', fontsize=18)
        plt.title('Position Error Comparison: EKF vs GPS', fontsize=20)
        plt.legend(fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.grid(True)

        # Add text with improvement percentage
        improvement = (1 - avg_ekf_error / avg_gps_error) * 100
        plt.text(0.5, 0.01, f'EKF improves accuracy by {improvement:.1f}%',
                 horizontalalignment='center', verticalalignment='bottom',
                 transform=plt.gca().transAxes, fontsize=14, bbox=dict(facecolor='white', alpha=0.5))
        if self.gps_outage:
            outage_start = self.gps_outage[0]
            outage_end = self.gps_outage[-1]
            plt.axvspan(outage_start, outage_end, color='lightblue', alpha=0.2, zorder=0)
            mid_x = (outage_start + outage_end) / 2
            max_y = np.nanmax([*self.ekf_error, *self.gps_error, *self.dsrc_error])
            plt.text(mid_x, max_y * 0.95, "GPS Outage", color='gray',
                     fontsize=12, ha='center', va='bottom', alpha=0.7)

        plt.show()

    def plot_cumulative_distribution(self):
        plt.figure(figsize=(10, 6))

        # Sort errors for CDF
        sorted_ekf_error = np.sort(self.ekf_error[~np.isnan(self.ekf_error)])
        sorted_gps_error = np.sort(self.gps_error[~np.isnan(self.gps_error)])

        # Calculate cumulative probabilities
        p_ekf = np.arange(1, len(sorted_ekf_error) + 1) / len(sorted_ekf_error)
        p_gps = np.arange(1, len(sorted_gps_error) + 1) / len(sorted_gps_error)

        # Plot CDF
        plt.plot(sorted_ekf_error, p_ekf, 'r-', label='EKF Error CDF')
        plt.plot(sorted_gps_error, p_gps, 'b--', label='GPS Error CDF')

        plt.xlabel('Position Error (m)', fontsize=18)
        plt.ylabel('Cumulative Probability', fontsize=18)
        plt.title('Error Distribution: EKF vs GPS', fontsize=20)
        plt.legend(fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.grid(True)

        # Calculate and display percentile values
        ekf_95 = np.percentile(sorted_ekf_error, 95)
        gps_95 = np.percentile(sorted_gps_error, 95)

        plt.axvline(x=ekf_95, color='r', linestyle=':', label=f'EKF 95th %: {ekf_95:.2f}m')
        plt.axvline(x=gps_95, color='b', linestyle=':', label=f'GPS 95th %: {gps_95:.2f}m')
        plt.legend(fontsize=16)

        plt.show()