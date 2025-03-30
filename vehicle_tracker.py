
import numpy as np
from utility_functions import add_gps_error_and_precision, add_communication_distance_error, calculate_distance, trilaterate_gps, calculate_weighted_position
class VehicleTracker:
    def __init__(self, rsu_manager,specific_car_id,error_std_dev,num_of_neighbors,proximity_radius,better_flag):
        self.vehicle_data = {}
        self.specific_car_id = specific_car_id
        self.rsu_manager = rsu_manager
        self.trilateration_data = {}  # For later use in trilateration
        self.sat_std=error_std_dev
        self.neighbors_number=num_of_neighbors
        self.proximity_radius=proximity_radius
        self.errors_results=[]
        self.sat_abs_error=None
        self.sat_mse=None
        self.MD_errors=[]
        self.FA_errors=[]
        self.better_values=[]
        self.not_better_values=[]
        self.better_flag=better_flag

    def update_vehicle_data(self, vehicle_id, geo_position, speed, step):
        if vehicle_id not in self.vehicle_data:
            self.vehicle_data[vehicle_id] = {
                'real_positions': [],  # Store the real positions without error
                'positions': [],  # This will now store positions with error
                'precision_radius': [],
                'speeds': [],
                'start_step': step,
                'nearby_vehicles': [],
                'estimated_position': [],
            }

        self.vehicle_data[vehicle_id]['real_positions'].append(geo_position)
        self.vehicle_data[vehicle_id]['speeds'].append(speed)

        # Add error to the geo_position and store it
        perturbed_position, precision_radius = add_gps_error_and_precision(geo_position,self.sat_std)
        self.vehicle_data[vehicle_id]['positions'].append(perturbed_position)
        self.vehicle_data[vehicle_id]['precision_radius'].append(precision_radius)

    def find_nearby_vehicles_and_check_rsus(self, vehicle_ids, step):
        # Using the real position ONLY in order to calculate the real distances
        specific_car_position = self.vehicle_data[self.specific_car_id]['real_positions'][-1] if self.specific_car_id in self.vehicle_data else None
        if specific_car_position is None:
            return  # If the specific car's position is not known, skip the check

        # Initialize data structure for the specific car if it's the first time running this check
        if step not in self.trilateration_data:
            self.trilateration_data[step] = ([], [], [])  # (positions, distances, precision_radius)

        # Iterate over RSUs
        for rsu_index, rsu_position in enumerate(self.rsu_manager.rsu_locations):
            distance_to_rsu = calculate_distance(specific_car_position, rsu_position)
            if distance_to_rsu <= self.proximity_radius:
                self.trilateration_data[step][0].append(rsu_position)
                self.trilateration_data[step][1].append(distance_to_rsu)
                self.trilateration_data[step][2].append(0.1)

        # Only iterate once per vehicle, avoiding redundant checks
        for other_vehicle_id in vehicle_ids:
            if other_vehicle_id == self.specific_car_id:
                continue  # Skip the specific car itself
            other_geo_position_for_dis = self.vehicle_data[other_vehicle_id]['real_positions'][-1]
            original_distance = calculate_distance(specific_car_position, other_geo_position_for_dis)
            # Note: for now the distance noise is 0
            perturbed_distance = add_communication_distance_error(original_distance)
            other_geo_position = self.vehicle_data[other_vehicle_id]['positions'][-1]
            precision_radius = self.vehicle_data[other_vehicle_id]['precision_radius'][-1]

            if perturbed_distance <= self.proximity_radius:
                self.vehicle_data[self.specific_car_id]['nearby_vehicles'].append(
                    (other_vehicle_id, perturbed_distance, other_geo_position, step))
                # Store data for trilateration
                self.trilateration_data[step][0].append(other_geo_position)
                self.trilateration_data[step][1].append(perturbed_distance)
                self.trilateration_data[step][2].append(precision_radius)


    def select_positions_for_triangulation(self, positions_distances_radii,step):
        # Sort positions by precision radius (ascending), so the best precision is first
        (positions, distances, precision_radius)=positions_distances_radii
        better=True
        num_of_parti=self.neighbors_number

        index=np.argsort(precision_radius)[:num_of_parti+1]
        index = index[:num_of_parti]
        selected_positions = [positions[i] for i in index]
        selected_distances = [distances[i] for i in index]
        selected_accr = [precision_radius[i] for i in index]

        if selected_accr[2]>self.vehicle_data[self.specific_car_id]['precision_radius'][step]:
            better=False
        if not(self.better_flag):
            better=True
        #print(better)
        accr = [float(x) for x in selected_accr]
        print(accr)
        return (selected_positions,selected_distances,better,selected_accr)


    def print_estimated_positions_and_errors(self,alpha,want_print):
        # Initialize variables for error accumulation
        squared_errors = []
        absolute_errors = []

        FA=0
        MD=0
        TP=0
        FN=0
        not_beter_count=0
        if want_print:
            print(f"\nEstimated Positions, Triangulation Errors, and Satellite Position Errors for {self.specific_car_id}:")

        for step, (positions, distances, precision_radius) in self.trilateration_data.items():
            if len(positions) >= 4:
                actual_position_index = step - self.vehicle_data[self.specific_car_id]['start_step']
                if actual_position_index < len(self.vehicle_data[self.specific_car_id]['real_positions']):
                    best_positions, best_distances, better, best_precision_radius = self.select_positions_for_triangulation(
                        (positions, distances, precision_radius), actual_position_index)
                    sat_pos=self.vehicle_data[self.specific_car_id]['positions'][actual_position_index]
                    estimated_position = trilaterate_gps(best_positions, best_distances, best_precision_radius,self.vehicle_data[self.specific_car_id]['positions'][actual_position_index], alpha)

                    actual_position = self.vehicle_data[self.specific_car_id]['real_positions'][actual_position_index]

                    # Calculate errors
                    triangulation_error = calculate_distance(estimated_position, actual_position)
                    satellite_position_error = calculate_distance(sat_pos, actual_position)

                    #if better=False so we use sat pos
                    #if better=True we use our method
                    if better and triangulation_error>satellite_position_error:
                        MD+=1
                        self.MD_errors.append(triangulation_error-satellite_position_error)
                    if better and triangulation_error<satellite_position_error:
                        TP+=1
                        self.better_values.append(satellite_position_error-triangulation_error)

                    if not(better) and triangulation_error<satellite_position_error:
                        FA+=1
                        self.FA_errors.append(satellite_position_error-triangulation_error)
                    if not (better) and triangulation_error > satellite_position_error:
                        FN+=1
                        self.not_better_values.append(triangulation_error-satellite_position_error)


                    if not(better):
                        estimated_position=sat_pos
                        not_beter_count+=1

                    self.vehicle_data[self.specific_car_id]['estimated_position'].append(estimated_position)

                    # Accumulate errors for MSE and absolute error calculations
                    squared_errors.append(triangulation_error ** 2)
                    absolute_errors.append(abs(triangulation_error))
                    if want_print:
                        print(
                            f"  Step {step}: Estimated Position - Lat: {estimated_position[0]}, Lon: {estimated_position[1]} | "
                            f"Triangulation Error: {triangulation_error:.2f} meters | "
                            f"Satellite Position Error: {satellite_position_error:.2f} meters | "
                            f"Number of participants: {len(positions)}")
                else:
                    print(f"  Step {step}: Data unavailable for real position.")
            else:
                print(f"  Step {step}: Not enough data for trilateration.")
        #plot_errors(absolute_errors,squared_errors)
        uses_of_better=100*not_beter_count / len(squared_errors)
        print(f"\n{uses_of_better:.2f}% of the position are the Satellite,{not_beter_count} Times")
        print(f"FA: {FA} Times, {FA*100/len(squared_errors):.2f}%")
        print(f"MD: {MD} Times, {MD*100/len(squared_errors):.2f}%")
        print(f"FN: {FN} Times, {FN*100/len(squared_errors):.2f}%")
        print(f"TP: {TP} Times, {TP*100/len(squared_errors):.2f}%")


        # Calculate and print the MSE and the average (absolute) error
        if squared_errors:
            mse = sum(squared_errors) / len(squared_errors)
            avg_absolute_error = sum(absolute_errors) / len(absolute_errors)
            print(f"\nMSE for Estimated Positions: {mse:.2f} meters squared")
            print(f"Average Absolute Error for Estimated Positions: {avg_absolute_error:.2f} meters")
        self.errors_results.append((avg_absolute_error, mse,uses_of_better ))

    def calculate_mse_satellite_positions(self):
        squared_errors = []
        avg=[]
        for step in range(len(self.vehicle_data[self.specific_car_id]['positions'])):
            perturbed_position = self.vehicle_data[self.specific_car_id]['positions'][step]
            if step < len(self.vehicle_data[self.specific_car_id]['real_positions']):
                actual_position = self.vehicle_data[self.specific_car_id]['real_positions'][step]
                error = calculate_distance(perturbed_position, actual_position)
                squared_errors.append(error ** 2)
                avg.append(error)
        if squared_errors:
            mse = np.mean(squared_errors)
            error=np.mean(avg)
            self.sat_abs_error=error
            self.sat_mse=mse
            print(f"MSE for Satellite Positions: {mse:.2f} meters squared")
            print(f"Abs Error for Satellite Positions: {error:.2f} meters ")

        else:
            print("No data available to calculate MSE for Satellite Positions.")
        print()

    def print_error_results_of_sweep(self,start,step):
        for i, (avg_absolute_error, mse, uses_of_better) in enumerate(self.errors_results):
            print(f"For alpha = {start+i*step}:")
            print(f"  Average Absolute Error: {avg_absolute_error:.2f}.     improve of: {(self.sat_abs_error-avg_absolute_error)*100/self.sat_abs_error:.2f}%")
            print(f"  Mean Squared Error: {mse:.2f}.           improve of: {(self.sat_mse-mse)*100/self.sat_mse:.2f}%")
            print(f"  Uses of Better: {uses_of_better:.2f}%")
            print()

    def print_error_results(self):

        for i, (avg_absolute_error, mse, uses_of_better) in enumerate(self.errors_results):
            print(f"  Average Absolute Error: {avg_absolute_error:.2f}.     improve of: {(self.sat_abs_error-avg_absolute_error)*100/self.sat_abs_error:.2f}%")
            print(f"  Mean Squared Error: {mse:.2f}.        improve of: {(self.sat_mse-mse)*100/self.sat_mse:.2f}%")
            print(f"  Uses of Better: {uses_of_better:.2f}%")
            print()
        print()
        print("MD- We think our result is better but it is not")
        print("TD- We think our result is better and rightly so")
        print("FA- We think our result is not better but it is")
        print("TA- We think our result is no better and it really isn't")
        print()
        total=len(self.MD_errors)+len(self.better_values)+len(self.FA_errors)+len(self.not_better_values)
        print(f"E[MD]- when we wrong we by:             {np.mean(self.MD_errors):.2f} meter     {len(self.MD_errors)*100/total:.2f}% of the time")
        print(f"E[TD]- when we right we by:             {np.mean(self.better_values):.2f} meter     {len(self.better_values)*100/total:.2f}% of the time")
        print(f"E[FA]- when better=false we wrong by:   {np.mean(self.FA_errors):.2f} meter     {len(self.FA_errors)*100/total:.2f}% of the time")
        print(f"E[TA]- using better, saves:             {np.mean(self.not_better_values):.2f} meter     {len(self.not_better_values)*100/total:.2f}% of the time")

    def sweep_alpha(self,vehicle_tracker,start,end,step,want_print):
        for i in np.arange(start, end + step, step):
            vehicle_tracker.print_estimated_positions_and_errors(1, want_print)
            print(i)

    def sweep_neighbors(self,vehicle_tracker,start,end,step,want_print):
        for i in np.arange(start, end + step, step):
            self.neighbors_number=i
            vehicle_tracker.print_estimated_positions_and_errors(1, want_print)
            print(i)

