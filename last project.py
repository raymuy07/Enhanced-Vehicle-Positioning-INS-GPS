def calculate_distance(pos1, pos2):
    """Calculate the geodesic distance between two GPS positions."""
    return geopy.distance.geodesic(pos1, pos2).meters

def add_gps_error_and_precision(gps_location, error_std_dev):
    """
    Adds a realistic error to a GPS location and returns the modified location
    with a precision radius that reflects the magnitude of the error introduced.

    Parameters:
    - gps_location: tuple, the original (latitude, longitude) GPS coordinates.
    - error_std_dev: float, the standard deviation of the error to add to the GPS coordinates (in meters).

    Returns:
    - perturbed_location: tuple, the GPS location after adding the error.
    - precision_radius: float, the precision radius of the perturbed location (in meters),
                         indicating the confidence level in the perturbed location.
    """
    # Convert error from meters to degrees approximately (rough approximation)
    error_in_degrees = error_std_dev / 111320

    # Adding random error to latitude and longitude
    error_lat = np.random.normal(0, error_in_degrees)
    error_lon = np.random.normal(0, error_in_degrees)

    perturbed_lat = gps_location[0] + error_lat
    perturbed_lon = gps_location[1] + error_lon
    perturbed_location = (perturbed_lat, perturbed_lon)

    # Calculate precision radius based on the magnitude of the error introduced
    error_magnitude = np.sqrt(error_lat**2 + error_lon**2) * 111320  # Convert back to meters for precision radius
    precision_radius = error_magnitude  # Directly use error magnitude as precision radius

    return perturbed_location, precision_radius

def add_communication_distance_error(original_distance, error_std_dev=2, systematic_bias=0.3):
    random_error = np.random.normal(0, error_std_dev)
    perturbed_distance = original_distance + random_error + systematic_bias
    return perturbed_distance

def trilaterate_gps(surrounding_positions, distances, error_radii,sat_pos,alpha):
    # Define the objective function for optimization
    def objective(x):
        est_pos = np.array(x)
        return sum(((geopy.distance.geodesic(est_pos, np.array(pos)).meters - dist) / error_radius**alpha) ** 2
                   for pos, dist, error_radius in zip(surrounding_positions, distances, error_radii))

    # Initial guess: sat pos
    initial_guess = sat_pos
    # Define bounds for latitude and longitude
    lat_bounds = (-90, 90)
    lon_bounds = (-180, 180)
    bounds = [lat_bounds, lon_bounds]

    # Perform the minimization with bounds
    result = minimize(objective, initial_guess, bounds=bounds)

    return result.x


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

class RSUManager:
    def __init__(self, rsu_locations, reception_radius):
        self.rsu_locations = rsu_locations
        self.reception_radius = reception_radius
        self.vehicle_proximity = {step: {rsu_index: [] for rsu_index in range(len(rsu_locations))} for step in
                                  range(100)}  # Assuming 100 steps for simplicity

    def update_vehicle_proximity(self, vehicle_id, vehicle_geo_position, step):
        """
        Updates the proximity information for a vehicle in relation to all RSUs.

        :param vehicle_id: ID of the vehicle.
        :param vehicle_geo_position: The geographical position (latitude, longitude) of the vehicle.
        :param step: The current simulation step.
        """
        for rsu_index, rsu_position in enumerate(self.rsu_locations):
            distance = calculate_distance(vehicle_geo_position, rsu_position)
            if distance <= self.reception_radius:
                if step not in self.vehicle_proximity:
                    self.vehicle_proximity[step] = {}
                if rsu_index not in self.vehicle_proximity[step]:
                    self.vehicle_proximity[step][rsu_index] = []
                self.vehicle_proximity[step][rsu_index].append((vehicle_id, distance))

    def print_vehicles_near_rsus(self):
        """
        Prints all vehicles that have been near each RSU for each step, including the distance if in range.
        """
        for step, rsus in self.vehicle_proximity.items():
            print(f"\nStep {step}:")
            for rsu_index, vehicles in rsus.items():
                if vehicles:  # Check if there are any vehicles near this RSU at this step
                    print(f"  RSU {rsu_index} - Vehicles in range:")
                    for vehicle_id, distance in vehicles:
                        print(f"    Vehicle {vehicle_id} at distance {distance:.2f} meters")


def generate_rsu_grid(point1, point2, point3, point4, interval_km=1):
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

def run_simulation():

    #sim parameters
    specific_car_id="veh1"
    error_std_dev= 8        #5- standart std, 10- very bad sat con
    num_of_neighbors= 8
    proximity_radius = 300
    use_RSU=True        #True if you want to use the RSU in the sim
    better_flag=True    #True if want to use to use the better methos
    want_print=True     #True if want you want to print
    number_of_steps=600

    sim=int(input("1- test line\n2- Loud_City_NY\n3- High_way\n"))
    if sim==1:
        simulation_path="Sumo/test_Line/osm.sumocfg"
        simulation = 'test_Line'
        specific_car_id = "f_0.1"
    if sim==2:
        simulation_path="Sumo/Loud_City_NY/osm.sumocfg"
        simulation = 'Loud_City_NY'
    if sim==3:
        simulation_path="Sumo/High_way/osm.sumocfg"
        simulation = 'High_way'

    print(simulation)

    if simulation == 'test_Line':
        point1 = (34.896294, 32.124676)
        point2 = (34.896367, 32.128195)
        point3 = (35.121167, 32.124688)
        point4 = (35.120992, 32.121777)

    elif simulation == 'Loud_City_NY':
        point1 = (-74.119840, 40.688979)
        point2 = (-74.119459, 40.723241)
        point3 = (-74.043905, 40.686053)
        point4 = (-74.043493, 40.724598)

    elif simulation == 'High_way':
        point1 = (34.890866,31.920025 )
        point2 = (35.023163,31.917199 )
        point3 = (35.019656,31.976297 )
        point4 = (34.883820,31.981850 )
    if use_RSU:
        rsu_locations = generate_rsu_grid(point1,point2,point3,point4)
    else:
        rsu_locations = []

    rsu_manager = RSUManager(rsu_locations, proximity_radius)
    vehicle_tracker = VehicleTracker(rsu_manager,specific_car_id,error_std_dev,num_of_neighbors,proximity_radius,better_flag)

    #Start SUMO simulation
    traci.start(["sumo", "-c", simulation_path])

    for step in range(number_of_steps):  # Adjust the number of steps as needed
        traci.simulationStep()
        vehicle_ids = traci.vehicle.getIDList()

        #Update data for all exsisting cars
        for vehicle_id in vehicle_ids:
            position = traci.vehicle.getPosition(vehicle_id)
            geo_position = traci.simulation.convertGeo(position[0], position[1])
            speed = traci.vehicle.getSpeed(vehicle_id)

            vehicle_tracker.update_vehicle_data(vehicle_id, geo_position, speed, step)
            #rsu_manager.update_vehicle_proximity(vehicle_id, geo_position, step)

        vehicle_tracker.find_nearby_vehicles_and_check_rsus(vehicle_ids, step)

    # After simulation loop

    #simple run of the sim
    vehicle_tracker.print_estimated_positions_and_errors(1,want_print)
    vehicle_tracker.calculate_mse_satellite_positions()

    vehicle_tracker.print_error_results()


    '''
There are several experiments that can be done
Sweep on the number Neighbors
Sweep on the optimize function
and more can be added

    #neighbors sweep
    start = 6
    end = 12
    step = 1
    vehicle_tracker.sweep_neighbors(vehicle_tracker,start,end,step,want_print)


    #alpha sweep
    start = 0.7
    end = 1.5
    step = 0.1
    want_print=True
    vehicle_tracker.sweep_alpha(vehicle_tracker,start,end,step, want_print)
    '''
    #rsu_manager.print_vehicles_near_rsus()

if __name__ == "__main__":
    run_simulation()



