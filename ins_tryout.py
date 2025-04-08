import traci
import geopy.distance
import numpy as np
from scipy.optimize import minimize
from geopy.distance import geodesic
from utility_functions import calculate_distance, add_gps_error_and_precision, estimate_next_position
from vehicle_tracker import VehicleTracker
import matplotlib.pyplot as plt

from rsu_manager_deprecated import RSUManager




def run_simulation():

    #sim parameters
    specific_car_id="veh1"
    error_std_dev= 8        #5- standart std, 10- very bad sat con
    num_of_neighbors= 8
    proximity_radius = 300
    use_RSU=True        #True if you want to use the RSU in the sim
    better_flag=False    #True if want to use to use the better methos
    want_print=True     #True if want you want to print
    number_of_steps=400

    sim=int(input("1- test line\n2- Loud_City_NY\n3- High_way\n"))
    if sim==1:
        simulation_path="C:/Users/barak/Sumo/test_Line/osm.sumocfg"
        simulation_type = 'test_Line'
        specific_car_id = "f_0.1"

    if sim==2:
        simulation_path="C:/Users/barak/Sumo/Loud_City_NY/osm.sumocfg"
        simulation_type = 'Loud_City_NY'

    if sim==3:
        simulation_path="C:/Users/barak/Sumo/High_way/osm.sumocfg"
        simulation_type = 'High_way'

    print(simulation_type)

    # Initialize the RSUManager and VehicleTracker
    rsu_manager = RSUManager(simulation_type, use_RSU, proximity_radius)

    vehicle_tracker = VehicleTracker(rsu_manager,specific_car_id,error_std_dev,num_of_neighbors,proximity_radius,better_flag)

    #Start SUMO simulation
    traci.start(["sumo", "-c", simulation_path])


    prev_position = None  # Initialize previous position to None
    previous_speed = 0  # Initialize previous speed to 0
    previous_heading = None  # Initialize previous heading to None
    previous_acceleration = 0  # Initialize previous acceleration to 0
    error_distances = []  # List to store the error distances
    ae = []
    first_step = True
    inertial_positions = []  # List to store inertial navigation positions

    for step in range(number_of_steps):  # Adjust the number of steps as needed
        traci.simulationStep()
        vehicle_ids = traci.vehicle.getIDList()

        #Update data for all exsisting cars
        for vehicle_id in vehicle_ids:
            position = traci.vehicle.getPosition(vehicle_id)
            geo_position = traci.simulation.convertGeo(position[0], position[1])
            speed = traci.vehicle.getSpeed(vehicle_id)

            vehicle_tracker.update_vehicle_data(vehicle_id, geo_position, speed, step)

            if vehicle_id==specific_car_id:
                heading = traci.vehicle.getAngle(specific_car_id)
                acceleration = traci.vehicle.getAcceleration(specific_car_id)
                if first_step:
                    # Add noise to the initial GPS position
                    prev_position = geo_position
                    prev_speed=speed
                    prev_heading=heading
                    prev_acceleration=acceleration
                    first_step = False

                else:
                    # Estimate the next position using inertial navigation
                    estimated_position = estimate_next_position(prev_position, prev_speed, prev_heading, prev_acceleration,step%10,step_length=1)
                    actual_position = geo_position

                    error_distance = calculate_distance(estimated_position, actual_position)
                    distance_traveled = calculate_distance(prev_position, actual_position)
                    # Store the squared error distance
                    error_distances.append(error_distance ** 2)
                    ae.append(error_distance)

                    inertial_positions.append(estimated_position)

                    # Print the error distance
                    print(f"Step {step} - Vehicle {specific_car_id} - Estimated Position: {estimated_position}, Actual Position: {actual_position}, Error Distance(actual vs inertial): {error_distance} meters")
                    print(f"Step {step} - Vehicle {specific_car_id} - Distance Traveled: {distance_traveled} meters")

                    prev_speed=speed
                    prev_heading=heading
                    prev_acceleration=acceleration

                    if step % 10 == 0:
                        prev_position,_ = add_gps_error_and_precision(actual_position,4)

                    else:
                        prev_position= estimated_position


        vehicle_tracker.find_nearby_vehicles_and_check_rsus(vehicle_ids, step)

    mse = np.mean(error_distances)
    abe = np.mean(ae)

    # Plotting the graph
    plt.figure(figsize=(10, 6))
    plt.plot(ae, marker='o')
    plt.title('Error Distances Over Steps')
    plt.xlabel('Step (Index)')
    plt.ylabel('Error Distance')
    plt.grid(True)
    plt.show()

    print("****************")
    print(f"Mean Squared Error (MSE) for the entire simulation (inertial navigation vs real pos): {mse} meters²")
    print("****************")

    print("****************")
    print(f"Average Absolute Error for the entire simulation (inertial navigation vs real pos): {abe} meters")
    print("****************")


    # After simulation loop
    vehicle_tracker.inertial_positions = inertial_positions  # Store the inertial positions in vehicle_tracker

    #simple run of the sim
    vehicle_tracker.print_estimated_positions_and_errors(1,want_print)
    vehicle_tracker.calculate_mse_satellite_positions()

    #vehicle_tracker.print_error_results()

if __name__ == "__main__":
    run_simulation()