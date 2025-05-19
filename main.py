import traci
import numpy as np

from core_classes import RSU, Position, Vehicle
from manager_classes import SimulationManager, CalculationManager, VehicleEKF
from error_classes import GPSErrorModel, CommunicationDistanceErrorModel

if __name__ == "__main__":

    # Simulation parameters
    simulation_params = {

        'gps_refresh_rate': 10,  # the rate in which the GPS is updated
        'dsrc_refresh_rate': 5,
        'ins_refresh_rate': 1,

        'number_of_steps': 600,
        'gps_error_model_std': 8,  # 5 - standard std, 10 - very bad sat con
        'communication_error_model_std': 2,
        'systematic_bias': 0.3,
        'proximity_radius': 300,
        'rsu_flag': True
    }

    # initialize the GPS error model
    gps_error_model = GPSErrorModel(simulation_params['gps_error_model_std'])
    comm_error_model = CommunicationDistanceErrorModel(simulation_params['communication_error_model_std'],
                                                       simulation_params['systematic_bias'])
    ##delay = DelayModel() #they values we want

    ##TODO switch it to more dynamic method
    specific_car_id = "veh1"

    # TODO find more pythonic way to do this
    better_flag = True  # True if you want to use the better method
    want_print = True  # True if you want to print

    # TODO add exception if sim is not 1,2,3
    #TODO change them to also high traffic and med traffice

    sim = int(input("1- Route 90\n2- Haifa\n3- Manhattan\n"))
    if sim == 1:
        simulation_path = "Sumo/route_90_high_traffic/osm.sumocfg"
        simulation_type = 'Route_90'
    elif sim == 2:
        simulation_path = "Sumo/Haifa/osm.sumocfg"
        simulation_type = 'Haifa'
    elif sim == 3:
        simulation_path = "Sumo/manhattan_high_traffic/osm.sumocfg"
        simulation_type = 'Manhattan'
    else:
        raise ValueError(f"Invalid simulation type: {sim}")

    print(f"Running simulation: {simulation_type}")

    # # Step 1: Choose simulation scenario
    # sim = int(input("Choose simulation scenario:\n1- Route 90\n2- Haifa\n3- Manhattan\n"))
    #
    # if sim == 1:
    #     simulation_path = "Sumo/route_90_high_traffic/osm.sumocfg"
    #     simulation_type = 'Route_90'
    #     gps_error_std = 5  # Low error – open highway
    # elif sim == 2:
    #     simulation_path = "Sumo/Haifa/osm.sumocfg"
    #     simulation_type = 'Haifa'
    #     gps_error_std = 8  # Medium error – mid-sized city
    # elif sim == 3:
    #     simulation_path = "Sumo/manhattan_high_traffic/osm.sumocfg"
    #     simulation_type = 'Manhattan'
    #     gps_error_std = 10  # High error – dense urban area
    # else:
    #     raise ValueError(f"Invalid simulation type: {sim}")
    #
    # print(f"Selected simulation: {simulation_type}")
    #
    # # Step 2: Choose traffic density
    # density = int(input("Choose traffic density:\n1- Low\n2- Medium\n3- High\n"))
    #
    # if density == 1:
    #     insertion_density = 1000
    # elif density == 2:
    #     insertion_density = 3000
    # elif density == 3:
    #     insertion_density = 5000
    # else:
    #     raise ValueError(f"Invalid traffic density: {density}")
    #
    # print(f"Selected traffic density: {insertion_density} vehicles/hour/km")
    #
    # # Step 3: Set simulation parameters
    # simulation_params = {
    #     'gps_refresh_rate': 10,
    #     'dsrc_refresh_rate': 5,
    #     'ins_refresh_rate': 1,
    #     'number_of_steps': 600,
    #     'gps_error_model_std': gps_error_std,  # Set based on scenario
    #     'communication_error_model_std': 2,
    #     'systematic_bias': 0.3,
    #     'proximity_radius': 300,
    #     'rsu_flag': True,
    #     'insertion_density': insertion_density  # New addition
    # }



    simulation_manager = SimulationManager(simulation_params, simulation_type, gps_error_model, comm_error_model)

    ##TODO change the specific_car_id method to be more dynamic
    main_vehicle = simulation_manager.run_simulation(simulation_path)

    ##TODO analyze results and print them
    # calc_manager = CalculationManager(main_vehicle)

    # for step_record in main_vehicle.position_history:
    #     if step_record.nearby_vehicles:
    #         print("hu")
    ekf = VehicleEKF(main_vehicle.position_history[0])

    for step_record in main_vehicle.position_history:
        ekf.process_step(step_record)

    ekf.plot_results()
        #
        # return ekf

    # Note: data_sequence should be a list of vehicle_data dictionaries as shown in your format
    # Run the simulation with your actual data from SUMO
    # run_simulation(data_sequence)
