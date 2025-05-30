from manager_classes import SimulationManager, VehicleEKF, PlottingManager
from error_classes import GPSErrorModel, CommunicationDistanceErrorModel

if __name__ == "__main__":

    # Step 1: Choose simulation scenario
    print("Choose simulation scenario:")
    print("1- Route 90 (Open highway)")
    print("2- Haifa (Mid-sized city)")
    print("3- Manhattan (Dense urban area)")

    sim = int(input("Enter your choice (1-3): "))

    if sim == 1:
        gps_error_std = 5  # Low error – open highway with good satellite visibility
        simulation_type = 'Route_90'
    elif sim == 2:
        gps_error_std = 8  # Medium error – mid-sized city with some signal obstruction
        simulation_type = 'haifa'
    elif sim == 3:
        gps_error_std = 12  # High error – dense urban area with significant multipath effects
        simulation_type = 'Manhattan'
    else:
        raise ValueError(f"Invalid simulation type: {sim}")
    print(f"Selected simulation: {simulation_type}")

    # Step 2: Choose traffic density

    print("Choose traffic density:")
    print("1- Low traffic")
    print("2- Medium traffic")
    print("3- High traffic")
    density = int(input("Enter your choice (1-3): "))

    if density == 1:
        traffic_type = "low_traffic"
    elif density == 2:
        traffic_type = "medium_traffic"
    elif density == 3:
        traffic_type = "high_traffic"
    else:

        raise ValueError(f"Invalid traffic density: {density}")

    print(f"Selected traffic density: {traffic_type}")

    # Set simulation path based on selections
    simulation_path = f"Sumo/{simulation_type.lower()}_{traffic_type}/osm.sumocfg"
    net_path = f"Sumo/{simulation_type.lower()}_{traffic_type}/osm.net.xml.gz"
    print(f"Using simulation path: {simulation_path}")

    # Step 3: Set simulation parameters
    simulation_params = {
        'gps_refresh_rate': 10,  # the rate at which the GPS is updated
        'dsrc_refresh_rate': 5,
        'ins_refresh_rate': 1,
        'number_of_steps': 2000,
        'gps_error_model_std': gps_error_std,  # Set based on scenario
        'communication_error_model_std': 2,
        'systematic_bias': 0.3,
        'proximity_radius': 300,
        'rsu_flag': True
    }
    print(f"GPS error standard deviation set to: {gps_error_std}")

    # initialize the GPS error model
    gps_error_model = GPSErrorModel(simulation_params['gps_error_model_std'])
    comm_error_model = CommunicationDistanceErrorModel(simulation_params['communication_error_model_std'],
                                                       simulation_params['systematic_bias'])


    simulation_manager = SimulationManager(simulation_params, simulation_type, gps_error_model, comm_error_model)

    ##TODO change the specific_car_id method to be more dynamic
    main_vehicle = simulation_manager.run_simulation(simulation_path)

    ##TODO analyze results and print them
    ekf = VehicleEKF(main_vehicle.position_history[0])

    for step_record in main_vehicle.position_history:
        ekf.process_step(step_record)

    # ekf.plot_results()
    plotter = PlottingManager(ekf, net_path)
    plotter.plot_trajectory_comparison()
    plotter.plot_error_comparison()
    plotter.plot_cumulative_distribution()
    print("Analysis complete")

        #
        # return ekf

    # Note: data_sequence should be a list of vehicle_data dictionaries as shown in your format
    # Run the simulation with your actual data from SUMO
    # run_simulation(data_sequence)
