from manager_classes import SimulationManager, VehicleEKF, PlottingManager, DSRCPositionEstimator
from error_classes import GPSErrorModel, CommunicationDistanceErrorModel
import pandas as pd

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
        simulation_type = 'Haifa'
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

    # Step 3: Simulation attributes
    print("Do you want to use DSRC protocol?")
    print("1 - YES")
    print("2 - NO")
    use_dsrc = int(input("Enter your choice (1-2): "))
    if use_dsrc == 1:
        use_dsrc = True
    elif use_dsrc == 2:
        use_dsrc = False
    else:
        raise ValueError(f"Invalid DSRC choice")
    if use_dsrc:
        print("Do you want to simulate RSUs?")
        print("1 - YES")
        print("2 - NO")
        use_rsu = int(input("Enter your choice (1-2): "))
        if use_rsu == 1:
            use_rsu = True
        elif use_rsu == 2:
            use_rsu = False
        else:
            raise ValueError(f"Invalid RSU choice")
    else:
        use_rsu = False

    print("Do you want to simulate GPS outage?")
    print("1 - YES")
    print("2 - NO")
    gps_outage = int(input("Enter your choice (1-2): "))
    if gps_outage == 1:
        gps_outage = True
    elif gps_outage == 2:
        gps_outage = False
    else:
        raise ValueError(f"Invalid GPS outage choice")

    # Set simulation path based on selections
    simulation_path = f"Sumo/{simulation_type.lower()}_{traffic_type}/osm.sumocfg"
    net_path = f"Sumo/{simulation_type.lower()}_{traffic_type}/osm.net.xml.gz"
    print(f"Using simulation path: {simulation_path}")

    # Step 3: Set simulation parameters
    simulation_params = {
        'num_vehicles_to_track': 100,
        'gps_refresh_rate': 10,  # the rate at which the GPS is updated
        'dsrc_refresh_rate': 2,
        'ins_refresh_rate': 1,
        'number_of_steps': 2000,
        'gps_error_model_std': gps_error_std,  # Set based on scenario
        'communication_error_model_std': 2,
        'systematic_bias': 0.3,
        'proximity_radius': 300,
        'rsu_flag': use_rsu
    }
    print(f"GPS error standard deviation set to: {gps_error_std}")

    # initialize the GPS error model
    gps_error_model = GPSErrorModel(simulation_params['gps_error_model_std'])
    comm_error_model = CommunicationDistanceErrorModel(simulation_params['communication_error_model_std'],
                                                       simulation_params['systematic_bias'])

    simulation_manager = SimulationManager(simulation_params, simulation_type,
                                           gps_error_model, comm_error_model, gps_outage)

    vehicle_objs = simulation_manager.run_simulation(simulation_path)
    ekf_objs = []

    for veh in vehicle_objs:
        dsrc_manager = DSRCPositionEstimator()
        ekf = VehicleEKF(veh.id, dsrc_manager, veh.position_history[0], use_dsrc)
        for step_record in veh.position_history:
            ekf.process_step(step_record)
        ekf_objs.append(ekf)

    frames = []
    for ekf in ekf_objs:
        # make a Date frame of processed data for each tracked vehicle
        df = pd.DataFrame(ekf.history)
        df["veh_id"] = ekf.vehicle_id
        frames.append(df)

    all_steps = pd.concat(frames, ignore_index=True)

    error_cols = ["gps_error", "ekf_error", "dsrc_error"]
    existing_cols = [c for c in error_cols if c in all_steps]

    agg = (all_steps.groupby("step")[existing_cols].agg(['mean', 'std', 'count']))

    mean_step = agg.xs('mean', level=1, axis=1)
    std_step = agg.xs('std', level=1, axis=1)
    count_step = agg.xs('count', level=1, axis=1)

    plotter = PlottingManager(mean_step, std_step, count_step,
                              all_steps,
                              net_file=net_path,
                              dsrc_flag=use_dsrc,
                              gps_outage=simulation_manager.gps_outage)
    for i in range(3):
        plotter.plot_trajectory_comparison(vehicle_objs[i].id, ekf_objs[i])
    plotter.plot_mean_error_with_band()
    plotter.plot_error_cdf()
    plotter.plot_summary_table()
    print("Analysis complete")

    # Note: data_sequence should be a list of vehicle_data dictionaries as shown in your format
    # Run the simulation with your actual data from SUMO
    # run_simulation(data_sequence)
