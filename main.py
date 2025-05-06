import traci
import numpy as np

from core_classes import RSU, Position, Vehicle
from manager_classes import SimulationManager
from error_classes import GPSErrorModel, CommunicationDistanceErrorModel

if __name__ == "__main__":

    # Simulation parameters
    simulation_params = {

        'gps_refresh_rate': 50,  # the rate in which the GPS is updated
        'dsrc_refresh_rate': 10,
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

    sim = int(input("1- test line\n2- Loud_City_NY\n3- High_way\n"))
    if sim == 1:
        simulation_path = "Sumo/test_Line/osm.sumocfg"
        simulation_type = 'test_Line'
        specific_car_id = "f_0.1"
    elif sim == 2:
        simulation_path = "Sumo/Loud_City_NY/osm.sumocfg"
        simulation_type = 'Loud_City_NY'
    elif sim == 3:
        simulation_path = "Sumo/High_way/osm.sumocfg"
        simulation_type = 'High_way'
    else:
        raise ValueError(f"Invalid simulation type: {sim}")

    print(f"Running simulation: {simulation_type}")

    simulation_manager = SimulationManager(simulation_params, simulation_type, gps_error_model, comm_error_model)

    ##TODO change the specific_car_id method to be more dynamic
    main_vehicle = simulation_manager.run_simulation(simulation_path)

    ##TODO analyze results and print them
    ##
