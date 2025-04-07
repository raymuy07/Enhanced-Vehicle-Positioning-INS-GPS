import traci

from manager_classes import GPSErrorModel
from rsu_manager import RSUManager
from vehicle_tracker import VehicleTracker


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
        simulation_type = 'test_Line'
        specific_car_id = "f_0.1"
    if sim==2:
        simulation_path="Sumo/Loud_City_NY/osm.sumocfg"
        simulation_type = 'Loud_City_NY'
    if sim==3:
        simulation_path="Sumo/High_way/osm.sumocfg"
        simulation_type = 'High_way'

    # TODO add exception if sim is not 1,2,3

    print(simulation_type)

    rsu_manager = RSUManager(simulation_type, use_RSU, proximity_radius)

    error_model = GPSErrorModel(error_std_dev)

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

