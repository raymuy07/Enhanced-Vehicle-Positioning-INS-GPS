import traci
import numpy as np

from core_classes import RSU, Position, Vehicle
from manager_classes import GPSErrorModel


# Import our new classes

def run_simulation():

    # Simulation parameters
    specific_car_id = "veh1"
    error_std_dev = 8  # 5 - standard std, 10 - very bad sat con
    num_of_neighbors = 8
    proximity_radius = 300
    use_RSU = True  # True if you want to use the RSU in the simulation
    better_flag = True  # True if you want to use the better method
    want_print = True  # True if you want to print
    number_of_steps = 600

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

    # Create error model
    error_model = GPSErrorModel(error_std_dev)

    # Create a simple dictionary to store vehicles
    vehicles = {}

    # Create a list to store RSUs
    rsus = []

    # TODO add the RSU Points.
    # # Initialize RSUs if enabled
    # if use_RSU:
    #     # In our new implementation, we would load RSU positions from a config file or similar
    #     # For now, we'll create a simple placeholder
    #     rsu_positions = [(0.0, 0.0), (1.0, 1.0)]  # Replace with actual positions
    #     for i, pos in enumerate(rsu_positions):
    #         rsu = RSU(f"rsu_{i}", Position(pos[0], pos[1]))
    #         rsus.append(rsu)

    # Start SUMO simulation
    traci.start(["sumo", "-c", simulation_path])

    # Results tracking
    results = {
        'better_values': [],
        'not_better_values': [],
        'errors': []
    }

    for step in range(number_of_steps):

        traci.simulationStep()
        vehicle_ids = traci.vehicle.getIDList()

        # Update data for all existing cars
        for vehicle_id in vehicle_ids:
            position = traci.vehicle.getPosition(vehicle_id)
            geo_position = traci.simulation.convertGeo(position[0], position[1])
            speed = traci.vehicle.getSpeed(vehicle_id)

            # Create or update vehicle
            if vehicle_id not in vehicles:
                vehicles[vehicle_id] = Vehicle(vehicle_id)

            # Create Position object
            real_position = Position(geo_position[0], geo_position[1])

            # Update vehicle with new data
            vehicles[vehicle_id].update(real_position, speed, step, error_model)



    # End simulation
    traci.close()

    # Simple summary of results
    if results['errors']:
        avg_error = sum(results['errors']) / len(results['errors'])
        print(f"\nSummary:")
        print(f"Average GPS error: {avg_error:.2f} meters")
        print(f"Min error: {min(results['errors']):.2f} meters")
        print(f"Max error: {max(results['errors']):.2f} meters")

    return results


if __name__ == "__main__":
    run_simulation()