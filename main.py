import traci
import geopy.distance
import numpy as np
from scipy.optimize import minimize
from geopy.distance import geodesic

import matplotlib.pyplot as plt

import argparse
import traci

import argparse
import traci

from rsu_manager import RSUManager

def initialize_simulation(simulation_path, use_RSU, proximity_radius):
    """Initializes SUMO and supporting components."""
    try:
        traci.start(["sumo", "-c", simulation_path] , numRetries = 3)
    except Exception as e:
        print(f"Error starting SUMO: {e}")
        return None, None

    rsu_locations = [] if not use_RSU else generate_rsu_grid()
    rsu_manager = RSUManager(rsu_locations, proximity_radius)
    vehicle_tracker = VehicleTracker(rsu_manager, "veh1", error_std_dev=8, num_of_neighbors=8, proximity_radius=300,
                                     better_flag=True)
    return rsu_manager, vehicle_tracker


def run_simulation():
    """Runs the SUMO simulation."""


    ####Debug
    simulation_path = "Sumo/simp_road/osm.sumocfg"
    print("Running simulation: simp_road")
    ####

    rsu_manager, vehicle_tracker = initialize_simulation(simulation_path, use_RSU=True, proximity_radius=300)
    if rsu_manager is None:
        print("Simulation initialization failed.")
        return

    for _ in range(600):
        traci.simulationStep()
    traci.close()
    print("Simulation finished!")



if __name__ == "__main__":
    run_simulation()

