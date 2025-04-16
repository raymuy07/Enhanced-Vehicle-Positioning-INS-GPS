import geopy.distance
import numpy as np
from scipy.optimize import minimize



def calculate_cartesian_distance(pos1, pos2):
    """Calculate the Euclidean distance between two Cartesian positions."""
    x1, y1 = pos1
    x2, y2 = pos2

    distance = ((x2 - x1)**2 + (y2 - y1)**2) ** 0.5
    return distance

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


def estimate_next_position(current_position, speed, heading, acceleration,step,step_length):
    """
    Estimate the next position based on current position, speed, heading, time step, and acceleration.
    """
    # Calculate the distance traveled using acceleration
    distance = (speed *1.3) + (0.7 * acceleration * (1**2))-0.3*step

    # Convert heading to radians
    heading_rad = np.radians(heading)

    # Calculate the change in position in meters
    delta_x = distance * np.cos(heading_rad)
    delta_y = distance * np.sin(heading_rad)

    # Convert the change in position to changes in latitude and longitude
    delta_lat = delta_y / 111320
    delta_lon = delta_x / (111320 * np.cos(np.radians(current_position[0])))

    next_lat = current_position[0] + delta_lat
    next_lon = current_position[1] + delta_lon

    return (next_lat, next_lon)

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


def calculate_weighted_position(estimated_position, trilateration_position, weight_inertial, weight_trilateration):
    """
    Calculate the weighted position based on inertial navigation and trilateration positions.

    Parameters:
    - estimated_position: tuple, the position estimated using inertial navigation (latitude, longitude).
    - trilateration_position: tuple, the position calculated using trilateration (latitude, longitude).
    - weight_inertial: float, the weight given to the inertial navigation method.
    - weight_trilateration: float, the weight given to the trilateration method.

    Returns:
    - weighted_position: tuple, the weighted position (latitude, longitude).
    """
    total_weight = weight_inertial + weight_trilateration
    weighted_lat = (estimated_position[0] * weight_inertial + trilateration_position[0] * weight_trilateration) / total_weight
    weighted_lon = (estimated_position[1] * weight_inertial + trilateration_position[1] * weight_trilateration) / total_weight

    weighted_position = (weighted_lat, weighted_lon)
    return weighted_position
