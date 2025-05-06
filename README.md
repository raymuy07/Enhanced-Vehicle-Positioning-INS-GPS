# GPS-INS Integration for Enhanced Vehicle Localization

## Overview
This project enhances vehicle positioning accuracy by combining GPS data with Inertial Navigation System (INS) algorithms. By leveraging real-time ECU data including speed and acceleration, we significantly reduce localization errors in GPS-challenged environments such as urban canyons, tunnels, and areas with dense foliage.

## Features
- **GPS-INS Fusion Algorithm**: Intelligent integration of satellite GPS data with inertial measurements  
- **Dynamic Error Correction**: Adaptive calibration based on environmental conditions and signal quality  
- **Real-time ECU Data Processing**: Utilizes vehicle speed, acceleration, and orientation metrics  
- **Advanced Filtering Techniques**: Implements Kalman filtering for optimal state estimation  
- **Degradation Resilience**: Maintains positioning accuracy during GPS signal loss or degradation  
- **Comprehensive Testing Framework**: Simulation environments for various challenging scenarios  

## Technical Architecture
The system is built with a modular design featuring:

### Core Classes
- `Position`: Represents geographical positions with precision metadata  
- `SimpleVehicle`: Lightweight vehicle representation for basic simulations  
- `Vehicle`: Comprehensive vehicle model with full sensor suite and state tracking  

### Key Components
- Sensor fusion module  
- Error estimation and correction algorithms  
- Simulation environment  
- Performance analytics system  

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/gps-ins-project.git

# Navigate to project directory
cd gps-ins-project

# Install dependencies
pip install -r requirements.txt
