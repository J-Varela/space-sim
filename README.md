# Space Domain Modeling & Simulation Platform

A Python-based orbital mechanics simulation platform for modeling satellite constellations, ground stations, and space mission scenarios. Features Keplerian orbit propagation, visibility analysis, and real-time 3D visualization.

## Features

- **Orbital Mechanics**: Keplerian orbit propagation with classical orbital elements (COE)
- **Satellite Management**: Multi-satellite constellation support with individual tracking
- **Ground Station Network**: Ground station visibility and access analysis
- **Visibility Analysis**: Satellite-to-ground station line-of-sight calculations
- **3D Visualization**: Interactive Three.js-based web viewer with Earth rendering
- **Scenario System**: Flexible scenario definition and simulation engine
- **State Recording**: Time-series data capture for analysis and replay

## Project Structure

```
space-sim/
├── src/space_sim/          # Main package
│   ├── core/               # Core utilities (frames, time, constants, propagator)
│   ├── physics/            # Physics models (orbit, gravity, visibility)
│   ├── objects/            # Domain objects (satellite, ground station)
│   ├── simulation/         # Simulation engine and systems
│   ├── analysis/           # Analysis tools
│   ├── visualization/      # Data visualization utilities
│   ├── ui/                 # Dashboard and UI components
│   └── scripts/            # Demo and run scripts
├── tests/                  # Unit tests
├── web/                    # Web-based 3D viewer
│   ├── index.html          # Main viewer page
│   ├── viewer.js           # Three.js visualization
│   └── assets/             # Web assets
└── spaceVenv/              # Virtual environment
```

## Installation

### Prerequisites

- Python 3.9 or higher
- pip

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd space-sim
```

2. Create and activate virtual environment:
```bash
# Windows
python -m venv spaceVenv
spaceVenv\Scripts\activate

# Linux/macOS
python -m venv spaceVenv
source spaceVenv/bin/activate
```

3. Install the package in editable mode:
```bash
pip install -e .
```

4. Install development dependencies:
```bash
pip install -e ".[dev]"
```

## Quick Start

### Basic Orbit Propagation

```python
import math
from space_sim.objects.satellite import Satellite
from space_sim.physics.orbit import OrbitalElements

def deg(x): return x * math.pi / 180.0

# Create a satellite with orbital elements
sat = Satellite(
    sat_id="SAT-001",
    name="DemoSat",
    elements=OrbitalElements(
        a_km=7000.0,           # Semi-major axis
        e=0.001,               # Eccentricity
        inc_rad=deg(51.6),     # Inclination
        raan_rad=deg(30.0),    # RAAN
        argp_rad=deg(40.0),    # Argument of perigee
        M0_rad=0.0,            # Mean anomaly at epoch
    )
)

# Get ECI position and velocity at different times
for t in [0, 600, 1200, 1800]:
    r, v = sat.state_eci_at(float(t))
    print(f"t={t}s: position={r}")
```

### Running Demo Scripts

```bash
# Basic propagation demo
python -m space_sim.scripts.propagate_demo

# Satellite access demo
python -m space_sim.scripts.access_demo

# Multi-satellite scenario
python -m space_sim.scripts.phase4_multisat_run

# Coverage analysis
python -m space_sim.scripts.phase5_coverage_run
```

### 3D Visualization

1. Run a simulation script that generates output data
2. Open `web/index.html` in a web browser
3. Use the viewer controls:
   - **Play/Pause**: Control simulation playback
   - **Time Slider**: Navigate through simulation time
   - **Satellite Selection**: Focus on specific satellites
   - **Visibility Mode**: Toggle ground station visibility

## Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_orbit.py

# Run with verbose output
pytest -v
```

## Core Concepts

### Orbital Elements
The simulation uses classical orbital elements (COE) for orbit definition:
- **a**: Semi-major axis (km)
- **e**: Eccentricity
- **i**: Inclination (radians)
- **Ω (RAAN)**: Right Ascension of Ascending Node (radians)
- **ω (argp)**: Argument of Perigee (radians)
- **M₀**: Mean Anomaly at Epoch (radians)

### Coordinate Frames
- **ECI**: Earth-Centered Inertial frame for orbit propagation
- **ECEF**: Earth-Centered Earth-Fixed frame for ground stations
- Frame transformations for visibility calculations

### Simulation Engine
The engine uses a system-based architecture:
- **Scenario**: Container for satellites and ground stations
- **Engine**: Time-stepping simulation controller
- **Systems**: Modular components (visibility, state recording)

## Dependencies

### Core
- `plotly>=5.0.0` - Data visualization and plotting

### Development
- `pytest>=7.0.0` - Testing framework
- `ipykernel>=6.0.0` - Jupyter notebook support

## Development

### Code Style
- Follow PEP 8 guidelines
- Use type hints where applicable
- Keep functions pure and modular

### Testing
- Write tests for new features
- Maintain test coverage
- Use descriptive test names

### Contributing
1. Create a feature branch
2. Make your changes
3. Run tests to ensure nothing breaks
4. Submit a pull request

## License

[Specify your license here]

## Acknowledgments

Built with Python, Plotly, and Three.js for space mission simulation and analysis.