"""
Comprehensive demonstration of new space simulation features.

This script showcases all the newly implemented capabilities:
1. J2 perturbation orbit propagation
2. Attitude dynamics and control
3. Power system modeling
4. Orbital maneuvers (Hohmann transfers)
5. Atmospheric drag
6. Link budget analysis
7. Collision detection
8. TLE import
9. Launch window calculation
10. Inter-satellite networking
"""

import math
from space_sim.physics.orbit import OrbitalElements
from space_sim.physics.perturbations import j2_perturbation_accel
from space_sim.core.propagator import NumericalPropagator
from space_sim.physics.attitude import (
    AttitudeState, Quaternion, compute_nadir_pointing_quaternion,
    AttitudeController, propagate_attitude
)
from space_sim.physics.power import (
    SolarPanel, Battery, PowerSystem, PowerBudget, simple_sun_position_eci
)
from space_sim.physics.maneuvers import (
    hohmann_transfer, compute_maneuver_sequence
)
from space_sim.physics.communications import compute_link_at_position
from space_sim.analysis.collision import conjunction_analysis
from space_sim.core.tle import tle_from_string, tle_to_orbital_elements, EXAMPLE_ISS_TLE
from space_sim.analysis.launch import (
    calculate_launch_windows, LAUNCH_SITES, launch_azimuth
)
from space_sim.analysis.isl_network import (
    build_network_topology, build_routing_tables, analyze_network
)


def demo_j2_perturbation():
    """Demonstrate J2 perturbation effects on orbit."""
    print("=" * 70)
    print("DEMO 1: J2 PERTURBATION ORBIT PROPAGATION")
    print("=" * 70)
    
    # Create LEO orbit
    elements = OrbitalElements(
        a_km=6900.0,  # 522 km altitude
        e=0.001,
        inc_rad=math.radians(51.6),
        raan_rad=math.radians(0.0),
        argp_rad=math.radians(0.0),
        M0_rad=0.0
    )
    
    print(f"Initial Orbit: a={elements.a_km:.1f} km, i={math.degrees(elements.inc_rad):.1f}°")
    
    # Get initial state
    from space_sim.physics.orbit import coe_to_rv_eci
    r0, v0 = coe_to_rv_eci(elements, 0.0)
    
    # Calculate J2 acceleration
    j2_accel = j2_perturbation_accel(r0)
    print(f"J2 Acceleration: {j2_accel[0]:.6e}, {j2_accel[1]:.6e}, {j2_accel[2]:.6e} km/s²")
    
    # Propagate with perturbations
    propagator = NumericalPropagator(include_j2=True, dt=60.0)
    trajectory = propagator.propagate(r0, v0, duration=86400.0)  # 1 day
    
    print(f"Propagated {len(trajectory)} time steps over 1 day")
    print(f"Final position: {trajectory[-1][1][0]:.1f}, {trajectory[-1][1][1]:.1f}, {trajectory[-1][1][2]:.1f} km")
    print()


def demo_attitude_control():
    """Demonstrate attitude dynamics and control."""
    print("=" * 70)
    print("DEMO 2: ATTITUDE DYNAMICS & CONTROL")
    print("=" * 70)
    
    # Initial attitude state
    state = AttitudeState.inertial()
    print("Initial attitude: Inertial (no rotation)")
    
    # Create controller
    controller = AttitudeController(kp=0.1, kd=0.5)
    
    # Simulate nadir pointing
    r_eci = (7000.0, 0.0, 0.0)
    v_eci = (0.0, 7.5, 0.0)
    
    desired_q = compute_nadir_pointing_quaternion(r_eci, v_eci)
    print(f"Desired quaternion (nadir): [{desired_q.q0:.3f}, {desired_q.q1:.3f}, {desired_q.q2:.3f}, {desired_q.q3:.3f}]")
    
    # Propagate attitude with control
    dt = 1.0
    for i in range(100):
        torque = controller.compute_control_torque(state, desired_q)
        state = propagate_attitude(state, dt, torque)
    
    roll, pitch, yaw = state.quaternion.to_euler_angles()
    print(f"After 100s of control:")
    print(f"  Euler angles: roll={math.degrees(roll):.2f}°, pitch={math.degrees(pitch):.2f}°, yaw={math.degrees(yaw):.2f}°")
    print(f"  Angular velocity: {state.angular_velocity[0]:.4f}, {state.angular_velocity[1]:.4f}, {state.angular_velocity[2]:.4f} rad/s")
    print()


def demo_power_system():
    """Demonstrate power system simulation."""
    print("=" * 70)
    print("DEMO 3: POWER SYSTEM MODELING")
    print("=" * 70)
    
    # Create power system
    solar_panel = SolarPanel(area_m2=4.0, efficiency=0.28, degradation=1.0)
    battery = Battery(capacity_wh=500.0, current_charge_wh=400.0)
    power_budget = PowerBudget(
        payload_w=50.0,
        communication_w=20.0,
        attitude_control_w=10.0,
        avionics_w=15.0
    )
    
    power_sys = PowerSystem(solar_panel, battery, power_budget)
    
    print(f"Solar Panel: {solar_panel.area_m2} m², {solar_panel.efficiency*100:.1f}% efficient")
    print(f"Battery: {battery.capacity_wh} Wh capacity, SOC={battery.state_of_charge()*100:.1f}%")
    print(f"Power Demand: {power_budget.total_consumption_w():.1f} W")
    
    # Simulate in sunlight
    r_sat = (7000.0, 0.0, 0.0)
    r_sun = simple_sun_position_eci(2451545.0)  # J2000
    
    status = power_sys.step(60.0, r_sat, r_sun)
    
    print(f"\nAfter 60s in sunlight:")
    print(f"  Solar Power: {status['solar_power_w']:.1f} W")
    print(f"  Battery SOC: {status['battery_soc']*100:.1f}%")
    print(f"  Power Balance: {status['solar_power_w'] - status['power_demand_w']:.1f} W")
    print(f"  Eclipse: {status['eclipse']}")
    print()


def demo_maneuver_planning():
    """Demonstrate orbital maneuver calculations."""
    print("=" * 70)
    print("DEMO 4: ORBITAL MANEUVER PLANNING")
    print("=" * 70)
    
    # Hohmann transfer from LEO to GEO
    r1_km = 6900.0  # LEO
    r2_km = 42164.0  # GEO
    
    dv1, dv2, transfer_time = hohmann_transfer(r1_km, r2_km)
    
    print(f"Hohmann Transfer: LEO ({r1_km:.0f} km) -> GEO ({r2_km:.0f} km)")
    print(f"  Burn 1: {dv1:.3f} km/s")
    print(f"  Burn 2: {dv2:.3f} km/s")
    print(f"  Total ΔV: {dv1 + dv2:.3f} km/s")
    print(f"  Transfer Time: {transfer_time/3600:.2f} hours")
    print()


def demo_link_budget():
    """Demonstrate communications link analysis."""
    print("=" * 70)
    print("DEMO 5: COMMUNICATIONS LINK BUDGET")
    print("=" * 70)
    
    r_sat = (7000.0, 0.0, 0.0)
    v_sat = (0.0, 7.5, 0.0)
    r_gs = (6378.137, 0.0, 0.0)
    
    link_params = compute_link_at_position(
        r_sat, v_sat, r_gs,
        tx_power_w=10.0,
        frequency_ghz=2.4,
        tx_gain_dbi=3.0,
        rx_gain_dbi=15.0
    )
    
    print(f"Satellite-to-Ground Link:")
    print(f"  Range: {link_params['range_km']:.1f} km")
    print(f"  Received Power: {link_params['received_power_dbw']:.2f} dBW")
    print(f"  C/N Ratio: {link_params['cn_ratio_db']:.2f} dB")
    print(f"  Link Margin: {link_params['link_margin_db']:.2f} dB")
    print(f"  Max Data Rate: {link_params['max_data_rate_mbps']:.2f} Mbps")
    print(f"  Doppler Shift: {link_params['doppler_shift_hz']:.1f} Hz")
    print()


def demo_collision_detection():
    """Demonstrate collision detection and analysis."""
    print("=" * 70)
    print("DEMO 6: COLLISION DETECTION & AVOIDANCE")
    print("=" * 70)
    
    # Two satellites on near-collision course
    r1 = (7000.0, 0.0, 0.0)
    v1 = (0.0, 7.5, 0.0)
    r2 = (7002.0, 0.5, 0.0)  # 2 km ahead, 0.5 km offset
    v2 = (0.0, 7.48, 0.0)  # Slightly slower, converging
    
    analysis = conjunction_analysis(r1, v1, r2, v2, 
                                   combined_radius_km=0.01,
                                   position_uncertainty_km=0.1)
    
    print(f"Conjunction Analysis:")
    print(f"  Current Separation: {analysis['current_separation_km']:.3f} km")
    print(f"  Miss Distance: {analysis['miss_distance_km']:.3f} km")
    print(f"  Time to TCA: {analysis['time_to_tca_s']:.1f} s")
    print(f"  Relative Velocity: {analysis['relative_velocity_km_s']:.3f} km/s")
    print(f"  Probability of Collision: {analysis['probability_of_collision']:.2e}")
    print(f"  Risk Level: {analysis['collision_risk']}")
    print()


def demo_tle_import():
    """Demonstrate TLE parsing and orbit conversion."""
    print("=" * 70)
    print("DEMO 7: TLE IMPORT & CONVERSION")
    print("=" * 70)
    
    tle = tle_from_string(EXAMPLE_ISS_TLE)
    print(f"Satellite: {tle.line0}")
    print(f"Catalog Number: {tle.catalog_number}")
    print(f"Epoch: Year {tle.epoch_year}, Day {tle.epoch_day:.2f}")
    print(f"Inclination: {tle.inclination_deg:.4f}°")
    print(f"Eccentricity: {tle.eccentricity:.6f}")
    print(f"Mean Motion: {tle.mean_motion_rev_day:.8f} rev/day")
    
    # Convert to orbital elements
    elements = tle_to_orbital_elements(tle)
    print(f"\nConverted Orbital Elements:")
    print(f"  Semi-major axis: {elements.a_km:.2f} km")
    print(f"  Altitude: {elements.a_km - 6378.137:.2f} km")
    print()


def demo_launch_windows():
    """Demonstrate launch window calculations."""
    print("=" * 70)
    print("DEMO 8: LAUNCH WINDOW CALCULATOR")
    print("=" * 70)
    
    site = LAUNCH_SITES["Kennedy Space Center"]
    target_inc = 51.6  # ISS inclination
    
    print(f"Launch Site: {site.name}")
    print(f"Location: {site.latitude_deg:.3f}°N, {abs(site.longitude_deg):.3f}°W")
    print(f"Target Inclination: {target_inc}°")
    
    # Calculate launch azimuth
    az = launch_azimuth(site.latitude_deg, target_inc, ascending=True)
    print(f"Launch Azimuth: {az:.2f}° from North")
    
    # Find launch windows
    windows = calculate_launch_windows(site, target_inc, search_duration_s=86400.0)
    print(f"\nLaunch Windows in next 24 hours: {len(windows)}")
    for i, window in enumerate(windows[:5]):
        hours = window.open_time_s / 3600.0
        print(f"  {i+1}. T+{hours:.2f}h: {window.description}")
    print()


def demo_isl_network():
    """Demonstrate inter-satellite link networking."""
    print("=" * 70)
    print("DEMO 9: INTER-SATELLITE LINK NETWORK")
    print("=" * 70)
    
    # Create small constellation
    positions = {}
    num_sats = 8
    altitude_km = 7000.0
    
    for i in range(num_sats):
        angle = 2.0 * math.pi * i / num_sats
        x = altitude_km * math.cos(angle)
        y = altitude_km * math.sin(angle)
        z = 0.0
        positions[f"SAT-{i:02d}"] = (x, y, z)
    
    print(f"Constellation: {num_sats} satellites in ring at {altitude_km} km")
    
    # Build network
    connections = build_network_topology(positions, max_range_km=3000.0, max_links_per_sat=4)
    nodes = build_routing_tables(connections)
    stats = analyze_network(nodes)
    
    print(f"Network Topology:")
    print(f"  Active ISL Connections: {stats.num_active_links}")
    print(f"  Average Links per Satellite: {stats.avg_links_per_satellite:.2f}")
    print(f"  Network Diameter: {stats.network_diameter} hops")
    print(f"  Average Hop Count: {stats.avg_hop_count:.2f}")
    print(f"  Connectivity: {stats.connectivity*100:.1f}%")
    print()


def run_all_demos():
    """Run all feature demonstrations."""
    print("\n")
    print("#" * 70)
    print("# SPACE SIMULATION PLATFORM - FEATURE SHOWCASE")
    print("#" * 70)
    print("\n")
    
    demo_j2_perturbation()
    demo_attitude_control()
    demo_power_system()
    demo_maneuver_planning()
    demo_link_budget()
    demo_collision_detection()
    demo_tle_import()
    demo_launch_windows()
    demo_isl_network()
    
    print("=" * 70)
    print("ALL DEMONSTRATIONS COMPLETE!")
    print("=" * 70)
    print("\nNew Features Added:")
    print("  ✓ J2 Perturbation Model")
    print("  ✓ Attitude Dynamics & Control")
    print("  ✓ Power System (Solar + Battery)")
    print("  ✓ Orbital Maneuvers (Hohmann, etc.)")
    print("  ✓ Atmospheric Drag")
    print("  ✓ Communications Link Budget")
    print("  ✓ Collision Detection")
    print("  ✓ TLE Import")
    print("  ✓ Launch Window Calculator")
    print("  ✓ Inter-Satellite Networking")
    print()


if __name__ == "__main__":
    run_all_demos()
