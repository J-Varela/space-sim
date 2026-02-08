"""
Inter-Satellite Link (ISL) networking and routing.

Includes:
- Link availability detection
- Network topology construction
- Data routing between satellites
- Link bandwidth allocation
- Latency calculations
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional
from space_sim.core.frames import Vector3


@dataclass
class ISLConnection:
    """Inter-satellite link connection."""
    sat1_id: str  # First satellite ID
    sat2_id: str  # Second satellite ID
    distance_km: float  # Link distance
    bandwidth_mbps: float  # Available bandwidth
    latency_ms: float  # One-way latency
    quality: float  # Link quality (0-1), 1 = perfect
    is_active: bool = True  # Whether link is currently active


@dataclass
class NetworkNode:
    """Network node representing a satellite."""
    sat_id: str
    position: Vector3  # Current position (km)
    neighbors: Set[str] = field(default_factory=set)  # Connected satellite IDs
    routing_table: Dict[str, str] = field(default_factory=dict)  # dest -> next_hop


@dataclass
class DataPacket:
    """Data packet for routing."""
    source_id: str
    destination_id: str
    payload_kb: float  # Payload size in kilobytes
    priority: int = 1  # Higher = more important
    path: List[str] = field(default_factory=list)  # Traversed nodes


def can_establish_isl(pos1: Vector3, pos2: Vector3, 
                     max_range_km: float = 5000.0,
                     min_elevation_deg: float = 10.0) -> bool:
    """
    Determine if two satellites can establish an inter-satellite link.
    
    Args:
        pos1: Position of satellite 1 (km)
        pos2: Position of satellite 2 (km)
        max_range_km: Maximum link range (km)
        min_elevation_deg: Minimum elevation above local horizon
        
    Returns:
        True if link is feasible
    """
    # Calculate distance
    dx = pos2[0] - pos1[0]
    dy = pos2[1] - pos1[1]
    dz = pos2[2] - pos1[2]
    distance = math.sqrt(dx*dx + dy*dy + dz*dz)
    
    # Check range constraint
    if distance > max_range_km:
        return False
    
    # Check if Earth blocks line of sight (simplified)
    # Midpoint check - if midpoint is below Earth surface, blocked
    mid_x = (pos1[0] + pos2[0]) / 2.0
    mid_y = (pos1[1] + pos2[1]) / 2.0
    mid_z = (pos1[2] + pos2[2]) / 2.0
    mid_radius = math.sqrt(mid_x*mid_x + mid_y*mid_y + mid_z*mid_z)
    
    # Earth radius
    from space_sim.core.constants import R_EARTH_KM
    
    if mid_radius < R_EARTH_KM + 100.0:  # 100 km margin
        return False
    
    return True


def compute_link_latency(distance_km: float) -> float:
    """
    Calculate one-way signal propagation latency.
    
    Args:
        distance_km: Link distance (km)
        
    Returns:
        Latency in milliseconds
    """
    # Speed of light
    c_km_ms = 299792.458 / 1000.0  # km/ms
    return distance_km / c_km_ms


def build_network_topology(satellite_positions: Dict[str, Vector3],
                          max_range_km: float = 5000.0,
                          max_links_per_sat: int = 4) -> List[ISLConnection]:
    """
    Build network topology for constellation.
    
    Args:
        satellite_positions: Dictionary of sat_id -> position
        max_range_km: Maximum ISL range
        max_links_per_sat: Maximum links per satellite
        
    Returns:
        List of active ISL connections
    """
    connections = []
    link_counts: Dict[str, int] = {sat_id: 0 for sat_id in satellite_positions.keys()}
    
    sat_ids = list(satellite_positions.keys())
    
    # Check all pairs
    for i, sat1_id in enumerate(sat_ids):
        if link_counts[sat1_id] >= max_links_per_sat:
            continue
            
        pos1 = satellite_positions[sat1_id]
        
        # Find closest satellites within range
        candidates = []
        
        for j, sat2_id in enumerate(sat_ids):
            if i >= j:  # Avoid duplicates and self
                continue
            
            if link_counts[sat2_id] >= max_links_per_sat:
                continue
            
            pos2 = satellite_positions[sat2_id]
            
            if can_establish_isl(pos1, pos2, max_range_km):
                dx = pos2[0] - pos1[0]
                dy = pos2[1] - pos1[1]
                dz = pos2[2] - pos1[2]
                distance = math.sqrt(dx*dx + dy*dy + dz*dz)
                candidates.append((sat2_id, distance))
        
        # Sort by distance, connect to closest
        candidates.sort(key=lambda x: x[1])
        
        for sat2_id, distance in candidates:
            if link_counts[sat1_id] >= max_links_per_sat:
                break
            if link_counts[sat2_id] >= max_links_per_sat:
                continue
            
            # Create connection
            latency = compute_link_latency(distance)
            
            # Bandwidth degrades with distance
            bandwidth_mbps = 1000.0 * (max_range_km / max(distance, 100.0))
            bandwidth_mbps = min(bandwidth_mbps, 1000.0)  # Cap at 1 Gbps
            
            # Link quality (simplified)
            quality = 1.0 - (distance / max_range_km) * 0.5
            
            connection = ISLConnection(
                sat1_id=sat1_id,
                sat2_id=sat2_id,
                distance_km=distance,
                bandwidth_mbps=bandwidth_mbps,
                latency_ms=latency,
                quality=quality
            )
            
            connections.append(connection)
            link_counts[sat1_id] += 1
            link_counts[sat2_id] += 1
    
    return connections


def dijkstra_shortest_path(nodes: Dict[str, NetworkNode], 
                          source_id: str, 
                          dest_id: str) -> Optional[List[str]]:
    """
    Find shortest path using Dijkstra's algorithm.
    
    Args:
        nodes: Dictionary of network nodes
        source_id: Source satellite ID
        dest_id: Destination satellite ID
        
    Returns:
        List of satellite IDs forming the path, or None if no path
    """
    if source_id not in nodes or dest_id not in nodes:
        return None
    
    if source_id == dest_id:
        return [source_id]
    
    # Initialize distances
    distances: Dict[str, float] = {sat_id: float('inf') for sat_id in nodes.keys()}
    distances[source_id] = 0.0
    
    previous: Dict[str, Optional[str]] = {sat_id: None for sat_id in nodes.keys()}
    unvisited = set(nodes.keys())
    
    while unvisited:
        # Find node with minimum distance
        current = min(unvisited, key=lambda x: distances[x])
        
        if distances[current] == float('inf'):
            break  # No path exists
        
        if current == dest_id:
            # Reconstruct path
            path = []
            node = dest_id
            while node is not None:
                path.insert(0, node)
                node = previous[node]
            return path
        
        unvisited.remove(current)
        
        # Check neighbors
        for neighbor_id in nodes[current].neighbors:
            if neighbor_id in unvisited:
                # Distance = number of hops (could be weighted by latency)
                alt_distance = distances[current] + 1.0
                
                if alt_distance < distances[neighbor_id]:
                    distances[neighbor_id] = alt_distance
                    previous[neighbor_id] = current
    
    return None  # No path found


def build_routing_tables(connections: List[ISLConnection]) -> Dict[str, NetworkNode]:
    """
    Build routing tables for all satellites in network.
    
    Args:
        connections: List of ISL connections
        
    Returns:
        Dictionary of sat_id -> NetworkNode with routing tables
    """
    # Create nodes
    sat_ids = set()
    for conn in connections:
        sat_ids.add(conn.sat1_id)
        sat_ids.add(conn.sat2_id)
    
    nodes = {sat_id: NetworkNode(sat_id=sat_id, position=(0, 0, 0)) 
             for sat_id in sat_ids}
    
    # Build neighbor lists
    for conn in connections:
        if conn.is_active:
            nodes[conn.sat1_id].neighbors.add(conn.sat2_id)
            nodes[conn.sat2_id].neighbors.add(conn.sat1_id)
    
    # Compute routing tables
    for source_id in sat_ids:
        for dest_id in sat_ids:
            if source_id != dest_id:
                path = dijkstra_shortest_path(nodes, source_id, dest_id)
                if path and len(path) > 1:
                    # Next hop is second node in path
                    next_hop = path[1]
                    nodes[source_id].routing_table[dest_id] = next_hop
    
    return nodes


def route_packet(nodes: Dict[str, NetworkNode], packet: DataPacket) -> Optional[List[str]]:
    """
    Route a packet through the network.
    
    Args:
        nodes: Network nodes with routing tables
        packet: Data packet to route
        
    Returns:
        Complete path from source to destination, or None if unreachable
    """
    if packet.source_id not in nodes or packet.destination_id not in nodes:
        return None
    
    path = [packet.source_id]
    current = packet.source_id
    max_hops = 20  # Prevent infinite loops
    
    for _ in range(max_hops):
        if current == packet.destination_id:
            return path
        
        # Look up next hop
        if packet.destination_id not in nodes[current].routing_table:
            return None  # No route
        
        next_hop = nodes[current].routing_table[packet.destination_id]
        path.append(next_hop)
        current = next_hop
    
    return None  # Too many hops


def calculate_end_to_end_latency(path: List[str], 
                                 connections: List[ISLConnection]) -> float:
    """
    Calculate total latency along a path.
    
    Args:
        path: List of satellite IDs forming the path
        connections: Available ISL connections
        
    Returns:
        Total latency in milliseconds
    """
    total_latency = 0.0
    
    # Build connection lookup
    conn_map = {}
    for conn in connections:
        key1 = (conn.sat1_id, conn.sat2_id)
        key2 = (conn.sat2_id, conn.sat1_id)
        conn_map[key1] = conn
        conn_map[key2] = conn
    
    # Sum latencies along path
    for i in range(len(path) - 1):
        from_sat = path[i]
        to_sat = path[i + 1]
        key = (from_sat, to_sat)
        
        if key in conn_map:
            total_latency += conn_map[key].latency_ms
        else:
            # No direct connection (shouldn't happen in valid path)
            total_latency += 1000.0  # Penalty
    
    # Add processing delay at each hop
    processing_delay_per_hop = 1.0  # ms
    total_latency += processing_delay_per_hop * (len(path) - 1)
    
    return total_latency


@dataclass
class NetworkStatistics:
    """Statistics about the network."""
    num_satellites: int
    num_active_links: int
    avg_links_per_satellite: float
    max_hop_count: int
    avg_hop_count: float
    network_diameter: int  # Maximum shortest path length
    connectivity: float  # Fraction of satellite pairs that can communicate


def analyze_network(nodes: Dict[str, NetworkNode]) -> NetworkStatistics:
    """
    Analyze network connectivity and performance.
    
    Args:
        nodes: Network nodes
        
    Returns:
        Network statistics
    """
    num_sats = len(nodes)
    
    # Count links
    total_links = 0
    for node in nodes.values():
        total_links += len(node.neighbors)
    num_links = total_links // 2  # Each link counted twice
    
    avg_links = total_links / num_sats if num_sats > 0 else 0
    
    # Check connectivity between all pairs
    reachable_pairs = 0
    total_pairs = 0
    hop_counts = []
    max_hops = 0
    
    sat_ids = list(nodes.keys())
    for i, src in enumerate(sat_ids):
        for dest in sat_ids[i+1:]:
            total_pairs += 1
            path = dijkstra_shortest_path(nodes, src, dest)
            if path:
                reachable_pairs += 1
                hops = len(path) - 1
                hop_counts.append(hops)
                max_hops = max(max_hops, hops)
    
    connectivity = reachable_pairs / total_pairs if total_pairs > 0 else 0.0
    avg_hops = sum(hop_counts) / len(hop_counts) if hop_counts else 0.0
    
    return NetworkStatistics(
        num_satellites=num_sats,
        num_active_links=num_links,
        avg_links_per_satellite=avg_links,
        max_hop_count=max_hops,
        avg_hop_count=avg_hops,
        network_diameter=max_hops,
        connectivity=connectivity
    )


def example_isl_network():
    """Example ISL network construction."""
    # Create simple constellation
    import math
    
    positions = {}
    num_sats = 12
    altitude_km = 7000.0
    
    # Simple ring constellation
    for i in range(num_sats):
        angle = 2.0 * math.pi * i / num_sats
        x = altitude_km * math.cos(angle)
        y = altitude_km * math.sin(angle)
        z = 0.0
        positions[f"SAT-{i:03d}"] = (x, y, z)
    
    print(f"Constellation: {num_sats} satellites at {altitude_km} km altitude")
    
    # Build network
    connections = build_network_topology(positions, max_range_km=3000.0, max_links_per_sat=4)
    print(f"ISL Connections: {len(connections)}")
    
    # Build routing
    nodes = build_routing_tables(connections)
    
    # Analyze
    stats = analyze_network(nodes)
    print(f"\nNetwork Statistics:")
    print(f"  Active Links: {stats.num_active_links}")
    print(f"  Avg Links/Sat: {stats.avg_links_per_satellite:.2f}")
    print(f"  Network Diameter: {stats.network_diameter} hops")
    print(f"  Avg Hops: {stats.avg_hop_count:.2f}")
    print(f"  Connectivity: {stats.connectivity*100:.1f}%")
    
    # Route a packet
    packet = DataPacket(source_id="SAT-000", destination_id="SAT-006", payload_kb=100.0)
    path = route_packet(nodes, packet)
    
    if path:
        print(f"\nRoute from {packet.source_id} to {packet.destination_id}:")
        print(f"  Path: {' -> '.join(path)}")
        latency = calculate_end_to_end_latency(path, connections)
        print(f"  Latency: {latency:.2f} ms")


if __name__ == "__main__":
    example_isl_network()
