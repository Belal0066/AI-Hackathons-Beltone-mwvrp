#!/usr/bin/env python3
"""
A*-based solver for the Robin Logistics Hackathon.

Rules followed:
1. The main function is named solver(env)
2. No caching is used
3. The environment is NOT imported or initialized inside solver
4. The main (local test) section is commented out for submission
"""

import heapq
import math
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


# ===============================================================
# A* Search Implementation
# ===============================================================
def a_star_search(adjacency_list: Dict, start: str, goal: str,
                  get_distance_fn=None,
                  coords: Optional[Dict[str, Tuple[float, float]]] = None,
                  time_limit_steps: int = 50000) -> Optional[List[str]]:
    """
    Compute a path using A* between start and goal.

    Args:
        adjacency_list: Graph dictionary {node: [neighbors]}
        start: start node ID
        goal: goal node ID
        get_distance_fn: callable(node1, node2) -> distance or None
        coords: optional dict of node -> (lat, lon)
        time_limit_steps: stop search if too long

    Returns:
        List of node IDs forming path or None if not found
    """

    if start == goal:
        return [start]

    # heuristic function (if coords exist)
    def heuristic(n1, n2):
        if not coords or n1 not in coords or n2 not in coords:
            return 0.0
        (lat1, lon1), (lat2, lon2) = coords[n1], coords[n2]
        R = 6371e3
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = phi2 - phi1
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # open set
    open_heap = [(0, start)]
    came_from = {}
    g_score = defaultdict(lambda: float("inf"))
    g_score[start] = 0

    steps = 0
    while open_heap:
        steps += 1
        if steps > time_limit_steps:
            return None

        _, current = heapq.heappop(open_heap)

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        for neighbor in adjacency_list.get(current, []):
            # determine edge cost
            cost = None
            if get_distance_fn:
                cost = get_distance_fn(current, neighbor)
            if cost is None:
                cost = 1.0  # fallback if missing
            tentative_g = g_score[current] + cost
            if tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_heap, (f_score, neighbor))

    return None


# ===============================================================
# Main Solver Function
# ===============================================================
def solver(env) -> Dict:
    """
    Generate a valid logistics plan using A* pathfinding.

    Args:
        env: LogisticsEnvironment instance

    Returns:
        Dict containing routes for all vehicles
    """
    solution = {"routes": []}

    # Get all environment data
    order_ids = env.get_all_order_ids()
    available_vehicle_ids = env.get_available_vehicles()
    road_network = env.get_road_network_data()
    adjacency_list = road_network.get("adjacency_list", {})

    # optional coords (if present in road network)
    coords = None
    if "nodes" in road_network and isinstance(road_network["nodes"], dict):
        nodes = road_network["nodes"]
        coords = {}
        for nid, val in nodes.items():
            if isinstance(val, dict) and "lat" in val and "lon" in val:
                coords[nid] = (val["lat"], val["lon"])

    # helper for distance lookup
    def get_distance(n1, n2):
        try:
            return env.get_distance(n1, n2)
        except Exception:
            return None

    # map vehicle objects
    all_vehicles = env.get_all_vehicles()
    vehicles_dict = {v.id: v for v in all_vehicles}

    # simple order-vehicle pairing
    for i, order_id in enumerate(order_ids):
        if i >= len(available_vehicle_ids):
            break

        vehicle_id = available_vehicle_ids[i]
        vehicle = vehicles_dict.get(vehicle_id)
        if not vehicle:
            continue

        # warehouse & customer node IDs
        warehouse_id = vehicle.home_warehouse_id
        warehouse = env.get_warehouse_by_id(warehouse_id)
        warehouse_node = warehouse.location.id
        customer_node = env.get_order_location(order_id)

        order = env.orders[order_id]
        order_items = order.requested_items

        # Use A* pathfinding
        path_to_customer = a_star_search(adjacency_list, warehouse_node, customer_node,
                                         get_distance_fn=get_distance, coords=coords)
        path_to_warehouse = a_star_search(adjacency_list, customer_node, warehouse_node,
                                          get_distance_fn=get_distance, coords=coords)

        if not path_to_customer or not path_to_warehouse:
            continue

        steps = []

        # Step 1: Pickup
        pickups = []
        for sku_id, quantity in order_items.items():
            pickups.append({
                "warehouse_id": warehouse_id,
                "sku_id": sku_id,
                "quantity": quantity
            })
        steps.append({"node_id": warehouse_node, "pickups": pickups,
                      "deliveries": [], "unloads": []})

        # Step 2: travel to customer
        for node in path_to_customer[1:-1]:
            steps.append({"node_id": node, "pickups": [], "deliveries": [], "unloads": []})

        # Step 3: deliver
        deliveries = []
        for sku_id, quantity in order_items.items():
            deliveries.append({
                "order_id": order_id,
                "sku_id": sku_id,
                "quantity": quantity
            })
        steps.append({"node_id": customer_node, "pickups": [],
                      "deliveries": deliveries, "unloads": []})

        # Step 4: return to warehouse
        for node in path_to_warehouse[1:-1]:
            steps.append({"node_id": node, "pickups": [], "deliveries": [], "unloads": []})

        # Step 5: arrive back
        steps.append({"node_id": warehouse_node, "pickups": [], "deliveries": [], "unloads": []})

        # Append final route
        solution["routes"].append({
            "vehicle_id": vehicle_id,
            "steps": steps
        })

    return solution


# ===============================================================
# Local test only (comment out before submission)
# ===============================================================
# if __name__ == "__main__":
#     from robin_logistics import LogisticsEnvironment
#     env = LogisticsEnvironment()
#     result = solver(env)
#     print(result)
#     print("Solution generated:")
#     print(f"Number of routes: {len(result.get('routes', []))}")
