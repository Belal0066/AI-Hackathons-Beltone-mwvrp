#!/usr/bin/env python3

# pip install robin-logistics-env  --- Before first run install in the terminal
"""
Contestant solver for the Robin Logistics Environment.

Generates a valid solution using A* for shortest-path routing.

IMPORTANT SUBMISSION RULES:
1. The main function MUST be named: solver(env)
2. Do NOT use any caching techniques
3. Do NOT import or initialize the environment inside the solver function
4. Comment out the main function when submitting the solver file
"""
from robin_logistics import LogisticsEnvironment
from typing import Dict, List, Optional, Any, Tuple
from collections import deque
import heapq
import math
import time

# ---------------------------
# Helpers for A*
# ---------------------------
def haversine_distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Great-circle distance in meters. a,b = (lat, lon) in degrees."""
    R = 6371000.0
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    sa = math.sin(dlat / 2.0)
    sb = math.sin(dlon / 2.0)
    c = 2 * math.asin(min(1.0, math.sqrt(sa * sa + math.cos(lat1) * math.cos(lat2) * sb * sb)))
    return R * c

def default_edge_info(entry: Any) -> Tuple[Any, float]:
    """
    Normalize adjacency list entry into (neighbor, cost).
    Accepts:
      - neighbor (primitive)
      - (neighbor, cost)
      - {'to': neighbor, 'length': cost} or similar small dicts
    """
    if entry is None:
        return (None, math.inf)
    if isinstance(entry, (list, tuple)) and len(entry) >= 2:
        return (entry[0], float(entry[1]))
    if isinstance(entry, dict):
        # common keys
        neigh = entry.get("to") or entry.get("neighbor") or entry.get("node") or entry.get("v") or entry.get("dst")
        cost = entry.get("length") or entry.get("weight") or entry.get("cost") or entry.get("distance") or 1.0
        return (neigh, float(cost))
    # fallback: entry is neighbor id, no cost given -> assume cost 1.0
    return (entry, 1.0)

def a_star_shortest_path(start: Any, goal: Any,
                         adjacency_list: Dict[Any, List[Any]],
                         node_coords: Optional[Dict[Any, Tuple[float, float]]] = None,
                         heuristic_type: str = "haversine",
                         time_limit_s: float = 2.0) -> Optional[List[Any]]:
    """
    A* search returning the path (list of nodes) from start -> goal, or None if not found.
    - adjacency_list: {node: [neighbor or (neighbor,cost) or {..}, ...], ...}
    - node_coords: {node: (lat, lon)} used for heuristic (optional)
    - heuristic_type: "haversine" or "zero"
    - time_limit_s: per-call time limit
    """
    if start == goal:
        return [start]

    t0 = time.time()
    # heuristic function
    if node_coords and heuristic_type == "haversine":
        def h(u):
            if u in node_coords and goal in node_coords:
                return haversine_distance(node_coords[u], node_coords[goal])
            return 0.0
    else:
        def h(u):
            return 0.0

    open_heap = []  # (f_score, g_score, node)
    g_score = {start: 0.0}
    parent = {}

    heapq.heappush(open_heap, (h(start), 0.0, start))
    closed = set()

    while open_heap:
        if time.time() - t0 > time_limit_s:
            return None  # timed out, caller should handle fallback
        f_curr, g_curr, node = heapq.heappop(open_heap)
        # stale entry check
        if g_curr > g_score.get(node, float("inf")):
            continue
        if node == goal:
            # reconstruct path
            path = [node]
            while node in parent:
                node = parent[node]
                path.append(node)
            path.reverse()
            return path
        closed.add(node)
        neighbors = adjacency_list.get(node, [])
        for entry in neighbors:
            nb, cost = default_edge_info(entry)
            if nb is None:
                continue
            tentative_g = g_curr + cost
            if tentative_g < g_score.get(nb, float("inf")):
                parent[nb] = node
                g_score[nb] = tentative_g
                f_nb = tentative_g + h(nb)
                heapq.heappush(open_heap, (f_nb, tentative_g, nb))
    return None  # not reachable

# ---------------------------
# Main solver (uses A*)
# ---------------------------
def solver(env) -> Dict:
    """Generate a simple, valid solution using the road network and A* for routing.

    Args:
        env: LogisticsEnvironment instance

    Returns:
        A complete solution dict with routes and sequential steps.
    """
    solution = {"routes": []}

    # Get scenario data (adapt to env API; these calls are expected to exist)
    order_ids: List[str] = env.get_all_order_ids()
    available_vehicle_ids: List[str] = env.get_available_vehicles()
    road_network = {}
    try:
        road_network = env.get_road_network_data() or {}
    except Exception:
        road_network = {}

    adjacency_list = road_network.get("adjacency_list", {}) or {}
    # node_coords may exist as node -> (lat, lon)
    node_coords = road_network.get("node_coords", None) or road_network.get("coords", None)

    # Get all vehicles
    try:
        all_vehicles = env.get_all_vehicles()
        vehicles_dict = {v.id: v for v in all_vehicles}
    except Exception:
        # fallback: if env exposes vehicles as list/dict
        vehicles_dict = {}
        if hasattr(env, "vehicles") and env.vehicles:
            for v in env.vehicles:
                # try both dict and object forms
                vid = v.get("id") if isinstance(v, dict) else getattr(v, "id", None)
                if vid:
                    vehicles_dict[vid] = v

    # Basic implementation: assign orders to vehicles (one-to-one)
    for i, order_id in enumerate(order_ids):
        if i >= len(available_vehicle_ids):
            break

        vehicle_id = available_vehicle_ids[i]

        # Get order and vehicle using robust accessors
        order = None
        try:
            # env.orders may be dict-like or have accessor
            if hasattr(env, "orders") and isinstance(env.orders, dict):
                order = env.orders[order_id]
            elif hasattr(env, "get_order"):
                order = env.get_order(order_id)
            elif hasattr(env, "get_order_by_id"):
                order = env.get_order_by_id(order_id)
            else:
                # try scanning list
                if hasattr(env, "orders"):
                    for o in env.orders:
                        if o.get("id") == order_id:
                            order = o
                            break
        except Exception:
            order = None

        vehicle = vehicles_dict.get(vehicle_id)
        if not vehicle or not order:
            continue

        # Access home warehouse id / nodes & customer node robustly
        try:
            warehouse_id = getattr(vehicle, "home_warehouse_id", None) or vehicle.get("home_warehouse_id", None) or vehicle.get("home", None)
        except Exception:
            warehouse_id = None

        warehouse = None
        warehouse_node = None
        try:
            if warehouse_id is not None and hasattr(env, "get_warehouse_by_id"):
                warehouse = env.get_warehouse_by_id(warehouse_id)
            elif hasattr(env, "warehouses"):
                # warehouses may be dict or list
                if isinstance(env.warehouses, dict):
                    warehouse = env.warehouses.get(warehouse_id)
                else:
                    for w in env.warehouses:
                        if w.get("id") == warehouse_id:
                            warehouse = w
                            break
            # robustly extract node id
            if warehouse is not None:
                warehouse_node = getattr(warehouse, "location", None)
                # If location is an object with id, try .id
                if warehouse_node is not None and not isinstance(warehouse_node, (str, int)):
                    warehouse_node = getattr(warehouse_node, "id", None) or warehouse_node
                # else maybe warehouse has 'node' field
                if warehouse_node is None:
                    warehouse_node = warehouse.get("node") if isinstance(warehouse, dict) else None
        except Exception:
            warehouse_node = None

        # customer node extraction
        customer_node = None
        try:
            # order could be object or dict
            if hasattr(order, "destination"):
                dest = getattr(order, "destination")
                if hasattr(dest, "id"):
                    customer_node = dest.id
                else:
                    # dest may be raw node id
                    customer_node = dest
            else:
                customer_node = order.get("destination") or order.get("customer_node") or order.get("node")
                if isinstance(customer_node, dict):
                    # try to extract id
                    customer_node = customer_node.get("id") or customer_node.get("node")
        except Exception:
            customer_node = None

        # order items robust
        try:
            if hasattr(order, "requested_items"):
                order_items = dict(getattr(order, "requested_items"))
            else:
                order_items = dict(order.get("requested_items") or order.get("items") or order.get("requested") or {})
        except Exception:
            order_items = {}

        # If required nodes missing, skip
        if warehouse_node is None or customer_node is None:
            continue

        # Use A* to get path to customer and back
        # time_limit per call small to avoid long blocking
        path_to_customer = a_star_shortest_path(warehouse_node, customer_node, adjacency_list, node_coords, time_limit_s=2.0)
        path_to_warehouse = a_star_shortest_path(customer_node, warehouse_node, adjacency_list, node_coords, time_limit_s=2.0)

        # if A* timed out or no path, try fallback to env.shortest_path if available
        if (path_to_customer is None or path_to_warehouse is None) and hasattr(env, "shortest_path"):
            try:
                # env.shortest_path may return (dist, path)
                p1 = env.shortest_path(warehouse_node, customer_node)
                if isinstance(p1, tuple) and len(p1) >= 2:
                    path_to_customer = p1[1]
                elif isinstance(p1, list):
                    path_to_customer = p1
                p2 = env.shortest_path(customer_node, warehouse_node)
                if isinstance(p2, tuple) and len(p2) >= 2:
                    path_to_warehouse = p2[1]
                elif isinstance(p2, list):
                    path_to_warehouse = p2
            except Exception:
                pass

        # if still missing, skip this order safely
        if not path_to_customer or not path_to_warehouse:
            continue

        # Build steps: start at warehouse, intermediate nodes to customer, deliver, return nodes, end at warehouse
        steps = []

        # Step 1: Pickup at warehouse
        pickups = []
        for sku_id, quantity in order_items.items():
            pickups.append({
                "warehouse_id": warehouse_id,
                "sku_id": sku_id,
                "quantity": quantity
            })
        steps.append({
            "node_id": warehouse_node,
            "pickups": pickups,
            "deliveries": [],
            "unloads": []
        })

        # Step 2: intermediate nodes to customer
        # skip first and last (warehouse_node and customer_node) to avoid duplicates
        for node in path_to_customer[1:-1]:
            steps.append({
                "node_id": node,
                "pickups": [],
                "deliveries": [],
                "unloads": []
            })

        # Step 3: Deliver at customer
        deliveries = []
        for sku_id, quantity in order_items.items():
            deliveries.append({
                "order_id": order_id,
                "sku_id": sku_id,
                "quantity": quantity
            })
        steps.append({
            "node_id": customer_node,
            "pickups": [],
            "deliveries": deliveries,
            "unloads": []
        })

        # Step 4: intermediate nodes back to warehouse
        for node in path_to_warehouse[1:-1]:
            steps.append({
                "node_id": node,
                "pickups": [],
                "deliveries": [],
                "unloads": []
            })

        # Step 5: Arrive back at warehouse
        steps.append({
            "node_id": warehouse_node,
            "pickups": [],
            "deliveries": [],
            "unloads": []
        })

        solution["routes"].append({
            "vehicle_id": vehicle_id,
            "steps": steps
        })

    return solution

# ---------------------------
# Do not initialize or create env inside solver() for submission
# ---------------------------
# if __name__ == '__main__':
#     env = LogisticsEnvironment()
#     sol = solver(env)
#     print(f"Generated {len(sol.get('routes', []))} routes")
