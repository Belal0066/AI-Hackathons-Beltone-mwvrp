#!/usr/bin/env python3
"""
Baseline MWVRP solver for Robin Logistics Hackathon.
Rules followed:
1) Exposes main entrypoint: solver(env)
2) No caching across runs
3) Does NOT import or initialize the environment inside solver()
4) Local test main is provided but commented out for submission
Strategy (baseline, simple and robust):
- Greedy order-to-vehicle assignment.
- For each order, allocate one or more vehicles until fully delivered or vehicles run out.
- For each allocated vehicle:
  - Prefer pickups from the vehicle's home warehouse; if insufficient, pull remainder from other warehouses with stock.
  - Respect vehicle capacity (weight and volume) at all times.
  - Build a valid, directed path using A* on the environment's adjacency list and distance function.
  - Sequence: Home WH -> (other WHs if needed) -> Customer -> Home WH.
- Returns a `{"routes": [...]}` plan that follows the step schema used in the examples.
This is a baseline to get valid solutions quickly. Improve heuristics for ranking:
- Better vehicle selection and batching
- Distance-aware multi-warehouse selection
- Order clustering and consolidation
- Capacity packing
"""

from __future__ import annotations

import heapq
import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any, Set


# ===============================================================
# A* Search Implementation
# ===============================================================

def a_star_search(
    adjacency_list: Dict[Any, List[Any]],
    start: Any,
    goal: Any,
    get_distance_fn=None,
    coords: Optional[Dict[Any, Tuple[float, float]]] = None,
    time_limit_steps: int = 200_000,
) -> Optional[List[Any]]:
    """
    Compute a path using A* between start and goal on a directed graph.
    Args:
        adjacency_list: Graph dictionary {node: [neighbors]}
        start: start node ID (hashable)
        goal: goal node ID (hashable)
        get_distance_fn: callable(node1, node2) -> distance or None
        coords: optional dict of node -> (lat, lon)
        time_limit_steps: guard against pathological searches
    Returns:
        List of node IDs forming a path (inclusive) or None if not found
    """
    if start == goal:
        return [start]

    # Haversine heuristic if coordinates exist; otherwise 0
    def heuristic(n1: Any, n2: Any) -> float:
        if not coords or n1 not in coords or n2 not in coords:
            return 0.0
        (lat1, lon1), (lat2, lon2) = coords[n1], coords[n2]
        R = 6371e3
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = phi2 - phi1
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    open_heap: List[Tuple[float, Any]] = [(0.0, start)]
    came_from: Dict[Any, Any] = {}
    g_score: Dict[Any, float] = defaultdict(lambda: float("inf"))
    g_score[start] = 0.0

    steps = 0
    while open_heap:
        steps += 1
        if steps > time_limit_steps:
            return None

        _, current = heapq.heappop(open_heap)
        if current == goal:
            path: List[Any] = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        for neighbor in adjacency_list.get(current, []):
            edge_cost = None
            if get_distance_fn is not None:
                try:
                    edge_cost = get_distance_fn(current, neighbor)
                except Exception:
                    edge_cost = None
            if edge_cost is None:
                # fallback to unit weight if not directly connected or unknown edge
                # This keeps search progressing in sparse data anomalies
                edge_cost = 1.0

            tentative_g = g_score[current] + edge_cost
            if tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_heap, (f_score, neighbor))

    return None


# ===============================================================
# Helper utilities
# ===============================================================

def safe_get_vehicle_home_warehouse_id(env: Any, vehicle: Any, warehouse_by_node: Dict[Any, str]) -> Optional[str]:
    """Get the vehicle home warehouse ID, robust across environment object variants."""
    # Prefer attribute if available
    if hasattr(vehicle, "home_warehouse_id"):
        try:
            return getattr(vehicle, "home_warehouse_id")
        except Exception:
            pass

    # Resolve by node lookup
    try:
        home_node = env.get_vehicle_home_warehouse(vehicle.id)
        return warehouse_by_node.get(home_node)
    except Exception:
        return None


def build_warehouse_maps(env: Any, sku_ids: Set[str]) -> Tuple[Dict[str, Any], Dict[Any, str]]:
    """
    Build maps of warehouse_id -> Warehouse object and node_id -> warehouse_id.
    Uses union of warehouses that hold any of the provided SKUs.
    """
    warehouse_ids: Set[str] = set()
    for sku_id in sku_ids:
        try:
            ids = env.get_warehouses_with_sku(sku_id, min_quantity=1) or []
            warehouse_ids.update(ids)
        except Exception:
            # If API behaves differently, continue best-effort
            continue

    warehouse_by_id: Dict[str, Any] = {}
    warehouse_by_node: Dict[Any, str] = {}
    for wid in warehouse_ids:
        try:
            wh = env.get_warehouse_by_id(wid)
            warehouse_by_id[wid] = wh
            node_id = getattr(wh.location, "id", None)
            if node_id is not None:
                warehouse_by_node[node_id] = wid
        except Exception:
            continue

    return warehouse_by_id, warehouse_by_node


def sum_remaining(remaining: Dict[str, int]) -> int:
    total = 0
    for v in remaining.values():
        total += max(0, int(v))
    return total


# ===============================================================
# Main Solver
# ===============================================================

def solver(env) -> Dict[str, Any]:
    """
    Generate a logistics plan with capacity-aware, multi-warehouse pickups and valid paths.
    Returns:
        Dict with key "routes": list of per-vehicle route dicts.
    """
    solution: Dict[str, Any] = {"routes": []}

    # Orders and vehicles
    try:
        order_ids: List[str] = env.get_all_order_ids() or []
    except Exception:
        order_ids = []

    try:
        available_vehicle_ids: List[str] = env.get_available_vehicles() or []
    except Exception:
        available_vehicle_ids = []

    if not order_ids or not available_vehicle_ids:
        return solution

    # Road network
    road_network = env.get_road_network_data() or {}
    adjacency_list: Dict[Any, List[Any]] = road_network.get("adjacency_list", {}) or {}

    # Optional coordinates for heuristic
    coords: Optional[Dict[Any, Tuple[float, float]]] = None
    try:
        nodes = road_network.get("nodes")
        if isinstance(nodes, dict):
            coords = {}
            for nid, val in nodes.items():
                if isinstance(val, dict) and "lat" in val and "lon" in val:
                    coords[nid] = (val["lat"], val["lon"])
    except Exception:
        coords = None

    def get_distance(n1: Any, n2: Any) -> Optional[float]:
        try:
            return env.get_distance(n1, n2)
        except Exception:
            return None

    # Vehicle objects map
    vehicles_dict: Dict[str, Any] = {}
    try:
        all_vehicles = env.get_all_vehicles() or []
        for v in all_vehicles:
            vehicles_dict[getattr(v, "id", None)] = v
    except Exception:
        pass

    # Gather SKU ids from all orders
    sku_ids: Set[str] = set()
    order_requirements: Dict[str, Dict[str, int]] = {}
    for oid in order_ids:
        try:
            req = env.get_order_requirements(oid) or {}
        except Exception:
            req = {}
        order_requirements[oid] = {str(k): int(v) for k, v in req.items()}
        sku_ids.update(order_requirements[oid].keys())

    # Build warehouse maps and inventories
    warehouse_by_id, warehouse_by_node = build_warehouse_maps(env, sku_ids)

    inventory_by_wh: Dict[str, Dict[str, int]] = {}
    for wid in warehouse_by_id.keys():
        try:
            inv = env.get_warehouse_inventory(wid) or {}
        except Exception:
            inv = {}
        inventory_by_wh[wid] = {str(k): int(v) for k, v in inv.items()}

    # Track planned reservations to avoid over-allocating the same stock
    reserved_by_wh: Dict[str, Dict[str, int]] = {wid: defaultdict(int) for wid in warehouse_by_id.keys()}

    # SKU details (weight, volume)
    sku_details: Dict[str, Dict[str, float]] = {}
    for sid in sku_ids:
        try:
            details = env.get_sku_details(sid) or {}
        except Exception:
            details = {}
        sku_details[sid] = details

    # Greedy assignment
    remaining_vehicle_ids = list(available_vehicle_ids)

    for order_id in order_ids:
        remaining_req = dict(order_requirements.get(order_id, {}))  # sku -> qty
        if sum_remaining(remaining_req) <= 0:
            continue

        customer_node = None
        try:
            customer_node = env.get_order_location(order_id)
        except Exception:
            customer_node = None
        if customer_node is None:
            # cannot route unknown location
            continue

        # Keep assigning vehicles until done or none left
        while sum_remaining(remaining_req) > 0 and remaining_vehicle_ids:
            vehicle_id = remaining_vehicle_ids.pop(0)
            vehicle = vehicles_dict.get(vehicle_id)
            if not vehicle:
                continue

            # Determine vehicle home warehouse and node
            home_warehouse_id = safe_get_vehicle_home_warehouse_id(env, vehicle, warehouse_by_node)
            home_node = None
            if home_warehouse_id is not None:
                try:
                    home_node = getattr(warehouse_by_id[home_warehouse_id].location, "id", None)
                except Exception:
                    home_node = None
            if home_node is None:
                # fallback to node from env and try to map back to id if needed later
                try:
                    home_node = env.get_vehicle_home_warehouse(vehicle_id)
                except Exception:
                    home_node = None

            if home_node is None:
                # cannot route without a start
                continue

            # Remaining capacity
            try:
                rem_weight, rem_volume = env.get_vehicle_remaining_capacity(vehicle_id)
            except Exception:
                rem_weight, rem_volume = (float("inf"), float("inf"))

            # Identify candidate warehouses per SKU (prefer home first, then others)
            def candidate_warehouses_for(sku: str) -> List[str]:
                cands = []
                if home_warehouse_id is not None:
                    cands.append(home_warehouse_id)
                try:
                    others = env.get_warehouses_with_sku(sku, min_quantity=1) or []
                except Exception:
                    others = []
                for w in others:
                    if w != home_warehouse_id:
                        cands.append(w)
                # Filter to known warehouses only (defensive)
                return [w for w in cands if w in warehouse_by_id]

            # Plan allocations for this vehicle within capacity and inventory
            allocations: Dict[str, int] = defaultdict(int)  # sku -> qty delivered by this vehicle
            pickups_by_wh: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

            for sku, need_qty in list(remaining_req.items()):
                if need_qty <= 0:
                    continue

                details = sku_details.get(sku, {})
                unit_w = float(details.get("weight", 0.0))
                unit_v = float(details.get("volume", 0.0))

                # Capacity-based maximum we can take of this SKU now
                cap_by_weight = int(rem_weight // unit_w) if unit_w > 0 else need_qty
                cap_by_volume = int(rem_volume // unit_v) if unit_v > 0 else need_qty
                cap_limit = max(0, min(need_qty, cap_by_weight, cap_by_volume))
                if cap_limit <= 0:
                    continue

                for wid in candidate_warehouses_for(sku):
                    available_in_wh = inventory_by_wh.get(wid, {}).get(sku, 0) - reserved_by_wh.get(wid, {}).get(sku, 0)
                    if available_in_wh <= 0:
                        continue

                    take = min(cap_limit, available_in_wh)
                    if take <= 0:
                        continue

                    # Apply capacity constraints again precisely
                    max_by_w = int(rem_weight // unit_w) if unit_w > 0 else take
                    max_by_v = int(rem_volume // unit_v) if unit_v > 0 else take
                    take = min(take, max_by_w, max_by_v)
                    if take <= 0:
                        continue

                    # Record planned pickup
                    pickups_by_wh[wid][sku] += take
                    reserved_by_wh[wid][sku] += take
                    allocations[sku] += take
                    remaining_req[sku] -= take

                    # Update remaining capacity
                    rem_weight -= unit_w * take
                    rem_volume -= unit_v * take

                    # Update cap_limit and stop if satisfied
                    cap_limit -= take
                    if remaining_req[sku] <= 0 or cap_limit <= 0:
                        break

            # If no allocations for this vehicle, skip routing to save time
            if sum_remaining(allocations) <= 0:
                continue

            # Determine stop sequence: home -> (other warehouses with pickups) -> customer -> home
            pickup_wh_ids = [wid for wid, items in pickups_by_wh.items() if sum_remaining(items) > 0]
            # Ensure home is first if it has pickups; otherwise, still start path from home_node and visit others
            ordered_wh_ids: List[str] = []
            if home_warehouse_id in pickup_wh_ids:
                ordered_wh_ids.append(home_warehouse_id)
            for wid in pickup_wh_ids:
                if wid != home_warehouse_id:
                    ordered_wh_ids.append(wid)

            # Build path segments
            def wh_node(wid: str) -> Any:
                try:
                    return getattr(warehouse_by_id[wid].location, "id", None)
                except Exception:
                    return None

            steps: List[Dict[str, Any]] = []
            current_node = home_node

            # Step at start (home), include pickups if any
            start_pickups = []
            if home_warehouse_id in pickups_by_wh:
                for sku, qty in pickups_by_wh[home_warehouse_id].items():
                    if qty > 0:
                        start_pickups.append({
                            "warehouse_id": home_warehouse_id,
                            "sku_id": sku,
                            "quantity": int(qty),
                        })
            steps.append({
                "node_id": current_node,
                "pickups": start_pickups,
                "deliveries": [],
                "unloads": [],
            })

            # Visit other warehouses with pickups
            for wid in ordered_wh_ids:
                if wid == home_warehouse_id:
                    # already added the start step, skip movement
                    continue
                target_node = wh_node(wid)
                if target_node is None:
                    continue
                path = a_star_search(adjacency_list, current_node, target_node, get_distance_fn=get_distance, coords=coords)
                if not path:
                    # cannot reach this warehouse; skip its pickups
                    continue
                # Traverse intermediate nodes (skip current, include intermediates, exclude target since we'll add a step with pickups)
                for mid in path[1:-1]:
                    steps.append({"node_id": mid, "pickups": [], "deliveries": [], "unloads": []})
                # Arrive at warehouse and pickup
                wh_pickups = []
                for sku, qty in pickups_by_wh.get(wid, {}).items():
                    if qty > 0:
                        wh_pickups.append({
                            "warehouse_id": wid,
                            "sku_id": sku,
                            "quantity": int(qty),
                        })
                steps.append({
                    "node_id": target_node,
                    "pickups": wh_pickups,
                    "deliveries": [],
                    "unloads": [],
                })
                current_node = target_node

            # Travel to customer
            path_to_customer = a_star_search(adjacency_list, current_node, customer_node, get_distance_fn=get_distance, coords=coords)
            if not path_to_customer:
                # If unreachable, attempt to return home; abandon this allocation
                if current_node != home_node:
                    path_back = a_star_search(adjacency_list, current_node, home_node, get_distance_fn=get_distance, coords=coords)
                    if path_back:
                        for mid in path_back[1:-1]:
                            steps.append({"node_id": mid, "pickups": [], "deliveries": [], "unloads": []})
                        steps.append({"node_id": home_node, "pickups": [], "deliveries": [], "unloads": []})
                # Revert reservations since we cannot deliver
                for wid, items in pickups_by_wh.items():
                    for sku, qty in items.items():
                        reserved_by_wh[wid][sku] -= qty
                        remaining_req[sku] += qty
                continue

            for mid in path_to_customer[1:-1]:
                steps.append({"node_id": mid, "pickups": [], "deliveries": [], "unloads": []})

            # Deliver at customer
            deliveries = []
            for sku, qty in allocations.items():
                if qty > 0:
                    deliveries.append({
                        "order_id": order_id,
                        "sku_id": sku,
                        "quantity": int(qty),
                    })
            steps.append({
                "node_id": customer_node,
                "pickups": [],
                "deliveries": deliveries,
                "unloads": [],
            })
            current_node = customer_node

            # Return home
            path_home = a_star_search(adjacency_list, current_node, home_node, get_distance_fn=get_distance, coords=coords)
            if path_home:
                for mid in path_home[1:-1]:
                    steps.append({"node_id": mid, "pickups": [], "deliveries": [], "unloads": []})
                steps.append({"node_id": home_node, "pickups": [], "deliveries": [], "unloads": []})
            else:
                # If can't find a path home, still finalize with current node (edge case)
                steps.append({"node_id": current_node, "pickups": [], "deliveries": [], "unloads": []})

            # Finalize route for this vehicle
            solution["routes"].append({
                "vehicle_id": vehicle_id,
                "steps": steps,
            })

        # Done with this order (either fulfilled or no vehicles left)

    return solution


# ===============================================================
# Local test harness (COMMENT OUT when submitting)
# ===============================================================
# if __name__ == "__main__":
#     # Example local run (requires the environment installed locally):
#     # pip install robin-logistics-env
#     # from robin_logistics_env import LogisticsEnvironment
#     # env = LogisticsEnvironment()
#     # result = solver(env)
#     # print("Routes generated:", len(result.get("routes", [])))
#     pass