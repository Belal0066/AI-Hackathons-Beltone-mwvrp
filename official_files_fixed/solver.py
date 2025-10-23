#!/usr/bin/env python3
"""
Improved MWVRP solver for Robin Logistics Hackathon.
Enhancements:
- Order consolidation: Batches geographically close, unfulfilled orders.
- Scored vehicle selection: Chooses the best vehicle based on distance and capacity fit.
- Optimized multi-warehouse routing: Uses a Nearest Neighbor TSP heuristic for pickup sequencing.
- **FIXED:** Robust home node handling and rigorous A* path failure handling to ensure valid routes.
"""
from __future__ import annotations
import heapq
import math
import itertools
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any, Set

# ===============================================================
# A* Search Implementation (Unchanged, it is robust)
# ===============================================================
def a_star_search(
    adjacency_list: Dict[Any, List[Any]],
    start: Any,
    goal: Any,
    get_distance_fn=None,
    coords: Optional[Dict[Any, Tuple[float, float]]] = None,
    time_limit_steps: int = 200_000,
) -> Optional[List[Any]]:
    """ Compute a path using A* between start and goal on a directed graph. """
    if start == goal:
        return [start]

    # Haversine heuristic
    def heuristic(n1: Any, n2: Any) -> float:
        if not coords or n1 not in coords or n2 not in coords:
            return 0.0
        (lat1, lon1), (lat2, lon2) = coords[n1], coords[n2]
        R = 6371e3  # Earth radius in meters
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
            
            if edge_cost is None or edge_cost < 0:
                edge_cost = 1.0  # Fallback unit cost

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
    try:
        if hasattr(vehicle, "home_warehouse_id"):
            return getattr(vehicle, "home_warehouse_id")
        home_node = env.get_vehicle_home_warehouse(vehicle.id)
        return warehouse_by_node.get(home_node)
    except Exception:
        return None

def build_warehouse_maps(env: Any, sku_ids: Set[str]) -> Tuple[Dict[str, Any], Dict[Any, str]]:
    """ Build maps of warehouse_id -> Warehouse object and node_id -> warehouse_id. """
    warehouse_ids: Set[str] = set()
    for sku_id in sku_ids:
        try:
            ids = env.get_warehouses_with_sku(sku_id, min_quantity=1) or []
            warehouse_ids.update(ids)
        except Exception:
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
    """ Sum of positive remaining quantities in a dict. """
    total = 0
    for v in remaining.values():
        total += max(0, int(v))
    return total

def haversine_distance(coords: Dict[Any, Tuple[float, float]], n1: Any, n2: Any) -> float:
    """ Calculate Haversine distance between two nodes (approximate). """
    R = 6371e3
    try:
        (lat1, lon1), (lat2, lon2) = coords[n1], coords[n2]
    except KeyError:
        return R * 1000.0 # Use a large value if coordinates are missing

    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def get_node_distance(env: Any, n1: Any, n2: Any, distance_cache: Dict[Tuple[Any, Any], float]) -> float:
    """ Get distance using env or cache, falling back to a large value if no path. """
    if n1 == n2:
        return 0.0
    
    key = tuple(sorted((n1, n2))) # Symmetric key for caching
    if key in distance_cache:
        return distance_cache[key]
        
    try:
        dist = env.get_distance(n1, n2)
        if dist is None or dist < 0:
             # Fallback: estimate if env returns bad data
            dist = 1e9 
        distance_cache[key] = dist
        return dist
    except Exception:
        # Fallback: very high cost if env fails
        return 1e9

# ===============================================================
# Main Solver
# ===============================================================
def solver(env) -> Dict[str, Any]:
    """ 
    Generate a logistics plan with consolidation, capacity-aware, multi-warehouse pickups 
    and valid paths using a VRP-like greedy approach.
    """
    solution: Dict[str, Any] = {"routes": []}
    distance_cache: Dict[Tuple[Any, Any], float] = {}

    try:
        order_ids: List[str] = env.get_all_order_ids() or []
        available_vehicle_ids: List[str] = env.get_available_vehicles() or []
    except Exception:
        return solution

    if not order_ids or not available_vehicle_ids:
        return solution

    # Road network setup
    road_network = env.get_road_network_data() or {}
    adjacency_list: Dict[Any, List[Any]] = road_network.get("adjacency_list", {}) or {}
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

    def get_distance_fn(n1: Any, n2: Any) -> Optional[float]:
        return get_node_distance(env, n1, n2, distance_cache)

    # Gather required data
    vehicles_dict: Dict[str, Any] = {getattr(v, "id"): v for v in env.get_all_vehicles() or [] if hasattr(v, "id")}
    
    order_requirements: Dict[str, Dict[str, int]] = {}
    customer_nodes: Dict[str, Any] = {}
    sku_ids: Set[str] = set()

    for oid in order_ids:
        try:
            req = env.get_order_requirements(oid) or {}
            order_requirements[oid] = {str(k): int(v) for k, v in req.items()}
            sku_ids.update(order_requirements[oid].keys())
            customer_nodes[oid] = env.get_order_location(oid)
        except Exception:
            pass
    
    warehouse_by_id, warehouse_by_node = build_warehouse_maps(env, sku_ids)
    
    inventory_by_wh: Dict[str, Dict[str, int]] = {}
    for wid in warehouse_by_id.keys():
        try:
            inv = env.get_warehouse_inventory(wid) or {}
            inventory_by_wh[wid] = {str(k): int(v) for k, v in inv.items()}
        except Exception:
            pass

    reserved_by_wh: Dict[str, Dict[str, int]] = {wid: defaultdict(int) for wid in warehouse_by_id.keys()}
    sku_details: Dict[str, Dict[str, float]] = {sid: env.get_sku_details(sid) or {} for sid in sku_ids}

    # Tracking remaining requirements
    remaining_order_req: Dict[str, Dict[str, int]] = {
        oid: dict(req) for oid, req in order_requirements.items() if customer_nodes.get(oid) is not None
    }
    
    available_vehicle_ids_set = set(available_vehicle_ids)
    
    while True:
        unfulfilled_orders = {oid: req for oid, req in remaining_order_req.items() if sum_remaining(req) > 0}
        if not unfulfilled_orders or not available_vehicle_ids_set:
            break

        # --- 1. Order Consolidation/Clustering (Simple Nearest-Neighbor Greedy) ---
        
        prime_order_id = next(iter(unfulfilled_orders))
        prime_customer_node = customer_nodes[prime_order_id]
        current_batch: Dict[str, Dict[str, int]] = {prime_order_id: unfulfilled_orders[prime_order_id]}
        
        batch_total_req = defaultdict(int)
        for req in current_batch.values():
            for sku, qty in req.items():
                batch_total_req[sku] += qty
        
        BATCH_DISTANCE_THRESHOLD = 20000.0 # 20km
        
        nearby_orders_ids = sorted([
            oid for oid in unfulfilled_orders if oid != prime_order_id and 
            haversine_distance(coords, prime_customer_node, customer_nodes[oid]) <= BATCH_DISTANCE_THRESHOLD
        ], key=lambda oid: haversine_distance(coords, prime_customer_node, customer_nodes[oid]))
        
        # Only add one more nearby order for conservative consolidation
        if nearby_orders_ids:
            oid = nearby_orders_ids[0]
            current_batch[oid] = unfulfilled_orders[oid]
            for sku, qty in unfulfilled_orders[oid].items():
                batch_total_req[sku] += qty


        # --- 2. Scored Vehicle Selection ---
        
        best_score = float('inf')
        best_vehicle_id = None
        best_vehicle = None
        
        for vehicle_id in available_vehicle_ids_set:
            vehicle = vehicles_dict.get(vehicle_id)
            if not vehicle: continue
            
            # **FIX 1A: Robustly get home node**
            try:
                home_node = env.get_vehicle_home_warehouse(vehicle_id)
            except Exception:
                home_node = None
            if home_node is None: continue
            
            home_warehouse_id = warehouse_by_node.get(home_node)

            try:
                max_w, max_v = env.get_vehicle_remaining_capacity(vehicle_id)
            except Exception:
                max_w, max_v = (float("inf"), float("inf"))

            total_weight, total_volume = 0.0, 0.0
            for sku, qty in batch_total_req.items():
                details = sku_details.get(sku, {})
                total_weight += qty * float(details.get("weight", 0.0))
                total_volume += qty * float(details.get("volume", 0.0))
                
            capacity_penalty = 0.0
            if total_weight > max_w or total_volume > max_v:
                capacity_penalty = 1e9 # Massive penalty
            elif max_w > 0 and total_weight > 0:
                capacity_penalty = 1000 * (1 - (total_weight / max_w)) 
            
            dist_to_cluster = haversine_distance(coords, home_node, prime_customer_node)
            
            home_stock_match_pct = 0.0
            if home_warehouse_id:
                home_inv = inventory_by_wh.get(home_warehouse_id, {})
                required_total = sum(batch_total_req.values())
                available_home = sum(min(qty, home_inv.get(sku, 0) - reserved_by_wh[home_warehouse_id].get(sku, 0))
                                     for sku, qty in batch_total_req.items())
                if required_total > 0:
                    home_stock_match_pct = available_home / required_total
            
            score = capacity_penalty + dist_to_cluster - (home_stock_match_pct * 10000)
            
            if score < best_score:
                best_score = score
                best_vehicle_id = vehicle_id
                best_vehicle = vehicle
                
        if best_vehicle_id is None or best_score >= 1e9:
             continue # No suitable vehicle, move to the next order

        available_vehicle_ids_set.remove(best_vehicle_id)
        
        # --- 3. Allocation and Capacity Check ---
        
        vehicle = best_vehicle
        home_node = env.get_vehicle_home_warehouse(best_vehicle_id)
        home_warehouse_id = warehouse_by_node.get(home_node)
        
        rem_weight, rem_volume = env.get_vehicle_remaining_capacity(best_vehicle_id)
            
        pickups_by_wh: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        allocations_by_order: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
        def candidate_warehouses_for(sku: str) -> List[str]:
            cands = []
            if home_warehouse_id is not None and home_warehouse_id in warehouse_by_id:
                cands.append(home_warehouse_id)
            try:
                others = env.get_warehouses_with_sku(sku, min_quantity=1) or []
            except Exception:
                others = []
            for w in others:
                if w != home_warehouse_id and w in warehouse_by_id:
                    cands.append(w)
            return cands

        sorted_skus = sorted(batch_total_req.keys(), 
                             key=lambda s: (sku_details.get(s, {}).get("weight", 0.0) + sku_details.get(s, {}).get("volume", 0.0)), reverse=True)
        
        current_batch_remaining = {oid: dict(req) for oid, req in current_batch.items()}
        
        for sku in sorted_skus:
            need_qty = batch_total_req[sku]
            if need_qty <= 0: continue
            
            details = sku_details.get(sku, {})
            unit_w = float(details.get("weight", 0.0))
            unit_v = float(details.get("volume", 0.0))
            
            while need_qty > 0 and (rem_weight > 0 or unit_w == 0) and (rem_volume > 0 or unit_v == 0):
                cap_by_weight = int(rem_weight // unit_w) if unit_w > 0 else need_qty
                cap_by_volume = int(rem_volume // unit_v) if unit_v > 0 else need_qty
                cap_limit = max(0, min(need_qty, cap_by_weight, cap_by_volume))
                
                if cap_limit <= 0: break
                
                for wid in candidate_warehouses_for(sku):
                    available_in_wh = inventory_by_wh.get(wid, {}).get(sku, 0) - reserved_by_wh.get(wid, {}).get(sku, 0)
                    if available_in_wh <= 0: continue
                    
                    take = min(cap_limit, available_in_wh)
                    if take <= 0: continue
                    
                    pickups_by_wh[wid][sku] += take
                    reserved_by_wh[wid][sku] += take
                    rem_weight -= unit_w * take
                    rem_volume -= unit_v * take
                    need_qty -= take
                    cap_limit -= take

                    qty_to_distribute = take
                    for oid in current_batch_remaining.keys():
                        qty_for_order = min(qty_to_distribute, current_batch_remaining[oid].get(sku, 0))
                        if qty_for_order > 0:
                            allocations_by_order[oid][sku] += qty_for_order
                            current_batch_remaining[oid][sku] -= qty_for_order
                            remaining_order_req[oid][sku] -= qty_for_order
                            qty_to_distribute -= qty_for_order
                            if qty_to_distribute <= 0: break
                            
                    if need_qty <= 0 or cap_limit <= 0: break

        total_allocations = sum(sum_remaining(allocs) for allocs in allocations_by_order.values())
        if total_allocations <= 0:
            for wid, items in pickups_by_wh.items():
                for sku, qty in items.items():
                    reserved_by_wh[wid][sku] -= qty
            continue

        # --- 4. Optimized Routing Sequence (TSP Nearest Neighbor) ---
        
        pickup_wh_nodes = {getattr(warehouse_by_id[wid].location, "id"): wid 
                           for wid, items in pickups_by_wh.items() if sum_remaining(items) > 0 and wid in warehouse_by_id}
        
        delivery_cust_nodes = {customer_nodes[oid]: oid 
                               for oid, allocs in allocations_by_order.items() if sum_remaining(allocs) > 0 and customer_nodes[oid] is not None}

        route_nodes: List[Any] = [home_node]
        tsp_targets = list(pickup_wh_nodes.keys()) + list(delivery_cust_nodes.keys())
        
        current_tsp_node = home_node
        while tsp_targets:
            best_next_node = None
            min_dist = float('inf')
            
            for next_node in tsp_targets:
                dist = get_distance_fn(current_tsp_node, next_node) or 1e9
                if dist < min_dist:
                    min_dist = dist
                    best_next_node = next_node
                    
            if best_next_node is not None:
                route_nodes.append(best_next_node)
                tsp_targets.remove(best_next_node)
                current_tsp_node = best_next_node
            else:
                break

        # **FIX 3: Add the final return to home node**
        if route_nodes[-1] != home_node:
            route_nodes.append(home_node)
            
        # --- 5. Build Route Steps ---
        steps: List[Dict[str, Any]] = []
        current_node = home_node
        route_failed = False
        
        # Start at home node (mandatory first step)
        steps.append({"node_id": home_node, "pickups": [], "deliveries": [], "unloads": []})
        
        for i, target_node in enumerate(route_nodes[1:]):
            
            # Route to target node
            path = a_star_search(adjacency_list, current_node, target_node, get_distance_fn=get_distance_fn, coords=coords)
            
            if not path:
                # **FIX 2: If path fails, clear steps and revert reservations**
                for wid, items in pickups_by_wh.items():
                    for sku, qty in items.items():
                        reserved_by_wh[wid][sku] -= qty
                steps = []
                route_failed = True
                break 

            # Add intermediate nodes in path (skip current and target)
            for mid in path[1:-1]:
                steps.append({"node_id": mid, "pickups": [], "deliveries": [], "unloads": []})
            
            # Action at target node
            current_node = target_node
            action_step = {"node_id": current_node, "pickups": [], "deliveries": [], "unloads": []}
            
            if current_node in pickup_wh_nodes:
                wid = pickup_wh_nodes[current_node]
                wh_pickups = []
                for sku, qty in pickups_by_wh.get(wid, {}).items():
                    if qty > 0:
                        wh_pickups.append({
                            "warehouse_id": wid, 
                            "sku_id": sku, 
                            "quantity": int(qty),
                        })
                action_step["pickups"] = wh_pickups
            
            if current_node in delivery_cust_nodes:
                oid = delivery_cust_nodes[current_node]
                deliveries = []
                for sku, qty in allocations_by_order[oid].items():
                    if qty > 0:
                        deliveries.append({
                            "order_id": oid,
                            "sku_id": sku,
                            "quantity": int(qty),
                        })
                action_step["deliveries"] = deliveries

            steps.append(action_step)

        # Finalize route
        if steps and not route_failed:
            solution["routes"].append({
                "vehicle_id": best_vehicle_id,
                "steps": steps,
            })
            
    return solution

# ===============================================================
# Local test harness (COMMENT OUT when submitting)
# ===============================================================
# if __name__ == "__main__":
#     # Example local run (requires the environment installed locally):
#     # pip install robin-logistics-env
#     from robin_logistics_env import LogisticsEnvironment
#     
#     env = LogisticsEnvironment()
#     
#     print("Starting solver...")
#     result = solver(env)
#     print("Routes generated:", len(result.get("routes", [])))
#     
#     # Optional: print the first route for inspection
#     if result.get("routes"):
#         print("\nFirst route steps:")
#         for step in result["routes"][0]["steps"]:
#             print(f"  Node: {step['node_id']} | Pickups: {len(step['pickups'])} | Deliveries: {len(step['deliveries'])}")
# pass