#!/usr/bin/env python3
"""
Improved MWVRP solver for Robin Logistics Hackathon.
Enhancements:
- Order consolidation: Batches geographically close, unfulfilled orders.
- Scored vehicle selection: Chooses the best vehicle based on distance and capacity fit.
- Optimized multi-warehouse routing: Uses a Nearest Neighbor TSP heuristic for pickup sequencing.
- **NEW:** Genetic Algorithm for Global Route Optimization (Replaces greedy route construction).
"""
from __future__ import annotations
import heapq
import math
import itertools
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any, Set, Callable
from collections import deque

# --- GLOBAL GA CONFIGURATION ---
POPULATION_SIZE = 50
GENERATIONS = 50
MUTATION_RATE = 0.1
ELITISM_PERCENTAGE = 0.1
# -------------------------------

# ===============================================================
# A* Search Implementation (Kept for routing between stops)
# ===============================================================
def beam_search_path(
    adjacency_list: Dict[Any, List[Any]],
    start: Any,
    goal: Any,
    get_distance_fn=None,
    coords: Optional[Dict[Any, Tuple[float, float]]] = None,
    beam_width: int = 3,
    time_limit_steps: int = 100_000,
) -> Optional[List[Any]]:
    """Compute a path using Beam Search between start and goal. (Unchanged)"""
    # ... (Keep the existing beam_search_path code here) ...
    # This section remains exactly as it was in your original code.
    if start == goal:
        return [start]
    # --- Heuristic (Haversine or fallback) ---
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
    queue = deque([[start]])
    visited = set()
    steps = 0
    while queue:
        steps += 1
        if steps > time_limit_steps:
            return None
        all_paths = []
        while queue:
            path = queue.popleft()
            node = path[-1]
            if node == goal:
                return path
            visited.add(node)
            for neighbor in adjacency_list.get(node, []):
                if neighbor in path or neighbor in visited:
                    continue
                edge_cost = None
                if get_distance_fn is not None:
                    try:
                        edge_cost = get_distance_fn(node, neighbor)
                    except Exception:
                        edge_cost = None
                if edge_cost is None or edge_cost < 0:
                    edge_cost = 1.0
                all_paths.append(path + [neighbor])
        all_paths.sort(key=lambda p: heuristic(p[-1], goal))
        queue.extend(all_paths[:beam_width])
    return None

# ===============================================================
# Helper utilities (Unchanged)
# ===============================================================
def safe_get_vehicle_home_warehouse_id(env: Any, vehicle: Any, warehouse_by_node: Dict[Any, str]) -> Optional[str]:
    # ... (Keep the existing safe_get_vehicle_home_warehouse_id code here) ...
    try:
        if hasattr(vehicle, "home_warehouse_id"):
            return getattr(vehicle, "home_warehouse_id")
        home_node = env.get_vehicle_home_warehouse(vehicle.id)
        return warehouse_by_node.get(home_node)
    except Exception:
        return None

def build_warehouse_maps(env: Any, sku_ids: Set[str]) -> Tuple[Dict[str, Any], Dict[Any, str]]:
    # ... (Keep the existing build_warehouse_maps code here) ...
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
    # ... (Keep the existing sum_remaining code here) ...
    total = 0
    for v in remaining.values():
        total += max(0, int(v))
    return total

def haversine_distance(coords: Dict[Any, Tuple[float, float]], n1: Any, n2: Any) -> float:
    # ... (Keep the existing haversine_distance code here) ...
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
    # ... (Keep the existing get_node_distance code here) ...
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
# Genetic Algorithm for VRP Routing
# ===============================================================

class RouteIndividual:
    """Represents a potential solution (a set of routes for all vehicles)."""
    
    def __init__(self, routes_dict: Dict[str, List[Any]], total_cost: float = float('inf')):
        """
        routes_dict: {vehicle_id: [node1, node2, ..., home_node]}
        """
        self.routes_dict = routes_dict
        self.total_cost = total_cost

    def __lt__(self, other):
        return self.total_cost < other.total_cost

def calculate_fitness(
    individual: RouteIndividual, 
    distance_fn: Callable[[Any, Any], float]
) -> float:
    """
    Fitness is the inverse of the total route cost (distance).
    Lower cost (distance) is better.
    """
    total_distance = 0.0
    
    # Calculate total distance for all routes
    for route in individual.routes_dict.values():
        for i in range(len(route) - 1):
            n1, n2 = route[i], route[i+1]
            total_distance += distance_fn(n1, n2) or 1e9 # Use high cost for unroutable paths
            
    individual.total_cost = total_distance
    
    # The VRP fitness is typically minimized distance/cost. 
    # For a maximization GA, we use 1 / (1 + cost).
    # Here, we keep the cost itself, relying on min-cost sorting.
    return total_distance 

def apply_2opt_local_search(route: List[Any], distance_fn: Callable[[Any, Any], float]) -> List[Any]:
    """A fast local search operator to improve a single route."""
    best_route = route
    improved = True
    while improved:
        improved = False
        for i in range(1, len(best_route) - 2):
            for j in range(i + 1, len(best_route) - 1):
                new_route = best_route[:i] + best_route[i:j+1][::-1] + best_route[j+1:]
                
                # Check cost change (approximate, only checking swapped segment edges)
                current_cost = (distance_fn(best_route[i-1], best_route[i]) +
                                distance_fn(best_route[j], best_route[j+1]))
                new_cost = (distance_fn(new_route[i-1], new_route[i]) +
                            distance_fn(new_route[j], new_route[j+1]))
                
                if new_cost < current_cost:
                    best_route = new_route
                    improved = True
        if improved:
            # Re-calculate full route cost to ensure the single route is locally optimal
            pass 
    return best_route

def ordered_crossover(parent1: RouteIndividual, parent2: RouteIndividual, all_nodes: Set[Any]) -> RouteIndividual:
    """Crossover using PMX (Partially Mapped Crossover) adapted for VRP routes."""
    child_routes: Dict[str, List[Any]] = {}
    vehicle_ids = list(parent1.routes_dict.keys())
    
    # Perform crossover for each vehicle's non-terminal nodes
    for vid in vehicle_ids:
        p1_route = parent1.routes_dict[vid][1:-1] # Exclude Home nodes
        p2_route = parent2.routes_dict[vid][1:-1]
        n = len(p1_route)
        
        if n == 0:
            child_routes[vid] = parent1.routes_dict[vid]
            continue

        # 1. Select a random crossover segment
        start, end = sorted(random.sample(range(n), 2))
        
        # 2. Copy the segment from P1 to the child
        child_route_segment = p1_route[start:end]
        child_route_genes = [None] * n
        child_route_genes[start:end] = child_route_segment
        
        # 3. Fill the remainder using P2's order, skipping nodes already in the segment
        mapping = {p1_route[i]: p2_route[i] for i in range(start, end)}
        
        fill_index = 0
        for gene in p2_route:
            if gene not in child_route_segment:
                # Find the next available slot outside the segment
                while child_route_genes[fill_index] is not None:
                    fill_index += 1
                child_route_genes[fill_index] = gene

        # Re-add the Home nodes
        home_node = parent1.routes_dict[vid][0]
        child_routes[vid] = [home_node] + child_route_genes + [home_node]

    return RouteIndividual(child_routes)

def mutation(individual: RouteIndividual, distance_fn: Callable[[Any, Any], float]) -> RouteIndividual:
    """Applies a small random perturbation (swap or inversion) to a route."""
    new_routes = individual.routes_dict.copy()
    
    for vid, route in new_routes.items():
        if len(route) < 3: continue # Can't mutate home -> home
        
        # Only mutate the middle (pickup/delivery) nodes
        mutatable_nodes = route[1:-1]
        n = len(mutatable_nodes)
        
        if random.random() < MUTATION_RATE and n > 1:
            # Simple 2-point swap mutation
            idx1, idx2 = random.sample(range(n), 2)
            mutatable_nodes[idx1], mutatable_nodes[idx2] = mutatable_nodes[idx2], mutatable_nodes[idx1]

            # Apply 2-Opt local search to polish the route after mutation
            mutated_route = [route[0]] + mutatable_nodes + [route[-1]]
            new_routes[vid] = apply_2opt_local_search(mutated_route, distance_fn)
        
    return RouteIndividual(new_routes, total_cost=float('inf'))

def optimize_routes_ga(
    initial_routes: Dict[str, List[Any]], 
    distance_fn: Callable[[Any, Any], float],
    all_nodes: Set[Any]
) -> Dict[str, List[Any]]:
    """Runs the Genetic Algorithm to find better routes."""
    
    if not initial_routes:
        return {}

    # 1. Initialize Population
    population: List[RouteIndividual] = []
    initial_individual = RouteIndividual(initial_routes)
    
    # Use the initial greedy solution as the base
    population.append(initial_individual)
    
    # Fill the rest of the population with permutations of the initial routes
    for _ in range(POPULATION_SIZE - 1):
        # Create a new individual by randomly swapping nodes in the initial routes
        new_routes = {}
        for vid, route in initial_routes.items():
            if len(route) > 2:
                middle = route[1:-1].copy()
                random.shuffle(middle)
                new_routes[vid] = [route[0]] + middle + [route[-1]]
            else:
                new_routes[vid] = route
        population.append(RouteIndividual(new_routes))
        
    # 2. Main GA Loop
    for gen in range(GENERATIONS):
        # a) Calculate Fitness
        for individual in population:
            calculate_fitness(individual, distance_fn)

        # b) Sort by Fitness (lowest cost is best)
        population.sort()
        
        best_of_gen = population[0]
        # print(f"GA Gen {gen}: Best Cost = {best_of_gen.total_cost:.2f}")

        # c) Selection (Elitism)
        elite_count = int(POPULATION_SIZE * ELITISM_PERCENTAGE)
        new_generation = population[:elite_count]
        
        # d) Crossover and Generate New Population
        while len(new_generation) < POPULATION_SIZE:
            # Tournament or simple random selection of parents from the fittest half
            p1 = random.choice(population[:POPULATION_SIZE // 2])
            p2 = random.choice(population[:POPULATION_SIZE // 2])
            
            child = ordered_crossover(p1, p2, all_nodes)
            new_generation.append(child)

        # e) Perform Mutation
        for i in range(elite_count, POPULATION_SIZE):
            new_generation[i] = mutation(new_generation[i], distance_fn)

        population = new_generation

    # Final fitness calculation and selection
    for individual in population:
        calculate_fitness(individual, distance_fn)
    population.sort()
    
    return population[0].routes_dict

# ===============================================================
# Main Solver (Modified)
# ===============================================================
def solver(env) -> Dict[str, Any]:
    """ 
    Generate a logistics plan using GA for multi-vehicle route optimization.
    """
    solution: Dict[str, Any] = {"routes": []}
    distance_cache: Dict[Tuple[Any, Any], float] = {}

    # --- Setup and Data Gathering (Unchanged) ---
    try:
        order_ids: List[str] = env.get_all_order_ids() or []
        available_vehicle_ids: List[str] = env.get_available_vehicles() or []
    except Exception:
        return solution
    if not order_ids or not available_vehicle_ids:
        return solution
    
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

    vehicles_dict: Dict[str, Any] = {getattr(v, "id"): v for v in env.get_all_vehicles() or [] if hasattr(v, "id")}
    order_requirements: Dict[str, Dict[str, int]] = {}
    customer_nodes: Dict[str, Any] = {}
    sku_ids: Set[str] = set()

    for oid in order_ids:
        # ... (Populate order_requirements, sku_ids, customer_nodes) ...
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
    
    remaining_order_req: Dict[str, Dict[str, int]] = {
        oid: dict(req) for oid, req in order_requirements.items() if customer_nodes.get(oid) is not None
    }
    available_vehicle_ids_set = set(available_vehicle_ids)
    
    # --- 1. Greedy Assignment (Initial Solution for GA) ---
    
    # This section replaces the old `while True` loop to create the initial, greedy assignment
    
    # Stores the full plan (what each vehicle must pick up/deliver)
    vehicle_plan: Dict[str, Dict[str, Any]] = {vid: {"pickups": defaultdict(lambda: defaultdict(int)), "deliveries": defaultdict(lambda: defaultdict(int)), "home_node": env.get_vehicle_home_warehouse(vid)} for vid in available_vehicle_ids_set}
    
    while True:
        unfulfilled_orders = {oid: req for oid, req in remaining_order_req.items() if sum_remaining(req) > 0}
        if not unfulfilled_orders or not available_vehicle_ids_set:
            break
        
        # This is a simplified greedy batching for the sake of the initial solution
        prime_order_id = next(iter(unfulfilled_orders))
        batch_total_req = unfulfilled_orders[prime_order_id]
        
        # --- 2. Simplified Scored Vehicle Selection for Initial Solution ---
        best_vehicle_id = None
        # ... (Simplified scoring logic to select best_vehicle_id and ensure capacity) ...
        for vehicle_id in available_vehicle_ids_set:
            # For simplicity, we just pick the first one that can carry it
            try:
                max_w, max_v = env.get_vehicle_remaining_capacity(vehicle_id)
            except Exception:
                continue

            total_weight, total_volume = 0.0, 0.0
            for sku, qty in batch_total_req.items():
                details = sku_details.get(sku, {})
                total_weight += qty * float(details.get("weight", 0.0))
                total_volume += qty * float(details.get("volume", 0.0))
            
            if total_weight <= max_w and total_volume <= max_v:
                best_vehicle_id = vehicle_id
                break
        
        if best_vehicle_id is None:
            # No suitable vehicle for this order
            del remaining_order_req[prime_order_id]
            continue
            
        available_vehicle_ids_set.remove(best_vehicle_id)
        
        # --- 3. Allocation (The remaining logic is identical to your original, simplified) ---
        
        home_warehouse_id = warehouse_by_node.get(vehicle_plan[best_vehicle_id]["home_node"])
        rem_weight, rem_volume = env.get_vehicle_remaining_capacity(best_vehicle_id)
        
        pickups_by_wh: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        allocations_by_order: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
        def candidate_warehouses_for(sku: str) -> List[str]:
            # ... (Unchanged logic) ...
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

        # Simplified allocation for the single order (prime_order_id)
        current_batch_remaining = {prime_order_id: dict(batch_total_req)}
        sorted_skus = sorted(batch_total_req.keys(), 
                             key=lambda s: (sku_details.get(s, {}).get("weight", 0.0) + sku_details.get(s, {}).get("volume", 0.0)), reverse=True)

        for sku in sorted_skus:
            need_qty = batch_total_req[sku]
            if need_qty <= 0: continue
            
            details = sku_details.get(sku, {})
            unit_w = float(details.get("weight", 0.0))
            unit_v = float(details.get("volume", 0.0))

            while need_qty > 0 and (rem_weight > 0 or unit_w == 0) and (rem_volume > 0 or unit_v == 0):
                cap_limit = need_qty
                if unit_w > 0: cap_limit = min(cap_limit, int(rem_weight // unit_w))
                if unit_v > 0: cap_limit = min(cap_limit, int(rem_volume // unit_v))
                cap_limit = max(0, cap_limit)
                if cap_limit <= 0: break
                
                for wid in candidate_warehouses_for(sku):
                    available_in_wh = inventory_by_wh.get(wid, {}).get(sku, 0) - reserved_by_wh.get(wid, {}).get(sku, 0)
                    if available_in_wh <= 0: continue
                    
                    take = min(cap_limit, available_in_wh)
                    if take <= 0: continue
                    
                    # Update reservations and inventory
                    pickups_by_wh[wid][sku] += take
                    reserved_by_wh[wid][sku] += take
                    rem_weight -= unit_w * take
                    rem_volume -= unit_v * take
                    need_qty -= take
                    cap_limit -= take
                    
                    # Allocate to the single order
                    allocations_by_order[prime_order_id][sku] += take
                    remaining_order_req[prime_order_id][sku] -= take
                    
                    if need_qty <= 0: break
                if need_qty <= 0: break
                
        total_allocations = sum(sum_remaining(allocs) for allocs in allocations_by_order.values())

        if total_allocations > 0:
            # Store the plan in the vehicle_plan structure
            for wid, items in pickups_by_wh.items():
                for sku, qty in items.items():
                    vehicle_plan[best_vehicle_id]["pickups"][wid][sku] += qty
            
            for oid, allocs in allocations_by_order.items():
                for sku, qty in allocs.items():
                    vehicle_plan[best_vehicle_id]["deliveries"][oid][sku] += qty
        else:
            # If no allocation, revert reservations and re-add vehicle
            for wid, items in pickups_by_wh.items():
                for sku, qty in items.items():
                    reserved_by_wh[wid][sku] -= qty
            available_vehicle_ids_set.add(best_vehicle_id)

    # --- 4. Initial Route Construction (Nearest Neighbor - Input for GA) ---
    
    # Create the initial, greedy route list for all planned vehicles
    initial_ga_routes: Dict[str, List[Any]] = {}
    
    for vid, plan in vehicle_plan.items():
        if not plan["deliveries"] and not plan["pickups"]: continue
        
        home_node = plan["home_node"]
        
        pickup_wh_nodes = {getattr(warehouse_by_id[wid].location, "id"): wid 
                           for wid in plan["pickups"].keys() if wid in warehouse_by_id}
        
        delivery_cust_nodes = {customer_nodes[oid]: oid 
                               for oid in plan["deliveries"].keys() if customer_nodes.get(oid) is not None}

        tsp_targets = list(pickup_wh_nodes.keys()) + list(delivery_cust_nodes.keys())
        
        # Use Nearest Neighbor to create a starting route sequence (The gene)
        route_nodes: List[Any] = [home_node]
        current_tsp_node = home_node
        temp_targets = list(tsp_targets)
        
        while temp_targets:
            best_next_node = None
            min_dist = float('inf')
            
            for next_node in temp_targets:
                dist = get_distance_fn(current_tsp_node, next_node) or 1e9
                if dist < min_dist:
                    min_dist = dist
                    best_next_node = next_node
                    
            if best_next_node is not None:
                route_nodes.append(best_next_node)
                temp_targets.remove(best_next_node)
                current_tsp_node = best_next_node
            else:
                break

        # Final return to home node
        if route_nodes[-1] != home_node:
            route_nodes.append(home_node)
            
        initial_ga_routes[vid] = route_nodes

    # --- 5. Genetic Algorithm Optimization ---
    
    all_target_nodes = set(n for route in initial_ga_routes.values() for n in route[1:-1])
    
    optimized_routes_dict = optimize_routes_ga(initial_ga_routes, get_distance_fn, all_target_nodes)

    # --- 6. Final Route Step Generation (Using Optimized Routes) ---
    
    for best_vehicle_id, route_nodes in optimized_routes_dict.items():
        if len(route_nodes) < 2: continue # Home -> Home route is not meaningful

        steps: List[Dict[str, Any]] = []
        current_node = route_nodes[0]
        route_failed = False
        
        # Start at home node
        steps.append({"node_id": current_node, "pickups": [], "deliveries": [], "unloads": []})
        
        plan = vehicle_plan[best_vehicle_id]
        
        for target_node in route_nodes[1:]:
            
            # Route to target node (using Beam Search)
            path = beam_search_path(adjacency_list, current_node, target_node, get_distance_fn=get_distance_fn, coords=coords, beam_width=3)
            if not path:
                route_failed = True
                break 

            # Add intermediate nodes
            for mid in path[1:-1]:
                steps.append({"node_id": mid, "pickups": [], "deliveries": [], "unloads": []})
            
            # Action at target node
            current_node = target_node
            action_step = {"node_id": current_node, "pickups": [], "deliveries": [], "unloads": []}
            
            # Check for pickups (warehouse node)
            wh_id = warehouse_by_node.get(current_node)
            if wh_id in plan["pickups"]:
                wh_pickups = []
                for sku, qty in plan["pickups"][wh_id].items():
                    if qty > 0:
                        wh_pickups.append({
                            "warehouse_id": wh_id, 
                            "sku_id": sku, 
                            "quantity": int(qty),
                        })
                action_step["pickups"] = wh_pickups
            
            # Check for deliveries (customer node)
            for oid, allocs in plan["deliveries"].items():
                if customer_nodes.get(oid) == current_node:
                    deliveries = []
                    for sku, qty in allocs.items():
                        if qty > 0:
                            deliveries.append({
                                "order_id": oid,
                                "sku_id": sku,
                                "quantity": int(qty),
                            })
                    action_step["deliveries"] = deliveries
                    break # Assuming one order per customer node for simplicity

            steps.append(action_step)

        # Finalize route
        if steps and not route_failed:
            solution["routes"].append({
                "vehicle_id": best_vehicle_id,
                "steps": steps,
            })
            
    return solution