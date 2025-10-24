#!/usr/bin/env python3
"""
Phase 1: Greedy Construction Solver with Full Constraint Validation
Beltone 2nd AI Hackathon - MWVRP Challenge

Strategy:
- Clarke-Wright Savings algorithm for initial route construction
- Full constraint validation (capacity, inventory, connectivity)
- Inventory-aware warehouse selection
- Deterministic and fast (<5 minutes expected)

Target: ≥95% fulfillment, establish cost baseline
"""

from robin_logistics import LogisticsEnvironment
from typing import Dict, List, Tuple, Optional, Set, Any
import heapq
from collections import defaultdict
import math


def solver(env: LogisticsEnvironment) -> Dict:
    """
    Main solver entry point using greedy construction.
    
    Args:
        env: LogisticsEnvironment instance
        
    Returns:
        Solution dictionary with routes
    """
    # Initialize solution builder
    builder = GreedyConstructionBuilder(env)
    
    # Use inventory-aware fallback for maximum fulfillment
    # (Clarke-Wright will be improved in Phase 2 with ALNS)
    solution = builder.construct_simple_fallback()
    
    # Validate before returning (returns 3-tuple: is_valid, message, details)
    validation_result = env.validate_solution_complete(solution)
    is_valid = validation_result[0] if isinstance(validation_result, tuple) else validation_result
    
    if not is_valid:
        message = validation_result[1] if isinstance(validation_result, tuple) and len(validation_result) > 1 else "Unknown error"
        print(f"Warning: Solution validation failed: {message}")
    
    return solution


class GreedyConstructionBuilder:
    """Builds initial solution using greedy heuristics with constraint validation."""
    
    def __init__(self, env: LogisticsEnvironment):
        self.env = env
        self.solution = {"routes": []}
        
        # Cache frequently accessed data
        self.orders = {oid: env.orders[oid] for oid in env.get_all_order_ids()}
        self.vehicles = {v.id: v for v in env.get_all_vehicles()}
        self.warehouses = dict(env.warehouses)
        self.skus = dict(env.skus)
        self.road_network = env.get_road_network_data()
        
        # Track assignments
        self.assigned_orders: Set[str] = set()
        self.used_vehicles: Set[str] = set()
        
        # Distance cache (within single solver call - allowed)
        # Note: node IDs are integers in the API
        self.distance_cache: Dict[Tuple[int, int], Optional[float]] = {}
        
    def construct_solution(self) -> Dict:
        """
        Main construction algorithm using Clarke-Wright savings.
        
        Complexity: O(orders² × log(orders) + orders × vehicles)
        Expected runtime: <3 minutes for 50 orders
        """
        # Step 1: Calculate savings for order pairs
        savings = self._calculate_savings()
        
        # Step 2: Greedily merge routes based on savings
        routes = self._merge_routes_by_savings(savings)
        
        # Step 3: Assign unassigned orders (greedy insertion)
        self._assign_remaining_orders(routes)
        
        # Step 4: Convert to solution format
        self.solution["routes"] = routes
        
        return self.solution
    
    def _calculate_savings(self) -> List[Tuple[float, str, str]]:
        """
        Calculate Clarke-Wright savings for all order pairs.
        
        Savings(i,j) = distance(depot, i) + distance(depot, j) - distance(i, j)
        Higher savings = more beneficial to serve together
        
        Returns:
            List of (savings_value, order_i, order_j) sorted descending
        """
        savings_heap = []
        order_ids = list(self.orders.keys())
        
        for i in range(len(order_ids)):
            for j in range(i + 1, len(order_ids)):
                oid_i, oid_j = order_ids[i], order_ids[j]
                
                # Get destinations
                dest_i = self.orders[oid_i].destination.id
                dest_j = self.orders[oid_j].destination.id
                
                # We'll compute savings relative to closest warehouse
                # (simplification: use first warehouse as reference depot)
                depot_node = list(self.warehouses.values())[0].location.id
                
                dist_depot_i = self._get_distance(depot_node, dest_i)
                dist_depot_j = self._get_distance(depot_node, dest_j)
                dist_i_j = self._get_distance(dest_i, dest_j)
                
                if dist_depot_i and dist_depot_j and dist_i_j:
                    savings_value = dist_depot_i + dist_depot_j - dist_i_j
                    # Use negative for max-heap behavior with heapq (min-heap)
                    heapq.heappush(savings_heap, (-savings_value, oid_i, oid_j))
        
        # Convert to sorted list (descending savings)
        return [(-s, i, j) for s, i, j in sorted(savings_heap)]
    
    def _merge_routes_by_savings(self, savings: List[Tuple[float, str, str]]) -> List[Dict]:
        """
        Merge orders into routes based on savings, respecting constraints.
        
        Args:
            savings: List of (savings_value, order_i, order_j)
            
        Returns:
            List of route dictionaries
        """
        # Initially, each order is in its own route (conceptually)
        order_to_route: Dict[str, int] = {}
        routes: List[List[str]] = []
        
        for savings_value, oid_i, oid_j in savings:
            # Skip if both already assigned to same route
            if oid_i in order_to_route and oid_j in order_to_route:
                if order_to_route[oid_i] == order_to_route[oid_j]:
                    continue
            
            # Try to merge
            if oid_i not in order_to_route and oid_j not in order_to_route:
                # Create new route with both orders
                if self._can_serve_together([oid_i, oid_j]):
                    route_idx = len(routes)
                    routes.append([oid_i, oid_j])
                    order_to_route[oid_i] = route_idx
                    order_to_route[oid_j] = route_idx
                    self.assigned_orders.update([oid_i, oid_j])
                    
            elif oid_i in order_to_route and oid_j not in order_to_route:
                # Add oid_j to oid_i's route
                route_idx = order_to_route[oid_i]
                candidate_route = routes[route_idx] + [oid_j]
                if self._can_serve_together(candidate_route):
                    routes[route_idx].append(oid_j)
                    order_to_route[oid_j] = route_idx
                    self.assigned_orders.add(oid_j)
                    
            elif oid_j in order_to_route and oid_i not in order_to_route:
                # Add oid_i to oid_j's route
                route_idx = order_to_route[oid_j]
                candidate_route = routes[route_idx] + [oid_i]
                if self._can_serve_together(candidate_route):
                    routes[route_idx].append(oid_i)
                    order_to_route[oid_i] = route_idx
                    self.assigned_orders.add(oid_i)
        
        # Convert order lists to proper route format
        return self._convert_order_lists_to_routes(routes)
    
    def _can_serve_together(self, order_ids: List[str]) -> bool:
        """
        Check if a set of orders can be served by one vehicle.
        
        Validates:
        - Total weight ≤ vehicle capacity
        - Total volume ≤ vehicle capacity
        - Inventory availability at selected warehouse
        - Route distance ≤ max vehicle distance
        
        Args:
            order_ids: List of order IDs to check
            
        Returns:
            True if feasible, False otherwise
        """
        # Calculate total demand
        total_weight = 0.0
        total_volume = 0.0
        sku_totals = defaultdict(int)
        
        for oid in order_ids:
            order = self.orders[oid]
            for sku_id, qty in order.requested_items.items():
                sku = self.skus[sku_id]
                total_weight += sku.weight * qty
                total_volume += sku.volume * qty
                sku_totals[sku_id] += qty
        
        # Find a vehicle that can handle this load
        available_vehicles = [
            v for v_id, v in self.vehicles.items()
            if v_id not in self.used_vehicles
        ]
        
        suitable_vehicle = None
        for v in available_vehicles:
            if v.capacity_weight >= total_weight and v.capacity_volume >= total_volume:
                suitable_vehicle = v
                break
        
        if not suitable_vehicle:
            return False
        
        # Check warehouse inventory
        warehouse = self.warehouses[suitable_vehicle.home_warehouse_id]
        for sku_id, qty in sku_totals.items():
            if warehouse.inventory.get(sku_id, 0) < qty:
                return False
        
        # Rough distance check (simplified: sum of distances from depot)
        depot_node = warehouse.location.id
        total_distance = 0.0
        for oid in order_ids:
            dest = self.orders[oid].destination.id
            dist = self._get_distance(depot_node, dest)
            if dist:
                total_distance += dist * 2  # Round trip estimate
        
        if total_distance > suitable_vehicle.max_distance:
            return False
        
        return True
    
    def _convert_order_lists_to_routes(self, order_lists: List[List[str]]) -> List[Dict]:
        """
        Convert lists of orders into proper route format with steps.
        
        Args:
            order_lists: List of [order_id, order_id, ...] for each route
            
        Returns:
            List of route dictionaries with vehicle_id and steps
        """
        routes = []
        
        for order_list in order_lists:
            if not order_list:
                continue
            
            # Find suitable vehicle
            vehicle = self._find_vehicle_for_orders(order_list)
            if not vehicle:
                continue
            
            # Build route steps
            steps = self._build_route_steps(vehicle, order_list)
            if steps:
                routes.append({
                    "vehicle_id": vehicle.id,  # type: ignore
                    "steps": steps
                })
                self.used_vehicles.add(vehicle.id)  # type: ignore
        
        return routes
    
    def _find_vehicle_for_orders(self, order_ids: List[str]) -> Optional[Any]:
        """Find a suitable unused vehicle for the given orders."""
        # Calculate total demand
        total_weight = 0.0
        total_volume = 0.0
        sku_totals = defaultdict(int)
        
        for oid in order_ids:
            order = self.orders[oid]
            for sku_id, qty in order.requested_items.items():
                sku = self.skus[sku_id]
                total_weight += sku.weight * qty
                total_volume += sku.volume * qty
                sku_totals[sku_id] += qty
        
        # Find suitable vehicle with inventory
        for v_id, v in self.vehicles.items():
            if v_id in self.used_vehicles:
                continue
            
            # Check capacity
            if v.capacity_weight < total_weight or v.capacity_volume < total_volume:
                continue
            
            # Check warehouse has inventory
            warehouse = self.warehouses[v.home_warehouse_id]
            has_inventory = all(
                warehouse.inventory.get(sku_id, 0) >= qty
                for sku_id, qty in sku_totals.items()
            )
            
            if has_inventory:
                return v
        
        return None
    
    def _build_route_steps(self, vehicle: Any, order_ids: List[str]) -> Optional[List[Dict]]:
        """
        Build step-by-step route for vehicle serving given orders.
        
        Strategy: Warehouse → nearest order → next nearest → ... → warehouse
        
        Args:
            vehicle: Vehicle object
            order_ids: List of order IDs to serve
            
        Returns:
            List of step dictionaries, or None if infeasible
        """
        warehouse = self.warehouses[vehicle.home_warehouse_id]  # type: ignore
        warehouse_node = warehouse.location.id
        
        # Collect all SKUs needed
        sku_totals = defaultdict(int)
        for oid in order_ids:
            for sku_id, qty in self.orders[oid].requested_items.items():
                sku_totals[sku_id] += qty
        
        steps = []
        
        # Step 1: Start at warehouse, pickup all items
        pickups = [
            {"warehouse_id": warehouse.id, "sku_id": sku_id, "quantity": qty}
            for sku_id, qty in sku_totals.items()
        ]
        steps.append({
            "node_id": warehouse_node,
            "pickups": pickups,
            "deliveries": [],
            "unloads": []
        })
        
        # Step 2: Visit orders in nearest-neighbor order
        current_node = warehouse_node
        remaining_orders = set(order_ids)
        
        while remaining_orders:
            # Find nearest unvisited order
            nearest_oid = None
            nearest_dist = float('inf')
            
            for oid in remaining_orders:
                dest = self.orders[oid].destination.id
                dist = self._get_distance(current_node, dest)
                if dist and dist < nearest_dist:
                    nearest_dist = dist
                    nearest_oid = oid
            
            if not nearest_oid:
                # No path found, abort this route
                return None
            
            # Get path to this order
            dest_node = self.orders[nearest_oid].destination.id
            path = self._get_path(current_node, dest_node)
            
            if not path:
                return None
            
            # Add intermediate nodes
            for node in path[1:-1]:
                steps.append({
                    "node_id": node,
                    "pickups": [],
                    "deliveries": [],
                    "unloads": []
                })
            
            # Deliver at destination
            deliveries = [
                {"order_id": nearest_oid, "sku_id": sku_id, "quantity": qty}
                for sku_id, qty in self.orders[nearest_oid].requested_items.items()
            ]
            steps.append({
                "node_id": dest_node,
                "pickups": [],
                "deliveries": deliveries,
                "unloads": []
            })
            
            current_node = dest_node
            remaining_orders.remove(nearest_oid)
        
        # Step 3: Return to warehouse
        return_path = self._get_path(current_node, warehouse_node)
        if not return_path:
            return None
        
        for node in return_path[1:-1]:
            steps.append({
                "node_id": node,
                "pickups": [],
                "deliveries": [],
                "unloads": []
            })
        
        steps.append({
            "node_id": warehouse_node,
            "pickups": [],
            "deliveries": [],
            "unloads": []
        })
        
        return steps
    
    def _assign_remaining_orders(self, routes: List[Dict]) -> None:
        """
        Greedy insertion of unassigned orders into existing routes or new routes.
        
        Args:
            routes: Current list of routes (modified in place)
        """
        unassigned = set(self.orders.keys()) - self.assigned_orders
        
        for oid in unassigned:
            # Try to insert into existing route
            inserted = False
            for route in routes:
                if self._try_insert_order(route, oid):
                    self.assigned_orders.add(oid)
                    inserted = True
                    break
            
            # If can't insert, create new single-order route
            if not inserted:
                # Reset used_vehicles to allow reassignment for single orders
                # This is a workaround: in Phase 1, we allow vehicles to be reused
                # if they haven't been assigned yet during this phase
                vehicle = self._find_vehicle_for_orders([oid])
                
                # If no vehicle found, try to find ANY vehicle (even if used)
                if not vehicle:
                    for v_id, v in self.vehicles.items():
                        # Check if vehicle's warehouse has inventory
                        warehouse = self.warehouses[v.home_warehouse_id]
                        order = self.orders[oid]
                        has_inventory = all(
                            warehouse.inventory.get(sku_id, 0) >= qty
                            for sku_id, qty in order.requested_items.items()
                        )
                        if has_inventory:
                            # Check capacity
                            total_weight = sum(
                                self.skus[sku_id].weight * qty
                                for sku_id, qty in order.requested_items.items()
                            )
                            total_volume = sum(
                                self.skus[sku_id].volume * qty
                                for sku_id, qty in order.requested_items.items()
                            )
                            if v.capacity_weight >= total_weight and v.capacity_volume >= total_volume:
                                vehicle = v
                                break
                
                if vehicle:
                    steps = self._build_route_steps(vehicle, [oid])
                    if steps:
                        routes.append({
                            "vehicle_id": vehicle.id,  # type: ignore
                            "steps": steps
                        })
                        # Don't add to used_vehicles - allow reuse
                        self.assigned_orders.add(oid)
    
    def _try_insert_order(self, route: Dict, order_id: str) -> bool:
        """
        Try to insert an order into an existing route.
        
        Simplified: Only append to end for now (full insertion later in ALNS).
        
        Args:
            route: Route dictionary
            order_id: Order ID to insert
            
        Returns:
            True if successfully inserted, False otherwise
        """
        # For Phase 1, skip complex insertion (will add in Phase 2)
        return False
    
    def _get_distance(self, node1: int, node2: int) -> Optional[float]:
        """
        Get distance between two nodes with caching.
        
        Args:
            node1: Source node ID (integer)
            node2: Destination node ID (integer)
            
        Returns:
            Distance in km, or None if no path exists
        """
        if node1 == node2:
            return 0.0
        
        key = (node1, node2)
        if key not in self.distance_cache:
            # Try API first
            dist = self.env.get_distance(node1, node2)
            
            # If API returns None, calculate using BFS path length (Euclidean approximation)
            if dist is None:
                path = self._get_path(node1, node2)
                if path and len(path) > 1:
                    # Approximate distance based on path length
                    # Use number of hops as rough distance (each hop ~500m average in Cairo)
                    dist = (len(path) - 1) * 0.5  # km
                else:
                    dist = None
            
            self.distance_cache[key] = dist
        
        return self.distance_cache[key]
    
    def _get_path(self, start: int, end: int) -> Optional[List[int]]:
        """
        Get shortest path between nodes using BFS.
        
        Args:
            start: Start node ID (integer)
            end: End node ID (integer)
            
        Returns:
            List of node IDs, or None if no path
        """
        if start == end:
            return [start]
        
        # Use BFS on adjacency list
        from collections import deque
        
        adjacency_list = self.road_network.get("adjacency_list", {})
        # Convert keys to int if needed (adjacency list may have int keys)
        queue = deque([(start, [start])])
        visited = {start}
        
        while queue:
            current, path = queue.popleft()
            
            # Handle both int and str keys in adjacency list
            neighbors = adjacency_list.get(current, adjacency_list.get(str(current), []))
            
            for neighbor in neighbors:
                # Ensure neighbor is int
                neighbor_int = int(neighbor) if isinstance(neighbor, str) else neighbor
                
                if neighbor_int == end:
                    return path + [neighbor_int]
                
                if neighbor_int not in visited:
                    visited.add(neighbor_int)
                    queue.append((neighbor_int, path + [neighbor_int]))
        
        return None
    
    def construct_simple_fallback(self) -> Dict:
        """
        Fallback to ultra-simple solution if main construction fails.
        
        Assigns each order to a vehicle from the warehouse with best inventory.
        Prioritizes fulfillment over cost optimization.
        
        Returns:
            Simple valid solution
        """
        routes = []
        order_ids = list(self.orders.keys())
        
        # Track inventory usage per warehouse
        warehouse_inventory_used = {wh_id: defaultdict(int) for wh_id in self.warehouses.keys()}
        # Track which vehicles have been used
        used_vehicles = set()
        
        for oid in order_ids:
            order = self.orders[oid]
            
            # Find warehouse with sufficient inventory for this order
            best_warehouse_id = None
            for wh_id, wh in self.warehouses.items():
                has_inventory = True
                for sku_id, qty in order.requested_items.items():
                    available = wh.inventory.get(sku_id, 0) - warehouse_inventory_used[wh_id][sku_id]
                    if available < qty:
                        has_inventory = False
                        break
                
                if has_inventory:
                    best_warehouse_id = wh_id
                    break
            
            if not best_warehouse_id:
                continue
            
            # Find an UNUSED vehicle from that warehouse
            warehouse = self.warehouses[best_warehouse_id]
            vehicle = None
            for v in warehouse.vehicles:
                # Skip if already used
                if v.id in used_vehicles:
                    continue
                    
                # Check capacity
                total_weight = sum(
                    self.skus[sku_id].weight * qty
                    for sku_id, qty in order.requested_items.items()
                )
                total_volume = sum(
                    self.skus[sku_id].volume * qty
                    for sku_id, qty in order.requested_items.items()
                )
                if v.capacity_weight >= total_weight and v.capacity_volume >= total_volume:
                    vehicle = v
                    break
            
            if not vehicle:
                continue
            
            # Build single-order route
            steps = self._build_route_steps(vehicle, [oid])
            if steps:
                routes.append({
                    "vehicle_id": vehicle.id,
                    "steps": steps
                })
                # Mark vehicle as used
                used_vehicles.add(vehicle.id)
                # Track inventory usage
                for sku_id, qty in order.requested_items.items():
                    warehouse_inventory_used[best_warehouse_id][sku_id] += qty
        
        return {"routes": routes}


# DO NOT CALL MAIN IN SUBMISSION
# if __name__ == '__main__':
#     env = LogisticsEnvironment()
#     solution = solver(env)
#     print(f"Solution generated with {len(solution['routes'])} routes")
