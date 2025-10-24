#!/usr/bin/env python3
"""
Overfitting_solver_16: Multi-Order Vehicle Routing with Local Search
Target: ~100% Fulfillment via 2-opt and Relocation

Strategy:
- Batch multiple orders per vehicle (capacity-aware)
- Apply 2-opt and relocation local search for route improvement
- Maintain fast execution (<5s) with smart heuristics
"""

from robin_logistics import LogisticsEnvironment
from typing import Dict, List, Optional, Set, Any, Tuple
from collections import defaultdict, deque
import random


def solver(env: LogisticsEnvironment) -> Dict:
    """Multi-order routing with local search."""
    return MultiOrderLocalSearchSolver(env).solve()


class MultiOrderLocalSearchSolver:
    """Solver that batches orders per vehicle and applies local search."""
    
    def __init__(self, env: LogisticsEnvironment):
        self.env = env
        
        # Pre-cache all data
        self.orders = {oid: env.orders[oid] for oid in env.get_all_order_ids()}
        self.vehicles = list(env.get_all_vehicles())
        self.warehouses = dict(env.warehouses)
        self.skus = dict(env.skus)
        
        # Inventory tracking
        self.inventory = {wh_id: dict(wh.inventory) for wh_id, wh in self.warehouses.items()}
        
        # Vehicle home nodes
        self.vehicle_homes = {}
        for v in self.vehicles:
            try:
                self.vehicle_homes[v.id] = env.get_vehicle_home_warehouse(v.id)
            except:
                self.vehicle_homes[v.id] = list(self.warehouses.values())[0].location.id
        
        # Road network
        self.road_network = env.get_road_network_data()
        self.adjacency_list = self.road_network.get("adjacency_list", {})
        
        # Caches
        self.distance_cache = {}
        self.path_cache = {}
        
        # Order assignment tracking
        self.assigned_orders = set()
        self.order_to_warehouse = {}
    
    def solve(self) -> Dict:
        """Main solving loop with multi-order batching and local search."""
        routes = []
        
        # Phase 1: Greedy assignment (batch multiple orders per vehicle)
        # Sort by total item count (not just heavy items) for better packing
        orders_sorted = sorted(
            self.orders.keys(),
            key=lambda oid: sum(self.orders[oid].requested_items.values()),
            reverse=True
        )
        
        # Try to pack orders more aggressively - allow multi-warehouse pickups
        for vehicle in self.vehicles:
            vehicle_orders = []
            vehicle_load_weight = 0
            vehicle_load_volume = 0
            order_warehouses = {}  # Track warehouse per order
            
            for oid in orders_sorted:
                if oid in self.assigned_orders:
                    continue
                
                order = self.orders[oid]
                order_demand = dict(order.requested_items)
                
                # Calculate order weight/volume
                order_weight = sum(
                    self.skus[sku_id].weight * qty
                    for sku_id, qty in order_demand.items()
                )
                order_volume = sum(
                    self.skus[sku_id].volume * qty
                    for sku_id, qty in order_demand.items()
                )
                
                # Check capacity
                if (vehicle_load_weight + order_weight > vehicle.capacity_weight or
                    vehicle_load_volume + order_volume > vehicle.capacity_volume):
                    continue
                
                # Find best warehouse for this order (independent per order)
                wh_id = self._find_nearest_warehouse_with_stock(order_demand, order.destination.id)
                if not wh_id:
                    continue
                
                # Assign order to vehicle
                vehicle_orders.append(oid)
                vehicle_load_weight += order_weight
                vehicle_load_volume += order_volume
                self.assigned_orders.add(oid)
                order_warehouses[oid] = wh_id
                
                # Update inventory
                for sku_id, qty in order_demand.items():
                    self.inventory[wh_id][sku_id] -= qty
            
            # Build route for this vehicle
            if vehicle_orders:
                route = self._build_multi_warehouse_route(vehicle, vehicle_orders, order_warehouses)
                if route:
                    # Phase 2: Apply local search on customer sequence
                    improved_route = self._apply_local_search_multi_wh(route, vehicle_orders, order_warehouses)
                    if improved_route:
                        routes.append(improved_route)
                    else:
                        routes.append(route)
        
        # Phase 3: Ultra-aggressive final pass - deliver ALL remaining orders
        # Sort remaining by smallest first (easier to fit)
        remaining_orders = [oid for oid in orders_sorted if oid not in self.assigned_orders]
        remaining_orders_sorted = sorted(
            remaining_orders,
            key=lambda oid: sum(self.orders[oid].requested_items.values())
        )
        
        for oid in remaining_orders_sorted:
            order = self.orders[oid]
            order_demand = dict(order.requested_items)
            
            # Try BOTH warehouses explicitly
            wh_id = None
            for candidate_wh_id in self.warehouses.keys():
                if all(self.inventory[candidate_wh_id].get(sku, 0) >= qty 
                      for sku, qty in order_demand.items()):
                    wh_id = candidate_wh_id
                    break
            
            if not wh_id:
                continue  # Truly no warehouse has stock
            
            # Calculate order weight/volume
            order_weight = sum(
                self.skus[sku_id].weight * qty
                for sku_id, qty in order_demand.items()
            )
            order_volume = sum(
                self.skus[sku_id].volume * qty
                for sku_id, qty in order_demand.items()
            )
            
            # Try EVERY vehicle until one works
            for vehicle in self.vehicles:
                if (order_weight > vehicle.capacity_weight or
                    order_volume > vehicle.capacity_volume):
                    continue  # Doesn't fit
                
                # Try to build route
                try:
                    route = self._build_multi_warehouse_route(
                        vehicle, 
                        [oid], 
                        {oid: wh_id}
                    )
                    if route:
                        routes.append(route)
                        self.assigned_orders.add(oid)
                        
                        # Update inventory
                        for sku_id, qty in order_demand.items():
                            self.inventory[wh_id][sku_id] -= qty
                        
                        break  # Success! Move to next order
                except Exception as e:
                    # Try next vehicle
                    continue
        
        return {"routes": routes}
    
    def _find_nearest_warehouse_with_stock(self, demand: Dict[str, int], customer_node: int) -> Optional[str]:
        """Find nearest warehouse with sufficient stock."""
        best_wh = None
        min_dist = float('inf')
        
        for wh_id, wh in self.warehouses.items():
            # Check stock
            if not all(self.inventory[wh_id].get(sku, 0) >= qty for sku, qty in demand.items()):
                continue
            
            # Calculate distance
            dist = self._get_distance(wh.location.id, customer_node)
            if dist < min_dist:
                min_dist = dist
                best_wh = wh_id
        
        return best_wh
    
    def _get_distance(self, node1: int, node2: int) -> float:
        """Cached distance lookup."""
        if node1 == node2:
            return 0.0
        
        key = (min(node1, node2), max(node1, node2))
        if key not in self.distance_cache:
            try:
                dist = self.env.get_distance(node1, node2)
                if dist is None or dist < 0:
                    dist = 10.0
            except:
                dist = 10.0
            self.distance_cache[key] = dist
        
        return self.distance_cache[key]
    
    def _get_path(self, start: int, end: int) -> Optional[List[int]]:
        """BFS pathfinding with caching."""
        if start == end:
            return [start]
        
        key = (start, end)
        if key in self.path_cache:
            return self.path_cache[key]
        
        queue = deque([(start, [start])])
        visited = {start}
        
        while queue:
            current, path = queue.popleft()
            neighbors = self.adjacency_list.get(current, self.adjacency_list.get(str(current), []))
            
            for neighbor in neighbors:
                neighbor_int = int(neighbor) if isinstance(neighbor, str) else neighbor
                
                if neighbor_int == end:
                    result = path + [neighbor_int]
                    self.path_cache[key] = result
                    return result
                
                if neighbor_int not in visited:
                    visited.add(neighbor_int)
                    queue.append((neighbor_int, path + [neighbor_int]))
        
        return None
    
    def _build_multi_warehouse_route(
        self,
        vehicle: Any,
        order_ids: List[str],
        order_warehouses: Dict[str, str]
    ) -> Optional[Dict]:
        """Build route with multiple orders from potentially different warehouses."""
        if not order_ids:
            return None
        
        steps = []
        
        # Group orders by warehouse
        warehouse_orders = defaultdict(list)
        for oid in order_ids:
            wh_id = order_warehouses[oid]
            warehouse_orders[wh_id].append(oid)
        
        # For simplicity, pick up from first warehouse then deliver all
        # (Multi-warehouse pickup complicates routing - keep simple for speed)
        primary_wh_id = list(warehouse_orders.keys())[0]
        primary_wh_node = self.warehouses[primary_wh_id].location.id
        
        # Step 1: Pickup all orders from primary warehouse
        all_pickups = []
        for oid in order_ids:
            wh_id = order_warehouses[oid]
            for sku_id, qty in self.orders[oid].requested_items.items():
                all_pickups.append({
                    "warehouse_id": wh_id,
                    "sku_id": sku_id,
                    "quantity": qty
                })
        
        steps.append({
            "node_id": primary_wh_node,
            "pickups": all_pickups,
            "deliveries": [],
            "unloads": []
        })
        
        # Step 2: Visit customers in nearest-neighbor order
        current_node = primary_wh_node
        remaining_orders = set(order_ids)
        
        while remaining_orders:
            # Find nearest unvisited order
            nearest_oid = None
            min_dist = float('inf')
            
            for oid in remaining_orders:
                dest_node = self.orders[oid].destination.id
                dist = self._get_distance(current_node, dest_node)
                if dist < min_dist:
                    min_dist = dist
                    nearest_oid = oid
            
            if not nearest_oid:
                return None
            
            # Get path to nearest customer
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
            
            # Deliver to customer
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
        
        # Step 3: Return to primary warehouse
        return_path = self._get_path(current_node, primary_wh_node)
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
            "node_id": primary_wh_node,
            "pickups": [],
            "deliveries": [],
            "unloads": []
        })
        
        return {"vehicle_id": vehicle.id, "steps": steps}
    
    def _apply_local_search_multi_wh(
        self,
        route: Dict,
        order_ids: List[str],
        order_warehouses: Dict[str, str]
    ) -> Optional[Dict]:
        """Apply 2-opt and relocation to improve route."""
        if len(order_ids) < 2:
            return route  # No improvement possible
        
        # Get primary warehouse
        primary_wh_id = list(set(order_warehouses.values()))[0]
        
        # Extract current order sequence from route
        current_sequence = list(order_ids)
        best_sequence = list(current_sequence)
        best_cost = self._calculate_route_cost(best_sequence, primary_wh_id)
        
        improved = True
        iterations = 0
        max_iterations = 10  # Limit iterations for speed
        
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            
            # Try 2-opt swaps
            for i in range(len(current_sequence) - 1):
                for j in range(i + 2, len(current_sequence) + 1):
                    # Reverse segment [i:j]
                    new_sequence = (current_sequence[:i] + 
                                   current_sequence[i:j][::-1] + 
                                   current_sequence[j:])
                    
                    new_cost = self._calculate_route_cost(new_sequence, primary_wh_id)
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_sequence = new_sequence
                        improved = True
            
            # Try relocations
            for i in range(len(current_sequence)):
                for j in range(len(current_sequence)):
                    if i == j:
                        continue
                    
                    # Move order from i to j
                    new_sequence = list(current_sequence)
                    order = new_sequence.pop(i)
                    new_sequence.insert(j, order)
                    
                    new_cost = self._calculate_route_cost(new_sequence, primary_wh_id)
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_sequence = new_sequence
                        improved = True
            
            current_sequence = list(best_sequence)
        
        # Rebuild route with improved sequence
        if best_sequence != order_ids:
            vehicle_id = route["vehicle_id"]
            vehicle = next(v for v in self.vehicles if v.id == vehicle_id)
            return self._build_multi_warehouse_route(vehicle, best_sequence, order_warehouses)
        
        return route
    
    def _calculate_route_cost(self, order_sequence: List[str], warehouse_id: str) -> float:
        """Calculate total distance for an order sequence."""
        warehouse_node = self.warehouses[warehouse_id].location.id
        total_dist = 0.0
        current_node = warehouse_node
        
        for oid in order_sequence:
            dest_node = self.orders[oid].destination.id
            total_dist += self._get_distance(current_node, dest_node)
            current_node = dest_node
        
        # Return to warehouse
        total_dist += self._get_distance(current_node, warehouse_node)
        
        return total_dist
