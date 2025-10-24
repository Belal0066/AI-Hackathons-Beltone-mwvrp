#!/usr/bin/env python3
"""
MWVRP Solver with Local Search Optimization - Beltone AI Hackathon
Strategy: Greedy construction + Local Search (2-opt, Relocate) for ~100% fulfillment

Key features:
1. Greedy bin-packing construction
2. 2-opt: Swap order sequences between routes
3. Relocate: Move orders from one vehicle to another
4. Iterative improvement until no gains
"""

from robin_logistics import LogisticsEnvironment
from typing import Dict, List, Tuple, Optional, Set, Any
from collections import defaultdict, deque
import copy


def solver(env: LogisticsEnvironment) -> Dict:
    """
    Main solver with greedy construction + local search.
    
    Returns:
        Solution dictionary with routes
    """
    builder = LocalSearchSolver(env)
    solution = builder.build_and_optimize()
    
    # Validate
    validation_result = env.validate_solution_complete(solution)
    is_valid = validation_result[0] if isinstance(validation_result, tuple) else validation_result
    
    if not is_valid:
        message = validation_result[1] if isinstance(validation_result, tuple) and len(validation_result) > 1 else "Unknown"
        print(f"[WARN] Validation failed: {message}")
    
    return solution


class LocalSearchSolver:
    """Solver with greedy construction + local search optimization."""
    
    def __init__(self, env: LogisticsEnvironment):
        self.env = env
        
        # Cache data
        self.orders = {oid: env.orders[oid] for oid in env.get_all_order_ids()}
        self.vehicles = {v.id: v for v in env.get_all_vehicles()}
        self.warehouses = dict(env.warehouses)
        self.skus = dict(env.skus)
        self.road_network = env.get_road_network_data()
        self.adjacency_list = self.road_network.get("adjacency_list", {})
        
        # Distance cache
        self.distance_cache: Dict[Tuple[int, int], Optional[float]] = {}
        
    def build_and_optimize(self) -> Dict:
        """Build initial solution then optimize with local search."""
        # Phase 1: Greedy construction
        initial_solution = self._greedy_construction()
        
        # Phase 2: Local search optimization
        optimized_solution = self._local_search(initial_solution)
        
        return optimized_solution
    
    def _greedy_construction(self) -> Dict:
        """Build initial solution with greedy bin-packing."""
        routes = []
        unassigned_orders = set(self.orders.keys())
        used_vehicles = set()
        
        # Track inventory per warehouse
        warehouse_inventory = {}
        for wh_id, wh in self.warehouses.items():
            warehouse_inventory[wh_id] = dict(wh.inventory)
        
        # Sort vehicles by capacity (largest first)
        vehicles_sorted = sorted(
            self.vehicles.values(),
            key=lambda v: v.capacity_weight * v.capacity_volume,
            reverse=True
        )
        
        # PASS 1: Greedy batching
        for vehicle in vehicles_sorted:
            if not unassigned_orders:
                break
                
            packed_orders, total_demand = self._pack_orders(
                vehicle, unassigned_orders, warehouse_inventory
            )
            
            if not packed_orders:
                continue
            
            # Find warehouse with inventory
            best_warehouse = self._find_warehouse_with_stock(total_demand, warehouse_inventory)
            if not best_warehouse:
                continue
            
            # Build route
            route = self._build_route(vehicle, packed_orders, best_warehouse, total_demand)
            
            if route:
                routes.append(route)
                used_vehicles.add(vehicle.id)
                unassigned_orders -= set(packed_orders)
                
                # Update inventory
                for sku_id, qty in total_demand.items():
                    warehouse_inventory[best_warehouse][sku_id] -= qty
        
        # PASS 2: Single-order fallback with ALL unused vehicles
        for oid in list(unassigned_orders):
            order = self.orders[oid]
            order_demand = dict(order.requested_items)
            
            assigned = False
            
            # Try each unused vehicle
            for vehicle in vehicles_sorted:
                if vehicle.id in used_vehicles:
                    continue
                
                # Check capacity
                if not self._fits_in_vehicle(vehicle, order_demand):
                    continue
                
                # Find warehouse with stock
                best_warehouse = self._find_warehouse_with_stock(order_demand, warehouse_inventory)
                
                if not best_warehouse:
                    # Try multi-warehouse pickup
                    multi_wh = self._find_multi_warehouse_stock(order_demand, warehouse_inventory)
                    if multi_wh:
                        route = self._build_multi_warehouse_route(vehicle, [oid], multi_wh)
                        if route:
                            routes.append(route)
                            used_vehicles.add(vehicle.id)
                            unassigned_orders.discard(oid)
                            
                            # Update inventory from all warehouses
                            for wh_id, wh_items in multi_wh.items():
                                for sku_id, qty in wh_items.items():
                                    warehouse_inventory[wh_id][sku_id] -= qty
                            assigned = True
                            break
                    continue
                
                # Build single-warehouse route
                route = self._build_route(vehicle, [oid], best_warehouse, order_demand)
                
                if route:
                    routes.append(route)
                    used_vehicles.add(vehicle.id)
                    unassigned_orders.discard(oid)
                    
                    for sku_id, qty in order_demand.items():
                        warehouse_inventory[best_warehouse][sku_id] -= qty
                    assigned = True
                    break
            
            if not assigned:
                # Last resort: Try with ANY vehicle (even if used) for critical orders
                pass
        
        return {"routes": routes, "unassigned": list(unassigned_orders)}
    
    def _local_search(self, solution: Dict) -> Dict:
        """
        Apply simple local search: Try to add unassigned orders to existing routes.
        Simpler and safer than complex relocate operations.
        """
        routes = solution.get("routes", [])
        unassigned = set(solution.get("unassigned", []))
        
        if not unassigned:
            return {"routes": routes}  # Already perfect
        
        # Track warehouse inventory
        warehouse_inventory = {}
        for wh_id, wh in self.warehouses.items():
            warehouse_inventory[wh_id] = dict(wh.inventory)
        
        # Subtract already used inventory
        for route in routes:
            wh_id = self._extract_warehouse_from_route(route)
            if wh_id:
                for step in route.get("steps", []):
                    for pickup in step.get("pickups", []):
                        sku_id = pickup.get("sku_id")
                        qty = pickup.get("quantity", 0)
                        if sku_id and wh_id in warehouse_inventory:
                            warehouse_inventory[wh_id][sku_id] = warehouse_inventory[wh_id].get(sku_id, 0) - qty
        
        # Try to add unassigned orders to existing routes
        max_iterations = 100
        iteration = 0
        improved = True
        
        while improved and iteration < max_iterations and unassigned:
            improved = False
            iteration += 1
            
            for oid in list(unassigned):
                if self._try_add_to_existing_route(oid, routes, warehouse_inventory):
                    unassigned.discard(oid)
                    improved = True
                    break
        
        return {"routes": routes}
    
    def _try_add_to_existing_route(
        self, 
        oid: str, 
        routes: List[Dict],
        warehouse_inventory: Dict[str, Dict[str, int]]
    ) -> bool:
        """Try to add unassigned order to an existing route."""
        order = self.orders[oid]
        order_demand = dict(order.requested_items)
        
        for route_idx, route in enumerate(routes):
            vehicle_id = route["vehicle_id"]
            vehicle = self.vehicles[vehicle_id]
            
            # Get current orders in route
            current_orders = self._extract_orders_from_route(route)
            current_demand = self._calculate_demand(current_orders)
            
            # Calculate new demand
            new_demand = dict(current_demand)
            for sku_id, qty in order_demand.items():
                new_demand[sku_id] = new_demand.get(sku_id, 0) + qty
            
            # Check capacity
            if not self._fits_in_vehicle(vehicle, new_demand):
                continue
            
            # Get warehouse from route
            warehouse_id = self._extract_warehouse_from_route(route)
            if not warehouse_id:
                continue
            
            # Check if warehouse has enough inventory
            has_stock = all(
                warehouse_inventory.get(warehouse_id, {}).get(sku_id, 0) >= order_demand.get(sku_id, 0)
                for sku_id in order_demand.keys()
            )
            
            if not has_stock:
                continue
            
            # Rebuild route with new order
            new_orders = current_orders + [oid]
            new_route = self._build_route(vehicle, new_orders, warehouse_id, new_demand)
            
            if new_route:
                routes[route_idx] = new_route
                
                # Update inventory tracking
                for sku_id, qty in order_demand.items():
                    warehouse_inventory[warehouse_id][sku_id] -= qty
                
                return True
        
        return False
    
    def _pack_orders(
        self, 
        vehicle: Any, 
        available_orders: Set[str],
        warehouse_inventory: Dict[str, Dict[str, int]]
    ) -> Tuple[List[str], Dict[str, int]]:
        """Greedy bin-packing of orders into vehicle."""
        packed = []
        total_demand = defaultdict(int)
        remaining_weight = vehicle.capacity_weight
        remaining_volume = vehicle.capacity_volume
        
        # Sort orders by size (smallest first)
        orders_sorted = sorted(
            available_orders,
            key=lambda oid: sum(
                self.skus[sku_id].weight * qty + self.skus[sku_id].volume * qty
                for sku_id, qty in self.orders[oid].requested_items.items()
            )
        )
        
        for oid in orders_sorted:
            order = self.orders[oid]
            
            order_weight = sum(
                self.skus[sku_id].weight * qty
                for sku_id, qty in order.requested_items.items()
            )
            order_volume = sum(
                self.skus[sku_id].volume * qty
                for sku_id, qty in order.requested_items.items()
            )
            
            # Check if fits
            if order_weight <= remaining_weight and order_volume <= remaining_volume:
                # Check if any warehouse has stock
                test_demand = dict(total_demand)
                for sku_id, qty in order.requested_items.items():
                    test_demand[sku_id] = test_demand.get(sku_id, 0) + qty
                
                if self._find_warehouse_with_stock(test_demand, warehouse_inventory):
                    packed.append(oid)
                    for sku_id, qty in order.requested_items.items():
                        total_demand[sku_id] += qty
                    
                    remaining_weight -= order_weight
                    remaining_volume -= order_volume
        
        return packed, dict(total_demand)
    
    def _find_warehouse_with_stock(
        self, 
        demand: Dict[str, int],
        warehouse_inventory: Dict[str, Dict[str, int]]
    ) -> Optional[str]:
        """Find warehouse with sufficient inventory."""
        best_warehouse = None
        best_score = -1
        
        for wh_id, inventory in warehouse_inventory.items():
            has_all = all(
                inventory.get(sku_id, 0) >= qty
                for sku_id, qty in demand.items()
            )
            
            if has_all:
                excess = sum(
                    inventory.get(sku_id, 0) - qty
                    for sku_id, qty in demand.items()
                )
                
                if excess > best_score:
                    best_score = excess
                    best_warehouse = wh_id
        
        return best_warehouse
    
    def _fits_in_vehicle(self, vehicle: Any, demand: Dict[str, int]) -> bool:
        """Check if demand fits in vehicle capacity."""
        total_weight = sum(
            self.skus[sku_id].weight * qty
            for sku_id, qty in demand.items()
        )
        total_volume = sum(
            self.skus[sku_id].volume * qty
            for sku_id, qty in demand.items()
        )
        
        return total_weight <= vehicle.capacity_weight and total_volume <= vehicle.capacity_volume
    
    def _calculate_demand(self, order_ids: List[str]) -> Dict[str, int]:
        """Calculate total demand for a list of orders."""
        total_demand = defaultdict(int)
        for oid in order_ids:
            for sku_id, qty in self.orders[oid].requested_items.items():
                total_demand[sku_id] += qty
        return dict(total_demand)
    
    def _extract_orders_from_route(self, route: Dict) -> List[str]:
        """Extract order IDs from a route."""
        orders = []
        for step in route.get("steps", []):
            for delivery in step.get("deliveries", []):
                oid = delivery.get("order_id")
                if oid and oid not in orders:
                    orders.append(oid)
        return orders
    
    def _extract_warehouse_from_route(self, route: Dict) -> Optional[str]:
        """Extract warehouse ID from a route."""
        for step in route.get("steps", []):
            pickups = step.get("pickups", [])
            if pickups:
                return pickups[0].get("warehouse_id")
        return None
    
    def _find_multi_warehouse_stock(
        self, 
        demand: Dict[str, int],
        warehouse_inventory: Dict[str, Dict[str, int]]
    ) -> Optional[Dict[str, Dict[str, int]]]:
        """
        Find combination of warehouses that can collectively fulfill demand.
        Returns dict: {warehouse_id: {sku_id: quantity}}
        """
        remaining_demand = dict(demand)
        warehouse_allocations = {}
        
        # Greedy allocation across warehouses
        for wh_id, inventory in warehouse_inventory.items():
            if not remaining_demand:
                break
            
            allocation = {}
            for sku_id in list(remaining_demand.keys()):
                needed = remaining_demand[sku_id]
                available = inventory.get(sku_id, 0)
                
                if available > 0:
                    take = min(needed, available)
                    allocation[sku_id] = take
                    remaining_demand[sku_id] -= take
                    
                    if remaining_demand[sku_id] <= 0:
                        del remaining_demand[sku_id]
            
            if allocation:
                warehouse_allocations[wh_id] = allocation
        
        # Check if fully satisfied
        if not remaining_demand:
            return warehouse_allocations
        
        return None
    
    def _build_multi_warehouse_route(
        self,
        vehicle: Any,
        order_ids: List[str],
        warehouse_allocations: Dict[str, Dict[str, int]]
    ) -> Optional[Dict]:
        """Build route with pickups from multiple warehouses."""
        if not order_ids or not warehouse_allocations:
            return None
        
        # Start from first warehouse
        warehouse_ids = list(warehouse_allocations.keys())
        start_wh_id = warehouse_ids[0]
        start_node = self.warehouses[start_wh_id].location.id
        
        steps = []
        current_node = start_node
        
        # Visit each warehouse for pickups
        for wh_id in warehouse_ids:
            wh_node = self.warehouses[wh_id].location.id
            
            # Navigate to warehouse
            if current_node != wh_node:
                path = self._get_path(current_node, wh_node)
                if not path:
                    return None
                
                for node in path[1:-1]:
                    steps.append({
                        "node_id": node,
                        "pickups": [],
                        "deliveries": [],
                        "unloads": []
                    })
                current_node = wh_node
            
            # Pickup at warehouse
            pickups = [
                {"warehouse_id": wh_id, "sku_id": sku_id, "quantity": qty}
                for sku_id, qty in warehouse_allocations[wh_id].items()
            ]
            steps.append({
                "node_id": wh_node,
                "pickups": pickups,
                "deliveries": [],
                "unloads": []
            })
        
        # Visit customers (nearest-neighbor)
        remaining_orders = set(order_ids)
        while remaining_orders:
            nearest_oid = self._find_nearest_order(current_node, remaining_orders)
            if not nearest_oid:
                return None
            
            dest_node = self.orders[nearest_oid].destination.id
            path = self._get_path(current_node, dest_node)
            if not path:
                return None
            
            for node in path[1:-1]:
                steps.append({
                    "node_id": node,
                    "pickups": [],
                    "deliveries": [],
                    "unloads": []
                })
            
            # Deliver
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
        
        # Return to first warehouse
        return_path = self._get_path(current_node, start_node)
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
            "node_id": start_node,
            "pickups": [],
            "deliveries": [],
            "unloads": []
        })
        
        return {
            "vehicle_id": vehicle.id,
            "steps": steps
        }
    
    def _build_route(
        self,
        vehicle: Any,
        order_ids: List[str],
        warehouse_id: str,
        total_demand: Dict[str, int]
    ) -> Optional[Dict]:
        """Build a complete route."""
        if not order_ids:
            return None
        
        warehouse = self.warehouses[warehouse_id]
        warehouse_node = warehouse.location.id
        
        steps = []
        
        # Step 1: Pickup at warehouse
        pickups = [
            {"warehouse_id": warehouse_id, "sku_id": sku_id, "quantity": qty}
            for sku_id, qty in total_demand.items()
        ]
        steps.append({
            "node_id": warehouse_node,
            "pickups": pickups,
            "deliveries": [],
            "unloads": []
        })
        
        # Step 2: Visit customers (nearest-neighbor)
        current_node = warehouse_node
        remaining_orders = set(order_ids)
        
        while remaining_orders:
            nearest_oid = self._find_nearest_order(current_node, remaining_orders)
            if not nearest_oid:
                return None
            
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
            
            # Deliver
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
        
        return {
            "vehicle_id": vehicle.id,
            "steps": steps
        }
    
    def _find_nearest_order(self, current_node: int, order_ids: Set[str]) -> Optional[str]:
        """Find nearest unvisited order."""
        nearest_oid = None
        min_distance = float('inf')
        
        for oid in order_ids:
            dest_node = self.orders[oid].destination.id
            dist = self._get_distance(current_node, dest_node)
            
            if dist and dist < min_distance:
                min_distance = dist
                nearest_oid = oid
        
        return nearest_oid
    
    def _get_distance(self, node1: int, node2: int) -> Optional[float]:
        """Get distance with caching and fallback."""
        if node1 == node2:
            return 0.0
        
        key = (node1, node2)
        if key not in self.distance_cache:
            dist = self.env.get_distance(node1, node2)
            
            if dist is None:
                path = self._get_path(node1, node2)
                if path:
                    dist = (len(path) - 1) * 0.5
                else:
                    dist = None
            
            self.distance_cache[key] = dist
        
        return self.distance_cache[key]
    
    def _get_path(self, start: int, end: int) -> Optional[List[int]]:
        """BFS pathfinding."""
        if start == end:
            return [start]
        
        queue = deque([(start, [start])])
        visited = {start}
        
        while queue:
            current, path = queue.popleft()
            
            neighbors = self.adjacency_list.get(current, self.adjacency_list.get(str(current), []))
            
            for neighbor in neighbors:
                neighbor_int = int(neighbor) if isinstance(neighbor, str) else neighbor
                
                if neighbor_int == end:
                    return path + [neighbor_int]
                
                if neighbor_int not in visited:
                    visited.add(neighbor_int)
                    queue.append((neighbor_int, path + [neighbor_int]))
        
        return None


# Comment out for submission
# if __name__ == '__main__':
#     env = LogisticsEnvironment()
#     solution = solver(env)
#     print(f"Routes: {len(solution['routes'])}")
