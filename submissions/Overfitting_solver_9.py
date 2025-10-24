#!/usr/bin/env python3
"""
High-Fulfillment MWVRP Solver - Beltone AI Hackathon
Strategy: Prioritize fulfillment ≥95%, then optimize cost

Key improvements:
1. Greedy bin-packing: Pack as many orders as possible per vehicle
2. Multi-warehouse inventory awareness
3. Capacity-constrained batching
4. BFS pathfinding with distance estimation fallback
5. Conservative validation before route creation
"""

from robin_logistics import LogisticsEnvironment
from typing import Dict, List, Tuple, Optional, Set, Any
from collections import defaultdict, deque
import heapq


def solver(env: LogisticsEnvironment) -> Dict:
    """
    Main solver with focus on maximum fulfillment.
    
    Returns:
        Solution dictionary with routes
    """
    builder = HighFulfillmentBuilder(env)
    solution = builder.build_solution()
    
    # Validate
    validation_result = env.validate_solution_complete(solution)
    is_valid = validation_result[0] if isinstance(validation_result, tuple) else validation_result
    
    if not is_valid:
        message = validation_result[1] if isinstance(validation_result, tuple) and len(validation_result) > 1 else "Unknown"
        print(f"[WARN] Validation failed: {message}")
    
    return solution


class HighFulfillmentBuilder:
    """Builds solution with maximum fulfillment priority."""
    
    def __init__(self, env: LogisticsEnvironment):
        self.env = env
        
        # Cache data
        self.orders = {oid: env.orders[oid] for oid in env.get_all_order_ids()}
        self.vehicles = {v.id: v for v in env.get_all_vehicles()}
        self.warehouses = dict(env.warehouses)
        self.skus = dict(env.skus)
        self.road_network = env.get_road_network_data()
        self.adjacency_list = self.road_network.get("adjacency_list", {})
        
        # Track usage
        self.used_vehicles: Set[str] = set()
        self.warehouse_inventory: Dict[str, Dict[str, int]] = {}
        for wh_id, wh in self.warehouses.items():
            self.warehouse_inventory[wh_id] = dict(wh.inventory)
        
        # Distance cache
        self.distance_cache: Dict[Tuple[int, int], Optional[float]] = {}
        
    def build_solution(self) -> Dict:
        """Build solution by greedily packing orders into vehicles."""
        routes = []
        unassigned_orders = set(self.orders.keys())
        
        # Sort vehicles by capacity (largest first for better packing)
        vehicles_sorted = sorted(
            self.vehicles.values(),
            key=lambda v: v.capacity_weight * v.capacity_volume,
            reverse=True
        )
        
        # PASS 1: Greedy bin-packing with large batches
        for vehicle in vehicles_sorted:
            if not unassigned_orders:
                break
                
            # Pack as many orders as possible into this vehicle
            packed_orders, total_demand = self._pack_orders_for_vehicle(
                vehicle, unassigned_orders
            )
            
            if not packed_orders:
                continue
            
            # Find best warehouse with inventory
            best_warehouse = self._find_best_warehouse(total_demand)
            if not best_warehouse:
                continue
            
            # Check if we can build a valid route
            route = self._build_vehicle_route(vehicle, packed_orders, best_warehouse, total_demand)
            
            if route:
                routes.append(route)
                self.used_vehicles.add(vehicle.id)
                unassigned_orders -= set(packed_orders)
                
                # Update inventory
                for sku_id, qty in total_demand.items():
                    self.warehouse_inventory[best_warehouse][sku_id] -= qty
        
        # PASS 2: Single-order routes for remaining orders (fallback)
        for oid in list(unassigned_orders):
            # Find ANY available vehicle that fits
            for vehicle in vehicles_sorted:
                if vehicle.id in self.used_vehicles:
                    continue
                
                order = self.orders[oid]
                order_demand = dict(order.requested_items)
                
                # Check capacity
                order_weight = sum(
                    self.skus[sku_id].weight * qty
                    for sku_id, qty in order_demand.items()
                )
                order_volume = sum(
                    self.skus[sku_id].volume * qty
                    for sku_id, qty in order_demand.items()
                )
                
                if order_weight > vehicle.capacity_weight or order_volume > vehicle.capacity_volume:
                    continue
                
                # Find warehouse with inventory
                best_warehouse = self._find_best_warehouse(order_demand)
                if not best_warehouse:
                    # Try multi-warehouse pickup
                    warehouses_needed = self._find_multi_warehouse_solution(order_demand)
                    if warehouses_needed:
                        route = self._build_multi_warehouse_route(
                            vehicle, [oid], warehouses_needed, order_demand
                        )
                        
                        if route:
                            routes.append(route)
                            self.used_vehicles.add(vehicle.id)
                            unassigned_orders.discard(oid)
                            
                            # Update inventory for all warehouses
                            for wh_id, wh_demand in warehouses_needed.items():
                                for sku_id, qty in wh_demand.items():
                                    self.warehouse_inventory[wh_id][sku_id] -= qty
                            break
                    continue
                
                # Build route
                route = self._build_vehicle_route(vehicle, [oid], best_warehouse, order_demand)
                
                if route:
                    routes.append(route)
                    self.used_vehicles.add(vehicle.id)
                    unassigned_orders.discard(oid)
                    
                    # Update inventory
                    for sku_id, qty in order_demand.items():
                        self.warehouse_inventory[best_warehouse][sku_id] -= qty
                    break
        
        return {"routes": routes}
    
    def _find_multi_warehouse_solution(
        self, 
        demand: Dict[str, int]
    ) -> Optional[Dict[str, Dict[str, int]]]:
        """
        Find combination of warehouses that can collectively fulfill demand.
        
        Returns dict mapping warehouse_id -> {sku_id: quantity} or None
        """
        remaining_demand = dict(demand)
        warehouse_allocations = {}
        
        # Greedy allocation: try each warehouse in order
        for wh_id, inventory in self.warehouse_inventory.items():
            if not remaining_demand:
                break
            
            allocation = {}
            for sku_id, needed_qty in list(remaining_demand.items()):
                available = inventory.get(sku_id, 0)
                if available > 0:
                    take_qty = min(needed_qty, available)
                    allocation[sku_id] = take_qty
                    remaining_demand[sku_id] -= take_qty
                    
                    if remaining_demand[sku_id] <= 0:
                        del remaining_demand[sku_id]
            
            if allocation:
                warehouse_allocations[wh_id] = allocation
        
        # Check if we fulfilled everything
        if not remaining_demand:
            return warehouse_allocations
        
        return None
    
    def _build_multi_warehouse_route(
        self,
        vehicle: Any,
        order_ids: List[str],
        warehouse_allocations: Dict[str, Dict[str, int]],
        total_demand: Dict[str, int]
    ) -> Optional[Dict]:
        """
        Build route with pickups from multiple warehouses.
        
        Returns route dict or None if route cannot be built
        """
        # Start from vehicle's home warehouse if possible, otherwise first warehouse
        warehouse_ids = list(warehouse_allocations.keys())
        start_wh_id = warehouse_ids[0]
        start_node = self.warehouses[start_wh_id].location.id
        
        steps = []
        current_node = start_node
        
        # Visit each warehouse for pickups
        for wh_id in warehouse_ids:
            wh_node = self.warehouses[wh_id].location.id
            
            # Navigate to warehouse if not already there
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
            
            # Pickup at this warehouse
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
        
        # Visit each customer
        for oid in order_ids:
            dest_node = self.orders[oid].destination.id
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
            
            deliveries = [
                {"order_id": oid, "sku_id": sku_id, "quantity": qty}
                for sku_id, qty in self.orders[oid].requested_items.items()
            ]
            steps.append({
                "node_id": dest_node,
                "pickups": [],
                "deliveries": deliveries,
                "unloads": []
            })
            current_node = dest_node
        
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
    
    def _pack_orders_for_vehicle(
        self, 
        vehicle: Any, 
        available_orders: Set[str]
    ) -> Tuple[List[str], Dict[str, int]]:
        """
        Greedy bin-packing: pack orders until capacity is reached.
        
        Returns:
            (list of order IDs, total demand dict)
        """
        packed = []
        total_demand = defaultdict(int)
        remaining_weight = vehicle.capacity_weight
        remaining_volume = vehicle.capacity_volume
        
        # Sort orders by total weight+volume (smallest first for better packing)
        orders_sorted = sorted(
            available_orders,
            key=lambda oid: sum(
                self.skus[sku_id].weight * qty + self.skus[sku_id].volume * qty
                for sku_id, qty in self.orders[oid].requested_items.items()
            )
        )
        
        for oid in orders_sorted:
            order = self.orders[oid]
            
            # Calculate order requirements
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
                packed.append(oid)
                for sku_id, qty in order.requested_items.items():
                    total_demand[sku_id] += qty
                
                remaining_weight -= order_weight
                remaining_volume -= order_volume
        
        return packed, dict(total_demand)
    
    def _find_best_warehouse(self, demand: Dict[str, int]) -> Optional[str]:
        """
        Find warehouse with sufficient inventory for the demand.
        Prioritizes warehouses with most excess inventory.
        
        Returns warehouse_id or None
        """
        best_warehouse = None
        best_score = -1
        
        for wh_id, inventory in self.warehouse_inventory.items():
            # Check if this warehouse has all required SKUs
            has_all = all(
                inventory.get(sku_id, 0) >= qty
                for sku_id, qty in demand.items()
            )
            
            if has_all:
                # Calculate excess inventory (prefer warehouses with more stock)
                excess = sum(
                    inventory.get(sku_id, 0) - qty
                    for sku_id, qty in demand.items()
                )
                
                if excess > best_score:
                    best_score = excess
                    best_warehouse = wh_id
        
        return best_warehouse
    
    def _build_vehicle_route(
        self,
        vehicle: Any,
        order_ids: List[str],
        warehouse_id: str,
        total_demand: Dict[str, int]
    ) -> Optional[Dict]:
        """
        Build a complete route for the vehicle.
        
        Returns route dict or None if route cannot be built
        """
        warehouse = self.warehouses[warehouse_id]
        warehouse_node = warehouse.location.id
        
        # Build path: warehouse -> order1 -> order2 -> ... -> warehouse
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
        
        # Step 2: Visit each customer using nearest-neighbor
        current_node = warehouse_node
        remaining_orders = set(order_ids)
        
        while remaining_orders:
            # Find nearest unvisited order
            nearest_oid = self._find_nearest_order(current_node, remaining_orders)
            if not nearest_oid:
                return None  # No path found
            
            dest_node = self.orders[nearest_oid].destination.id
            
            # Get path
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
        
        return {
            "vehicle_id": vehicle.id,
            "steps": steps
        }
    
    def _find_nearest_order(self, current_node: int, order_ids: Set[str]) -> Optional[str]:
        """Find the nearest unvisited order."""
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
            # Try API
            dist = self.env.get_distance(node1, node2)
            
            # Fallback: BFS hop count * 0.5km
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
            
            # Handle both int and str keys
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
