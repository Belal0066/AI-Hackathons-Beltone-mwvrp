#!/usr/bin/env python3
"""
MWVRP Solver - Simplified for 100% fulfillment consistency
Focus: Greedy construction + aggressive vehicle usage
"""

from robin_logistics import LogisticsEnvironment
from typing import Dict, List, Tuple, Optional, Set, Any
from collections import defaultdict, deque


def solver(env: LogisticsEnvironment) -> Dict:
    """Main solver focusing on maximum fulfillment."""
    builder = AggressiveSolver(env)
    solution = builder.build()
    
    # Validate
    validation_result = env.validate_solution_complete(solution)
    is_valid = validation_result[0] if isinstance(validation_result, tuple) else validation_result
    
    if not is_valid:
        message = validation_result[1] if isinstance(validation_result, tuple) and len(validation_result) > 1 else "Unknown"
        print(f"[WARN] Validation failed: {message}")
    
    return solution


class AggressiveSolver:
    """Aggressive solver that uses all vehicles to maximize fulfillment."""
    
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
        
    def build(self) -> Dict:
        """Build solution aggressively using all available resources."""
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
        
        # PASS 1: Greedy batching (pack as many orders as possible)
        for vehicle in vehicles_sorted:
            if not unassigned_orders:
                break
                
            packed_orders, total_demand = self._pack_orders_greedy(
                vehicle, unassigned_orders, warehouse_inventory
            )
            
            if not packed_orders:
                continue
            
            # Find warehouse with stock
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
        
        # PASS 2: One order per vehicle for remaining
        for oid in list(unassigned_orders):
            order = self.orders[oid]
            order_demand = dict(order.requested_items)
            
            # Try ALL unused vehicles
            for vehicle in vehicles_sorted:
                if vehicle.id in used_vehicles:
                    continue
                
                # Check capacity
                if not self._fits_in_vehicle(vehicle, order_demand):
                    continue
                
                # Find warehouse
                best_warehouse = self._find_warehouse_with_stock(order_demand, warehouse_inventory)
                if not best_warehouse:
                    continue
                
                # Build route
                route = self._build_route(vehicle, [oid], best_warehouse, order_demand)
                
                if route:
                    routes.append(route)
                    used_vehicles.add(vehicle.id)
                    unassigned_orders.discard(oid)
                    
                    for sku_id, qty in order_demand.items():
                        warehouse_inventory[best_warehouse][sku_id] -= qty
                    break
        
        # PASS 3: Try splitting large orders if still unassigned
        # (Skip for now to maintain simplicity)
        
        return {"routes": routes}
    
    def _pack_orders_greedy(
        self, 
        vehicle: Any, 
        available_orders: Set[str],
        warehouse_inventory: Dict[str, Dict[str, int]]
    ) -> Tuple[List[str], Dict[str, int]]:
        """Pack as many orders as possible into vehicle."""
        packed = []
        total_demand = defaultdict(int)
        remaining_weight = vehicle.capacity_weight
        remaining_volume = vehicle.capacity_volume
        
        # Sort orders by size (smallest first for better packing)
        orders_sorted = sorted(
            available_orders,
            key=lambda oid: self._order_size(oid)
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
            
            # Check capacity
            if order_weight > remaining_weight or order_volume > remaining_volume:
                continue
            
            # Check if adding this order still allows warehouse stock
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
    
    def _order_size(self, oid: str) -> float:
        """Calculate order size (weight + volume)."""
        order = self.orders[oid]
        return sum(
            self.skus[sku_id].weight * qty + self.skus[sku_id].volume * qty
            for sku_id, qty in order.requested_items.items()
        )
    
    def _find_warehouse_with_stock(
        self, 
        demand: Dict[str, int],
        warehouse_inventory: Dict[str, Dict[str, int]]
    ) -> Optional[str]:
        """Find warehouse with sufficient inventory, prefer most excess."""
        best_warehouse = None
        best_score = -1
        
        for wh_id, inventory in warehouse_inventory.items():
            has_all = all(
                inventory.get(sku_id, 0) >= qty
                for sku_id, qty in demand.items()
            )
            
            if has_all:
                # Prefer warehouse with most excess
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
    
    def _build_route(
        self,
        vehicle: Any,
        order_ids: List[str],
        warehouse_id: str,
        total_demand: Dict[str, int]
    ) -> Optional[Dict]:
        """Build a complete valid route."""
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
