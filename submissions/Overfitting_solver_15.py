#!/usr/bin/env python3
"""
Overfitting_solver_15: Ultra-Fast Aggressive Heuristic Solver
Target: Top 100 ranking (Fulfillment >90%, Cost/Order <$50, Time <1s)

Strategy:
- Drop complex pathfinding (use direct distance only)
- Aggressive fulfillment (deliver everything possible)
- Simple heuristics: nearest warehouse + nearest vehicle
- Cache everything, no nested loops
- Partial deliveries > zero deliveries
"""

from robin_logistics import LogisticsEnvironment
from typing import Dict, List, Optional, Set, Any
from collections import defaultdict, deque


def solver(env: LogisticsEnvironment) -> Dict:
    """Ultra-fast aggressive heuristic solver."""
    return AggressiveHeuristicSolver(env).solve()


class AggressiveHeuristicSolver:
    """Fast heuristic solver prioritizing fulfillment over cost."""
    
    def __init__(self, env: LogisticsEnvironment):
        self.env = env
        
        # Pre-cache all data (do this ONCE)
        self.orders = {oid: env.orders[oid] for oid in env.get_all_order_ids()}
        self.vehicles = list(env.get_all_vehicles())
        self.warehouses = dict(env.warehouses)
        self.skus = dict(env.skus)
        
        # Pre-cache inventories
        self.inventory = {wh_id: dict(wh.inventory) for wh_id, wh in self.warehouses.items()}
        
        # Pre-cache vehicle home nodes
        self.vehicle_homes = {}
        for v in self.vehicles:
            try:
                home_node = env.get_vehicle_home_warehouse(v.id)
                self.vehicle_homes[v.id] = home_node
            except:
                # Fallback: assume first warehouse
                self.vehicle_homes[v.id] = list(self.warehouses.values())[0].location.id
        
        # Distance cache
        self.distance_cache = {}
        
        # Road network for pathfinding
        self.road_network = env.get_road_network_data()
        self.adjacency_list = self.road_network.get("adjacency_list", {})
        
        # Available vehicles pool
        self.available_vehicles = list(self.vehicles)
    
    def solve(self) -> Dict:
        """Main solving loop."""
        routes = []
        
        # Sort orders by priority (descending by total weight - heavier first)
        orders_sorted = sorted(
            self.orders.keys(),
            key=lambda oid: self._order_priority(oid),
            reverse=True
        )
        
        # Process each order
        for oid in orders_sorted:
            if not self.available_vehicles:
                break  # No more vehicles
            
            order = self.orders[oid]
            order_demand = dict(order.requested_items)
            
            # Find best warehouse with stock
            warehouse_id = self._find_nearest_warehouse_with_stock(order_demand, order.destination.id)
            if not warehouse_id:
                continue  # No warehouse has stock
            
            # Find best vehicle
            vehicle = self._find_nearest_available_vehicle(warehouse_id, order_demand)
            if not vehicle:
                continue  # No vehicle fits
            
            # Build simple route
            route = self._build_simple_route(vehicle, warehouse_id, oid, order_demand)
            if route:
                routes.append(route)
                self.available_vehicles.remove(vehicle)
                
                # Update inventory
                for sku_id, qty in order_demand.items():
                    self.inventory[warehouse_id][sku_id] -= qty
        
        return {"routes": routes}
    
    def _order_priority(self, oid: str) -> float:
        """Calculate order priority (higher = more important)."""
        order = self.orders[oid]
        # Prioritize by total weight (proxy for value)
        return sum(
            self.skus[sku_id].weight * qty
            for sku_id, qty in order.requested_items.items()
        )
    
    def _find_nearest_warehouse_with_stock(
        self, 
        demand: Dict[str, int],
        customer_node: int
    ) -> Optional[str]:
        """Find nearest warehouse that has all required stock."""
        best_warehouse = None
        min_distance = float('inf')
        
        for wh_id, wh in self.warehouses.items():
            # Check stock availability
            has_stock = all(
                self.inventory[wh_id].get(sku_id, 0) >= qty
                for sku_id, qty in demand.items()
            )
            
            if not has_stock:
                continue
            
            # Calculate distance (warehouse -> customer)
            wh_node = wh.location.id
            dist = self._get_distance_fast(wh_node, customer_node)
            
            if dist < min_distance:
                min_distance = dist
                best_warehouse = wh_id
        
        return best_warehouse
    
    def _find_nearest_available_vehicle(
        self,
        warehouse_id: str,
        demand: Dict[str, int]
    ) -> Optional[Any]:
        """Find nearest available vehicle that fits demand."""
        warehouse_node = self.warehouses[warehouse_id].location.id
        
        # Calculate demand size
        total_weight = sum(
            self.skus[sku_id].weight * qty
            for sku_id, qty in demand.items()
        )
        total_volume = sum(
            self.skus[sku_id].volume * qty
            for sku_id, qty in demand.items()
        )
        
        best_vehicle = None
        min_distance = float('inf')
        
        for vehicle in self.available_vehicles:
            # Check capacity
            if total_weight > vehicle.capacity_weight or total_volume > vehicle.capacity_volume:
                continue
            
            # Get vehicle home node
            home_node = self.vehicle_homes.get(vehicle.id)
            if home_node is None:
                continue
            
            # Calculate distance (home -> warehouse)
            dist = self._get_distance_fast(home_node, warehouse_node)
            
            if dist < min_distance:
                min_distance = dist
                best_vehicle = vehicle
        
        return best_vehicle
    
    def _get_distance_fast(self, node1: int, node2: int) -> float:
        """Fast distance lookup with caching and fallback."""
        if node1 == node2:
            return 0.0
        
        key = (min(node1, node2), max(node1, node2))
        
        if key not in self.distance_cache:
            try:
                dist = self.env.get_distance(node1, node2)
                if dist is None or dist < 0:
                    # Fallback: assume reasonable distance
                    dist = 10.0  # 10km penalty
            except:
                dist = 10.0
            
            self.distance_cache[key] = dist
        
        return self.distance_cache[key]
    
    def _get_path(self, start: int, end: int) -> Optional[List[int]]:
        """BFS pathfinding for valid route."""
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
    
    def _build_simple_route(
        self,
        vehicle: Any,
        warehouse_id: str,
        order_id: str,
        demand: Dict[str, int]
    ) -> Optional[Dict]:
        """
        Build route with pathfinding for validation.
        """
        try:
            warehouse = self.warehouses[warehouse_id]
            warehouse_node = warehouse.location.id
            customer_node = self.orders[order_id].destination.id
            
            steps = []
            
            # Step 1: Pickup at warehouse
            pickups = [
                {"warehouse_id": warehouse_id, "sku_id": sku_id, "quantity": qty}
                for sku_id, qty in demand.items()
            ]
            steps.append({
                "node_id": warehouse_node,
                "pickups": pickups,
                "deliveries": [],
                "unloads": []
            })
            
            # Step 2: Path to customer (with intermediate nodes)
            path_to_customer = self._get_path(warehouse_node, customer_node)
            if not path_to_customer:
                return None
            
            # Add intermediate nodes
            for node in path_to_customer[1:-1]:
                steps.append({
                    "node_id": node,
                    "pickups": [],
                    "deliveries": [],
                    "unloads": []
                })
            
            # Delivery at customer
            deliveries = [
                {"order_id": order_id, "sku_id": sku_id, "quantity": qty}
                for sku_id, qty in demand.items()
            ]
            steps.append({
                "node_id": customer_node,
                "pickups": [],
                "deliveries": deliveries,
                "unloads": []
            })
            
            # Step 3: Path back to warehouse (with intermediate nodes)
            path_return = self._get_path(customer_node, warehouse_node)
            if not path_return:
                return None
            
            # Add intermediate return nodes
            for node in path_return[1:-1]:
                steps.append({
                    "node_id": node,
                    "pickups": [],
                    "deliveries": [],
                    "unloads": []
                })
            
            # Final return to warehouse
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
        
        except Exception as e:
            # Fail gracefully - skip this route
            return None


# Comment out for submission
# if __name__ == '__main__':
#     env = LogisticsEnvironment()
#     solution = solver(env)
#     print(f"Routes: {len(solution['routes'])}")
