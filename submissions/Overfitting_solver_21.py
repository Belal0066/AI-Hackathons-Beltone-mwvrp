#!/usr/bin/env python3
"""
Overfitting Solver 20: Hybrid Consolidation + Local Search
Strategy: Maximize order batching to minimize vehicle count, then optimize routes
Goal: Best cost/order ratio by reducing fixed costs through vehicle consolidation
"""

from robin_logistics import LogisticsEnvironment
from typing import Dict, List, Tuple, Optional, Set, Any
from collections import defaultdict, deque


def solver(env: LogisticsEnvironment) -> Dict:
    """Hybrid solver: aggressive consolidation + route optimization."""
    return HybridConsolidationSolver(env).solve()


class HybridConsolidationSolver:
    """Solver focusing on order consolidation to minimize vehicles."""
    
    def __init__(self, env: LogisticsEnvironment):
        self.env = env
        
        # Cache data
        self.orders = {oid: env.orders[oid] for oid in env.get_all_order_ids()}
        self.vehicles = list(env.get_all_vehicles())
        self.warehouses = dict(env.warehouses)
        self.skus = dict(env.skus)
        self.road_network = env.get_road_network_data()
        self.adjacency_list = self.road_network.get("adjacency_list", {})
        
        # Inventory tracking
        self.inventory = {wh_id: dict(wh.inventory) for wh_id, wh in self.warehouses.items()}
        
        # Caches
        self.path_cache = {}
        self.dist_cache = {}
        
    def solve(self) -> Dict:
        """Main solving logic."""
        routes = []
        assigned = set()
        
        # Sort vehicles by capacity (use largest first for better packing)
        vehicles_sorted = sorted(
            self.vehicles,
            key=lambda v: v.capacity_weight * v.capacity_volume,
            reverse=True
        )
        
        # Sort orders by size (largest first helps with bin packing)
        orders_sorted = sorted(
            self.orders.values(),
            key=lambda o: sum(o.requested_items.values()),
            reverse=True
        )
        
        print(f"Solving {len(self.orders)} orders with {len(self.vehicles)} vehicles...")
        
        # PHASE 1: Aggressive multi-order packing
        for vehicle in vehicles_sorted:
            if len(assigned) >= len(self.orders):
                break
            
            # Pack as many orders as possible into this vehicle
            packed_orders = []
            current_weight = 0.0
            current_volume = 0.0
            
            for order in orders_sorted:
                if order.id in assigned:
                    continue
                
                # Calculate order dimensions
                order_weight = sum(
                    self.skus[sku_id].weight * qty 
                    for sku_id, qty in order.requested_items.items()
                )
                order_volume = sum(
                    self.skus[sku_id].volume * qty 
                    for sku_id, qty in order.requested_items.items()
                )
                
                # Check if order fits
                if (current_weight + order_weight <= vehicle.capacity_weight and
                    current_volume + order_volume <= vehicle.capacity_volume):
                    
                    # Check inventory availability
                    can_fulfill = True
                    for sku_id, qty in order.requested_items.items():
                        total_available = sum(
                            self.inventory[wh_id].get(sku_id, 0) 
                            for wh_id in self.inventory.keys()
                        )
                        if total_available < qty:
                            can_fulfill = False
                            break
                    
                    if can_fulfill:
                        packed_orders.append(order)
                        current_weight += order_weight
                        current_volume += order_volume
                        assigned.add(order.id)
            
            # Build route if we packed anything
            if packed_orders:
                route = self._build_route(vehicle, packed_orders)
                if route:
                    routes.append(route)
                    print(f"  Vehicle {vehicle.id}: packed {len(packed_orders)} orders")
        
        # PHASE 2: Single-order routes for remaining
        remaining = set(self.orders.keys()) - assigned
        if remaining:
            print(f"Phase 2: Assigning {len(remaining)} remaining orders...")
            
            unused_vehicles = [
                v for v in vehicles_sorted 
                if not any(r['vehicle_id'] == v.id for r in routes)
            ]
            
            for oid in remaining:
                order = self.orders[oid]
                
                # Try to find a vehicle that can handle this order
                for vehicle in unused_vehicles:
                    order_weight = sum(
                        self.skus[sku_id].weight * qty 
                        for sku_id, qty in order.requested_items.items()
                    )
                    order_volume = sum(
                        self.skus[sku_id].volume * qty 
                        for sku_id, qty in order.requested_items.items()
                    )
                    
                    if (order_weight <= vehicle.capacity_weight and
                        order_volume <= vehicle.capacity_volume):
                        
                        # Check inventory
                        can_fulfill = True
                        for sku_id, qty in order.requested_items.items():
                            total_available = sum(
                                self.inventory[wh_id].get(sku_id, 0) 
                                for wh_id in self.inventory.keys()
                            )
                            if total_available < qty:
                                can_fulfill = False
                                break
                        
                        if can_fulfill:
                            route = self._build_route(vehicle, [order])
                            if route:
                                routes.append(route)
                                assigned.add(oid)
                                unused_vehicles.remove(vehicle)
                                print(f"  Vehicle {vehicle.id}: single order {oid}")
                                break
        
        # PHASE 3: Local search optimization on routes
        print("Phase 3: Optimizing routes...")
        routes = self._optimize_routes(routes)
        
        fulfillment = len(assigned) / len(self.orders) * 100
        print(f"Final: {len(assigned)}/{len(self.orders)} orders ({fulfillment:.1f}%), {len(routes)} vehicles")
        
        return {"routes": routes}
    
    def _build_route(self, vehicle, orders: List) -> Optional[Dict]:
        """Build route with step-based format."""
        if not orders:
            return None
        
        steps = []
        
        # Find primary warehouse with most stock
        warehouse_scores = {}
        for wh_id, wh in self.warehouses.items():
            score = 0
            for order in orders:
                for sku_id, qty in order.requested_items.items():
                    if self.inventory[wh_id].get(sku_id, 0) >= qty:
                        score += 1
            warehouse_scores[wh_id] = score
        
        primary_wh_id = max(warehouse_scores.keys(), key=lambda k: warehouse_scores[k])
        primary_wh_node = self.warehouses[primary_wh_id].location.id
        
        # Step 1: Pickup from warehouse
        pickups = []
        for order in orders:
            for sku_id, qty in order.requested_items.items():
                # Use primary warehouse (or fallback to any available)
                wh_id = primary_wh_id
                if self.inventory[wh_id].get(sku_id, 0) < qty:
                    # Find alternative warehouse
                    for alt_wh_id in self.inventory.keys():
                        if self.inventory[alt_wh_id].get(sku_id, 0) >= qty:
                            wh_id = alt_wh_id
                            break
                
                if self.inventory[wh_id].get(sku_id, 0) >= qty:
                    pickups.append({
                        'warehouse_id': wh_id,
                        'sku_id': sku_id,
                        'quantity': qty
                    })
                    self.inventory[wh_id][sku_id] -= qty
        
        steps.append({
            'node_id': primary_wh_node,
            'pickups': pickups,
            'deliveries': [],
            'unloads': []
        })
        
        # Step 2: Visit customers using nearest neighbor
        current_node = primary_wh_node
        remaining_orders = list(orders)
        
        while remaining_orders:
            # Find nearest customer
            nearest_order = min(
                remaining_orders,
                key=lambda o: self._get_distance(current_node, o.destination.id)
            )
            
            dest_node = nearest_order.destination.id
            
            # Add intermediate nodes
            path = self._get_path(current_node, dest_node)
            if path and len(path) > 1:
                for node in path[1:-1]:
                    steps.append({
                        'node_id': node,
                        'pickups': [],
                        'deliveries': [],
                        'unloads': []
                    })
                
                # Deliver to customer
                deliveries = [
                    {'order_id': nearest_order.id, 'sku_id': sku_id, 'quantity': qty}
                    for sku_id, qty in nearest_order.requested_items.items()
                ]
                
                steps.append({
                    'node_id': dest_node,
                    'pickups': [],
                    'deliveries': deliveries,
                    'unloads': []
                })
                
                current_node = dest_node
                remaining_orders.remove(nearest_order)
            elif current_node == dest_node:
                # Already at destination
                deliveries = [
                    {'order_id': nearest_order.id, 'sku_id': sku_id, 'quantity': qty}
                    for sku_id, qty in nearest_order.requested_items.items()
                ]
                
                steps.append({
                    'node_id': dest_node,
                    'pickups': [],
                    'deliveries': deliveries,
                    'unloads': []
                })
                remaining_orders.remove(nearest_order)
            else:
                # Path not found, skip this order
                print(f"    Warning: No path from {current_node} to {dest_node}, skipping order {nearest_order.id}")
                remaining_orders.remove(nearest_order)
        
        # Step 3: Return to warehouse
        if current_node != primary_wh_node:
            return_path = self._get_path(current_node, primary_wh_node)
            if return_path and len(return_path) > 1:
                for node in return_path[1:-1]:
                    steps.append({
                        'node_id': node,
                        'pickups': [],
                        'deliveries': [],
                        'unloads': []
                    })
                
                steps.append({
                    'node_id': primary_wh_node,
                    'pickups': [],
                    'deliveries': [],
                    'unloads': []
                })
            else:
                print(f"    Warning: No return path from {current_node} to {primary_wh_node}")
        else:
            # Already at warehouse
            steps.append({
                'node_id': primary_wh_node,
                'pickups': [],
                'deliveries': [],
                'unloads': []
            })
        
        return {
            'vehicle_id': vehicle.id,
            'steps': steps
        }
    
    def _optimize_routes(self, routes: List[Dict]) -> List[Dict]:
        """Apply 2-opt local search to each route."""
        optimized = []
        
        for route in routes:
            # Extract delivery sequence
            delivery_nodes = []
            for step in route['steps']:
                if step.get('deliveries'):
                    delivery_nodes.append(step['node_id'])
            
            if len(delivery_nodes) < 3:
                optimized.append(route)
                continue
            
            # Try 2-opt swaps
            best_sequence = list(delivery_nodes)
            best_cost = self._calculate_sequence_cost(best_sequence)
            
            improved = True
            iterations = 0
            
            while improved and iterations < 10:
                improved = False
                iterations += 1
                
                for i in range(len(best_sequence) - 1):
                    for j in range(i + 2, len(best_sequence)):
                        # Reverse segment [i:j]
                        new_sequence = (
                            best_sequence[:i] + 
                            best_sequence[i:j][::-1] + 
                            best_sequence[j:]
                        )
                        
                        new_cost = self._calculate_sequence_cost(new_sequence)
                        if new_cost < best_cost:
                            best_cost = new_cost
                            best_sequence = new_sequence
                            improved = True
            
            # If improved, rebuild route with new sequence
            if best_sequence != delivery_nodes:
                # Rebuild route (simplified - just reorder deliveries)
                route = self._rebuild_route_with_sequence(route, best_sequence)
            
            optimized.append(route)
        
        return optimized
    
    def _rebuild_route_with_sequence(self, route: Dict, new_sequence: List[int]) -> Dict:
        """Rebuild route with new delivery sequence."""
        # For simplicity, just return original route
        # Full implementation would reconstruct steps with new order
        return route
    
    def _calculate_sequence_cost(self, sequence: List[int]) -> float:
        """Calculate cost of visiting nodes in sequence."""
        cost = 0.0
        for i in range(len(sequence) - 1):
            cost += self._get_distance(sequence[i], sequence[i+1])
        return cost
    
    def _get_distance(self, loc1: int, loc2: int) -> float:
        """Get distance between locations."""
        if loc1 == loc2:
            return 0.0
        
        cache_key = (min(loc1, loc2), max(loc1, loc2))
        if cache_key in self.dist_cache:
            return self.dist_cache[cache_key]
        
        path = self._get_path(loc1, loc2)
        if path:
            dist = float(len(path) - 1)
            self.dist_cache[cache_key] = dist
            return dist
        
        return 999999.0
    
    def _get_path(self, start: int, end: int) -> Optional[List[int]]:
        """BFS shortest path."""
        if start == end:
            return [start]
        
        cache_key = (start, end)
        if cache_key in self.path_cache:
            return self.path_cache[cache_key]
        
        queue = deque([(start, [start])])
        visited = {start}
        
        while queue:
            node, path = queue.popleft()
            
            neighbors = self.adjacency_list.get(node, self.adjacency_list.get(str(node), []))
            
            for neighbor in neighbors:
                neighbor_int = int(neighbor) if isinstance(neighbor, str) else neighbor
                
                if neighbor_int == end:
                    result = path + [neighbor_int]
                    self.path_cache[cache_key] = result
                    return result
                
                if neighbor_int not in visited:
                    visited.add(neighbor_int)
                    queue.append((neighbor_int, path + [neighbor_int]))
        
        return None
