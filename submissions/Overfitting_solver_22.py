#!/usr/bin/env python3
"""
Overfitting Solver 22: Geographical Clustering + Route Optimization
Strategy: Cluster orders by proximity, pack clusters into vehicles, optimize routes
Goal: Beat Solver 14's $41.72/order by combining low distance with vehicle consolidation
"""

from robin_logistics import LogisticsEnvironment
from typing import Dict, List, Tuple, Optional, Set, Any
from collections import defaultdict, deque
import math


def solver(env: LogisticsEnvironment) -> Dict:
    """Geographical clustering solver with route optimization."""
    return GeographicalClusteringSolver(env).solve()


class GeographicalClusteringSolver:
    """Solver using geographical clustering for efficient routing."""
    
    def __init__(self, env: LogisticsEnvironment):
        self.env = env
        
        # Cache data
        self.orders = {oid: env.orders[oid] for oid in env.get_all_order_ids()}
        self.vehicles = list(env.get_all_vehicles())
        self.warehouses = dict(env.warehouses)
        self.skus = dict(env.skus)
        self.road_network = env.get_road_network_data()
        self.adjacency_list = self.road_network.get("adjacency_list", {})
        self.nodes = dict(env.nodes)
        
        # Inventory tracking
        self.inventory = {wh_id: dict(wh.inventory) for wh_id, wh in self.warehouses.items()}
        
        # Caches
        self.path_cache = {}
        self.dist_cache = {}
        
    def solve(self) -> Dict:
        """Main solving logic with geographical clustering."""
        routes = []
        assigned = set()
        
        print(f"Solving {len(self.orders)} orders with {len(self.vehicles)} vehicles...")
        
        # Sort vehicles by capacity (use largest first)
        vehicles_sorted = sorted(
            self.vehicles,
            key=lambda v: v.capacity_weight * v.capacity_volume,
            reverse=True
        )
        
        # PHASE 1: Cluster orders by geographical proximity
        print("Phase 1: Clustering orders by location...")
        clusters = self._cluster_orders_geographically(num_clusters=4)
        
        print(f"  Created {len(clusters)} clusters: {[len(c) for c in clusters]}")
        
        # PHASE 2: Pack orders greedily from clusters (ignore cluster boundaries if needed)
        print("Phase 2: Packing orders into vehicles...")
        
        # Flatten all cluster orders
        all_clustered_orders = []
        for cluster in clusters:
            all_clustered_orders.extend(cluster)
        
        for vehicle in vehicles_sorted:
            if len(assigned) >= len(self.orders):
                break
            
            # Pack orders into this vehicle
            vehicle_orders = []
            current_weight = 0.0
            current_volume = 0.0
            
            for order in all_clustered_orders:
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
                
                # Check if fits
                if (current_weight + order_weight <= vehicle.capacity_weight and
                    current_volume + order_volume <= vehicle.capacity_volume):
                    
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
                        vehicle_orders.append(order)
                        current_weight += order_weight
                        current_volume += order_volume
                        assigned.add(order.id)
            
            # Build route if we packed anything
            if vehicle_orders:
                route = self._build_optimized_route(vehicle, vehicle_orders)
                if route:
                    routes.append(route)
                    print(f"  Vehicle {vehicle.id}: {len(vehicle_orders)} orders")
        
        # PHASE 3: Handle remaining orders individually
        remaining = set(self.orders.keys()) - assigned
        if remaining:
            print(f"Phase 3: Assigning {len(remaining)} remaining orders...")
            
            unused_vehicles = [
                v for v in vehicles_sorted 
                if not any(r['vehicle_id'] == v.id for r in routes)
            ]
            
            for oid in remaining:
                order = self.orders[oid]
                
                for vehicle in unused_vehicles:
                    if self._order_fits_vehicle(order, vehicle):
                        route = self._build_optimized_route(vehicle, [order])
                        if route:
                            routes.append(route)
                            assigned.add(oid)
                            unused_vehicles.remove(vehicle)
                            break
        
        fulfillment = len(assigned) / len(self.orders) * 100
        print(f"Final: {len(assigned)}/{len(self.orders)} orders ({fulfillment:.1f}%), {len(routes)} vehicles")
        
        return {"routes": routes}
    
    def _cluster_orders_geographically(self, num_clusters: int = 4) -> List[List]:
        """Cluster orders by geographical proximity using simple k-means-like approach."""
        orders_list = list(self.orders.values())
        
        if len(orders_list) <= num_clusters:
            return [[o] for o in orders_list]
        
        # Get order locations
        order_locations = {}
        for order in orders_list:
            node = self.nodes.get(order.destination.id)
            if node:
                order_locations[order.id] = (node.lat, node.lon)
            else:
                order_locations[order.id] = (0, 0)
        
        # Initialize cluster centers (spread across range)
        lats = [loc[0] for loc in order_locations.values()]
        lons = [loc[1] for loc in order_locations.values()]
        
        min_lat, max_lat = min(lats), max(lats)
        min_lon, max_lon = min(lons), max(lons)
        
        # Create initial centers in a grid
        centers = []
        grid_size = int(math.sqrt(num_clusters))
        for i in range(grid_size):
            for j in range(grid_size):
                if len(centers) >= num_clusters:
                    break
                lat = min_lat + (max_lat - min_lat) * (i + 0.5) / grid_size
                lon = min_lon + (max_lon - min_lon) * (j + 0.5) / grid_size
                centers.append((lat, lon))
        
        # Simple k-means (3 iterations)
        clusters = [[] for _ in range(len(centers))]
        
        for iteration in range(3):
            # Assign orders to nearest center
            clusters = [[] for _ in range(len(centers))]
            
            for order in orders_list:
                loc = order_locations[order.id]
                nearest_idx = min(
                    range(len(centers)),
                    key=lambda i: self._euclidean_distance(loc, centers[i])
                )
                clusters[nearest_idx].append(order)
            
            # Update centers
            for i, cluster in enumerate(clusters):
                if cluster:
                    avg_lat = sum(order_locations[o.id][0] for o in cluster) / len(cluster)
                    avg_lon = sum(order_locations[o.id][1] for o in cluster) / len(cluster)
                    centers[i] = (avg_lat, avg_lon)
        
        # Remove empty clusters
        return [c for c in clusters if c]
    
    def _euclidean_distance(self, loc1: Tuple[float, float], loc2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two lat/lon points."""
        return math.sqrt((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2)
    
    def _order_fits_vehicle(self, order, vehicle) -> bool:
        """Check if order fits in vehicle."""
        order_weight = sum(
            self.skus[sku_id].weight * qty 
            for sku_id, qty in order.requested_items.items()
        )
        order_volume = sum(
            self.skus[sku_id].volume * qty 
            for sku_id, qty in order.requested_items.items()
        )
        
        if order_weight > vehicle.capacity_weight or order_volume > vehicle.capacity_volume:
            return False
        
        # Check inventory
        for sku_id, qty in order.requested_items.items():
            total_available = sum(
                self.inventory[wh_id].get(sku_id, 0) 
                for wh_id in self.inventory.keys()
            )
            if total_available < qty:
                return False
        
        return True
    
    def _build_optimized_route(self, vehicle, orders: List) -> Optional[Dict]:
        """Build route with TSP optimization for order sequence."""
        if not orders:
            return None
        
        steps = []
        
        # Find primary warehouse
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
        
        # Pickup from warehouse
        pickups = []
        for order in orders:
            for sku_id, qty in order.requested_items.items():
                wh_id = primary_wh_id
                if self.inventory[wh_id].get(sku_id, 0) < qty:
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
        
        # Optimize order sequence using nearest neighbor TSP
        current_node = primary_wh_node
        remaining_orders = list(orders)
        
        while remaining_orders:
            # Find nearest order
            nearest_order = min(
                remaining_orders,
                key=lambda o: self._get_distance(current_node, o.destination.id)
            )
            
            dest_node = nearest_order.destination.id
            path = self._get_path(current_node, dest_node)
            
            if path and len(path) > 1:
                # Add intermediate nodes
                for node in path[1:-1]:
                    steps.append({
                        'node_id': node,
                        'pickups': [],
                        'deliveries': [],
                        'unloads': []
                    })
                
                # Deliver
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
        
        # Return to warehouse
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
        
        return {
            'vehicle_id': vehicle.id,
            'steps': steps
        }
    
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
