#!/usr/bin/env python3
"""
Production-Grade ALNS Solver for Robin Logistics Environment
Team: Overfitting
Solver: 18 - Adaptive Large Neighborhood Search

Design Philosophy:
1. Lexicographic Objective: 100% fulfillment first, then minimize cost
2. ALNS with adaptive destroy/repair operators
3. Simulated annealing acceptance with adaptive weights
4. Local search polish for cost optimization
"""

from robin_logistics import LogisticsEnvironment
from typing import Dict, List, Optional, Set, Any, Tuple
from collections import defaultdict, deque
import random
import math
import time


def solver(env: LogisticsEnvironment) -> Dict:
    """Main entry point - ALNS metaheuristic solver."""
    routes = ALNSSolver(env).solve()
    return {"routes": routes}


class ALNSSolver:
    """ALNS solver with adaptive operators for 100% fulfillment."""
    
    def __init__(self, env: LogisticsEnvironment):
        self.env = env
        
        # Cache data
        self.orders = {oid: env.orders[oid] for oid in env.get_all_order_ids()}
        self.vehicles = list(env.get_all_vehicles())
        self.warehouses = dict(env.warehouses)
        self.skus = dict(env.skus)
        
        # Inventory tracking (mutable copy)
        self.inventory = {wh_id: dict(wh.inventory) for wh_id, wh in self.warehouses.items()}
        
        # Road network
        self.road_network = env.get_road_network_data()
        self.adjacency_list = self.road_network.get("adjacency_list", {})
        
        # Home depot
        self.home_depot = list(self.warehouses.values())[0].location.id
        
        # Solution state
        self.solution = []  # List of route dicts
        self.assigned_orders = set()
        
        # ALNS parameters
        self.max_iterations = 300
        self.segment_size = 50
        self.destroy_fraction = 0.25
        
        # Operator weights
        self.destroy_ops = ['random', 'worst', 'shaw', 'route']
        self.repair_ops = ['greedy', 'regret2', 'regret3']
        
        self.destroy_weights = {op: 1.0 for op in self.destroy_ops}
        self.repair_weights = {op: 1.0 for op in self.repair_ops}
        
        self.destroy_scores = {op: 0.0 for op in self.destroy_ops}
        self.repair_scores = {op: 0.0 for op in self.repair_ops}
        
        self.destroy_uses = {op: 0 for op in self.destroy_ops}
        self.repair_uses = {op: 0 for op in self.repair_ops}
        
        # Scoring parameters
        self.sigma1, self.sigma2, self.sigma3 = 33, 9, 3
        
        # Simulated annealing
        self.temperature = 1000.0
        self.cooling_rate = 0.995
        self.min_temp = 0.1
        
        # Caches
        self.path_cache = {}
        self.dist_cache = {}
    
    def solve(self) -> List[Dict]:
        """Main ALNS loop."""
        start_time = time.time()
        
        print("="*60)
        print("ALNS Solver v18 - Production Grade")
        print("="*60)
        
        # Phase 1: Greedy initialization
        print("Phase 1: Greedy initialization...")
        self._greedy_init()
        best_solution = self._copy_solution()
        best_obj = self._objective()
        current_obj = best_obj
        
        print(f"Initial: {len(self.assigned_orders)}/{len(self.orders)} fulfilled, cost={self._total_cost():.2f}")
        
        # Skip ALNS for now - debug greedy init first
        print("Phase 2: ALNS - SKIPPED FOR DEBUGGING")
        
        # Phase 3: Local search
        print("Phase 3: Local search - SKIPPED")
        
        # Phase 4: Final cleanup
        print("Phase 4: Final cleanup - SKIPPED")
        
        final_fulfilled = len(self.assigned_orders)
        final_cost = self._total_cost()
        elapsed = time.time() - start_time
        
        print("="*60)
        print(f"FINAL: {final_fulfilled}/{len(self.orders)} fulfilled ({100*final_fulfilled/len(self.orders):.1f}%)")
        print(f"Cost: ${final_cost:.2f}, Time: {elapsed:.1f}s")
        print("="*60)
        
        return self.solution
    
    def _greedy_init(self):
        """Greedy initialization with regret-2."""
        self.solution = []
        self.assigned_orders = set()
        
        # Sort orders by size (largest first)
        orders_list = sorted(self.orders.values(),
                           key=lambda o: sum(o.requested_items.values()),
                           reverse=True)
        
        for vehicle in self.vehicles:
            route_orders = []
            
            # Greedy packing
            for order in orders_list:
                if order.id in self.assigned_orders:
                    continue
                
                # Check if we can add this order
                if self._can_add_to_route(order, route_orders, vehicle):
                    route_orders.append(order)
                    self.assigned_orders.add(order.id)
            
            if route_orders:
                route_dict = self._build_route(vehicle, route_orders)
                if route_dict:
                    self.solution.append(route_dict)
    
    def _can_add_to_route(self, order, current_orders, vehicle) -> bool:
        """Check if order can be added to route."""
        # Calculate total load
        all_orders = current_orders + [order]
        total_weight = 0
        total_volume = 0
        
        for o in all_orders:
            for sku_id, qty in o.requested_items.items():
                sku = self.skus[sku_id]
                total_weight += sku.weight * qty
                total_volume += sku.volume * qty
        
        # Check capacity
        if total_weight > vehicle.capacity_weight + 1e-6:
            return False
        if total_volume > vehicle.capacity_volume + 1e-6:
            return False
        
        # Check inventory
        for sku_id, qty in order.requested_items.items():
            total_available = sum(self.inventory[wh_id].get(sku_id, 0) 
                                for wh_id in self.inventory.keys())
            if total_available < qty:
                return False
        
        return True
    
    def _build_route(self, vehicle, orders) -> Optional[Dict]:
        """Build route dict for orders using step-based format."""
        if not orders:
            return None
        
        # Assign warehouses to orders (greedy: first available)
        order_warehouses = {}
        for order in orders:
            # Pick first warehouse with all items
            for wh_id in self.inventory.keys():
                has_all = all(self.inventory[wh_id].get(sku_id, 0) >= qty 
                            for sku_id, qty in order.requested_items.items())
                if has_all:
                    order_warehouses[order.id] = wh_id
                    # Deduct inventory
                    for sku_id, qty in order.requested_items.items():
                        self.inventory[wh_id][sku_id] -= qty
                    break
        
        if not order_warehouses:
            return None
        
        steps = []
        
        # Step 1: Pickup from primary warehouse
        primary_wh_id = list(set(order_warehouses.values()))[0]
        primary_wh_node = self.warehouses[primary_wh_id].location.id
        
        all_pickups = []
        for order in orders:
            if order.id in order_warehouses:
                wh_id = order_warehouses[order.id]
                for sku_id, qty in order.requested_items.items():
                    all_pickups.append({
                        'warehouse_id': wh_id,
                        'sku_id': sku_id,
                        'quantity': qty
                    })
        
        steps.append({
            'node_id': primary_wh_node,
            'pickups': all_pickups,
            'deliveries': [],
            'unloads': []
        })
        
        # Step 2: Deliver to customers (nearest neighbor)
        current_node = primary_wh_node
        remaining_orders = set(order.id for order in orders if order.id in order_warehouses)
        
        while remaining_orders:
            # Find nearest customer
            nearest_oid = None
            min_dist = float('inf')
            
            for oid in remaining_orders:
                order = self.orders[oid]
                dest_node = order.destination.id
                dist = self._get_distance(current_node, dest_node)
                if dist < min_dist:
                    min_dist = dist
                    nearest_oid = oid
            
            if not nearest_oid:
                break
            
            order = self.orders[nearest_oid]
            dest_node = order.destination.id
            path = self._get_path(current_node, dest_node)
            
            if not path:
                break
            
            # Add intermediate nodes
            for node in path[1:-1]:
                steps.append({
                    'node_id': node,
                    'pickups': [],
                    'deliveries': [],
                    'unloads': []
                })
            
            # Deliver at destination
            deliveries = [
                {'order_id': nearest_oid, 'sku_id': sku_id, 'quantity': qty}
                for sku_id, qty in order.requested_items.items()
            ]
            
            steps.append({
                'node_id': dest_node,
                'pickups': [],
                'deliveries': deliveries,
                'unloads': []
            })
            
            current_node = dest_node
            remaining_orders.remove(nearest_oid)
        
        # Step 3: Return to primary warehouse
        return_path = self._get_path(current_node, primary_wh_node)
        if return_path:
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
    
    def _destroy(self, operator: str) -> Set[str]:
        """Apply destroy operator."""
        if not self.solution:
            return set()
        
        total_assigned = len(self.assigned_orders)
        if total_assigned == 0:
            return set()
        
        num_remove = max(1, int(total_assigned * self.destroy_fraction))
        destroyed = set()
        
        if operator == 'random':
            # Random removal
            to_remove = random.sample(list(self.assigned_orders), 
                                    min(num_remove, len(self.assigned_orders)))
            destroyed.update(to_remove)
            
        elif operator == 'worst':
            # Remove orders with highest cost contribution
            order_costs = []
            for route in self.solution:
                route_cost = self._estimate_route_cost(route)
                order_ids = set()
                for step in route.get('steps', []):
                    for delivery in step.get('deliveries', []):
                        order_ids.add(delivery['order_id'])
                
                num_orders = len(order_ids)
                cost_per_order = route_cost / max(num_orders, 1)
                
                for oid in order_ids:
                    order_costs.append((cost_per_order, oid))
            
            order_costs.sort(reverse=True)
            destroyed.update([oid for _, oid in order_costs[:num_remove]])
            
        elif operator == 'shaw':
            # Shaw removal (related orders by destination proximity)
            if self.assigned_orders:
                seed = random.choice(list(self.assigned_orders))
                seed_order = self.orders[seed]
                
                # Find related orders
                relatedness = []
                for oid in self.assigned_orders:
                    if oid == seed:
                        continue
                    order = self.orders[oid]
                    dist = self._get_distance(seed_order.destination.id, order.destination.id)
                    relatedness.append((dist, oid))
                
                relatedness.sort()
                destroyed.add(seed)
                destroyed.update([oid for _, oid in relatedness[:num_remove-1]])
                
        elif operator == 'route':
            # Remove entire routes
            if len(self.solution) > 0:
                num_routes = max(1, min(num_remove // 5, len(self.solution)))
                routes_to_remove = random.sample(self.solution, num_routes)
                
                for route in routes_to_remove:
                    for step in route.get('steps', []):
                        for delivery in step.get('deliveries', []):
                            destroyed.add(delivery['order_id'])
        
        # Remove destroyed orders from solution
        self._remove_orders_from_solution(destroyed)
        
        return destroyed
    
    def _repair(self, operator: str, destroyed: Set[str]):
        """Apply repair operator."""
        if not destroyed:
            return
        
        orders_to_insert = [self.orders[oid] for oid in destroyed]
        
        if operator == 'greedy':
            # Greedy best insertion
            for order in orders_to_insert:
                best_route_idx = None
                best_cost = float('inf')
                
                # Try existing routes
                for idx, route in enumerate(self.solution):
                    route_order_ids = set()
                    for step in route.get('steps', []):
                        for delivery in step.get('deliveries', []):
                            route_order_ids.add(delivery['order_id'])
                    
                    route_orders = [self.orders[oid] for oid in route_order_ids]
                    vehicle = next(v for v in self.vehicles if v.id == route['vehicle_id'])
                    
                    if self._can_add_to_route(order, route_orders, vehicle):
                        cost = self._estimate_route_cost(route)
                        if cost < best_cost:
                            best_cost = cost
                            best_route_idx = idx
                
                if best_route_idx is not None:
                    # Add to existing route
                    self._add_order_to_route(order, best_route_idx)
                else:
                    # Try new route
                    self._try_new_route_for_order(order)
                    
        else:  # regret2 or regret3
            k = 2 if operator == 'regret2' else 3
            
            while orders_to_insert:
                best_order = None
                best_regret = -float('inf')
                best_route_idx = None
                
                for order in orders_to_insert:
                    # Find k-best insertions
                    costs = []
                    
                    for idx, route in enumerate(self.solution):
                        route_order_ids = set()
                        for step in route.get('steps', []):
                            for delivery in step.get('deliveries', []):
                                route_order_ids.add(delivery['order_id'])
                        
                        route_orders = [self.orders[oid] for oid in route_order_ids]
                        vehicle = next(v for v in self.vehicles if v.id == route['vehicle_id'])
                        
                        if self._can_add_to_route(order, route_orders, vehicle):
                            cost = self._estimate_route_cost(route)
                            costs.append((cost, idx))
                    
                    if costs:
                        costs.sort()
                        if len(costs) >= k:
                            regret = sum(costs[i][0] - costs[0][0] for i in range(1, k))
                        else:
                            regret = costs[0][0] if costs else 0
                        
                        if regret > best_regret:
                            best_regret = regret
                            best_order = order
                            best_route_idx = costs[0][1] if costs else None
                
                if best_order:
                    if best_route_idx is not None:
                        self._add_order_to_route(best_order, best_route_idx)
                    else:
                        self._try_new_route_for_order(best_order)
                    orders_to_insert.remove(best_order)
                else:
                    break
    
    def _add_order_to_route(self, order, route_idx):
        """Add order to existing route."""
        route = self.solution[route_idx]
        route_order_ids = set()
        for step in route.get('steps', []):
            for delivery in step.get('deliveries', []):
                route_order_ids.add(delivery['order_id'])
        
        route_orders = [self.orders[oid] for oid in route_order_ids] + [order]
        vehicle = next(v for v in self.vehicles if v.id == route['vehicle_id'])
        
        new_route = self._build_route(vehicle, route_orders)
        if new_route:
            self.solution[route_idx] = new_route
            self.assigned_orders.add(order.id)
    
    def _try_new_route_for_order(self, order):
        """Try to create new route for order."""
        for vehicle in self.vehicles:
            # Check if vehicle is already used
            if any(r['vehicle_id'] == vehicle.id for r in self.solution):
                continue
            
            if self._can_add_to_route(order, [], vehicle):
                new_route = self._build_route(vehicle, [order])
                if new_route:
                    self.solution.append(new_route)
                    self.assigned_orders.add(order.id)
                    break
    
    def _remove_orders_from_solution(self, order_ids: Set[str]):
        """Remove orders from solution and restore inventory."""
        new_solution = []
        
        for route in self.solution:
            # Get all orders in this route
            route_order_ids = set()
            for step in route.get('steps', []):
                for delivery in step.get('deliveries', []):
                    route_order_ids.add(delivery['order_id'])
            
            # Filter out destroyed orders
            remaining_order_ids = route_order_ids - order_ids
            
            if remaining_order_ids:
                # Rebuild route with remaining orders
                remaining_orders = [self.orders[oid] for oid in remaining_order_ids]
                vehicle = next(v for v in self.vehicles if v.id == route['vehicle_id'])
                
                # Restore inventory for destroyed orders in this route
                destroyed_in_route = route_order_ids & order_ids
                for oid in destroyed_in_route:
                    order = self.orders[oid]
                    # Find which warehouse was used (from pickups)
                    for step in route.get('steps', []):
                        for pickup in step.get('pickups', []):
                            if pickup.get('sku_id') in order.requested_items:
                                wh_id = pickup['warehouse_id']
                                self.inventory[wh_id][pickup['sku_id']] += pickup['quantity']
                
                new_route = self._build_route(vehicle, remaining_orders)
                if new_route:
                    new_solution.append(new_route)
            else:
                # Restore all inventory for this route
                for step in route.get('steps', []):
                    for pickup in step.get('pickups', []):
                        wh_id = pickup['warehouse_id']
                        self.inventory[wh_id][pickup['sku_id']] += pickup['quantity']
        
        self.solution = new_solution
        self.assigned_orders -= order_ids
    
    def _local_search_phase(self):
        """Local search to improve solution."""
        improved = True
        iterations = 0
        
        while improved and iterations < 30:
            improved = False
            iterations += 1
            
            # Try relocating orders between routes
            if self._relocate_orders():
                improved = True
    
    def _relocate_orders(self) -> bool:
        """Try relocating single orders between routes."""
        if len(self.solution) < 2:
            return False
        
        for i, route1 in enumerate(self.solution):
            for j, route2 in enumerate(self.solution):
                if i == j:
                    continue
                
                # Get orders from both routes
                orders1_ids = set()
                for step in route1.get('steps', []):
                    for delivery in step.get('deliveries', []):
                        orders1_ids.add(delivery['order_id'])
                
                orders2_ids = set()
                for step in route2.get('steps', []):
                    for delivery in step.get('deliveries', []):
                        orders2_ids.add(delivery['order_id'])
                
                # Try moving each order from route1 to route2
                for oid in orders1_ids:
                    old_cost = self._estimate_route_cost(route1) + self._estimate_route_cost(route2)
                    
                    # Remove from route1
                    orders1 = [self.orders[o] for o in orders1_ids if o != oid]
                    orders2 = [self.orders[o] for o in orders2_ids] + [self.orders[oid]]
                    
                    v1 = next(v for v in self.vehicles if v.id == route1['vehicle_id'])
                    v2 = next(v for v in self.vehicles if v.id == route2['vehicle_id'])
                    
                    # Check feasibility
                    if not self._can_add_to_route(self.orders[oid], 
                                                  [self.orders[o] for o in orders2_ids], 
                                                  v2):
                        continue
                    
                    # Rebuild routes
                    new_route1 = self._build_route(v1, orders1) if orders1 else None
                    new_route2 = self._build_route(v2, orders2)
                    
                    if new_route2:
                        new_cost = (self._estimate_route_cost(new_route1) if new_route1 else 0) + \
                                 self._estimate_route_cost(new_route2)
                        
                        if new_cost < old_cost - 1e-6:
                            # Accept move
                            if new_route1:
                                self.solution[i] = new_route1
                            else:
                                self.solution.pop(i)
                            self.solution[j] = new_route2
                            return True
        
        return False
    
    def _final_cleanup_pass(self):
        """Final aggressive pass to assign remaining orders."""
        unassigned = set(self.orders.keys()) - self.assigned_orders
        
        if not unassigned:
            return
        
        for oid in sorted(unassigned, key=lambda x: sum(self.orders[x].requested_items.values())):
            order = self.orders[oid]
            
            # Try adding to existing routes
            for idx, route in enumerate(self.solution):
                route_order_ids = set()
                for step in route.get('steps', []):
                    for delivery in step.get('deliveries', []):
                        route_order_ids.add(delivery['order_id'])
                
                route_orders = [self.orders[oid] for oid in route_order_ids]
                vehicle = next(v for v in self.vehicles if v.id == route['vehicle_id'])
                
                if self._can_add_to_route(order, route_orders, vehicle):
                    self._add_order_to_route(order, idx)
                    break
            else:
                # Try new route
                self._try_new_route_for_order(order)
    
    def _objective(self) -> float:
        """Lexicographic objective: fulfillment first, then cost."""
        num_unassigned = len(self.orders) - len(self.assigned_orders)
        total_cost = self._total_cost()
        
        # Huge penalty for each unassigned order
        return 1000000.0 * num_unassigned + total_cost
    
    def _total_cost(self) -> float:
        """Calculate total solution cost."""
        return sum(self._estimate_route_cost(route) for route in self.solution)
    
    def _estimate_route_cost(self, route: Dict) -> float:
        """Estimate route cost."""
        if not route or not route.get('steps'):
            return 0.0
        
        vehicle = next(v for v in self.vehicles if v.id == route['vehicle_id'])
        total_dist = 0.0
        
        steps = route['steps']
        for i in range(len(steps) - 1):
            total_dist += self._get_distance(steps[i]['node_id'], steps[i+1]['node_id'])
        
        return vehicle.cost_per_km * total_dist + vehicle.fixed_cost
    
    def _get_distance(self, loc1: int, loc2: int) -> float:
        """Get distance between locations."""
        if loc1 == loc2:
            return 0.0
        
        cache_key = (min(loc1, loc2), max(loc1, loc2))
        if cache_key in self.dist_cache:
            return self.dist_cache[cache_key]
        
        path = self._get_path(loc1, loc2)
        if path:
            # Use path length as proxy for distance (can be improved)
            dist = float(len(path) - 1)
            self.dist_cache[cache_key] = dist
            return dist
        
        return float('inf')
    
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
            
            # Get neighbors - adjacency list stores as list
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
    
    def _copy_solution(self) -> List[Dict]:
        """Deep copy solution."""
        return [dict(route) for route in self.solution]
    
    def _recalc_assigned(self):
        """Recalculate assigned orders from solution."""
        self.assigned_orders = set()
        for route in self.solution:
            for step in route.get('steps', []):
                for delivery in step.get('deliveries', []):
                    self.assigned_orders.add(delivery['order_id'])
    
    def _select_operator(self, weights: Dict[str, float]) -> str:
        """Roulette wheel operator selection."""
        total = sum(weights.values())
        
        if total < 1e-9:
            return random.choice(list(weights.keys()))
        
        r = random.uniform(0, total)
        cumsum = 0
        
        for op, w in weights.items():
            cumsum += w
            if r <= cumsum:
                return op
        
        return list(weights.keys())[-1]
    
    def _update_weights(self):
        """Update operator weights based on scores."""
        for op in self.destroy_ops:
            if self.destroy_uses[op] > 0:
                avg_score = self.destroy_scores[op] / self.destroy_uses[op]
                self.destroy_weights[op] = 0.8 * self.destroy_weights[op] + 0.2 * avg_score
            self.destroy_scores[op] = 0.0
            self.destroy_uses[op] = 0
        
        for op in self.repair_ops:
            if self.repair_uses[op] > 0:
                avg_score = self.repair_scores[op] / self.repair_uses[op]
                self.repair_weights[op] = 0.8 * self.repair_weights[op] + 0.2 * avg_score
            self.repair_scores[op] = 0.0
            self.repair_uses[op] = 0
