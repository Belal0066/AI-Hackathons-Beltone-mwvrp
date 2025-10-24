#!/usr/bin/env python3
"""
MWVRP Aggressive Solver (improved)
- Uses A* pathfinding (with env.get_distance as heuristic when available)
- Normalizes adjacency / node id types
- Safer vehicle sorting and capacity checks
- More robust inventory and validation handling
"""

from robin_logistics import LogisticsEnvironment
from typing import Dict, List, Tuple, Optional, Set, Any
from collections import defaultdict, deque
import heapq
import math


def solver(env: LogisticsEnvironment) -> Dict:
    """Main solver focusing on maximum fulfillment."""
    builder = AggressiveSolver(env)
    solution = builder.build()

    # Validate (some envs return tuple (bool, message) — handle both)
    try:
        validation_result = env.validate_solution_complete(solution)
    except Exception as e:
        print(f"[WARN] Validation call raised: {e}")
        validation_result = (False, f"validation exception: {e}")

    is_valid = validation_result[0] if isinstance(validation_result, tuple) else validation_result

    if not is_valid:
        message = validation_result[1] if isinstance(validation_result, tuple) and len(validation_result) > 1 else "Unknown"
        print(f"[WARN] Validation failed: {message}")
    else:
        print("[INFO] Solution validated as complete/valid")

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

        # Raw road network; we'll normalize adjacency to integers
        self.road_network = env.get_road_network_data()
        raw_adj = self.road_network.get("adjacency_list", {}) if self.road_network else {}
        self.adjacency_list = self._normalize_adjacency(raw_adj)

        # Distance cache
        self.distance_cache: Dict[Tuple[int, int], Optional[float]] = {}

    def _normalize_adjacency(self, raw_adj: Dict) -> Dict[int, List[int]]:
        """Turn adjacency keys & values into ints for stable graph traversal."""
        adj = {}
        for k, neighbors in raw_adj.items():
            try:
                node = int(k)
            except Exception:
                # If it can't convert, skip
                continue
            norm_neighs = []
            for n in neighbors:
                try:
                    norm_neighs.append(int(n))
                except Exception:
                    continue
            # remove duplicates
            adj[node] = list(dict.fromkeys(norm_neighs))
        return adj

    def build(self) -> Dict:
        """Build solution aggressively using all available resources."""
        routes = []
        unassigned_orders: Set[str] = set(self.orders.keys())
        used_vehicles: Set[Any] = set()

        # Track inventory per warehouse (copy)
        warehouse_inventory: Dict[str, Dict[str, int]] = {}
        for wh_id, wh in self.warehouses.items():
            warehouse_inventory[wh_id] = dict(getattr(wh, "inventory", {}))

        # Sort vehicles by a robust capacity metric (weight + volume scaled)
        # Use a small scaling factor to put weight and volume on similar scale if needed.
        vehicles_sorted = sorted(
            self.vehicles.values(),
            key=lambda v: (getattr(v, "capacity_weight", 0) + getattr(v, "capacity_volume", 0) * 0.001),
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

                # Update inventory safely
                for sku_id, qty in total_demand.items():
                    prev = warehouse_inventory[best_warehouse].get(sku_id, 0)
                    warehouse_inventory[best_warehouse][sku_id] = max(prev - qty, 0)

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
                        prev = warehouse_inventory[best_warehouse].get(sku_id, 0)
                        warehouse_inventory[best_warehouse][sku_id] = max(prev - qty, 0)
                    break

        # PASS 3: Could split orders across vehicles — skipped to keep deterministic/simple

        return {"routes": routes}

    def _pack_orders_greedy(
        self,
        vehicle: Any,
        available_orders: Set[str],
        warehouse_inventory: Dict[str, Dict[str, int]]
    ) -> Tuple[List[str], Dict[str, int]]:
        """Pack as many orders as possible into vehicle (smallest orders first)."""
        packed = []
        total_demand = defaultdict(int)
        remaining_weight = getattr(vehicle, "capacity_weight", 0)
        remaining_volume = getattr(vehicle, "capacity_volume", 0)

        # Sort orders by size (smallest first)
        orders_sorted = sorted(
            available_orders,
            key=lambda oid: self._order_size(oid)
        )

        for oid in orders_sorted:
            order = self.orders[oid]

            order_weight = 0.0
            order_volume = 0.0
            valid = True

            for sku_id, qty in order.requested_items.items():
                if sku_id not in self.skus:
                    # SKU unknown: skip order for safety
                    valid = False
                    break
                sku = self.skus[sku_id]
                order_weight += getattr(sku, "weight", 0) * qty
                order_volume += getattr(sku, "volume", 0) * qty

            if not valid:
                continue

            # Check capacity remaining
            if order_weight > remaining_weight or order_volume > remaining_volume:
                continue

            # Check if adding this order still allows warehouse stock (some warehouse)
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
        total = 0.0
        for sku_id, qty in order.requested_items.items():
            sku = self.skus.get(sku_id)
            if sku is None:
                continue
            total += getattr(sku, "weight", 0) * qty + getattr(sku, "volume", 0) * qty
        return total

    def _find_warehouse_with_stock(
        self,
        demand: Dict[str, int],
        warehouse_inventory: Dict[str, Dict[str, int]]
    ) -> Optional[str]:
        """Find warehouse with sufficient inventory, prefer one with largest surplus."""
        best_wh = None
        best_score = -math.inf
        for wh_id, inventory in warehouse_inventory.items():
            has_all = True
            for sku_id, qty in demand.items():
                if inventory.get(sku_id, 0) < qty:
                    has_all = False
                    break
            if not has_all:
                continue
            # score = total excess across demanded SKUs (higher -> prefer)
            excess = sum(inventory.get(sku_id, 0) - qty for sku_id, qty in demand.items())
            if excess > best_score:
                best_score = excess
                best_wh = wh_id
        return best_wh

    def _fits_in_vehicle(self, vehicle: Any, demand: Dict[str, int]) -> bool:
        """Check if demand fits in vehicle capacity."""
        total_weight = 0.0
        total_volume = 0.0
        for sku_id, qty in demand.items():
            sku = self.skus.get(sku_id)
            if sku is None:
                # if SKU unknown, be conservative and treat as not fitting
                return False
            total_weight += getattr(sku, "weight", 0) * qty
            total_volume += getattr(sku, "volume", 0) * qty

        return total_weight <= getattr(vehicle, "capacity_weight", 0) and total_volume <= getattr(vehicle, "capacity_volume", 0)

    def _build_route(
        self,
        vehicle: Any,
        order_ids: List[str],
        warehouse_id: str,
        total_demand: Dict[str, int]
    ) -> Optional[Dict]:
        """Build a complete valid route (pickup -> multiple deliveries -> return)."""
        if not order_ids:
            return None

        warehouse = self.warehouses.get(warehouse_id)
        if warehouse is None:
            return None

        # Attempt to get node id; be robust if different attributes
        warehouse_node = getattr(warehouse, "location", None)
        if hasattr(warehouse_node, "id"):
            warehouse_node = warehouse_node.id
        try:
            warehouse_node = int(warehouse_node)
        except Exception:
            return None

        steps = []

        # Step 1: Pickup at warehouse (simple representation)
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

        # Step 2: Visit customers in nearest-neighbor order (using _get_distance)
        current_node = warehouse_node
        remaining_orders = set(order_ids)

        while remaining_orders:
            nearest_oid = self._find_nearest_order(current_node, remaining_orders)
            if not nearest_oid:
                # If we cannot find a reachable order, fail this route
                return None

            dest_node_attr = getattr(self.orders[nearest_oid].destination, "id", None)
            try:
                dest_node = int(dest_node_attr)
            except Exception:
                return None

            path = self._get_path(current_node, dest_node)
            if not path:
                return None

            # Add intermediate nodes (skip first and last)
            for node in path[1:-1]:
                steps.append({
                    "node_id": node,
                    "pickups": [],
                    "deliveries": [],
                    "unloads": []
                })

            # Deliver at final node
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
        """Find nearest unvisited order using cached distances."""
        nearest_oid = None
        min_distance = float('inf')
        for oid in order_ids:
            dest_node_attr = getattr(self.orders[oid].destination, "id", None)
            try:
                dest_node = int(dest_node_attr)
            except Exception:
                continue
            dist = self._get_distance(current_node, dest_node)
            if dist is None:
                # treat unreachable as infinite
                continue
            if dist < min_distance:
                min_distance = dist
                nearest_oid = oid
        return nearest_oid

    def _get_distance(self, node1: int, node2: int) -> Optional[float]:
        """Get distance with caching and fallback to path-length heuristic."""
        if node1 == node2:
            return 0.0

        key = (int(node1), int(node2))
        if key in self.distance_cache:
            return self.distance_cache[key]

        # Ask environment first (may be None)
        try:
            dist = self.env.get_distance(node1, node2)
        except Exception:
            dist = None

        if dist is None:
            # Try to compute shortest path length using actual path (A* will compute an exact path)
            path = self._get_path(node1, node2)
            if path:
                # estimate distance as number of edges * default_edge_length
                dist = (len(path) - 1) * 0.5
            else:
                dist = None

        self.distance_cache[key] = dist
        return dist

    def _get_path(self, start: int, end: int) -> Optional[List[int]]:
        """A* pathfinding on adjacency_list (normalized)."""

        start = int(start)
        end = int(end)
        if start == end:
            return [start]

        # Heuristic: if env can provide a distance estimate, use it; otherwise 0
        def heuristic(a: int, b: int) -> float:
            try:
                h = self.env.get_distance(a, b)
                if h is None:
                    return 0.0
                return float(h)
            except Exception:
                return 0.0

        # A* using adjacency_list
        open_heap = []
        heapq.heappush(open_heap, (0 + heuristic(start, end), 0, start, [start]))  # (f, g, node, path)
        closed = {}

        while open_heap:
            f, g, node, path = heapq.heappop(open_heap)
            if node == end:
                return path

            if node in closed and closed[node] <= g:
                continue
            closed[node] = g

            neighbors = self.adjacency_list.get(node, [])
            for nbr in neighbors:
                tentative_g = g + 1  # assume uniform edge cost (1 per hop)
                if nbr in closed and closed[nbr] <= tentative_g:
                    continue
                new_path = path + [nbr]
                heapq.heappush(open_heap, (tentative_g + heuristic(nbr, end), tentative_g, nbr, new_path))

        # No path found
        return None
