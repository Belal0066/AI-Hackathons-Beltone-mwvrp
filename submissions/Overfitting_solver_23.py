#!/usr/bin/env python3
"""
MWVRP Solver — Improved High-Fulfillment Version (vehicle-reuse fixed)
- Vehicles are never assigned to more than one route.
- Reuse is implemented by EXTENDING an existing vehicle route.
- Keeps cost/distance optimizations and fulfillment-improving passes.
"""

from robin_logistics import LogisticsEnvironment
from typing import Dict, List, Tuple, Optional, Set, Any
from collections import defaultdict, deque
import math
import logging

# Configure simple logging
logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger("solver_improved")

# Tunable parameters
SOFT_OVERFLOW_FACTOR = 1.05  # allow 5% overflow on weight/volume when beneficial
NEARBY_REASSIGN_DISTANCE = 2.0  # km — threshold to consider "nearby" warehouses for last-resort moves


def solver(env: LogisticsEnvironment) -> Dict:
    """
    Entry point expected by the competition environment.
    Returns solution dict { "routes": [...] }.
    """
    builder = ImprovedHighFulfillmentSolver(env)
    solution = builder.build_solution()

    # Validate (some envs return tuple)
    try:
        validation_result = env.validate_solution_complete(solution)
    except Exception as e:
        LOG.warning("Validation call raised: %s", e)
        validation_result = (False, f"validation exception: {e}")

    is_valid = validation_result[0] if isinstance(validation_result, tuple) else validation_result
    if not is_valid:
        msg = validation_result[1] if isinstance(validation_result, tuple) and len(validation_result) > 1 else "Unknown"
        LOG.warning("Solution validation failed: %s", msg)
    else:
        LOG.info("Solution validated successfully.")

    return solution


class ImprovedHighFulfillmentSolver:
    """High-fulfillment solver that preserves cost/distance optimizations."""

    def __init__(self, env: LogisticsEnvironment):
        self.env = env

        # Cached environment data
        self.orders: Dict[str, Any] = {oid: env.orders[oid] for oid in env.get_all_order_ids()}
        self.vehicles: Dict[Any, Any] = {v.id: v for v in env.get_all_vehicles()}
        self.warehouses: Dict[str, Any] = dict(env.warehouses)
        self.skus: Dict[str, Any] = dict(env.skus)

        # Make a working copy of inventory (we mutate this)
        self.warehouse_inventory: Dict[str, Dict[str, int]] = {
            wh_id: dict(getattr(wh, "inventory", {})) for wh_id, wh in self.warehouses.items()
        }

        # Road network adjacency (may have string keys) — keep original but handle both types in _get_path
        raw_net = getattr(env, "get_road_network_data", lambda: {})()
        self.adjacency_list = raw_net.get("adjacency_list", {}) if isinstance(raw_net, dict) else {}

        # Distance cache for speed
        self.distance_cache: Dict[Tuple[int, int], Optional[float]] = {}

        # Bookkeeping
        self.routes: List[Dict] = []
        self.used_vehicle_ids: Set[str] = set()
        # Track vehicle remaining capacity (weight, volume)
        self.vehicle_remaining: Dict[str, Dict[str, float]] = {}
        # Map vehicle_id -> index in self.routes for quick extension
        self.vehicle_route_index: Dict[str, int] = {}

    def build_solution(self) -> Dict:
        """
        Build the final solution with multiple improvement passes.
        """
        LOG.info("Starting initial greedy packing pass.")
        self._initial_greedy_packing()

        LOG.info("Attempting reassignment and vehicle reuse passes to boost fulfillment.")
        self._reassignment_and_reuse_pass()

        LOG.info("Attempting last-resort redistribution to nearby warehouses.")
        self._nearby_warehouse_redistribution()

        LOG.info("Final validation-safe cleanup.")
        # Ensure every route is valid structure; (env.validate will check further)
        # Run per-route validation and prune invalid ones (and rollback their effects).
        try:
            self._validate_and_prune_routes()
        except Exception as e:
            LOG.warning("Per-route validation/pruning raised exception: %s", e)

        return {"routes": self.routes}

        # Ensure every route is valid structure; (env.validate will check further)
        return {"routes": self.routes}

        # -------------------------
    # Validation & rollback helpers
    # -------------------------
    def _validate_and_prune_routes(self):
        """
        Validate each route individually using env.validate_solution_complete on single-route solutions.
        If a route is invalid, remove it and rollback inventory and vehicle capacity changes
        previously applied for that route (best-effort).
        This both helps identify the offending route(s) and ensures the final solution is valid.
        """
        if not self.routes:
            return

        # iterate a copy since we may remove entries
        idx = 0
        while idx < len(self.routes):
            route = self.routes[idx]
            single_solution = {"routes": [route]}

            try:
                valid = self.env.validate_solution_complete(single_solution)
            except Exception as e:
                LOG.warning("Per-route validation raised exception for route index %d: %s", idx, e)
                # treat as invalid and remove
                valid = (False, f"validation exception: {e}")

            is_valid = valid[0] if isinstance(valid, tuple) else valid
            if not is_valid:
                msg = valid[1] if isinstance(valid, tuple) and len(valid) > 1 else "Unknown"
                LOG.warning("Route at index %d invalid: %s. Rolling back and removing route.", idx, msg)

                # Attempt rollback: need to infer the allocations and vehicle id used for that route.
                # We'll best-effort:
                self._rollback_route(idx, route)
                # Remove route from lists and adjust vehicle mappings
                vid = str(route.get("vehicle_id"))
                if vid in self.used_vehicle_ids:
                    self.used_vehicle_ids.discard(vid)
                # Remove route index mapping if present
                if vid in self.vehicle_route_index:
                    del self.vehicle_route_index[vid]

                # remove the route
                self.routes.pop(idx)
                # After pop we do NOT increment idx (we want next element that shifted into this index)
                continue
            else:
                LOG.debug("Route at index %d validated ok.", idx)
            idx += 1

    def _rollback_route(self, route_index: int, route: Dict):
        """
        Attempt to undo the inventory and vehicle_remaining changes that were applied when committing this route.
        We don't have exact allocation maps saved per-route in existing code, so we will best-effort:
         - For pickups: sum up pickups steps and add quantities back to corresponding warehouse_inventory.
         - For deliveries: sum deliveries and add used weight/volume back into vehicle_remaining if applicable.
        Note: this is a best-effort approach to restore consistency so the final solution is valid.
        """
        try:
            # 1) restore warehouse inventory from pickups in the route
            steps = route.get("steps", [])
            pickups_by_wh = defaultdict(lambda: defaultdict(int))
            for step in steps:
                for p in step.get("pickups", []):
                    wh_id = p.get("warehouse_id")
                    sku = p.get("sku_id")
                    qty = p.get("quantity", 0)
                    if wh_id is not None and sku is not None and qty:
                        pickups_by_wh[wh_id][sku] += qty

            for wh_id, sku_map in pickups_by_wh.items():
                inv = self.warehouse_inventory.setdefault(wh_id, {})
                for sku_id, qty in sku_map.items():
                    inv[sku_id] = inv.get(sku_id, 0) + qty

            # 2) adjust vehicle_remaining by adding back weight/volume used by deliveries on this route
            vid = str(route.get("vehicle_id"))
            # compute delivered totals from route deliveries
            weight_released = 0.0
            volume_released = 0.0
            for step in route.get("steps", []):
                for d in step.get("deliveries", []):
                    sku_id = d.get("sku_id")
                    qty = d.get("quantity", 0)
                    sku = self.skus.get(sku_id)
                    if sku and qty:
                        weight_released += getattr(sku, "weight", 0) * qty
                        volume_released += getattr(sku, "volume", 0) * qty

            # If we already tracked remaining, add back; else set to full capacity
            vehicle_obj = self.vehicles.get(vid) or self.vehicles.get(int(vid)) if isinstance(vid, str) and vid.isdigit() else None
            if not vehicle_obj:
                # try find by matching id types
                for v in self.vehicles.values():
                    if str(v.id) == vid:
                        vehicle_obj = v
                        break

            if vehicle_obj:
                cap_w = getattr(vehicle_obj, "capacity_weight", 0)
                cap_v = getattr(vehicle_obj, "capacity_volume", 0)
                prev_rem = self.vehicle_remaining.get(vid)
                if prev_rem:
                    # add back the previously used amounts
                    new_w = min(cap_w, prev_rem["weight"] + weight_released)
                    new_v = min(cap_v, prev_rem["volume"] + volume_released)
                else:
                    # route had used some capacity; restore to full (safe fallback)
                    new_w = cap_w
                    new_v = cap_v
                self.vehicle_remaining[vid] = {"weight": new_w, "volume": new_v}
        except Exception as e:
            LOG.exception("Exception during rollback of route at index %s: %s", route_index, e)


    # -------------------------
    # PASS 1 - Initial greedy packing (cost-aware)
    # -------------------------
    def _initial_greedy_packing(self):
        unassigned = set(self.orders.keys())

        # Robust vehicle sort: combine weight & volume scaled (avoid zero collapse)
        vehicles_sorted = sorted(
            self.vehicles.values(),
            key=lambda v: (getattr(v, "capacity_weight", 0) + getattr(v, "capacity_volume", 0) * 0.001),
            reverse=True
        )

        for vehicle in vehicles_sorted:
            if not unassigned:
                break

            # Pack greedily
            packed_orders, total_demand = self._pack_orders_for_vehicle(vehicle, unassigned)

            if not packed_orders:
                continue

            # Try to find a single warehouse that can satisfy total_demand
            best_wh = self._find_best_warehouse(total_demand)
            if not best_wh:
                # Try multi-warehouse allocation (greedy)
                allocations = self._find_multi_warehouse_solution(total_demand)
                if allocations:
                    route = self._build_multi_warehouse_route(vehicle, packed_orders, allocations, total_demand)
                    if route:
                        self._commit_route(route, total_demand, allocations, vehicle)
                        unassigned -= set(packed_orders)
                continue

            # Build a single-warehouse route
            route = self._build_vehicle_route(vehicle, packed_orders, best_wh, total_demand)
            if route:
                self._commit_route(route, total_demand, {best_wh: total_demand}, vehicle)
                unassigned -= set(packed_orders)

        # Save remaining unassigned orders for later passes
        self.unassigned_orders = set(unassigned)
        LOG.info("Initial packing done. Unassigned orders: %d", len(self.unassigned_orders))

    def _pack_orders_for_vehicle(self, vehicle: Any, available_orders: Set[str]) -> Tuple[List[str], Dict[str, int]]:
        """Greedy bin-packing using smallest-first heuristic (weight+volume)."""
        packed = []
        total_demand = defaultdict(int)

        rem_w = getattr(vehicle, "capacity_weight", 0)
        rem_v = getattr(vehicle, "capacity_volume", 0)

        # Sort orders by size (weight+volume)
        def order_metric(oid):
            order = self.orders[oid]
            total = 0.0
            for sku_id, qty in order.requested_items.items():
                sku = self.skus.get(sku_id)
                if not sku:
                    total += 1e6
                else:
                    total += getattr(sku, "weight", 0) * qty + getattr(sku, "volume", 0) * qty
            return total

        for oid in sorted(available_orders, key=order_metric):
            order = self.orders[oid]
            order_w = 0.0
            order_v = 0.0
            skip = False
            for sku_id, qty in order.requested_items.items():
                sku = self.skus.get(sku_id)
                if not sku:
                    skip = True
                    break
                order_w += getattr(sku, "weight", 0) * qty
                order_v += getattr(sku, "volume", 0) * qty
            if skip:
                continue

            if order_w <= rem_w and order_v <= rem_v:
                # Simulate if some warehouse(s) can satisfy
                test_demand = dict(total_demand)
                for sku_id, qty in order.requested_items.items():
                    test_demand[sku_id] = test_demand.get(sku_id, 0) + qty
                if self._find_best_warehouse(test_demand) or self._find_multi_warehouse_solution(test_demand):
                    packed.append(oid)
                    for sku_id, qty in order.requested_items.items():
                        total_demand[sku_id] += qty
                    rem_w -= order_w
                    rem_v -= order_v

        return packed, dict(total_demand)

    # -------------------------
    # Helper: Multi-warehouse allocation (greedy)
    # -------------------------
    def _find_multi_warehouse_solution(self, demand: Dict[str, int]) -> Optional[Dict[str, Dict[str, int]]]:
        """
        Greedily allocate demand across warehouses. Return mapping wh_id -> {sku:qty}
        """
        remaining = dict(demand)
        allocation: Dict[str, Dict[str, int]] = {}

        # Iterate warehouses in descending total available for demanded SKUs (heuristic)
        def wh_score(wh_id):
            inv = self.warehouse_inventory.get(wh_id, {})
            return sum(inv.get(k, 0) for k in remaining.keys())

        for wh_id in sorted(self.warehouse_inventory.keys(), key=wh_score, reverse=True):
            inv = self.warehouse_inventory[wh_id]
            take = {}
            for sku_id in list(remaining.keys()):
                avail = inv.get(sku_id, 0)
                if avail <= 0:
                    continue
                take_qty = min(avail, remaining[sku_id])
                if take_qty > 0:
                    take[sku_id] = take_qty
                    remaining[sku_id] -= take_qty
                    if remaining[sku_id] <= 0:
                        del remaining[sku_id]
            if take:
                allocation[wh_id] = take
            if not remaining:
                return allocation
        return None

    # -------------------------
    # Route builders (single & multi-warehouse)
    # -------------------------
    def _build_vehicle_route(self, vehicle: Any, order_ids: List[str], warehouse_id: str, total_demand: Dict[str, int]) -> Optional[Dict]:
        """Construct route with single pickup warehouse."""
        # Validate nodes
        try:
            wh_node = self.warehouses[warehouse_id].location.id
        except Exception:
            return None

        steps = []
        # pickup step
        pickups = [{"warehouse_id": warehouse_id, "sku_id": sku, "quantity": qty} for sku, qty in total_demand.items()]
        steps.append({"node_id": wh_node, "pickups": pickups, "deliveries": [], "unloads": []})

        current_node = wh_node
        remaining = set(order_ids)

        while remaining:
            nearest = self._find_nearest_order(current_node, remaining)
            if not nearest:
                return None
            dest_node = self.orders[nearest].destination.id
            path = self._get_path(current_node, dest_node)
            if not path:
                return None
            # add intermediates
            for node in path[1:-1]:
                steps.append({"node_id": node, "pickups": [], "deliveries": [], "unloads": []})
            # delivery
            deliveries = [{"order_id": nearest, "sku_id": sku, "quantity": qty}
                          for sku, qty in self.orders[nearest].requested_items.items()]
            steps.append({"node_id": dest_node, "pickups": [], "deliveries": deliveries, "unloads": []})
            current_node = dest_node
            remaining.remove(nearest)

        # return to warehouse
        return_path = self._get_path(current_node, wh_node)
        if not return_path:
            return None
        for node in return_path[1:-1]:
            steps.append({"node_id": node, "pickups": [], "deliveries": [], "unloads": []})
        steps.append({"node_id": wh_node, "pickups": [], "deliveries": [], "unloads": []})

        return {"vehicle_id": vehicle.id, "steps": steps}

    def _build_multi_warehouse_route(self, vehicle: Any, order_ids: List[str],
                                     warehouse_allocations: Dict[str, Dict[str, int]],
                                     total_demand: Dict[str, int]) -> Optional[Dict]:
        """
        Build route that visits multiple warehouses to pick allocated SKUs, then delivers orders.
        Warehouse visit ordering is chosen greedily by nearest-next to minimize travel.
        """
        # Choose a starting warehouse (the one with largest allocation by total qty)
        whs = list(warehouse_allocations.keys())
        if not whs:
            return None

        whs_sorted = sorted(whs, key=lambda w: sum(warehouse_allocations[w].values()), reverse=True)
        start_wh = whs_sorted[0]
        start_node = self.warehouses[start_wh].location.id

        steps = []
        current_node = start_node

        # iterate through warehouses in nearest-neighbor order starting from start_wh
        remaining_whs = set(whs)
        while remaining_whs:
            # pick nearest warehouse
            nearest_wh = None
            min_dist = float("inf")
            for wh_id in remaining_whs:
                node = self.warehouses[wh_id].location.id
                d = self._get_distance(current_node, node)
                if d is None:
                    continue
                if d < min_dist:
                    min_dist = d
                    nearest_wh = wh_id
            if nearest_wh is None:
                # cannot reach remaining warehouses
                return None

            wh_node = self.warehouses[nearest_wh].location.id
            path = self._get_path(current_node, wh_node)
            if not path:
                return None
            for node in path[1:-1]:
                steps.append({"node_id": node, "pickups": [], "deliveries": [], "unloads": []})

            pickups = [{"warehouse_id": nearest_wh, "sku_id": sku, "quantity": qty}
                       for sku, qty in warehouse_allocations[nearest_wh].items()]
            steps.append({"node_id": wh_node, "pickups": pickups, "deliveries": [], "unloads": []})
            current_node = wh_node
            remaining_whs.remove(nearest_wh)

        # After collecting, deliver orders (nearest-neighbor)
        remaining_orders = set(order_ids)
        while remaining_orders:
            nearest = self._find_nearest_order(current_node, remaining_orders)
            if not nearest:
                return None
            dest_node = self.orders[nearest].destination.id
            path = self._get_path(current_node, dest_node)
            if not path:
                return None
            for node in path[1:-1]:
                steps.append({"node_id": node, "pickups": [], "deliveries": [], "unloads": []})
            deliveries = [{"order_id": nearest, "sku_id": sku, "quantity": qty}
                          for sku, qty in self.orders[nearest].requested_items.items()]
            steps.append({"node_id": dest_node, "pickups": [], "deliveries": deliveries, "unloads": []})
            current_node = dest_node
            remaining_orders.remove(nearest)

        # Return to start warehouse
        return_path = self._get_path(current_node, start_node)
        if not return_path:
            return None
        for node in return_path[1:-1]:
            steps.append({"node_id": node, "pickups": [], "deliveries": [], "unloads": []})
        steps.append({"node_id": start_node, "pickups": [], "deliveries": [], "unloads": []})

        return {"vehicle_id": vehicle.id, "steps": steps}

    # -------------------------
    # Commit route and update inventories & vehicle remaining capacities
    # -------------------------
    def _commit_route(self, route: Dict, total_demand: Dict[str, int], allocations: Dict[str, Dict[str, int]], vehicle: Any):
        """
        Append route to self.routes, subtract inventory as per allocations,
        or extend existing route if vehicle already assigned.
        allocations: mapping warehouse_id -> sku->qty (for multi-warehouse) — for single-wh, it's {wh: total_demand}
        """
        vid = str(vehicle.id)

        if vid in self.used_vehicle_ids:
            # Extend existing route instead of creating a second entry
            idx = self.vehicle_route_index.get(vid)
            if idx is None:
                LOG.warning("Vehicle %s marked used but no route index found — skipping extension", vid)
                return
            LOG.info("Extending route for vehicle %s (route index %d)", vid, idx)
            success = self._extend_route(idx, allocations, total_demand, vehicle)
            if not success:
                LOG.warning("Failed to extend existing route for vehicle %s — keeping original route", vid)
                return
        else:
            # New route: append and record index
            route_index = len(self.routes)
            self.routes.append(route)
            self.used_vehicle_ids.add(vid)
            self.vehicle_route_index[vid] = route_index

        # Update inventories (subtract allocated quantities)
        for wh_id, sku_map in allocations.items():
            inv = self.warehouse_inventory.get(wh_id, {})
            for sku_id, qty in sku_map.items():
                prev = inv.get(sku_id, 0)
                inv[sku_id] = max(prev - qty, 0)

        # Compute remaining capacity for the vehicle after fulfilling total_demand
        weight_used = 0.0
        volume_used = 0.0
        for sku_id, qty in total_demand.items():
            sku = self.skus.get(sku_id)
            if sku:
                weight_used += getattr(sku, "weight", 0) * qty
                volume_used += getattr(sku, "volume", 0) * qty

        cap_w = getattr(vehicle, "capacity_weight", 0)
        cap_v = getattr(vehicle, "capacity_volume", 0)
        # If vehicle was previously used and had remaining capacity, subtract newly used amounts
        prev_rem = self.vehicle_remaining.get(vid)
        if prev_rem:
            remaining_w = max(0.0, prev_rem["weight"] - weight_used)
            remaining_v = max(0.0, prev_rem["volume"] - volume_used)
        else:
            remaining_w = max(0.0, cap_w - weight_used)
            remaining_v = max(0.0, cap_v - volume_used)

        self.vehicle_remaining[vid] = {"weight": remaining_w, "volume": remaining_v}
        LOG.debug("Vehicle %s remaining capacity updated: w=%s v=%s", vid, remaining_w, remaining_v)

    def _extend_route(self, route_index: int, allocations: Dict[str, Dict[str, int]], total_demand: Dict[str, int], vehicle: Any) -> bool:
        """
        Attempt to extend an existing route (at routes[route_index]) by
        removing its final return-to-warehouse step and appending pickups/deliveries for the new allocations,
        then adding the return-to-warehouse back. Returns True on success.
        """
        if route_index < 0 or route_index >= len(self.routes):
            return False

        existing_route = self.routes[route_index]
        # Ensure this route belongs to the same vehicle
        if str(existing_route.get("vehicle_id")) != str(vehicle.id):
            LOG.warning("Route-vehicle mismatch when extending route: expected %s, found %s", vehicle.id, existing_route.get("vehicle_id"))
            return False

        steps = existing_route.get("steps", [])
        if not steps:
            return False

        # We expect the last step to be the return-to-warehouse node; record warehouse node
        last_step = steps[-1]
        warehouse_node = last_step.get("node_id")

        # Remove final warehouse step and any immediate preceding intermediate nodes that are part of return path
        # For simplicity, remove only the final step (the warehouse node) and keep everything else.
        # This assumes the route ends with a single warehouse step (as in our builder).
        steps = steps[:-1]

        # Current node is now the last node in steps (if none, start at warehouse_node)
        current_node = steps[-1]["node_id"] if steps else warehouse_node

        # Build pickups from allocations: visit warehouses needed and pick allocated SKUs
        warehouse_ids = list(allocations.keys())
        # We'll visit warehouses in nearest order from current_node
        remaining_whs = set(warehouse_ids)
        while remaining_whs:
            nearest_wh = None
            min_dist = float("inf")
            for wh in remaining_whs:
                try:
                    wh_node = self.warehouses[wh].location.id
                except Exception:
                    continue
                d = self._get_distance(current_node, wh_node)
                if d is None:
                    continue
                if d < min_dist:
                    min_dist = d
                    nearest_wh = wh
            if nearest_wh is None:
                # inability to reach some warehouse — abort extension
                LOG.debug("Cannot reach remaining warehouses when extending route.")
                return False

            wh_node = self.warehouses[nearest_wh].location.id
            path = self._get_path(current_node, wh_node)
            if not path:
                LOG.debug("No path to warehouse %s when extending route.", nearest_wh)
                return False
            # append intermediate nodes (exclude start and end)
            for node in path[1:-1]:
                steps.append({"node_id": node, "pickups": [], "deliveries": [], "unloads": []})
            # append pickup step
            pickups = [{"warehouse_id": nearest_wh, "sku_id": sku, "quantity": qty} for sku, qty in allocations[nearest_wh].items()]
            steps.append({"node_id": wh_node, "pickups": pickups, "deliveries": [], "unloads": []})
            current_node = wh_node
            remaining_whs.remove(nearest_wh)

        # After pickups, deliver the orders implied by total_demand (we do not have order IDs here in extension context)
        # To attach deliveries properly, the caller should ideally pass order ids. But in our usage we call extend when
        # we have order ids forming a simple single-order or small set. For safety, we will append deliveries by matching
        # orders whose requested_items are subset of total_demand — in reuse code we call _commit_route with single-order demand.
        # Simpler: assume caller used single-order allocation in reuse/nearby routines; the reuse logic does pass single-order demands.
        # So we will search for orders matching the exact SKU quantities equal to total_demand to find the order id(s).
        matched_order_ids = []
        for oid, order in self.orders.items():
            # skip if order already delivered earlier in route (we can't easily detect deliveries done in step list),
            # but we rely on high-level bookkeeping (unassigned_orders) to avoid duplicates.
            # Here we look for exact match (order.requested_items == total_demand)
            if dict(order.requested_items) == dict(total_demand):
                matched_order_ids.append(oid)
        # If not found, we will still add delivery steps using SKU-level deliveries (no order_id) — env may accept them.
        if matched_order_ids:
            # deliver matched orders (nearest neighbor)
            remaining_orders = set(matched_order_ids)
            while remaining_orders:
                nearest_oid = self._find_nearest_order(current_node, remaining_orders)
                if not nearest_oid:
                    return False
                dest_node = self.orders[nearest_oid].destination.id
                path = self._get_path(current_node, dest_node)
                if not path:
                    return False
                for node in path[1:-1]:
                    steps.append({"node_id": node, "pickups": [], "deliveries": [], "unloads": []})
                deliveries = [{"order_id": nearest_oid, "sku_id": sku, "quantity": qty}
                              for sku, qty in self.orders[nearest_oid].requested_items.items()]
                steps.append({"node_id": dest_node, "pickups": [], "deliveries": deliveries, "unloads": []})
                current_node = dest_node
                remaining_orders.remove(nearest_oid)
        else:
            # fallback: create a generic delivery step using SKU-level deliveries (no order_id) at nearest reachable customer node
            # Try to find the single order among unassigned that corresponds to these SKUs (likely the intended one)
            target_oid = None
            for oid, order in self.orders.items():
                if oid in self.unassigned_orders:
                    # check if order.requested_items is subset of total_demand
                    ok = True
                    for sku_id, qty in order.requested_items.items():
                        if total_demand.get(sku_id, 0) < qty:
                            ok = False
                            break
                    if ok:
                        target_oid = oid
                        break
            if target_oid:
                dest_node = self.orders[target_oid].destination.id
                path = self._get_path(current_node, dest_node)
                if not path:
                    return False
                for node in path[1:-1]:
                    steps.append({"node_id": node, "pickups": [], "deliveries": [], "unloads": []})
                deliveries = [{"order_id": target_oid, "sku_id": sku, "quantity": qty}
                              for sku, qty in self.orders[target_oid].requested_items.items()]
                steps.append({"node_id": dest_node, "pickups": [], "deliveries": deliveries, "unloads": []})
                current_node = dest_node
            else:
                # cannot identify a target order, abort extension (safer)
                LOG.debug("Could not identify order to attach deliveries when extending route.")
                return False

        # Finally, return to warehouse_node
        return_path = self._get_path(current_node, warehouse_node)
        if not return_path:
            LOG.debug("No path to return to warehouse when extending route.")
            return False
        for node in return_path[1:-1]:
            steps.append({"node_id": node, "pickups": [], "deliveries": [], "unloads": []})
        steps.append({"node_id": warehouse_node, "pickups": [], "deliveries": [], "unloads": []})

        # Save modified steps back to route
        existing_route["steps"] = steps
        # update the route in the main list
        self.routes[route_index] = existing_route
        LOG.info("Route at index %d successfully extended for vehicle %s", route_index, vehicle.id)
        return True

    # -------------------------
    # Reassignment and vehicle reuse passes
    # -------------------------
    def _reassignment_and_reuse_pass(self):
        """
        Try to push up fulfillment by:
         - attempting to assign leftover orders to vehicles that still have remaining capacity,
         - trying split deliveries across warehouses if single-warehouse can't satisfy,
         - trying soft overflow when a small overflow yields fulfillment without large travel overhead.
        """
        still_unassigned = set(self.unassigned_orders)
        if not still_unassigned:
            LOG.info("No unassigned orders; skipping reassignment pass.")
            return

        # Build a list of vehicles available for reuse (those not used OR with remaining capacity)
        all_vehicles = list(self.vehicles.values())
        veh_lookup = {v.id: v for v in all_vehicles}

        # Sort remaining orders by 'hardness' (largest first) to try hard items earlier
        def order_hardness(oid):
            order = self.orders[oid]
            total = 0.0
            for sku_id, qty in order.requested_items.items():
                sku = self.skus.get(sku_id)
                if not sku:
                    total += 1e6
                else:
                    total += getattr(sku, "weight", 0) * qty + getattr(sku, "volume", 0) * qty
            return -total  # largest first

        for oid in sorted(still_unassigned, key=order_hardness):
            assigned = False
            order = self.orders[oid]
            order_demand = dict(order.requested_items)

            # Try vehicles that have remaining capacity recorded (including unused)
            candidate_vehicle_ids = list(self.vehicle_remaining.keys()) + [v.id for v in all_vehicles if v.id not in self.vehicle_remaining]
            # Remove duplicates while preserving order
            seen = set()
            candidate_vehicle_ids = [x for x in candidate_vehicle_ids if not (x in seen or seen.add(x))]

            for vid in candidate_vehicle_ids:
                v = veh_lookup.get(vid)
                if v is None:
                    continue

                # Get current remaining capacity (if not recorded -> full capacity)
                rem_cap = self.vehicle_remaining.get(str(v.id), {"weight": getattr(v, "capacity_weight", 0), "volume": getattr(v, "capacity_volume", 0)})
                rem_w = rem_cap["weight"]
                rem_v = rem_cap["volume"]

                # Allow soft overflow attempt
                cap_w = getattr(v, "capacity_weight", 0) * SOFT_OVERFLOW_FACTOR
                cap_v = getattr(v, "capacity_volume", 0) * SOFT_OVERFLOW_FACTOR

                # Compute order weight/volume
                order_w = 0.0
                order_v = 0.0
                valid = True
                for sku_id, qty in order_demand.items():
                    sku = self.skus.get(sku_id)
                    if not sku:
                        valid = False
                        break
                    order_w += getattr(sku, "weight", 0) * qty
                    order_v += getattr(sku, "volume", 0) * qty
                if not valid:
                    continue

                # If we have enough rem capacity (or overflow allowed), continue
                if order_w <= rem_w or (order_w <= cap_w and order_v <= cap_v):
                    # Find single warehouse or multi-warehouse allocation
                    best_wh = self._find_best_warehouse(order_demand)
                    allocations = None
                    if best_wh:
                        allocations = {best_wh: order_demand}
                    else:
                        allocations = self._find_multi_warehouse_solution(order_demand)

                    if not allocations:
                        continue

                    # Build appropriate route extension or new route depending on vehicle usage
                    if str(v.id) in self.used_vehicle_ids:
                        # will extend existing route
                        # We still construct a route object for validation/step correctness, but _commit_route will extend instead of append
                        if len(allocations) == 1:
                            wh_id = next(iter(allocations.keys()))
                            route = self._build_vehicle_route(v, [oid], wh_id, order_demand)
                            alloc_for_commit = {wh_id: order_demand}
                        else:
                            route = self._build_multi_warehouse_route(v, [oid], allocations, order_demand)
                            alloc_for_commit = allocations
                    else:
                        # new vehicle -> new route
                        if len(allocations) == 1:
                            wh_id = next(iter(allocations.keys()))
                            route = self._build_vehicle_route(v, [oid], wh_id, order_demand)
                            alloc_for_commit = {wh_id: order_demand}
                        else:
                            route = self._build_multi_warehouse_route(v, [oid], allocations, order_demand)
                            alloc_for_commit = allocations

                    if not route:
                        continue

                    # Commit route (will extend if vehicle already has a route)
                    self._commit_route(route, order_demand, alloc_for_commit, v)
                    # remove order from unassigned set
                    self.unassigned_orders.discard(oid)
                    assigned = True
                    LOG.info("Assigned order %s using vehicle %s in reuse pass.", oid, v.id)
                    break  # order assigned

            if not assigned:
                LOG.debug("Could not assign order %s in reuse pass.", oid)

        LOG.info("Reassignment pass done. Remaining unassigned: %d", len(self.unassigned_orders))

    # -------------------------
    # Last-resort redistribution: move some orders to nearby warehouses and try again
    # -------------------------
    def _nearby_warehouse_redistribution(self):
        """
        For orders still unassigned, attempt to find nearby warehouses that collectively
        can fulfill them even if not the best warehouse originally chosen.
        """
        if not self.unassigned_orders:
            return

        for oid in list(self.unassigned_orders):
            order = self.orders[oid]
            order_demand = dict(order.requested_items)

            # Try to find any allocation across warehouses regardless of being the 'best'
            allocations = self._find_multi_warehouse_solution(order_demand)
            if allocations:
                # But check that warehouses are reasonably close to each other or to the customer
                wh_ids = list(allocations.keys())
                dest_node = order.destination.id

                def is_nearby_wh_set(wh_list):
                    for w in wh_list:
                        try:
                            w_node = self.warehouses[w].location.id
                        except Exception:
                            return False
                        d_to_dest = self._get_distance(w_node, dest_node)
                        if d_to_dest is not None and d_to_dest <= NEARBY_REASSIGN_DISTANCE:
                            continue
                        close_to_other = False
                        for other in wh_list:
                            if other == w:
                                continue
                            other_node = self.warehouses[other].location.id
                            d = self._get_distance(w_node, other_node)
                            if d is not None and d <= NEARBY_REASSIGN_DISTANCE:
                                close_to_other = True
                                break
                        if not close_to_other:
                            return False
                    return True

                if not is_nearby_wh_set(wh_ids):
                    continue

                # Now attempt to assign using any available vehicle
                assigned = False
                for v in self.vehicles.values():
                    if v.id in self.used_vehicle_ids:
                        # still can reuse if we have remaining capacity recorded
                        rem = self.vehicle_remaining.get(str(v.id))
                        if not rem:
                            continue
                        cap_w = getattr(v, "capacity_weight", 0) * SOFT_OVERFLOW_FACTOR
                        cap_v = getattr(v, "capacity_volume", 0) * SOFT_OVERFLOW_FACTOR
                        order_w = sum(self.skus[s].weight * q for s, q in order_demand.items() if s in self.skus)
                        order_v = sum(self.skus[s].volume * q for s, q in order_demand.items() if s in self.skus)
                        if order_w <= cap_w and order_v <= cap_v:
                            route = self._build_multi_warehouse_route(v, [oid], allocations, order_demand)
                            if route:
                                self._commit_route(route, order_demand, allocations, v)
                                self.unassigned_orders.discard(oid)
                                assigned = True
                                LOG.info("Assigned order %s via nearby redistribution with vehicle %s", oid, v.id)
                                break
                    else:
                        # unused vehicle — try assign
                        order_w = sum(self.skus[s].weight * q for s, q in order_demand.items() if s in self.skus)
                        order_v = sum(self.skus[s].volume * q for s, q in order_demand.items() if s in self.skus)
                        if order_w <= getattr(v, "capacity_weight", 0) * SOFT_OVERFLOW_FACTOR and \
                           order_v <= getattr(v, "capacity_volume", 0) * SOFT_OVERFLOW_FACTOR:
                            route = self._build_multi_warehouse_route(v, [oid], allocations, order_demand)
                            if route:
                                self._commit_route(route, order_demand, allocations, v)
                                self.unassigned_orders.discard(oid)
                                assigned = True
                                LOG.info("Assigned order %s via nearby redistribution with new vehicle %s", oid, v.id)
                                break
                if assigned:
                    continue

        LOG.info("Nearby redistribution done. Remaining unassigned: %d", len(self.unassigned_orders))

    # -------------------------
    # Helper: find single best warehouse for a demand
    # -------------------------
    def _find_best_warehouse(self, demand: Dict[str, int]) -> Optional[str]:
        best_wh = None
        best_score = -math.inf
        for wh_id, inv in self.warehouse_inventory.items():
            has_all = True
            for sku_id, qty in demand.items():
                if inv.get(sku_id, 0) < qty:
                    has_all = False
                    break
            if not has_all:
                continue
            # Heuristic score: total excess inventory across demanded SKUs (prefer more-stocked)
            excess = sum(inv.get(sku_id, 0) - qty for sku_id, qty in demand.items())
            if excess > best_score:
                best_score = excess
                best_wh = wh_id
        return best_wh

    # -------------------------
    # Distance and path utilities (BFS pathfinding + caching)
    # -------------------------
    def _get_distance(self, node1: int, node2: int) -> Optional[float]:
        """Try env.get_distance first, fallback to BFS hop-estimate."""
        if node1 == node2:
            return 0.0
        key = (int(node1), int(node2))
        if key in self.distance_cache:
            return self.distance_cache[key]
        try:
            dist = self.env.get_distance(node1, node2)
        except Exception:
            dist = None
        if dist is None:
            path = self._get_path(node1, node2)
            if path:
                dist = (len(path) - 1) * 0.5  # default edge length estimate
            else:
                dist = None
        self.distance_cache[key] = dist
        return dist

    def _get_path(self, start: int, end: int) -> Optional[List[int]]:
        """BFS; supports adjacency keys as ints or strings."""
        if start == end:
            return [start]
        queue = deque([(start, [start])])
        visited = {start}
        while queue:
            current, path = queue.popleft()
            neighbors = self.adjacency_list.get(current, self.adjacency_list.get(str(current), []))
            for nbr in neighbors:
                try:
                    nbr_int = int(nbr) if isinstance(nbr, str) else nbr
                except Exception:
                    continue
                if nbr_int == end:
                    return path + [nbr_int]
                if nbr_int not in visited:
                    visited.add(nbr_int)
                    queue.append((nbr_int, path + [nbr_int]))
        return None

    def _find_nearest_order(self, current_node: int, order_ids: Set[str]) -> Optional[str]:
        """Pick nearest order by estimated distance (cache-aware)."""
        best = None
        best_d = float("inf")
        for oid in order_ids:
            dest_node = self.orders[oid].destination.id
            d = self._get_distance(current_node, dest_node)
            if d is None:
                continue
            if d < best_d:
                best_d = d
                best = oid
        return best
