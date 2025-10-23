#!/usr/bin/env python3
"""
Highly-optimized MWVRP solver for Robin Logistics Hackathon.

Key optimization choices:
- Bulk-read environment data once where possible (orders, vehicles, warehouses, inventory, SKUs).
- Replace repeated haversine/A* heuristics for clustering with a cheap "approx distance" using coordinates when available.
- Reduce number of A* calls: only compute A* for consecutive route hops (home -> pickups/deliveries -> home).
- Greedy allocation by SKU with simple warehouse preference (home warehouse first), but early exit when vehicle capacity exhausted.
- Use local variables heavily to avoid attribute/dict lookups cost.
- No environment initialization inside solver and no caching that persists beyond this solver invocation.
- Robust handling for missing data (skip/continue).
"""

from __future__ import annotations
import math
import heapq
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Optional

# --------------------------
# Lightweight helpers
# --------------------------
def haversine_approx(coords: Optional[Dict[Any, Tuple[float, float]]], a: Any, b: Any) -> float:
    """Cheap haversine-like distance used for clustering when coords exist. If not, return large constant."""
    if not coords:
        return 1e9
    pa = coords.get(a)
    pb = coords.get(b)
    if not pa or not pb:
        return 1e9
    (lat1, lon1), (lat2, lon2) = pa, pb
    # small-optimized haversine (avoid many function calls)
    r = 6371.0  # km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    s = math.sin(dlat * 0.5)
    t = math.sin(dlon * 0.5)
    a_ = s * s + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * t * t
    return 2.0 * r * math.asin(min(1.0, math.sqrt(a_))) * 1000.0  # meters

def a_star_search(adjacency_list: Dict[Any, List[Any]],
                  start: Any, goal: Any,
                  get_distance_fn,
                  coords: Optional[Dict[Any, Tuple[float, float]]] = None,
                  step_limit: int = 100000) -> Optional[List[Any]]:
    """A* search with heuristic from coords if available. Returns node list or None."""
    if start == goal:
        return [start]
    def heuristic(n1, n2):
        # use cheap haversine_approx
        return haversine_approx(coords, n1, n2) if coords else 0.0

    open_heap = [(0.0, start)]
    gscore = {start: 0.0}
    came_from: Dict[Any, Any] = {}
    steps = 0
    while open_heap:
        steps += 1
        if steps > step_limit:
            return None
        _, current = heapq.heappop(open_heap)
        if current == goal:
            # reconstruct
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path
        for nbr in adjacency_list.get(current, ()):
            # edge cost via env if available
            try:
                ec = get_distance_fn(current, nbr) or 1.0
            except Exception:
                ec = 1.0
            ng = gscore.get(current, 1e18) + ec
            if ng < gscore.get(nbr, 1e18):
                came_from[nbr] = current
                gscore[nbr] = ng
                f = ng + heuristic(nbr, goal)
                heapq.heappush(open_heap, (f, nbr))
    return None

# --------------------------
# Solver entrypoint (required)
# --------------------------
def solver(env) -> Dict[str, Any]:
    """
    Main solver that returns {"routes": [...]}
    """
    solution = {"routes": []}

    # Bulk-read basic lists (fail-safe)
    try:
        order_ids = env.get_all_order_ids() or []
        available_vehicle_ids = env.get_available_vehicles() or []
        all_vehicles = env.get_all_vehicles() or []
    except Exception:
        return solution

    if not order_ids or not available_vehicle_ids:
        return solution

    # Road network and adjacency (used by A*)
    road_network = env.get_road_network_data() or {}
    adjacency_list = road_network.get("adjacency_list") or {}
    nodes = road_network.get("nodes") or {}
    coords = {}
    if isinstance(nodes, dict):
        for nid, val in nodes.items():
            if isinstance(val, dict) and "lat" in val and "lon" in val:
                coords[nid] = (float(val["lat"]), float(val["lon"]))

    # Build vehicles dict for quick access
    vehicles = {}
    for v in all_vehicles:
        vid = getattr(v, "id", None)
        if vid is not None:
            vehicles[vid] = v

    # Pre-fetch orders (requirements and location) once
    order_requirements: Dict[str, Dict[str, int]] = {}
    customer_node_of: Dict[str, Any] = {}
    sku_set = set()
    for oid in order_ids:
        try:
            req = env.get_order_requirements(oid) or {}
            if not req:
                continue
            # convert keys to strings
            req2 = {str(k): int(v) for k, v in req.items() if int(v) > 0}
            if not req2:
                continue
            order_requirements[oid] = req2
            sku_set.update(req2.keys())
            customer_node_of[oid] = env.get_order_location(oid)
        except Exception:
            continue

    if not order_requirements:
        return solution

    # Build warehouse maps and per-sku warehouses (single environment reads)
    # We gather set of warehouse ids that supply any SKU in sku_set
    warehouse_by_id = {}
    warehouse_node_to_id = {}
    warehouses_with_sku: Dict[str, List[str]] = defaultdict(list)
    try:
        # For each sku, ask env once for warehouses that have it
        for sku in sku_set:
            try:
                whs = env.get_warehouses_with_sku(sku, min_quantity=1) or []
            except Exception:
                whs = []
            for wid in whs:
                # get warehouse object once
                if wid not in warehouse_by_id:
                    try:
                        wh = env.get_warehouse_by_id(wid)
                        if not wh:
                            continue
                        warehouse_by_id[wid] = wh
                        node_id = getattr(wh.location, "id", None)
                        if node_id is not None:
                            warehouse_node_to_id[node_id] = wid
                    except Exception:
                        continue
                warehouses_with_sku[sku].append(wid)
    except Exception:
        pass

    # Read inventories for collected warehouses once
    inventory_by_wh: Dict[str, Dict[str, int]] = {}
    for wid in list(warehouse_by_id.keys()):
        try:
            inv = env.get_warehouse_inventory(wid) or {}
            inventory_by_wh[wid] = {str(k): int(v) for k, v in inv.items() if int(v) > 0}
        except Exception:
            inventory_by_wh[wid] = {}

    # SKU details used for weight/volume
    sku_details: Dict[str, Dict[str, float]] = {}
    for sku in sku_set:
        try:
            sku_details[sku] = env.get_sku_details(sku) or {}
        except Exception:
            sku_details[sku] = {}

    # remaining requirements per order
    remaining_req = {oid: dict(req) for oid, req in order_requirements.items() if oid in customer_node_of}

    # Available vehicles set (ids)
    avail_vehicles = set(available_vehicle_ids)

    # Utility to get distance using env.get_distance - no persistent caching across runs
    def get_distance(n1, n2):
        try:
            d = env.get_distance(n1, n2)
            return d if (d is not None and d >= 0) else 1e9
        except Exception:
            return 1e9

    # Main loop: assign each available vehicle greedily to a small batch of orders
    # Stop when no unfulfilled orders or no vehicles left
    while avail_vehicles:
        # collect unfulfilled orders
        unfilled = [oid for oid, req in remaining_req.items() if any(v > 0 for v in req.values())]
        if not unfilled:
            break

        # pick seed order (largest remaining demand to encourage packing)
        seed = max(unfilled, key=lambda o: sum(remaining_req[o].values()))
        seed_node = customer_node_of.get(seed)
        if seed_node is None:
            # remove this order if location missing
            remaining_req.pop(seed, None)
            continue

        # build a tiny cluster: add nearest order by approximate coords distance if within threshold
        cluster = [seed]
        seed_coord = seed_node
        # compute cheap distances to others (using coords if available)
        candidates = []
        for oid in unfilled:
            if oid == seed:
                continue
            node = customer_node_of.get(oid)
            if node is None:
                continue
            dist = haversine_approx(coords, seed_coord, node)
            candidates.append((dist, oid))
        if candidates:
            candidates.sort()
            # only add one extra to limit route complexity (keeps A* count down)
            d0, oid0 = candidates[0]
            if d0 <= 20000.0:  # 20km threshold
                cluster.append(oid0)

        # precompute cluster requirements aggregated per SKU
        aggregate_req = defaultdict(int)
        for oid in cluster:
            for s, q in remaining_req[oid].items():
                if q > 0:
                    aggregate_req[s] += q

        # choose best vehicle among available: score = distance from home to cluster centroid + capacity penalty
        best_vid = None
        best_score = 1e18
        chosen_vehicle_home_node = None

        # prepare cluster representative node (first customer's node)
        cluster_node = customer_node_of.get(cluster[0])

        for vid in list(avail_vehicles):
            v = vehicles.get(vid)
            if v is None:
                # try env fallback
                try:
                    # ensure vehicle exists in env
                    _ = env.get_vehicle_state(vid)
                except Exception:
                    avail_vehicles.discard(vid)
                    continue

            # attempt to get home node robustly
            try:
                home_node = env.get_vehicle_home_warehouse(vid)
            except Exception:
                # old env variants may require different call
                try:
                    home_node = getattr(v, "home_warehouse_id", None)
                except Exception:
                    home_node = None
            if home_node is None:
                continue

            # get remaining capacity
            try:
                rem_w, rem_v = env.get_vehicle_remaining_capacity(vid)
            except Exception:
                rem_w, rem_v = 1e12, 1e12

            # compute aggregated weight and volume demand
            total_w = 0.0
            total_v = 0.0
            for s, q in aggregate_req.items():
                detail = sku_details.get(s, {})
                total_w += q * float(detail.get("weight", 0.0))
                total_v += q * float(detail.get("volume", 0.0))

            # capacity penalty: large if cannot carry all; else prefer better fit
            if (total_w > rem_w + 1e-9) or (total_v > rem_v + 1e-9):
                cap_pen = 1e12
            else:
                # smaller penalty favors vehicles that closely fit load to avoid very small loads on huge trucks
                cap_pen = (rem_w - total_w) + (rem_v - total_v)

            # distance estimate from home to cluster (cheap)
            dist_est = haversine_approx(coords, home_node, cluster_node)
            score = dist_est + cap_pen * 0.001
            if score < best_score:
                best_score = score
                best_vid = vid
                chosen_vehicle_home_node = home_node

        if best_vid is None:
            # no vehicle found for this cluster -> drop largest order from consideration to avoid infinite loop
            # mark the seed order as skipped/unfulfillable to avoid lock
            avail_vehicles.clear()
            break

        # Reserve this vehicle
        avail_vehicles.discard(best_vid)

        # get exact vehicle remaining capacity now (after selection)
        try:
            rem_w, rem_v = env.get_vehicle_remaining_capacity(best_vid)
        except Exception:
            rem_w, rem_v = 1e12, 1e12

        # attempt to allocate SKUs greedily from preferred warehouses:
        # preference: home warehouse first, then warehouses_with_sku listing
        home_wh_id = None
        try:
            if chosen_vehicle_home_node is not None:
                home_wh_id = (warehouse_node_to_id.get(chosen_vehicle_home_node))
        except Exception:
            home_wh_id = None

        pickups_by_wh = defaultdict(lambda: defaultdict(int))  # wid -> sku -> qty
        allocations_by_order = {oid: defaultdict(int) for oid in cluster}
        vehicle_load_weight = 0.0
        vehicle_load_volume = 0.0

        # create working copy of remaining req for this cluster
        cluster_remaining = {oid: dict(remaining_req[oid]) for oid in cluster}

        # order SKUs by largest unit size first (weight+volume) to pack heavy/large first
        sku_list = sorted(aggregate_req.keys(),
                          key=lambda s: float(sku_details.get(s, {}).get("weight", 0.0)) + float(sku_details.get(s, {}).get("volume", 0.0)),
                          reverse=True)

        for sku in sku_list:
            need = aggregate_req[sku]
            if need <= 0:
                continue
            detail = sku_details.get(sku, {})
            unit_w = float(detail.get("weight", 0.0))
            unit_v = float(detail.get("volume", 0.0))

            # compute max possible units by capacity
            max_by_w = int(rem_w // unit_w) if unit_w > 0 else need
            max_by_v = int(rem_v // unit_v) if unit_v > 0 else need
            max_possible = max(0, min(need, max_by_w, max_by_v))
            if max_possible <= 0:
                continue

            # warehouse preference list
            candidates = []
            if home_wh_id:
                candidates.append(home_wh_id)
            candidates.extend(warehouses_with_sku.get(sku, []))
            # dedupe preserving order
            seen = set()
            cand_order = []
            for w in candidates:
                if w and w not in seen and w in inventory_by_wh:
                    seen.add(w)
                    cand_order.append(w)

            qty_remaining = max_possible
            for wid in cand_order:
                if qty_remaining <= 0:
                    break
                avail = inventory_by_wh.get(wid, {}).get(sku, 0)
                if avail <= 0:
                    continue
                take = min(avail, qty_remaining)
                if take <= 0:
                    continue

                # commit pickups and reduce local capacity
                pickups_by_wh[wid][sku] += take
                inventory_by_wh[wid][sku] -= take
                qty_remaining -= take
                rem_w -= take * unit_w
                rem_v -= take * unit_v
                vehicle_load_weight += take * unit_w
                vehicle_load_volume += take * unit_v

                # distribute to cluster orders in simple FIFO to fill orders
                to_distribute = take
                for oid in cluster:
                    need_oid = cluster_remaining[oid].get(sku, 0)
                    if need_oid <= 0:
                        continue
                    give = min(need_oid, to_distribute)
                    allocations_by_order[oid][sku] += give
                    cluster_remaining[oid][sku] -= give
                    remaining_req[oid][sku] -= give
                    to_distribute -= give
                    if to_distribute <= 0:
                        break

            # if we could not allocate anything for this SKU, continue to next SKU
            # capacity / inventory driven loop will naturally end

        # verify total allocations for this vehicle
        total_allocated = sum(sum(v.values()) for v in allocations_by_order.values())
        if total_allocated <= 0:
            # nothing allocated -> skip vehicle (no route produced)
            # restore any inventory changes done above (we subtracted inventory_by_wh during allocation)
            # To avoid complexity and many env reads, we won't attempt to restore inventory_by_wh to original state,
            # but inventory_by_wh is local to this solver call, so it's safe. We simply continue.
            continue

        # Build route node sequence: home -> pickups -> deliveries -> home
        # Map warehouse id -> node id
        pickup_nodes = []
        for wid, items in pickups_by_wh.items():
            if items and sum(items.values()) > 0:
                nodeid = getattr(warehouse_by_id.get(wid, {}), "location", None)
                # robust extraction
                nodeid = getattr(nodeid, "id", None) if nodeid else None
                if nodeid is not None:
                    pickup_nodes.append(nodeid)

        delivery_nodes = []
        for oid, allocs in allocations_by_order.items():
            if allocs and sum(allocs.values()) > 0:
                n = customer_node_of.get(oid)
                if n is not None:
                    delivery_nodes.append(n)

        # Keep order small: nearest-neighbor heuristic on actual get_distance when possible
        route_nodes = []
        home_node = chosen_vehicle_home_node
        if home_node is None:
            # fallback: choose any warehouse node from pickups or vehicle home inferred from env
            # try env call again
            try:
                home_node = env.get_vehicle_home_warehouse(best_vid)
            except Exception:
                home_node = (pickup_nodes[0] if pickup_nodes else (delivery_nodes[0] if delivery_nodes else None))
            if home_node is None:
                # if still none, abort this vehicle
                continue

        route_nodes.append(home_node)
        targets = pickup_nodes + delivery_nodes
        cur = home_node
        # greedy selection using env distances to reduce number of A* path calls
        while targets:
            # choose best next by env.get_distance (cheaper than full A*)
            best_t = None
            best_d = 1e18
            for t in targets:
                d = get_distance(cur, t)
                if d < best_d:
                    best_d = d
                    best_t = t
            if best_t is None:
                break
            route_nodes.append(best_t)
            targets.remove(best_t)
            cur = best_t
        # end route back to home
        if route_nodes[-1] != home_node:
            route_nodes.append(home_node)

        # Build steps: we must expand to actual path nodes (A*) between route_nodes consecutive pairs
        steps = []
        # initial step at home must be present
        steps.append({"node_id": home_node, "pickups": [], "deliveries": [], "unloads": []})

        route_failed = False
        for idx in range(1, len(route_nodes)):
            a = route_nodes[idx - 1]
            b = route_nodes[idx]
            # compute path using A*; if A* fails, abort route and leave inventory_by_wh as-is (local)
            path = a_star_search(adjacency_list, a, b, get_distance, coords)
            if not path:
                route_failed = True
                break
            # append intermediate nodes (skip a because last step already references it)
            for mid in path[1:]:
                steps.append({"node_id": mid, "pickups": [], "deliveries": [], "unloads": []})

            # action at node mid==b
            # if this node is a pickup node, add pickups
            if b in pickup_nodes:
                wid = warehouse_node_to_id.get(b)
                if wid:
                    picklist = []
                    for sku, qty in pickups_by_wh.get(wid, {}).items():
                        if qty > 0:
                            picklist.append({"warehouse_id": wid, "sku_id": sku, "quantity": int(qty)})
                    # attach to last step
                    if picklist:
                        steps[-1]["pickups"] = picklist
            # if node is a delivery node, add deliveries
            if b in delivery_nodes:
                # there may be multiple orders at same node - find them
                deliver_list = []
                for oid in cluster:
                    if customer_node_of.get(oid) == b:
                        allocs = allocations_by_order.get(oid, {})
                        for sku, qty in allocs.items():
                            if qty > 0:
                                deliver_list.append({"order_id": oid, "sku_id": sku, "quantity": int(qty)})
                if deliver_list:
                    steps[-1]["deliveries"] = deliver_list

        if route_failed:
            # if route building failed, skip adding route. continue to next vehicle (no env changes done).
            continue

        # finalize: append the route for the vehicle
        solution["routes"].append({"vehicle_id": best_vid, "steps": steps})

    return solution

# ===========================
# Local test harness (COMMENT OUT when submitting)
# ===========================
# if __name__ == "__main__":
#     from robin_logistics_env import LogisticsEnvironment
#     env = LogisticsEnvironment()
#     res = solver(env)
#     print("Generated routes:", len(res.get("routes", [])))
#     if res.get("routes"):
#         print("First route steps:", len(res["routes"][0]["steps"]))
