#!/usr/bin/env python3

# pip install robin-logistics-env  --- Before first run install in the terminal
"""
Contestant solver for the Robin Logistics Environment.
 
Generates a valid solution using basic assignment and BFS-based routing.

IMPORTANT SUBMISSION RULES:
1. The main function MUST be named: solver(env)
2. Do NOT use any caching techniques
3. Do NOT import or initialize the environment inside the solver function
4. Comment out the main function when submitting the solver file
"""
from robin_logistics import LogisticsEnvironment
from typing import Dict, List, Optional
from collections import deque
 
 
def solver(env) -> Dict:
    """Generate a simple, valid solution using the road network.
 
    Args:
        env: LogisticsEnvironment instance
 
    Returns:
        A complete solution dict with routes and sequential steps.
    """
    solution = {"routes": []}
 
    # Get all necessary data
    order_ids: List[str] = env.get_all_order_ids()
    available_vehicle_ids: List[str] = env.get_available_vehicles()
    road_network = env.get_road_network_data()
    adjacency_list = road_network.get("adjacency_list", {})
    
    # Get all vehicles
    all_vehicles = env.get_all_vehicles()
    vehicles_dict = {v.id: v for v in all_vehicles}
    
    # Basic implementation: assign orders to vehicles (one-to-one)
    for i, order_id in enumerate(order_ids):
        if i >= len(available_vehicle_ids):
            break
            
        vehicle_id = available_vehicle_ids[i]
        
        # Get order and vehicle using correct API
        order = env.orders[order_id]
        vehicle = vehicles_dict.get(vehicle_id)
        
        if not vehicle:
            continue
        
        # Get warehouse and customer locations
        warehouse_id = vehicle.home_warehouse_id
        warehouse = env.get_warehouse_by_id(warehouse_id)
        warehouse_node = warehouse.location.id
        customer_node = order.destination.id
        
        # Get order items
        order_items = order.requested_items
        
        # Build a simple route: warehouse -> customer -> warehouse
        path_to_customer = bfs_shortest_path(adjacency_list, warehouse_node, customer_node)
        path_to_warehouse = bfs_shortest_path(adjacency_list, customer_node, warehouse_node)
        
        if path_to_customer and path_to_warehouse:
            steps = []
            
            # Step 1: Pickup at warehouse
            pickups = []
            for sku_id, quantity in order_items.items():
                pickups.append({
                    "warehouse_id": warehouse_id,
                    "sku_id": sku_id,
                    "quantity": quantity
                })
            
            steps.append({
                "node_id": warehouse_node,
                "pickups": pickups,
                "deliveries": [],
                "unloads": []
            })
            
            # Step 2: Travel to customer (intermediate nodes)
            for node in path_to_customer[1:-1]:
                steps.append({
                    "node_id": node,
                    "pickups": [],
                    "deliveries": [],
                    "unloads": []
                })
            
            # Step 3: Deliver at customer
            deliveries = []
            for sku_id, quantity in order_items.items():
                deliveries.append({
                    "order_id": order_id,
                    "sku_id": sku_id,
                    "quantity": quantity
                })
            
            steps.append({
                "node_id": customer_node,
                "pickups": [],
                "deliveries": deliveries,
                "unloads": []
            })
            
            # Step 4: Return to warehouse (intermediate nodes)
            for node in path_to_warehouse[1:-1]:
                steps.append({
                    "node_id": node,
                    "pickups": [],
                    "deliveries": [],
                    "unloads": []
                })
            
            # Step 5: Arrive back at warehouse
            steps.append({
                "node_id": warehouse_node,
                "pickups": [],
                "deliveries": [],
                "unloads": []
            })
            
            solution["routes"].append({
                "vehicle_id": vehicle_id,
                "steps": steps
            })
    
    return solution


def bfs_shortest_path(adjacency_list: Dict, start: str, goal: str) -> Optional[List[str]]:
    """Find shortest path using BFS.
    
    Args:
        adjacency_list: Graph adjacency list
        start: Start node ID
        goal: Goal node ID
        
    Returns:
        List of node IDs representing the path, or None if no path exists
    """
    if start == goal:
        return [start]
    
    queue = deque([(start, [start])])
    visited = {start}
    
    while queue:
        current, path = queue.popleft()
        
        for neighbor in adjacency_list.get(current, []):
            if neighbor == goal:
                return path + [neighbor]
            
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    
    return None



if __name__ == '__main__':
    env = LogisticsEnvironment()
    solution = solver(env)
    print("Solution generated:")
    print(f"Number of routes: {len(solution.get('routes', []))}")

 
