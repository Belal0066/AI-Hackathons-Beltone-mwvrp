#!/usr/bin/env python3
"""
Manual Assignments Solver for the Robin Logistics Environment.

This solver allows you to manually define vehicle-to-order assignments
and uses BFS pathfinding for routing.

IMPORTANT SUBMISSION RULES:
1. The main function MUST be named: solver(env)
2. Do NOT use any caching techniques
3. Do NOT import or initialize the environment inside the solver function
4. Comment out the main function when submitting the solver file
"""

from robin_logistics import LogisticsEnvironment
from typing import Dict, List, Optional
from collections import deque


# Define your manual assignments here
# Format: list of dicts with 'vehicle_id' and 'order_ids'
ASSIGNMENTS = [
    # Example:
    # {
    #     "vehicle_id": "VEH_001",
    #     "order_ids": ["ORD_001", "ORD_002", "ORD_003"]
    # },
    # {
    #     "vehicle_id": "VEH_002", 
    #     "order_ids": ["ORD_004", "ORD_005"]
    # }
]


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


def manual_assignments_solver_main(env) -> Dict:
    """Generate solution based on manual assignments.
    
    Args:
        env: LogisticsEnvironment instance
        
    Returns:
        A complete solution dict with routes and sequential steps.
    """
    solution = {"routes": []}
    
    # Get road network data
    road_network = env.get_road_network_data()
    adjacency_list = road_network.get("adjacency_list", {})
    
    # If no manual assignments, fall back to simple assignment
    if not ASSIGNMENTS:
        print("Warning: No manual assignments defined. Using automatic assignment.")
        return automatic_fallback_solver(env)
    
    # Get all vehicles
    all_vehicles = env.get_all_vehicles()
    vehicles_dict = {v.id: v for v in all_vehicles}
    
    # Process each manual assignment
    for assignment in ASSIGNMENTS:
        vehicle_id = assignment.get("vehicle_id")
        order_ids = assignment.get("order_ids", [])
        
        if not vehicle_id or not order_ids:
            continue
        
        # Get vehicle and warehouse info
        vehicle = vehicles_dict.get(vehicle_id)
        if not vehicle:
            print(f"Warning: Vehicle {vehicle_id} not found, skipping...")
            continue
        
        warehouse_id = vehicle.home_warehouse_id
        warehouse = env.get_warehouse_by_id(warehouse_id)
        warehouse_node = warehouse.location.id
        
        steps = []
        current_location = warehouse_node
        all_pickups = []
        
        # Collect all items to load
        for order_id in order_ids:
            if order_id not in env.orders:
                print(f"Warning: Order {order_id} not found, skipping...")
                continue
            
            order = env.orders[order_id]
            for sku_id, quantity in order.requested_items.items():
                all_pickups.append({
                    "warehouse_id": warehouse_id,
                    "sku_id": sku_id,
                    "quantity": quantity
                })
        
        if not all_pickups:
            continue
        
        # Step 1: Load all items at warehouse
        steps.append({
            "node_id": warehouse_node,
            "pickups": all_pickups,
            "deliveries": [],
            "unloads": []
        })
        
        # Visit each customer to deliver
        for order_id in order_ids:
            if order_id not in env.orders:
                continue
            
            order = env.orders[order_id]
            customer_node = order.destination.id
            
            # Find path from current location to customer
            path = bfs_shortest_path(adjacency_list, current_location, customer_node)
            
            if not path:
                print(f"Warning: No path from {current_location} to {customer_node}")
                continue
            
            # Travel to customer (intermediate nodes)
            for node in path[1:-1]:
                steps.append({
                    "node_id": node,
                    "pickups": [],
                    "deliveries": [],
                    "unloads": []
                })
            
            # Deliver order
            deliveries = []
            for sku_id, quantity in order.requested_items.items():
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
            
            current_location = customer_node
        
        # Return to warehouse
        path_home = bfs_shortest_path(adjacency_list, current_location, warehouse_node)
        if path_home:
            # Intermediate nodes
            for node in path_home[1:-1]:
                steps.append({
                    "node_id": node,
                    "pickups": [],
                    "deliveries": [],
                    "unloads": []
                })
            
            # Arrive at warehouse
            steps.append({
                "node_id": warehouse_node,
                "pickups": [],
                "deliveries": [],
                "unloads": []
            })
        
        # Add route to solution
        if steps:
            solution["routes"].append({
                "vehicle_id": vehicle_id,
                "steps": steps
            })
    
    return solution


def automatic_fallback_solver(env) -> Dict:
    """Simple fallback solver when no manual assignments provided."""
    solution = {"routes": []}
    
    order_ids = env.get_all_order_ids()
    vehicle_ids = env.get_available_vehicles()
    road_network = env.get_road_network_data()
    adjacency_list = road_network.get("adjacency_list", {})
    
    # Get all vehicles
    all_vehicles = env.get_all_vehicles()
    vehicles_dict = {v.id: v for v in all_vehicles}
    
    # Simple one-to-one assignment
    for i, order_id in enumerate(order_ids):
        if i >= len(vehicle_ids):
            break
        
        vehicle_id = vehicle_ids[i]
        order = env.orders[order_id]
        vehicle = vehicles_dict.get(vehicle_id)
        
        if not vehicle:
            continue
        
        warehouse_id = vehicle.home_warehouse_id
        warehouse = env.get_warehouse_by_id(warehouse_id)
        warehouse_node = warehouse.location.id
        customer_node = order.destination.id
        
        path_to_customer = bfs_shortest_path(adjacency_list, warehouse_node, customer_node)
        path_to_warehouse = bfs_shortest_path(adjacency_list, customer_node, warehouse_node)
        
        if path_to_customer and path_to_warehouse:
            steps = []
            
            # Pickup at warehouse
            pickups = []
            for sku_id, quantity in order.requested_items.items():
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
            
            # Travel to customer (intermediate nodes)
            for node in path_to_customer[1:-1]:
                steps.append({
                    "node_id": node,
                    "pickups": [],
                    "deliveries": [],
                    "unloads": []
                })
            
            # Deliver at customer
            deliveries = []
            for sku_id, quantity in order.requested_items.items():
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
            
            # Return to warehouse (intermediate nodes)
            for node in path_to_warehouse[1:-1]:
                steps.append({
                    "node_id": node,
                    "pickups": [],
                    "deliveries": [],
                    "unloads": []
                })
            
            # Arrive at warehouse
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


# IMPORTANT: Comment out this main section when submitting!
# Uncomment for local testing only
"""
if __name__ == '__main__':
    env = LogisticsEnvironment()
    env.set_solver(manual_assignments_solver_main)
    env.launch_dashboard()
"""
