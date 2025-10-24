#!/usr/bin/env python3
"""Debug script to check what's happening with order fulfillment."""

from robin_logistics import LogisticsEnvironment
from solver import solver

# Initialize environment
env = LogisticsEnvironment()

# Get initial order count
initial_orders = len(env.get_all_order_ids())
print(f"Initial orders: {initial_orders}")

# Run solver
print("\nRunning solver...")
solution = solver(env)
print(f"Routes created: {len(solution['routes'])}")

# Check what orders are in the solution
orders_in_solution = set()
for route in solution['routes']:
    for step in route['steps']:
        for delivery in step.get('deliveries', []):
            orders_in_solution.add(delivery['order_id'])

print(f"Unique orders in solution: {len(orders_in_solution)}")
print(f"Sample orders: {list(orders_in_solution)[:5]}")

# Validate
print("\nValidating...")
is_valid, message, _ = env.validate_solution_complete(solution)
print(f"Valid: {is_valid}")
if not is_valid:
    print(f"Message: {message}")

# Execute
print("\nExecuting...")
success, exec_message = env.execute_solution(solution)
print(f"Success: {success}")
if not success:
    print(f"Message: {exec_message}")

# Check order statuses
print("\nChecking order statuses...")
fulfilled_count = 0
for oid in env.get_all_order_ids():
    status = env.get_order_fulfillment_status(oid)
    if status.get('is_fulfilled', False):
        fulfilled_count += 1

print(f"Fulfilled orders: {fulfilled_count}/{initial_orders}")

# Check solution fulfillment summary
fulfillment = env.get_solution_fulfillment_summary(solution)
print(f"\nSolution fulfillment summary: {fulfillment}")
