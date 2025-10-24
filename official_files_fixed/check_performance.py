#!/usr/bin/env python3
"""Quick performance check script for the solver."""

from robin_logistics import LogisticsEnvironment
from solver import solver

# Initialize environment
env = LogisticsEnvironment()

# Run solver
print("Running solver...")
solution = solver(env)

# Validate solution
print("\nValidating solution...")
validation_result = env.validate_solution_complete(solution)
is_valid = validation_result[0] if isinstance(validation_result, tuple) else validation_result

if not is_valid:
    message = validation_result[1] if isinstance(validation_result, tuple) and len(validation_result) > 1 else "Unknown"
    print(f"❌ Solution is INVALID: {message}")
    exit(1)

print("✓ Solution is valid")

# Execute solution
print("\nExecuting solution...")
success, message = env.execute_solution(solution)
if not success:
    print(f"❌ Execution failed: {message}")
    exit(1)

print("✓ Solution executed successfully")

# Get metrics
print("\n" + "="*60)
print("PERFORMANCE SUMMARY")
print("="*60)

# Basic stats
num_routes = len(solution.get('routes', []))
print(f"Routes generated: {num_routes}")

# Get fulfillment
fulfillment_summary = env.get_solution_fulfillment_summary(solution)
print(f"\nFulfillment:")
print(f"  Total orders: {fulfillment_summary.get('total_orders', 0)}")
print(f"  Fulfilled orders: {fulfillment_summary.get('fulfilled_orders', 0)}")
print(f"  Fulfillment rate: {fulfillment_summary.get('fulfillment_percentage', 0):.1f}%")

# Get cost
cost = env.calculate_solution_cost(solution)
print(f"\nTotal Cost: ${cost:,.2f}")

# Get cost breakdown
cost_breakdown = env.metrics_calculator.calculate_cost_breakdown(solution)
print(f"  Fixed cost: ${cost_breakdown.get('fixed_cost_total', 0):,.2f}")
print(f"  Variable cost: ${cost_breakdown.get('variable_cost_total', 0):,.2f}")

# Calculate scenario score (assuming benchmark cost = $9000)
BENCHMARK_COST = 9000
fulfillment_pct = fulfillment_summary.get('fulfillment_percentage', 0)
scenario_score = cost + (BENCHMARK_COST * (100 - fulfillment_pct))
print(f"\nScenario Score (estimate): {scenario_score:,.0f}")
print(f"  (Assumes benchmark cost = ${BENCHMARK_COST:,})")

print("="*60)
