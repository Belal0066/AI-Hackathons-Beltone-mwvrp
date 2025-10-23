#!/usr/bin/env python3
"""
Headless runner for the Robin Logistics Environment.
 
Runs the solver without the dashboard UI and prints basic statistics.
Launch with: python run_headless.py
"""
 
import os
import sys
 
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
 
from robin_logistics import LogisticsEnvironment
from solver import solver
 
 
def main():
    """Run solver in headless mode and display results."""
    env = LogisticsEnvironment()
    
    print("=" * 60)
    print("Running solver in headless mode...")
    print("=" * 60)
    
    # Generate solution
    solution = solver(env)
    
    # Validate solution
    print("\nValidating solution...")
    is_valid, message, _ = env.validate_solution_complete(solution)
    
    if is_valid:
        print("✓ Solution is VALID")
        
        # Execute solution to get metrics
        print("\nExecuting solution...")
        success, result_message = env.execute_solution(solution)
        
        if success:
            # Get metrics after execution
            stats = env.get_solution_statistics(solution)
            cost = env.calculate_solution_cost(solution)
            
            print("\n" + "=" * 60)
            print("SOLUTION METRICS")
            print("=" * 60)
            print(f"Total Cost: ${cost:,.2f}")
            print(f"Number of Routes: {len(solution.get('routes', []))}")
            print(f"Orders: {len(env.get_all_order_ids())}")
            print(f"Vehicles Used: {len(solution.get('routes', []))}")
            print("=" * 60)
        else:
            print(f"✗ Execution failed: {result_message}")
    else:
        print("✗ Solution is INVALID")
        print(f"Validation message: {message}")
    
    print("\nHeadless run complete!")
 
 
if __name__ == "__main__":
    main()
