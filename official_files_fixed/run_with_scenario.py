#!/usr/bin/env python3
"""
Run solver with a saved scenario file.
 
This allows you to:
1. Save scenario data to a JSON file
2. Load and rerun with the same scenario for testing
3. Share scenarios with team members
 
Usage:
    python run_with_scenario.py <scenario_file.json>
    
Example:
    python run_with_scenario.py saved_scenario.json
"""
 
import os
import sys
import json
import argparse
 
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
 
from robin_logistics import LogisticsEnvironment
from solver import solver
 
 
def save_scenario(env, filename):
    """Save current scenario to a JSON file.
    
    Args:
        env: LogisticsEnvironment instance
        filename: Output filename
    """
    print(f"Saving scenario to {filename}...")
    scenario = env.export_scenario()
    
    with open(filename, 'w') as f:
        json.dump(scenario, f, indent=2)
    
    print(f"✓ Scenario saved successfully")
    
    # Print scenario info
    config = env.get_stored_generation_config()
    print(f"\nScenario Info:")
    print(f"  - Random Seed: {config.get('random_seed')}")
    print(f"  - Orders: {config.get('num_orders')}")
    print(f"  - Radius: {config.get('distance_control', {}).get('radius_km')} km")
    print(f"  - Strategy: {config.get('distance_control', {}).get('density_strategy')}")


def load_and_run_scenario(filename, mode='headless'):
    """Load a scenario from file and run solver.
    
    Args:
        filename: Input scenario JSON file
        mode: 'headless' or 'dashboard'
    """
    print(f"Loading scenario from {filename}...")
    
    if not os.path.exists(filename):
        print(f"✗ File not found: {filename}")
        return
    
    try:
        with open(filename, 'r') as f:
            scenario = json.load(f)
        
        print("✓ Scenario loaded successfully")
        
        # Initialize environment and load scenario
        env = LogisticsEnvironment()
        env.load_scenario(scenario)
        
        # Print scenario info
        config = env.get_stored_generation_config()
        if config:
            print(f"\nScenario Configuration:")
            print(f"  - Random Seed: {config.get('random_seed')}")
            print(f"  - Orders: {config.get('num_orders')}")
            print(f"  - Radius: {config.get('distance_control', {}).get('radius_km')} km")
            print(f"  - Strategy: {config.get('distance_control', {}).get('density_strategy')}")
        
        if mode == 'dashboard':
            print("\nLaunching dashboard...")
            env.set_solver(solver)
            env.launch_dashboard()
        else:
            print("\nRunning solver in headless mode...")
            solution = solver(env)
            
            # Validate
            is_valid, message, _ = env.validate_solution_complete(solution)
            
            if is_valid:
                print("✓ Solution is VALID")
                
                # Execute
                success, result_message = env.execute_solution(solution)
                
                if success:
                    cost = env.calculate_solution_cost(solution)
                    print(f"\nMetrics:")
                    print(f"  - Total Cost: ${cost:,.2f}")
                    print(f"  - Routes: {len(solution.get('routes', []))}")
                else:
                    print(f"✗ Execution failed: {result_message}")
            else:
                print("✗ Solution is INVALID")
                print(f"Validation message: {message}")
    
    except json.JSONDecodeError as e:
        print(f"✗ Invalid JSON file: {e}")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description='Run solver with a saved scenario file'
    )
    parser.add_argument(
        'scenario_file',
        nargs='?',
        help='Path to scenario JSON file'
    )
    parser.add_argument(
        '--save',
        action='store_true',
        help='Save current scenario to file'
    )
    parser.add_argument(
        '--dashboard',
        action='store_true',
        help='Launch dashboard instead of headless mode'
    )
    
    args = parser.parse_args()
    
    if args.save:
        # Generate and save a new scenario
        env = LogisticsEnvironment()
        filename = args.scenario_file or 'scenario.json'
        save_scenario(env, filename)
    elif args.scenario_file:
        # Load and run with scenario
        mode = 'dashboard' if args.dashboard else 'headless'
        load_and_run_scenario(args.scenario_file, mode)
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python run_with_scenario.py --save my_scenario.json")
        print("  python run_with_scenario.py my_scenario.json")
        print("  python run_with_scenario.py my_scenario.json --dashboard")


if __name__ == "__main__":
    main()
