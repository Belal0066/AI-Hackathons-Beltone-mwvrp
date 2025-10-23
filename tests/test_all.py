#!/usr/bin/env python3
"""
Test script to verify the Robin Logistics Environment setup.
 
Tests:
1. Package installation
2. Environment initialization
3. Basic API access
4. Solver function
5. Solution validation
 
Launch with: python test_all.py
"""
 
import os
import sys

## TODO m7tag sovler, fel src ba2a
 
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
 
 
def test_package_installation():
    """Test if robin-logistics-env package is installed."""
    print("Testing package installation...")
    try:
        import robin_logistics
        print("✓ Package 'robin-logistics-env' is installed")
        return True
    except ImportError:
        print("✗ Package 'robin-logistics-env' is NOT installed")
        print("  Run: pip install robin-logistics-env")
        return False
 
 
def test_environment_initialization():
    """Test if environment can be initialized."""
    print("\nTesting environment initialization...")
    try:
        from robin_logistics import LogisticsEnvironment
        env = LogisticsEnvironment()
        print("✓ Environment initialized successfully")
        return True, env
    except Exception as e:
        print(f"✗ Failed to initialize environment: {e}")
        return False, None
 
 
def test_basic_api_access(env):
    """Test basic API methods."""
    print("\nTesting basic API access...")
    try:
        # Test various API methods
        orders = env.get_all_order_ids()
        vehicles = env.get_available_vehicles()
        warehouses = env.warehouses
        road_network = env.get_road_network_data()
        
        print(f"✓ Found {len(orders)} orders")
        print(f"✓ Found {len(vehicles)} vehicles")
        print(f"✓ Found {len(warehouses)} warehouses")
        print(f"✓ Road network has {len(road_network.get('adjacency_list', {}))} nodes")
        return True
    except Exception as e:
        print(f"✗ API access failed: {e}")
        return False
 
 
def test_solver_function():
    """Test if solver function exists and can be imported."""
    print("\nTesting solver function...")
    try:
        from solver import solver
        print("✓ Solver function imported successfully")
        
        # Check if it's callable
        if callable(solver):
            print("✓ Solver is callable")
            return True, solver
        else:
            print("✗ Solver is not callable")
            return False, None
    except ImportError as e:
        print(f"✗ Failed to import solver: {e}")
        return False, None
    except Exception as e:
        print(f"✗ Error with solver: {e}")
        return False, None
 
 
def test_solver_execution(env, solver):
    """Test if solver can generate a solution."""
    print("\nTesting solver execution...")
    try:
        solution = solver(env)
        
        if not isinstance(solution, dict):
            print(f"✗ Solver must return a dict, got {type(solution)}")
            return False, None
        
        if "routes" not in solution:
            print("✗ Solution missing 'routes' key")
            return False, None
        
        num_routes = len(solution.get("routes", []))
        print(f"✓ Solver generated solution with {num_routes} routes")
        return True, solution
    except Exception as e:
        print(f"✗ Solver execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None
 
 
def test_solution_validation(env, solution):
    """Test if solution can be validated."""
    print("\nTesting solution validation...")
    try:
        is_valid, message, _ = env.validate_solution_complete(solution)
        
        if is_valid:
            print("✓ Solution is VALID")
            return True
        else:
            print("✗ Solution is INVALID")
            print(f"  Validation message: {message}")
            return False
    except Exception as e:
        print(f"✗ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
 
 
def main():
    """Run all tests."""
    print("=" * 60)
    print("Robin Logistics Environment - Setup Test")
    print("=" * 60)
    
    all_passed = True
    
    # Test 1: Package installation
    if not test_package_installation():
        all_passed = False
        print("\n" + "=" * 60)
        print("TESTING ABORTED: Package not installed")
        print("=" * 60)
        return
    
    # Test 2: Environment initialization
    success, env = test_environment_initialization()
    if not success:
        all_passed = False
        print("\n" + "=" * 60)
        print("TESTING ABORTED: Environment initialization failed")
        print("=" * 60)
        return
    
    # Test 3: Basic API access
    if not test_basic_api_access(env):
        all_passed = False
    
    # Test 4: Solver function
    success, solver = test_solver_function()
    if not success:
        all_passed = False
        print("\n" + "=" * 60)
        print("TESTING ABORTED: Solver function not available")
        print("=" * 60)
        return
    
    # Test 5: Solver execution
    success, solution = test_solver_execution(env, solver)
    if not success:
        all_passed = False
    
    # Test 6: Solution validation
    if solution is not None:
        if not test_solution_validation(env, solution):
            all_passed = False
    
    # Final summary
    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
        print("\nYou're ready to start! Try running:")
        print("  python run_headless.py    # Run without UI")
        print("  python run_dashboard.py   # Run with dashboard")
    else:
        print("SOME TESTS FAILED ✗")
        print("=" * 60)
        print("\nPlease fix the issues above before proceeding.")
    print("=" * 60)
 
 
if __name__ == "__main__":
    main()
