#!/usr/bin/env python3
"""Debug inventory issues."""

from robin_logistics import LogisticsEnvironment

env = LogisticsEnvironment()

print("="*60)
print("WAREHOUSE INVENTORY")
print("="*60)

for wh_id, wh in env.warehouses.items():
    print(f"\n{wh_id}:")
    print(f"  Location: {wh.location.id}")
    print(f"  Inventory:")
    for sku_id, qty in wh.inventory.items():
        print(f"    {sku_id}: {qty}")
    print(f"  Vehicles: {len(wh.vehicles)}")
    for v in wh.vehicles:
        print(f"    {v.id}: {v.type} (cap: {v.capacity_weight}kg, {v.capacity_volume}m³)")

print("\n" + "="*60)
print("TOTAL DEMAND")
print("="*60)

from collections import defaultdict
total_demand = defaultdict(int)

for oid, order in env.orders.items():
    for sku_id, qty in order.requested_items.items():
        total_demand[sku_id] += qty

for sku_id, qty in total_demand.items():
    print(f"{sku_id}: {qty}")

print("\n" + "="*60)
print("INVENTORY VS DEMAND")
print("="*60)

total_inventory = defaultdict(int)
for wh in env.warehouses.values():
    for sku_id, qty in wh.inventory.items():
        total_inventory[sku_id] += qty

for sku_id in total_demand.keys():
    inv = total_inventory[sku_id]
    dem = total_demand[sku_id]
    ratio = (inv / dem * 100) if dem > 0 else 0
    status = "✓" if inv >= dem else "✗ SHORTAGE!"
    print(f"{sku_id}: {inv}/{dem} ({ratio:.0f}%) {status}")
