# AI Hackathons - Beltone MWVRP Challenge

Multi-Warehouse Vehicle Routing Problem (MWVRP) solver for the Beltone AI Hackathon using the Robin Logistics Environment.


---

## Table of Contents

- [About](#about)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [Development](#development)
- [Submission Guide](#submission-guide)
- [Documentation](#documentation)
- [Contributing](#contributing)

---

## About

This repository contains solutions for the Beltone AI Hackathon's Vehicle Routing Problem challenge. The goal is to optimize logistics operations by efficiently routing vehicles from warehouses to customer locations while minimizing costs and meeting all constraints.

### Challenge Overview

- **Problem Type**: Multi-Warehouse Vehicle Routing Problem (MWVRP)
- **Environment**: Robin Logistics Environment
- **Objective**: Minimize operational costs while delivering all orders
- **Constraints**: Vehicle capacity, warehouse inventory, road network connectivity

---

## Project Structure

```
AI-Hackathons-Beltone-mwvrp/
├── .venv/                          # Virtual environment
├── data/                           # Data files (scenarios)
├── docs/                           # Documentation
│   ├── API_REFERENCE.md           # Complete API documentation
├── notebooks/                      # Jupyter notebooks for exploration
│   └── ayExpWkeda.ipynb
├── official_files_fixed/          # Main working directory
│   ├── solver.py                  #  Main solver implementation
│   ├── test_all.py                #  Comprehensive test suite
│   ├── run_dashboard.py           #  Dashboard runner
│   ├── run_headless.py            #  Headless runner
│   ├── run_with_scenario.py       #  Scenario manager
│   ├── manual_assignments_dashboard.py  #  Alternative solver
│   ├── API_REFERENCE.md           # API documentation
│   └── README.md                  # Official files guide
├── submissions/                    # Submission files directory
│   └── Overfitting_solver_1.py    # Example submission
├── src/                           # Source code
│   └── main.py
├── tests/                         # Test files
│   └── test_all.py
├── requirements.txt               # Python dependencies
├── setup.sh                       # Setup script
└── README.md                      # This file
```

---

## Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Setup

1. Clone the repository
```sh
git clone https://github.com/Belal0066/AI-Hackathons-Beltone-mwvrp.git
cd AI-Hackathons-Beltone-mwvrp
```

2. Create and activate virtual environment
```sh
python3 -m venv .venv
source .venv/bin/activate  
# On Windows: .venv\Scripts\activate
```

3. Install dependencies
Install the Robin Logistics Environment
```bash
pip install robin-logistics-env
```

Install from requirements.txt (if it has something added)
```
pip install -r requirements.txt
```

```bash
pip install --upgrade robin-logistics-env
```



## Usage

### Running the Solver

#### 1. **Test Everything** (Recommended First Step)

```bash
cd official_files_fixed
python3 test_all.py
```

**Output:**
```
✓ Package 'robin-logistics-env' is installed
✓ Environment initialized successfully
✓ Found 50 orders, 12 vehicles, 2 warehouses
✓ Road network has 7522 nodes
✓ Solver function imported successfully
✓ Solver generated solution with 12 routes
✓ Solution is VALID
```

#### 2. **Run Headless** (Fast Execution, No UI)

```bash
python3 run_headless.py
```

**Output:**
```
✓ Solution is VALID
Total Cost: $6,858.16
Number of Routes: 12
Orders: 50
Vehicles Used: 12
```

#### 3. **Run Dashboard** (Visual Interface)

```bash
python3 run_dashboard.py
```

Opens interactive dashboard on `http://localhost:8501`

#### 4. **Test with Scenarios**

```bash
# Save current scenario
python3 run_with_scenario.py --save my_test_scenario.json

# Load and run saved scenario
python3 run_with_scenario.py my_test_scenario.json

# Load scenario with dashboard
python3 run_with_scenario.py my_test_scenario.json --dashboard
```

---


### Submission Portal

Submit your file to: [AI Hackathon Submission Portal](https://submission-portal-link)

View results on: [Results Dashboard](https://results-dashboard-link)

---


