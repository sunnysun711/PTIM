# PTIM - Passenger Trajectory Inference Model for Urban Rail Transit

A data-driven framework for inferring metro passenger trajectories under physical constraints and congestion dynamics.

## Author

**Yichen Sun**

## Overview

PTIM is a comprehensive passenger trajectory inference system designed for urban rail transit networks. The model combines physical network constraints, timetable information, and congestion dynamics to reconstruct detailed passenger movements from AFC (Automated Fare Collection) transaction data.

## Project Structure

```text
PTIM/
├── configs/                         # Configuration files
│   ├── config1.yaml                 # Primary configuration file
│   └── config2.yaml                 # Alternative configuration
├── data/                            # Input data files
│   ├── AFC.pkl                      # Automated Fare Collection transaction data
│   ├── TT.pkl                       # Train timetable data
│   ├── STA.pkl                      # Station data
│   ├── platform.json                # Platform layout information
│   └── coordinates.csv              # Station geographic coordinates
├── src/                             # Core source modules
│   ├── config.py                    # Configuration loader and manager
│   ├── congest_penal.py             # Congestion penalty calculations
│   ├── globals.py                   # Global variables and constants
│   ├── itinerary.py                 # Itinerary generation logic
│   ├── metro_net.py                 # Metro network representation
│   ├── passenger.py                 # Passenger data structures
│   ├── timetable.py                 # Timetable processing
│   ├── trajectory.py                # Trajectory inference algorithms
│   ├── utils.py                     # Utility functions
│   ├── walk_time_dis_calculator.py  # Walking time distribution calculator
│   ├── walk_time_dis_fit.py         # Distribution fitting for walking times
│   ├── walk_time_filter.py          # Walking time data filtering
│   └── walk_time_plot.py            # Walking time visualization
├── scripts/                         # Pipeline execution scripts
│   ├── prep_network.py              # Network preprocessing
│   ├── find_feas_iti.py             # Feasible itinerary generation
│   ├── split_feas_iti.py            # Trajectory splitting
│   ├── analyze_walk_time.py         # Walking time analysis
│   └── calculate_distribution.py    # Distribution calculation
├── figures/                         # Output visualizations
├── results/                         # Processing results (generated)
│   ├── network/                     # Network files (nodes, links, paths)
│   ├── itinerary/                   # Feasible itineraries
│   ├── trajectory/                  # Inferred trajectories
│   ├── egress/                      # Egress time distributions
│   └── transfer/                    # Transfer time distributions
├── run.py                           # Main execution script
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## Installation

### Prerequisites

- Python 3.7+
- pip package manager

### Dependencies

Install required packages using:

```bash
pip install -r requirements.txt
```

Required packages:
- numpy
- pandas
- networkx
- matplotlib
- tqdm
- seaborn
- scipy
- pyyaml
- joblib

## Example Usage

### Basic Usage

Run the complete trajectory inference pipeline with the default configuration:

```bash
python run.py
```

### Custom Configuration

Specify a custom configuration file:

```bash
python run.py --config configs/config2.yaml
```

Or use the short form:

```bash
python run.py -c configs/config2.yaml
```

### Configuration Options

The configuration file ([config1.yaml](configs/config1.yaml)) controls the entire pipeline:

**Data Settings:**

- Input data folder and file paths
- Output results folder structure

**Pipeline Control:**

- `use_existing.network`: Skip network preprocessing if already generated
- `use_existing.itinerary`: Skip itinerary generation if already generated
- `use_existing.trajectory`: Skip trajectory splitting if already generated
- `use_existing.walk_times`: Skip walking time analysis if already generated

**Model Parameters:**

- `k`: Number of candidate paths (default: 10)
- `theta1`, `theta2`: Path selection thresholds
- `transfer_deviation`: Transfer time deviation tolerance
- `distribution_type`: Walking time distribution ("kde", "gamma", or "lognorm")
- `penalty_type`: Congestion penalty function type
- `strategy`: Trajectory assignment strategy ("random" or "greedy")
- Train capacity parameters for different line types

### Pipeline Stages

The model executes the following stages:

1. **Network Preprocessing** ([prep_network.py](scripts/prep_network.py))
   - Constructs the metro network graph
   - Identifies transfer stations and terminals
   - Computes k-shortest paths between all station pairs

2. **Feasible Itinerary Generation** ([find_feas_iti.py](scripts/find_feas_iti.py))
   - Matches AFC transactions to candidate paths
   - Integrates timetable information
   - Generates feasible itineraries for each passenger

3. **Trajectory Splitting** ([split_feas_iti.py](scripts/split_feas_iti.py))
   - Splits itineraries into manageable batches
   - Handles edge cases and outliers

4. **Walking Time Analysis** ([analyze_walk_time.py](scripts/analyze_walk_time.py))
   - Extracts egress and transfer times
   - Fits probability distributions

5. **Distribution Calculation** ([calculate_distribution.py](scripts/calculate_distribution.py))
   - Computes walking time distributions
   - Generates statistical summaries

## Output Files

Results are organized in the `results/` folder:

### Network Files

- `node.csv`: Network nodes with station and line information
- `link.csv`: Network links with weights
- `platform.csv`: Platform-to-node mappings
- `path.pkl`: k-shortest paths between OD pairs
- `pathvia.pkl`: Detailed path segments

### Itinerary Files

- `feas_iti.pkl`: Feasible itineraries with train assignments
- `AFC_no_iti.pkl`: AFC records without feasible itineraries

### Trajectory Files

- `assigned.pkl`: Successfully assigned trajectories
- `left.pkl`: Unassigned trajectories
- `stashed.pkl`: Outlier trajectories
- `dis.pkl`: Detailed trajectory distributions

### Walking Time Files

- `egress_times.pkl`: Egress time observations
- `transfer_times.pkl`: Transfer time observations
- `etd.csv`: Egress time distributions with fit statistics
- `ttd.csv`: Transfer time distributions with fit statistics

## Contact

For questions, suggestions, or collaborations:

- **Yichen Sun**
- Email: <yichensun@my.swjtu.edu.cn>
- Email: <sun_yichen@foxmail.com>

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{LiuTITS2026DataDrivenMetroPassenger,
  title = {Data-{{Driven Metro Passenger Trajectory Inference Under Physical Constraints}} and {{Congestion Dynamics}}},
  author = {Liu, Zhanru and Shuai, Bin and Sun, Yichen and Chen, Dingjun and Zhang, Qingpeng and Lv, Miaomiao},
  year = 2026,
  journal = {IEEE Transactions on Intelligent Transportation Systems},
  pages = {1--18},
  issn = {1558-0016},
  doi = {10.1109/TITS.2026.3656019},
  urldate = {2026-02-03}
}
```

## Acknowledgments

This work was published in IEEE Transactions on Intelligent Transportation Systems. The methodology incorporates physical network constraints, real-time congestion dynamics, and statistical inference techniques to achieve accurate passenger trajectory reconstruction in complex urban rail transit systems.
