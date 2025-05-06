"""
This script assigns feasible itineraries to passengers using different strategies
(e.g., greedy assignment, dynamic assignment) and updates the assigned trajectories.

Key Outputs:
- assigned.pkl: Assigned passenger itineraries
- left.pkl: Remaining passengers without assignment
- stashed.pkl: Passengers with excessive feasible itineraries

Dependencies:
- src.trajectory
"""
from src import config
from src.trajectory import assign_greedy_all, dynamic_assignment

def main():
    print("\033[33m"
      "======================================================================================\n"
      "[INFO] This script assigns feasible itineraries to passengers using the specified strategy.\n"
      "       Supported strategies:\n"
      "         - greedy: Assign passengers all at once based on the most probable feasible itineraries.\n"
      "         - dynamic: Iteratively assign passengers with probability adjustment and rollback.\n"
      "       Key Outputs:\n"
      "         - assigned_*.pkl: [rid, iti_id, path_id, seg_id, train_id, board_ts, alight_ts].\n"
      "         - left.pkl: Updated remaining passengers without assigned itinerary.\n"
      "         - assignment.log: log info automatically generated.\n"
      "======================================================================================"
      "\033[0m")

    if config.CONFIG["parameters"]["strategy"] == "greedy":
        assign_greedy_all()
    elif config.CONFIG["parameters"]["strategy"] == "dynamic":
        dynamic_assignment()
    else:
        raise NotImplementedError
    ...
    
    
if __name__ == "__main__":
    config.load_config("configs/config.yaml")
    # main()
    pass