def step1():
    """
    Step 1: Prepare the metro network structure for pathfinding.
    """
    from scripts.prep_network import main as prep_network_main
    prep_network_main()


def step2():
    """
    Step 2: Find feasible itineraries for each trajectory.
    """
    from scripts.find_feas_iti import main as find_feas_iti_main
    find_feas_iti_main()


def step3():
    """
    Step 3:
    """
    pass


if __name__ == "__main__":
    step1()
    step2()
    pass
