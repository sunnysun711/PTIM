# to test and implement GPT-generated code
from src.walk_time_dis import *

if __name__ == '__main__':
    data = get_egress_time_from_feas_iti_left()
    print(data.shape)
    print(data[np.random.choice(len(data), size=10)])

    data = get_egress_time_from_feas_iti_assigned()
    print(data.shape)
    print(data[np.random.choice(len(data), size=10)])
    pass
