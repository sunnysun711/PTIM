# to test and implement GPT-generated code
from src.walk_time_dis import *

if __name__ == '__main__':
    data1 = get_egress_time_from_feas_iti_left()
    data2 = get_egress_time_from_feas_iti_assigned()
    data = np.vstack((data1, data2))
    print(data.shape)

    uid2time: dict[int, np.ndarray] = {uid: np.array([]) for uid in range(1001, 1137)}
    for uid in range(1001, 1137):
        uid2time[uid] = data[data[:, 1] == uid, -1]
        print(uid, uid2time[uid].shape)

    print(min([uid2time[uid].shape[0] for uid in range(1001, 1137)]))  # 1004 (94,)

    # nd1nd22time: dict[tuple[int, int], np.ndarray] = {}
    df = pd.DataFrame(data, columns=["node1", "node2", "alight_ts", "ts2", "egress_time"])
    nd1nd22time = df.groupby(["node1", "node2"])["egress_time"].apply(np.array).to_dict()
    print([(key, nd1nd22time[key].shape[0]) for key in nd1nd22time])
    pass
