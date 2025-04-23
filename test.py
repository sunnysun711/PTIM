# to test and implement GPT-generated code
import numpy as np
import time

from src import config


def test_performance():
    from src.globals import get_etd, get_pl_info
    # 测试不同规模的 walk_time
    sizes = [100, 1000, 10000, 100000, 1000000, 10000000, 100000000]
    etd = get_etd()
    print(np.unique(etd[:, 0]))
    pl_id = np.random.choice(get_pl_info()[:, 0], size=1)[0]
    print(get_pl_info())
    print(f"pl_id: {pl_id}")
    etd = etd[etd[:, 0] == pl_id][:, [1, 2]]  # Keep only x and pdf columns
    print(etd)
    x2pdf = dict(zip(etd[:, 0], etd[:, 1]))  # Create a dictionary mapping x to pdf
    print(x2pdf)
    for size in sizes:
        print(f"\nTesting with walk_time size: {size}")
        walk_time = np.random.randint(0, 500, size=size)
        # x2pdf = {i: i / 500 for i in range(500)}

        # 方法 1: np.vectorize
        start = time.time()
        pdf_values_1 = np.vectorize(x2pdf.get, otypes=[float])(walk_time, None)
        print("np.vectorize:", time.time() - start)

        # 方法 2: 列表推导式
        start = time.time()
        pdf_values_2 = np.array([x2pdf[wt] for wt in walk_time])
        print("列表推导式:", time.time() - start)

        # 方法 3: np.fromiter
        start = time.time()
        pdf_values_3 = np.fromiter((x2pdf[wt] for wt in walk_time), dtype=float)
        print("np.fromiter:", time.time() - start)


def test():
    from src.utils import read_, get_latest_file_index, get_file_path

    config.load_config()

    from src.walk_time_dis import get_pdf, get_cdf, node_id_to_pl_id

    res = get_pdf(walk_time=np.random.randint(low=0, high=600, size=100), pl_id=40)
    print(res)

    print(res.shape)

    print(get_pdf(walk_time=182, pl_id=40))

    res = get_cdf(
        pl_id=40,
        t_start=np.random.randint(low=0, high=500, size=100),
        t_end=np.random.randint(low=0, high=500, size=100),
    )
    print(res)

    print(get_cdf(
        pl_id=40,
        t_start=18,
        t_end=200
    ))

    # print(node_id_to_pl_id(node_id=107511))


if __name__ == '__main__':
    config.load_config()
    # test_performance()
    test()
    pass
