# to test and implement GPT-generated code
import numpy as np

import numpy as np
import time


def test_performance():
    # 模拟数据
    # 测试不同规模，如 100, 1000, 100000
    walk_time = np.random.randint(0, 500, size=100000)
    x2pdf = {i: i / 500 for i in range(500)}

    # 方法 1: np.vectorize
    start = time.time()
    pdf_values_1 = np.vectorize(x2pdf.get, otypes=[float])(walk_time, 0)
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
    from src import config

    config.load_config()

    from src.walk_time_dis import get_pdf, get_cdf, node_id_to_pl_id

    res = get_pdf(walk_time=[np.random.randint(low=0, high=500)
                  for _ in range(50)], pl_id=40)
    print(res)

    print(res.shape)

    print(get_pdf(walk_time=182, pl_id=40))

    res = get_cdf(
        pl_id=40,
        t_start=[np.random.randint(low=0, high=500) for _ in range(50)],
        t_end=[np.random.randint(low=0, high=500) for _ in range(50)]
    )
    print(res)

    print(get_cdf(
        pl_id=40,
        t_start=18,
        t_end=200
    ))

    print(node_id_to_pl_id(node_id=107511))


if __name__ == '__main__':
    config.load_config()
    test_performance()
    pass
