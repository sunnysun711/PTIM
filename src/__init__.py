# change when switchin devices
# DATA_DIR = r"E:\OneDrive\Documents\00-MyResearch\_20200626_PTAM\PTIM2025\data\cd"
DATA_DIR = r"D:\cd"

# B型车定员1468，超员1820；A型车定员1828, 超员2488
# 1,2,3,4号线用车为B型车6辆编组，编组形式为：+TC-MP-M1-M2-MP-TC+。
# TC为司机室拖车，座位数36，面积31.75平方米；MP、M1、M2座位数42，面积33.5平方米。
TRAIN_B_SEAT, TRAIN_B_AREA = 240, 197.5  # 座位数，立席面积

# 7， 10号线为A型车，因为以2018年4月11日数据来算，不可能超员，因此直接按照网上的定员和超员数据来反推一下座位数和立席面积
TRAIN_A_SEAT, TRAIN_A_AREA = 240, 265

# 普通乘客耐受立席密度，通勤乘客耐受立席密度。
# 普通乘客由陈伟的论文和列车定员得到，取为6.57；通勤乘客参考的是调研时火车南站-高新区间的车厢客流情况（一个车厢325人），立席密度为8.4478
STD_DENSITY, STD_DENSITY_COMMUTER = 6.57, 8.45