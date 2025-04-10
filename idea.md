# 接下来的思路

## 文件管理
- feas_iti这个文件不动，首先根据iti的数量进行筛选，产生feas_iti_stashed.pkl (iti数量大于500)、feas_iti_assigned.pkl (已分配的)、feas_iti_left.pkl (还未分配的）。三个文件应该字段完全一致，其中feas_iti_assigned应该存在多个文件，由后缀_1, _2, _3等表示第几次分配得到的结果。另外存在一个iti_prob.parquet文件，存储每个(rid, iti_id)组合的概率，每一次更新计算就增加一列。

- 步行时间分布计算：不仅仅是唯一可行乘车方案的乘客来统计步行时间，还有所有可行乘车方案中最后一个seg的train都是同一列车的乘客，也可以拿来统计步行时间。所以本质上来说，对于feas_iti这个文件来说，应该是：“同一rid下的所有iti中最后一个seg列车train id相同的”，这些都可以拿来做步行时间分布计算。

- 步行时间分布计算方法：假设存在k个gamma分布，估计k个gamma分布的参数，要求shape相关的参数相同，然后遍历k取值得到一个最终的参数估计。但这存在问题：**用的时候该如何确定乘客是从哪个gamma分布的进站口进站的呢？**

- python文件结构：passenger负责找可行方案，计算可行方案的概率；walk_time_dis负责找到出站步行时间，估计gamma分布参数（增加lognormal等分布参数估计方法，用于对比），计算分布概率公式（pdf、cdf，但是注意执行效率，最好能够按列执行）；timetable负责统计客流，快速得到列车断面客流；trajectory.py(*)负责处理feas_iti和三个拆分文件的读取、写入操作，还负责生成完整出行链文件trajectory.pkl