"""
This script analyzes assigned passenger walking times (egress and transfer) 
and generates their statistical distributions.

Key Outputs:
- egress_times.pkl: Raw egress time data
- transfer_times.pkl: Raw transfer time data
- ETD figures: Egress time distribution plots
- TTD figures: Transfer time distribution plots
- etd.csv: Fitted egress time distribution parameters
- ttd.csv: Fitted transfer time distribution parameters

Dependencies:
- src.walk_time_filter
- src.walk_time_plot
- src.walk_time_dis_fit
- src.utils
"""
from src import config
from src.utils import save_, read_, pd
from src.walk_time_filter import filter_egress_all, filter_transfer_all
from src.walk_time_plot import plot_egress_all, plot_transfer_all
from src.walk_time_dis_fit import (
    fit_platform_egress_time_dis_all_parallel,
    fit_transfer_time_dis_all,
)


def analyze_assigned_walk_times():
    # eg_t = filter_egress_all()
    tr_t = filter_transfer_all()
    
    # plot_egress_all(eg_t, save_on=True, save_subfolder="ETD541")
    plot_transfer_all(tr_t, save_on=True, save_subfolder="TTD541", abs_max=2000)
    ...


def main():
    """
    Main function to save filtered walk times and physical links information.
    """
    message = (
        '\033[33m'
        '======================================================================================\n'
        '[INFO] This script performs the following operations:\n'
        '       1. Generate egress time data file in `egress_times_1.pkl`.\n'
        '          - ["rid"(index), "physical_platform_id", "alight_ts", "egress_time"].\n'
        '       2. Generate transfer time data file in `transfer_times_1.pkl`.\n'
        '          - ["rid" (index), "path_id", "seg_id", "pp_id1", "pp_id2", "alight_ts", \n'
        '             "transfer_time", "transfer_type"]\n'
        '       3. Generate egress time distribution plots in `figures/ETD0/`, transfer time\n'
        '          distribution plots in `figures/TTD0/`.\n'
        '       4. Save ETD (Egress Time Distribution) and TTD (Transfer Time Distribution) results\n'
        '          to `results/egress/etd_1.csv` and `results/transfer/ttd_1.csv`.\n'
        '          - ETD: ["pp_id", "x", \n'
        '                  "kde_pdf", "kde_cdf", "kde_ks_stat", "kde_ks_p_value", \n'
        '                  "gamma_pdf", "gamma_cdf", "gamma_ks_stat", "gamma_ks_p_value", \n'
        '                  "lognorm_pdf", "lognorm_cdf", "lognorm_ks_stat", "lognorm_ks_p_value" \n'
        '                 ].\n'
        '          - TTD:  ["pp_id_min", "pp_id_max", "x", "kde_cdf", "gamma_cdf", "lognorm_cdf"].\n'
        '======================================================================================\n'
        '\033[0m'
    )
    print(message)
    # Save filtered walk times
    eg_t = filter_egress_all()
    tr_t = filter_transfer_all()
    save_(fn=config.CONFIG["results"]["egress_times"], data=eg_t, auto_index_on=True)
    save_(fn=config.CONFIG["results"]["transfer_times"], data=tr_t, auto_index_on=True)

    # Plot egress times
    plot_egress_all(eg_t, save_on=True, save_subfolder="ETD0")

    # Plot transfer times
    plot_transfer_all(tr_t, save_on=True, save_subfolder="TTD0")

    # Save distributions: ETD, TTD
    etd = fit_platform_egress_time_dis_all_parallel(eg_t=eg_t, n_jobs=-1)
    save_(fn=config.CONFIG["results"]["etd"], data=etd, auto_index_on=True)

    ttd = fit_transfer_time_dis_all(tr_t=tr_t)
    save_(fn=config.CONFIG["results"]["ttd"], data=ttd, auto_index_on=True)


# This is the printed progress of the script when run:
"""
[INFO] Created subfolder: results\egress
[INFO] Created subfolder: results\transfer
[INFO] Reading file: results\trajectory\left.pkl
[INFO] Reading file: data\AFC.pkl
[INFO] Reading file: results\network\pathvia.pkl
[INFO] Reading 1 versioned files: results\trajectory\assigned.pkl
[INFO] Reading file: results\network\platform.csv
[INFO] Reading 1 versioned files: results\trajectory\assigned.pkl
         physical_platform_id  alight_ts  egress_time
rid
718308                 103301      39102          236
1711124                110001      68734           89
1165334                105101      56028           91
1800846                110101      73460          124
1829148                104201      71082           53
1562519                109901      65922           70
298760                 107701      30274           44
1467177                112101      64108          105
986344                 108901      49982           58
489282                 112901      32984          163
<class 'pandas.core.frame.DataFrame'>
Index: 893024 entries, 317 to 2047060
Data columns (total 3 columns):
 #   Column                Non-Null Count   Dtype
---  ------                --------------   -----
 0   physical_platform_id  893024 non-null  int64
 1   alight_ts             893024 non-null  int32
 2   egress_time           893024 non-null  int64
dtypes: int32(1), int64(2)
memory usage: 20.4 MB
[INFO] File saved to: results\egress\egress_times_1.pkl
            path_id  seg_id  pp_id1  pp_id2  alight_ts  transfer_time  transfer_type
rid
1708192  1007105701       1  105201  105202      68316            144   egress-entry
650826   1048107001       1  103202  103202      35793            113  platform_swap
1160180  1085111201       1  112901  112902      56658            294   egress-entry
418318   1013100601       1  103801  103802      31167            220   egress-entry
1204078  1015113001       1  102801  102802      57167            242   egress-entry
197487   1130111801       1  112901  112902      30204             99   egress-entry
890930   1011103301       1  102202  102201      45875            289   egress-entry
1610067  1027105401       1  103802  103802      66930            139  platform_swap
1309305  1007104301       1  109702  109701      60483             92   egress-entry
805825   1019104901       1  102801  102802      43720            109   egress-entry
<class 'pandas.core.frame.DataFrame'>
Index: 133308 entries, 321 to 2046455
Data columns (total 7 columns):
 #   Column         Non-Null Count   Dtype
---  ------         --------------   -----
 0   path_id        133308 non-null  object
 1   seg_id         133308 non-null  int8
 2   pp_id1         133308 non-null  int64
 3   pp_id2         133308 non-null  int64
 4   alight_ts      133308 non-null  int32
 5   transfer_time  133308 non-null  int32
 6   transfer_type  133308 non-null  category
dtypes: category(1), int32(2), int64(2), int8(1), object(1)
memory usage: 4.8+ MB
[INFO] File saved to: results\transfer\transfer_times_1.pkl
[INFO] Plotting ETD...
(Egress): 100101_101200-101201 | Data size: 2107 / 2127 | BD: [0.0000, 338.2869]
(Egress): 100201_101300-101301 | Data size: 6869 / 6948 | BD: [0.0000, 167.7614]
(Egress): 100301_101470-101471 | Data size: 417 / 435 | BD: [0.0000, 500.0000]
(Egress): 100401_101480-101481 | Data size: 91 / 94 | BD: [0.0000, 461.0978]
(Egress): 100501_102250-102251 | Data size: 3622 / 3674 | BD: [0.0000, 193.2035]
(Egress): 100601_103310-103311 | Data size: 5314 / 5374 | BD: [0.0000, 244.2687]
(Egress): 100701_103320-103321 | Data size: 5095 / 5147 | BD: [0.0000, 217.4646]
(Egress): 100801_104140-104141 | Data size: 6523 / 6566 | BD: [0.0000, 248.7122]
(Egress): 100901_104240-104241 | Data size: 9597 / 9709 | BD: [30.1045, 185.3603]
(Egress): 101001_104370-104371 | Data size: 3818 / 3869 | BD: [22.4688, 258.8953]
(Egress): 101002_107290-107291 | Data size: 3010 / 3018 | BD: [0.0000, 308.1035]
(Egress): 101101_104380-104381 | Data size: 2343 / 2387 | BD: [11.3957, 176.8858]
(Egress): 101201_107270-107271 | Data size: 2388 / 2467 | BD: [0.0000, 357.0262]
(Egress): 101301_107420-107421 | Data size: 8642 / 8753 | BD: [50.8773, 233.0094]
(Egress): 101401_107450-107451 | Data size: 2786 / 2825 | BD: [25.9778, 211.7121]
(Egress): 101501_101240-101241 | Data size: 6287 / 6374 | BD: [0.0000, 180.1097]
(Egress): 101601_101250-101251 | Data size: 6573 / 6637 | BD: [0.0000, 226.1119]
(Egress): 101602_104310-104311 | Data size: 9288 / 9402 | BD: [29.4466, 255.7699]
(Egress): 101701_101280-101281 | Data size: 10262 / 10522 | BD: [0.0000, 256.1250]
(Egress): 101801_101380-101381 | Data size: 16275 / 16498 | BD: [0.0000, 325.4171]
(Egress): 101901_101450-101451 | Data size: 6849 / 7145 | BD: [0.0000, 430.0358]
(Egress): 102001_103220-103221 | Data size: 3611 / 3633 | BD: [0.0000, 225.7499]
(Egress): 102101_103230-103231 | Data size: 8756 / 8814 | BD: [0.0000, 243.7327]
(Egress): 102201_103290-103291 | Data size: 6706 / 6765 | BD: [0.0000, 261.3502]
(Egress): 102202_104330-104331 | Data size: 7753 / 7870 | BD: [7.6378, 211.5236]
(Egress): 102301_103350-103351 | Data size: 7246 / 7353 | BD: [0.0000, 239.6208]
(Egress): 102401_104260-104261 | Data size: 6463 / 6528 | BD: [0.0000, 275.6638]
(Egress): 102402_107440-107441 | Data size: 3583 / 3605 | BD: [31.2836, 303.2235]
(Egress): 102501_104400-104401 | Data size: 2915 / 2952 | BD: [13.2195, 188.4390]
(Egress): 102601_104420-104421 | Data size: 5295 / 5370 | BD: [25.6302, 191.9739]
(Egress): 102701_110260-110261 | Data size: 5404 / 5483 | BD: [0.0000, 236.0961]
(Egress): 102801_101260-101261 | Data size: 7515 / 7649 | BD: [0.0000, 191.4890]
(Egress): 102802_102350-102351 | Data size: 8576 / 8669 | BD: [0.0000, 272.5854]
(Egress): 102901_101330-101331 | Data size: 8674 / 8771 | BD: [0.0000, 200.3900]
(Egress): 103001_102220-102221 | Data size: 5072 / 5224 | BD: [0.0000, 291.1183]
(Egress): 103101_102240-102241 | Data size: 4379 / 4448 | BD: [0.0000, 200.2174]
(Egress): 103201_104290-102320 | Data size: 11006 / 11207 | BD: [0.0000, 256.6432]
(Egress): 103202_104291-102321 | Data size: 6318 / 6428 | BD: [0.0000, 230.6178]
(Egress): 103301_102360-102361 | Data size: 21259 / 21389 | BD: [0.0000, 289.6949]
(Egress): 103302_103300 | Data size: 9527 / 9622 | BD: [0.0000, 279.7040]
(Egress): 103303_103301 | Data size: 7948 / 8004 | BD: [0.0000, 294.6545]
(Egress): 103401_102390-102391 | Data size: 8732 / 8845 | BD: [0.0000, 196.0886]
(Egress): 103501_102480-102481 | Data size: 4452 / 4489 | BD: [0.0000, 189.2440]
(Egress): 103601_103270-103271 | Data size: 5848 / 5895 | BD: [0.0000, 231.4684]
(Egress): 103701_103280-103281 | Data size: 7838 / 7922 | BD: [0.0000, 214.4155]
(Egress): 103801_107400-107401 | Data size: 3745 / 3773 | BD: [34.3260, 310.3557]
(Egress): 103802_103371-110211 | Data size: 916 / 940 | BD: [0.0000, 283.5871]

[WARNING] No egress data for pp_id: 103803.

(Egress): 103804_103370 | Data size: 2955 / 2995 | BD: [0.0000, 265.0252]
(Egress): 103901_107430-107431 | Data size: 4913 / 4983 | BD: [27.9474, 204.0598]
(Egress): 104001_101340-101341 | Data size: 7056 / 7124 | BD: [0.0000, 284.4131]
(Egress): 104101_101350-101351 | Data size: 7501 / 7616 | BD: [0.0000, 211.9061]
(Egress): 104201_101370-101371 | Data size: 9830 / 10000 | BD: [0.0000, 211.4347]
(Egress): 104301_101410 | Data size: 6490 / 6666 | BD: [0.0000, 214.4469]
(Egress): 104302_101411 | Data size: 519 / 538 | BD: [0.0000, 468.6284]
(Egress): 104401_101490-101491 | Data size: 117 / 124 | BD: [0.0000, 500.0000]
(Egress): 104501_101500-101501 | Data size: 484 / 514 | BD: [0.0000, 500.0000]
(Egress): 104601_101520-101521 | Data size: 190 / 198 | BD: [0.0000, 500.0000]
(Egress): 104701_101530-101531 | Data size: 1265 / 1317 | BD: [0.0000, 438.5340]
(Egress): 104801_102330-102331 | Data size: 4458 / 4507 | BD: [0.0000, 186.7286]
(Egress): 104901_102340-102341 | Data size: 9732 / 9833 | BD: [0.0000, 198.8494]
(Egress): 105001_102440-102441 | Data size: 3286 / 3336 | BD: [0.0000, 193.9578]
(Egress): 105101_103240-103241 | Data size: 6844 / 6893 | BD: [0.0000, 235.9387]
(Egress): 105201_103250-103251 | Data size: 1250 / 1259 | BD: [0.0000, 292.0363]
(Egress): 105202_107220-107221 | Data size: 1482 / 1502 | BD: [4.9383, 261.7861]
(Egress): 105301_103260-103261 | Data size: 4552 / 4591 | BD: [0.0000, 200.0373]
(Egress): 105401_103340-103341 | Data size: 7604 / 7686 | BD: [0.0000, 215.0826]
(Egress): 105501_104170-104171 | Data size: 5049 / 5131 | BD: [21.7955, 199.1649]
(Egress): 105601_104210-104211 | Data size: 3338 / 3390 | BD: [0.0000, 133.1402]
(Egress): 105701_107250-107251 | Data size: 2469 / 2502 | BD: [16.4709, 199.0479]
(Egress): 105801_107350-107351 | Data size: 5873 / 5932 | BD: [5.5645, 234.9783]
(Egress): 105901_107500-107501 | Data size: 3892 / 3937 | BD: [23.0368, 204.7814]
(Egress): 106001_110250-110251 | Data size: 2289 / 2382 | BD: [0.0000, 500.0000]
(Egress): 106101_101220-101221 | Data size: 6192 / 6250 | BD: [0.0000, 237.4789]
(Egress): 106102_107210-107211 | Data size: 4003 / 4037 | BD: [41.5302, 287.9511]
(Egress): 106201_101230-101231 | Data size: 6931 / 6998 | BD: [0.0000, 187.7753]
(Egress): 106301_101320-101321 | Data size: 5248 / 5325 | BD: [0.0000, 226.0060]
(Egress): 106302_107370-107371 | Data size: 6284 / 6318 | BD: [16.9550, 304.8133]
(Egress): 106401_101390-101391 | Data size: 9005 / 9100 | BD: [0.0000, 401.3518]
(Egress): 106501_101460-101461 | Data size: 1649 / 1724 | BD: [0.0000, 500.0000]
(Egress): 106601_102300-102301 | Data size: 8723 / 8812 | BD: [0.0000, 183.5106]
(Egress): 106701_104130-104131 | Data size: 8177 / 8276 | BD: [0.0000, 246.5717]
(Egress): 106801_104190-104191 | Data size: 7567 / 7655 | BD: [33.1492, 186.8456]
(Egress): 106901_104220-104221 | Data size: 1362 / 1381 | BD: [24.1476, 165.8843]
(Egress): 107001_104230-104231 | Data size: 17042 / 17209 | BD: [23.9192, 198.7308]
(Egress): 107101_104300-104301 | Data size: 8975 / 9104 | BD: [38.0215, 239.4221]
(Egress): 107201_107280-107281 | Data size: 1015 / 1033 | BD: [18.7471, 207.2625]
(Egress): 107301_107510-107511 | Data size: 1106 / 1123 | BD: [4.3499, 221.1069]
(Egress): 107401_110230-110231 | Data size: 3254 / 3321 | BD: [0.0000, 245.3896]
(Egress): 107501_101270-101271 | Data size: 4378 / 4425 | BD: [0.0000, 170.3655]
(Egress): 107601_101310-101311 | Data size: 6601 / 6646 | BD: [61.6678, 262.5443]
(Egress): 107701_101360-101361 | Data size: 6065 / 6132 | BD: [0.0000, 188.9579]
(Egress): 107801_101540-101541 | Data size: 913 / 975 | BD: [0.0000, 500.0000]
(Egress): 107901_102230-102231 | Data size: 7818 / 7951 | BD: [0.0000, 207.3587]
(Egress): 108001_102410-102411 | Data size: 3987 / 4037 | BD: [0.0000, 196.3016]
(Egress): 108101_102430-102431 | Data size: 6807 / 6890 | BD: [0.0000, 241.1428]
(Egress): 108201_102450-102451 | Data size: 7746 / 7870 | BD: [0.0000, 205.0325]
(Egress): 108301_102470-102471 | Data size: 2639 / 2668 | BD: [0.0000, 187.6259]
(Egress): 108401_102490-102491 | Data size: 1618 / 1643 | BD: [0.0000, 178.8379]
(Egress): 108501_102520-102521 | Data size: 12561 / 12694 | BD: [0.0000, 211.9535]
(Egress): 108601_103360-103361 | Data size: 8851 / 8957 | BD: [0.0000, 221.3892]
(Egress): 108701_104160-104161 | Data size: 5839 / 5909 | BD: [0.0000, 252.5088]
(Egress): 108801_104270-104271 | Data size: 6032 / 6133 | BD: [18.8343, 190.2849]
(Egress): 108901_104320-104321 | Data size: 11724 / 11825 | BD: [16.7746, 198.8699]
(Egress): 109001_104350-104351 | Data size: 6881 / 6993 | BD: [23.4955, 179.8993]
(Egress): 109101_104390-104391 | Data size: 2965 / 3014 | BD: [22.3194, 171.6859]
(Egress): 109201_107230-107231 | Data size: 1905 / 1935 | BD: [32.7958, 197.9009]
(Egress): 109301_107300-107301 | Data size: 1873 / 1899 | BD: [20.0731, 209.3877]
(Egress): 109401_107320-107321 | Data size: 2291 / 2328 | BD: [10.8181, 209.2876]
(Egress): 109501_110220-110221 | Data size: 2293 / 2333 | BD: [0.0000, 212.8284]
(Egress): 109601_101210-101211 | Data size: 3786 / 3844 | BD: [0.0000, 173.3451]
(Egress): 109701_101290-101291 | Data size: 6163 / 6259 | BD: [0.0000, 227.0582]
(Egress): 109702_103330-103331 | Data size: 4160 / 4192 | BD: [0.0000, 302.2236]
(Egress): 109801_101400-101401 | Data size: 8557 / 8661 | BD: [0.0000, 203.9425]
(Egress): 109901_101430-101431 | Data size: 5269 / 5326 | BD: [0.0000, 226.4775]
(Egress): 110001_101440-101441 | Data size: 7426 / 7666 | BD: [0.0000, 359.6229]
(Egress): 110101_102210-102211 | Data size: 20937 / 21137 | BD: [0.0000, 281.3973]
(Egress): 110201_102280-102281 | Data size: 7761 / 7864 | BD: [0.0000, 199.6142]
(Egress): 110301_102310-102311 | Data size: 5813 / 5901 | BD: [0.0000, 200.1639]
(Egress): 110401_102370-102371 | Data size: 11579 / 11745 | BD: [0.0000, 229.0449]
(Egress): 110501_102400-102401 | Data size: 2092 / 2124 | BD: [0.0000, 210.8175]
(Egress): 110601_102460-102461 | Data size: 5386 / 5460 | BD: [0.0000, 198.5227]
(Egress): 110701_102500-102501 | Data size: 7018 / 7096 | BD: [0.0000, 191.4832]
(Egress): 110801_103210-103211 | Data size: 10957 / 11233 | BD: [0.0000, 311.7099]
(Egress): 110901_104180-104181 | Data size: 6085 / 6185 | BD: [28.1480, 192.7726]
(Egress): 111001_104250-104251 | Data size: 10753 / 10896 | BD: [24.2663, 183.1238]
(Egress): 111101_104340-104341 | Data size: 6920 / 6999 | BD: [9.4983, 198.6497]
(Egress): 111201_104360-104361 | Data size: 5585 / 5695 | BD: [19.7236, 189.3222]
(Egress): 111301_104410-104411 | Data size: 1298 / 1320 | BD: [16.2244, 180.4089]
(Egress): 111401_107240-107241 | Data size: 1849 / 1873 | BD: [12.2198, 208.5277]
(Egress): 111501_107330-107331 | Data size: 4828 / 4897 | BD: [32.2526, 206.9196]
(Egress): 111601_107340-107341 | Data size: 6616 / 6712 | BD: [51.8659, 231.7509]
(Egress): 111701_107360-107361 | Data size: 6294 / 6383 | BD: [32.9464, 208.6529]
(Egress): 111801_107380-107381 | Data size: 4206 / 4262 | BD: [32.6431, 209.6793]
(Egress): 111901_107390-107391 | Data size: 5008 / 5060 | BD: [24.9831, 218.5138]
(Egress): 112001_107410-107411 | Data size: 8544 / 8667 | BD: [33.4802, 216.6322]
(Egress): 112101_107480-107481 | Data size: 3448 / 3503 | BD: [24.8721, 206.0860]
(Egress): 112201_107490-107491 | Data size: 4046 / 4108 | BD: [35.7092, 204.9919]
(Egress): 112301_101420-101421 | Data size: 4411 / 4509 | BD: [0.0000, 500.0000]
(Egress): 112401_101510-101511 | Data size: 267 / 280 | BD: [0.0000, 459.6261]
(Egress): 112501_102260-102261 | Data size: 2510 / 2553 | BD: [0.0000, 182.3610]
(Egress): 112601_102270-102271 | Data size: 6775 / 7020 | BD: [0.0000, 256.4835]
(Egress): 112701_102290-102291 | Data size: 4862 / 4894 | BD: [0.0000, 275.5558]
(Egress): 112702_107460-107461 | Data size: 4180 / 4205 | BD: [40.4948, 318.3175]
(Egress): 112801_102380-102381 | Data size: 10224 / 10352 | BD: [0.0000, 218.7796]
(Egress): 112901_102420-102421 | Data size: 15447 / 15594 | BD: [0.0000, 250.1930]
(Egress): 112902_107310-107311 | Data size: 10593 / 10721 | BD: [16.1557, 285.7225]
(Egress): 113001_102510-102511 | Data size: 6384 / 6456 | BD: [0.0000, 186.9475]
(Egress): 113101_104150-104151 | Data size: 3852 / 3905 | BD: [0.0000, 247.8083]
(Egress): 113201_104200-104201 | Data size: 1346 / 1370 | BD: [18.7542, 190.6064]
(Egress): 113301_104280-104281 | Data size: 6298 / 6398 | BD: [24.0776, 184.6632]
(Egress): 113401_107260-107261 | Data size: 6205 / 6242 | BD: [5.0777, 243.6906]
(Egress): 113501_107470-107471 | Data size: 2917 / 2957 | BD: [20.4149, 224.9615]
(Egress): 113601_110240-110241 | Data size: 2945 / 3002 | BD: [0.0000, 177.5811]
[INFO] Plotting TTD...
(Transfer egress-entry): 101001-101002 | Data size: 3909 / 3928 | BD: [0.0000, 434.7822]
(Transfer egress-entry): 101601-101602 | Data size: 3839 / 3839 | BD: [0.0000, 358.8388]
(Transfer egress-entry): 102201-102202 | Data size: 8240 / 8315 | BD: [0.0000, 413.8516]
(Transfer egress-entry): 102401-102402 | Data size: 13507 / 13670 | BD: [0.0000, 424.0221]
(Transfer egress-entry): 102801-102802 | Data size: 10832 / 10844 | BD: [0.0000, 324.6178]
(Transfer egress-entry): 103201-103202 | Data size: 236 / 236 | BD: [0.0000, 396.2689]
(Transfer egress-entry): 103301-103302 | Data size: 3371 / 3379 | BD: [0.0000, 371.7357]
(Transfer egress-entry): 103301-103303 | Data size: 3641 / 3643 | BD: [0.0000, 391.8434]
(Transfer egress-entry): 103801-103802 | Data size: 7456 / 7467 | BD: [0.0000, 432.0963]
(Transfer egress-entry): 103801-103803 | Data size: 2145 / 2145 | BD: [0.0000, 473.6715]
(Transfer egress-entry): 103801-103804 | Data size: 1588 / 1588 | BD: [0.0000, 493.5478]
(Transfer egress-entry): 103803-103804 | Data size: 1608 / 1608 | BD: [0.0000, 488.6036]
(Transfer egress-entry): 104301-104302 | Data size: 94 / 94 | BD: [0.0000, 500.0000]
(Transfer egress-entry): 105201-105202 | Data size: 6706 / 6747 | BD: [0.0000, 446.6028]
(Transfer egress-entry): 106101-106102 | Data size: 2075 / 2077 | BD: [0.0000, 357.6043]
(Transfer egress-entry): 106301-106302 | Data size: 15243 / 15332 | BD: [0.0000, 384.8286]
(Transfer egress-entry): 109701-109702 | Data size: 8875 / 8900 | BD: [0.0000, 357.3832]
(Transfer egress-entry): 112701-112702 | Data size: 7978 / 7979 | BD: [0.0000, 399.6716]
(Transfer egress-entry): 112901-112902 | Data size: 8188 / 8209 | BD: [0.0000, 393.6754]
(Transfer platform_swap): 103201-103201 | Data size: 7746 / 7780 | BD: [0, 311.8238671831109]
(Transfer platform_swap): 103202-103202 | Data size: 7844 / 7896 | BD: [0, 302.73119860972014]
(Transfer platform_swap): 103802-103802 | Data size: 7571 / 7632 | BD: [0, 417.93765489981615]
[INFO] Start fitting egress time distribution using -1 threads...
Egress time distribution fitting for each physical platform: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 154/154 [00:52<00:00,  2.93it/s] 
          pl_id      x       kde_pdf   kde_cdf  kde_ks_stat  kde_ks_p_value     gamma_pdf  gamma_cdf  gamma_ks_stat  gamma_ks_p_value   lognorm_pdf   lognorm_cdf  lognorm_ks_stat  lognorm_ks_p_value
52491  109101.0  387.0  0.000000e+00  1.000000     0.019607    2.017585e-01  1.196356e-18   1.000000       0.037419      4.823625e-04  3.781275e-14  1.000000e+00         0.031171            0.006157
64820  111501.0  191.0  7.499727e-04  0.992304     0.012170    4.683418e-01  4.194997e-04   0.995118       0.023239      1.069902e-02  5.218880e-04  9.926749e-01         0.014916            0.230711
62157  111001.0   33.0  1.228971e-06  0.000002     0.013923    3.064380e-02  1.283456e-06   0.000003       0.033991      3.145259e-11  1.177343e-07  1.733093e-07         0.024979            0.000003
58244  110201.0  128.0  2.437651e-03  0.933611     0.015122    5.690045e-02  2.834803e-03   0.910286       0.069503      4.813478e-33  2.837651e-03  9.336720e-01         0.020957            0.002158
49400  108501.0  302.0  7.584551e-67  1.000000     0.008954    2.646564e-01  8.553269e-06   0.999805       0.053563      9.203493e-32  2.864808e-07  9.999962e-01         0.009517            0.203961
1194   100301.0  192.0  1.275766e-03  0.685871     0.285336    1.532847e-30  1.322942e-03   0.831523       0.119102      1.302779e-05  8.347978e-04  8.821935e-01         0.118898            0.000014
76555  113601.0  403.0  0.000000e+00  1.000000     0.017112    3.504836e-01  5.143282e-12   1.000000       0.028506      1.636676e-02  1.053552e-10  1.000000e+00         0.019452            0.212328
13968  102402.0  441.0  4.878048e-63  1.000000     0.015972    3.167487e-01  1.734179e-07   0.999997       0.034439      3.974307e-04  8.507708e-07  9.999816e-01         0.033468            0.000638
44804  107601.0  215.0  2.400978e-03  0.942409     0.012563    2.464316e-01  2.627918e-03   0.955153       0.030797      7.132411e-06  2.591001e-03  9.512137e-01         0.023166            0.001648
29868  104901.0  309.0  1.378672e-95  1.000000     0.011834    1.299277e-01  1.943071e-06   0.999960       0.042794      6.341660e-16  3.617866e-07  9.999936e-01         0.016925            0.007493
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 76653 entries, 0 to 76652
Data columns (total 14 columns):
 #   Column              Non-Null Count  Dtype
---  ------              --------------  -----
 0   pl_id               76653 non-null  float64
 1   x                   76653 non-null  float64
 2   kde_pdf             76653 non-null  float64
 3   kde_cdf             76653 non-null  float64
 4   kde_ks_stat         76653 non-null  float64
 5   kde_ks_p_value      76653 non-null  float64
 6   gamma_pdf           76653 non-null  float64
 7   gamma_cdf           76653 non-null  float64
 8   gamma_ks_stat       76653 non-null  float64
 9   gamma_ks_p_value    76653 non-null  float64
 10  lognorm_pdf         76653 non-null  float64
 11  lognorm_cdf         76653 non-null  float64
 12  lognorm_ks_stat     76653 non-null  float64
 13  lognorm_ks_p_value  76653 non-null  float64
dtypes: float64(14)
memory usage: 8.2 MB
[INFO] File saved to: results\egress\etd_1.csv
[INFO] Start fitting transfer time distribution...
101001 101002 3928 -> 3909       | kde: 0.0228 0.0336 | gamma: 0.0730 0.0000 | lognorm: 0.0323 0.0006 | 
101601 101602 3839 -> 3839       | kde: 0.0263 0.0096 | gamma: 0.0998 0.0000 | lognorm: 0.0445 0.0000 | 
102201 102202 8315 -> 8240       | kde: 0.0205 0.0020 | gamma: 0.0800 0.0000 | lognorm: 0.0263 0.0000 | 
102401 102402 13670 -> 13507     | kde: 0.0177 0.0004 | gamma: 0.0536 0.0000 | lognorm: 0.0260 0.0000 | 
102801 102802 10844 -> 10832     | kde: 0.0201 0.0003 | gamma: 0.0923 0.0000 | lognorm: 0.0308 0.0000 | 
103201 103202 236 -> 236         | kde: 0.0633 0.2887 | gamma: 0.1080 0.0075 | lognorm: 0.0646 0.2671 | 
103301 103302 3379 -> 3371       | kde: 0.0309 0.0031 | gamma: 0.0847 0.0000 | lognorm: 0.0462 0.0000 | 
103301 103303 3643 -> 3641       | kde: 0.0288 0.0046 | gamma: 0.0801 0.0000 | lognorm: 0.0377 0.0001 | 
103801 103802 7467 -> 7456       | kde: 0.0288 0.0000 | gamma: 0.0461 0.0000 | lognorm: 0.0431 0.0000 | 
103801 103803 2145 -> 2145       | kde: 0.0443 0.0004 | gamma: 0.1194 0.0000 | lognorm: 0.1055 0.0000 | 
103801 103804 1588 -> 1588       | kde: 0.0767 0.0000 | gamma: 0.1537 0.0000 | lognorm: 0.1027 0.0000 | 
103803 103804 1608 -> 1608       | kde: 0.0486 0.0010 | gamma: 0.0650 0.0000 | lognorm: 0.0725 0.0000 | 
104301 104302 94 -> 94   | kde: 0.1322 0.0681 | gamma: 0.0789 0.5746 | lognorm: 0.0690 0.7358 | 
105201 105202 6747 -> 6706       | kde: 0.0201 0.0089 | gamma: 0.0897 0.0000 | lognorm: 0.0331 0.0000 | 
106101 106102 2077 -> 2075       | kde: 0.0387 0.0038 | gamma: 0.1261 0.0000 | lognorm: 0.0661 0.0000 | 
106301 106302 15332 -> 15243     | kde: 0.0248 0.0000 | gamma: 0.0850 0.0000 | lognorm: 0.0591 0.0000 | 
109701 109702 8900 -> 8875       | kde: 0.0221 0.0003 | gamma: 0.0830 0.0000 | lognorm: 0.0386 0.0000 | 
112701 112702 7979 -> 7978       | kde: 0.0190 0.0061 | gamma: 0.0795 0.0000 | lognorm: 0.0377 0.0000 | 
112901 112902 8209 -> 8188       | kde: 0.0213 0.0012 | gamma: 0.0554 0.0000 | lognorm: 0.0536 0.0000 | 
103201 103201 7780
103202 103202 7896
103802 103802 7632
      pp_id_min  "pp_id_max"  "x" x"   kde_"cd"f  gamma_"cd"f"  lognorm_cdf
9239     112901     112902  221  0.668560   0.689024     0.669378
7345     106101     106102  331  0.999566   0.961268     0.995663
6507     104301     104302  495  0.995563   0.998607     0.998662
664      101601     101602  163  0.503202   0.588136     0.522394
9178     112901     112902  160  0.321233   0.363736     0.310174
4598     103801     103803   89  0.038826   0.024627     0.031012
9495     112901     112902  477  1.000000   0.999243     0.999987
9094     112901     112902   76  0.032152   0.023304     0.034789
923      101601     101602  422  1.000000   0.993519     0.999961
6829     105201     105202  316  0.934319   0.907696     0.931750
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 9522 entries, 0 to 9521
Data columns (total 6 columns):
 #   Column       Non-Null Count  Dtype
---  ------       --------------  -----
 0   pp_id_min    9522 non-null   int32
 1   "pp_id_max"  "x" 9"522 non"-n"ull   int"32"
 2"   x            9522 non-null   int32
 3   kde_cdf      9522 non-null   float64
 4   gamma_cdf    9522 non-null   float64
 5   lognorm_cdf  9522 non-null   float64
dtypes: float64(3), int32(3)
memory usage: 334.9 KB
[INFO] File saved to: results\transfer\ttd_1.csv
"""


if __name__ == "__main__":
    config.load_config()
    # main()
    analyze_assigned_walk_times()
    ...
