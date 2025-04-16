# to test and implement GPT-generated code

import matplotlib

matplotlib.use('TkAgg')


def plot_egress_time_distribution2():
    import numpy as np
    from scipy.stats import gaussian_kde
    from scipy.integrate import quad
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Generate two groups of normal distribution data
    np.random.seed(42)
    data1 = np.random.normal(loc=10, scale=2, size=300)
    data2 = np.random.normal(loc=20, scale=3, size=200)
    combined_data = np.concatenate([data1, data2])

    # Perform KDE
    kde = gaussian_kde(combined_data)

    # Define the range for the variable
    x_min, x_max = 5, 25  # Limit the range to [5, 25]
    x_values = np.linspace(x_min, x_max, 1000)

    # Compute PDF
    pdf_values = kde(x_values)

    # Compute CDF using numerical integration
    cdf_values = [quad(kde, x_min, x)[0] for x in x_values]

    # Plot PDF
    plt.figure(figsize=(10, 6))
    sns.histplot(combined_data, kde=False, bins=30,
                 stat='density', label='Histogram')
    plt.plot(x_values, pdf_values, label='PDF (KDE)', color='red')
    plt.title('PDF with Kernel Density Estimation')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.show()

    # Plot CDF
    # plt.figure(figsize=(10, 6))
    # plt.plot(x_values, cdf_values, label='CDF', color='blue')
    # plt.title('CDF from Kernel Density Estimation')
    # plt.xlabel('Value')
    # plt.ylabel('Cumulative Probability')
    # plt.legend()
    # plt.show()

    # Print bandwidth
    print(f"Bandwidth of KDE: {kde.factor}")
    return


if __name__ == '__main__':
    # plot_egress_time_distribution2()
    from src import config

    config.load_config()
    from scripts.analyze_egress import _check_egress_time_outlier_rejects
    _check_egress_time_outlier_rejects(uid=1024)
    pass
