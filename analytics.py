import datetime
import os
import sqlite3
from matplotlib.colors import ListedColormap

import numpy as np
import re
import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
from scipy import stats
import statsmodels.api as sm

RESULTS_AGENTS = "truck_agents"

titles = {
    "base_deliveries.db": "bazinis",
    "florida_deliveries.db": "Tampos",
    "georgia_deliveries.db": "Atlantos",
    "texas_deliveries.db": "Dalaso"
}


def get_warehouse_deliveries(db_files: List[str]):
    for idx, db_file in enumerate(db_files):
        conn = sqlite3.connect(db_file)
        deliveries = get_generated_agents(conn)
        deliveries_through_warehouse = deliveries[deliveries['through_warehouse'] == 1]
        num_deliveries_through_warehouse = len(deliveries_through_warehouse)
        
        print(db_file)
        print(f"Number of deliveries made through warehouse: {num_deliveries_through_warehouse}")


def compare_seasonality(original_data_file: str, db_file: str):
    """
    Plot the comparison of ship time trend between a database and data from a file.
    
    Args:
        db_file (str): path to the SQLite database file.
        data_file (str): Path to the data file.

    Returns:
        None
    """
    # Original data
    data = pd.read_csv(original_data_file)
    data['Ship Day'] = pd.to_datetime(data['Ship Day'], format='%Y-%m-%d')
    data.set_index('Ship Day', inplace=True)
    data_count = data.resample('D').size()


    fig, axs = plt.subplots(1, 2, figsize=(12, 8))

    conn = sqlite3.connect(db_file)
    deliveries = get_generated_agents(conn)
    deliveries['ship_day'] = pd.to_datetime(deliveries['ship_day'], format='%Y-%m-%d')
    deliveries.set_index('ship_day', inplace=True)
    deliveries_count = deliveries.resample('D').size()

    # Time series decompositions
    decomposition_db = sm.tsa.seasonal_decompose(deliveries_count, model='additive')
    seasonal_db = decomposition_db.seasonal

    decomposition_data = sm.tsa.seasonal_decompose(data_count, model='additive')
    seasonal_data = decomposition_data.seasonal

    # Plot the seasonal component of the selected database's ship_day
    axs[0].plot(seasonal_db, color='blue')
    axs[0].set_title(f"Krovinių išsiuntimų sezoniniškumas: Dalaso scenarijus", fontsize=16)
    axs[0].set_xlabel('Data', fontsize=12)
    axs[0].set_ylabel('Kiekis', fontsize=12)

    # Plot the seasonal component of the data from the file
    axs[1].plot(seasonal_data, color='green')
    axs[1].set_title(f"Krovinių išsiuntimų sezoniniškumas: pradiniai duomenys", fontsize=16)
    axs[1].set_xlabel('Data', fontsize=12)
    axs[1].set_ylabel('Kiekis', fontsize=12)

    plt.tight_layout()
    plt.show()



def compare_series(original_data_file: str, db_file: str):
    """
    Plot the comparison of ship time trend between a database and data from a file.

    Args:
        db_file (str): path to the SQLite database file.
        data_file (str): Path to the data file.

    Returns:
        None
    """
    # Original data
    data = pd.read_csv(original_data_file)
    data['Ship Day'] = pd.to_datetime(data['Ship Day'], format='%Y-%m-%d')
    data.set_index('Ship Day', inplace=True)
    data_count = data.resample('D').size()

    # Create subplots for the selected database only
    fig, ax = plt.subplots(figsize=(8, 6))

    conn = sqlite3.connect(db_file)
    deliveries = get_generated_agents(conn)
    deliveries['ship_day'] = pd.to_datetime(deliveries['ship_day'], format='%Y-%m-%d')
    deliveries.set_index('ship_day', inplace=True)
    deliveries_count = deliveries.resample('D').size()

    # Perform the ADF test for the selected database's ship_day
    result_db = sm.tsa.stattools.adfuller(deliveries_count)

    # Perform the ADF test for the data from the file
    result_data = sm.tsa.stattools.adfuller(data_count)

    # Extract the test statistics and p-values
    test_statistic_db = result_db[0]
    p_value_db = result_db[1]
    test_statistic_data = result_data[0]
    p_value_data = result_data[1]

    print(f"Results for {db_file}:")
    print(f"Test statistic (Database): {test_statistic_db}")
    print(f"P-value (Database): {p_value_db}")
    print(f"Test statistic (Data from file): {test_statistic_data}")
    print(f"P-value (Data from file): {p_value_data}")
    print("")

    # Plot the distribution of the selected database's ship_day
    ax.plot(deliveries_count, label='Dalaso scenarijus')
    # Plot the distribution of the data from the file
    ax.plot(data_count, label='Pradiniai duomenys')

    ax.set_title(f'Krovinių išsiuntimo dažnumas: {titles[db_file]} scenarijus', fontsize=16)
    ax.set_xlabel('Data', fontsize=12)
    ax.set_ylabel('Išsiųsti kroviniai', fontsize=12)
    ax.legend()
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.show()


def plot_ship_time_distribution(db_files):
    """
    Plot the distribution of ship time for multiple databases.

    Args:
        db_files (list): List of paths to the SQLite database files.

    Returns:
        None
    """
    # Create subplots for each database
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    for idx, db_file in enumerate(db_files):
        
        conn = sqlite3.connect(db_file)

        deliveries = get_generated_agents(conn)
        deliveries['ship_day'] = pd.to_datetime(deliveries['ship_day'], format='%Y-%m-%d')
        deliveries.set_index('ship_day', inplace=True)
        deliveries_count = deliveries.resample('D').size()

        # Perform the ADF test
        result = sm.tsa.stattools.adfuller(deliveries_count)

        # Extract the test statistics and p-value
        test_statistic = result[0]
        p_value = result[1]

        print(f"Results for {db_file}:")
        print(f"Test statistic: {test_statistic}")
        print(f"P-value: {p_value}")
        print("")


        # Plot the distribution using a histogram
        axs[idx//2, idx%2].plot(deliveries_count)
        axs[idx//2, idx%2].set_title(f'Krovinių išsiuntimo dažnumas: {titles[db_file]} scenarijus', fontsize=16)
        axs[idx//2, idx%2].set_xlabel('Data', fontsize=12)
        axs[idx//2, idx%2].set_ylabel('Išsiųsti kroviniai', fontsize=12)


    plt.tight_layout()

    plt.show()



def check_lognormal_distribution(data):
    alpha = 0.05  
    # Take the logarithm of the weight data
    # Convert your data to a log scale
    data_copy = data.copy()
    data_copy = data_copy[data_copy['weight'] > 0]

    log_data = np.log(data_copy['weight'])

    # # Plot the histogram
    # plt.figure(figsize=(10, 6))
    # plt.hist(log_data, bins=50, alpha=0.5, color='g')
    # plt.title('Log of Weights Distribution')
    # plt.xlabel('Log of Weights')
    # plt.ylabel('Frequency')
    # plt.grid(True)
    # plt.show()

    mu, std = stats.norm.fit(log_data)

    ks_statistic, p_value = stats.kstest(log_data, 'norm', args=(mu, std))

    print(f"KS statistic: {ks_statistic}")
    print(f"P-value: {p_value}")
    
    if p_value > alpha:
        print(f"The p-value is {p_value:.4f}, which is greater than the significance level of {alpha}. Cannot reject the null hypothesis that the data follows a log-normal distribution.")
    else:
        print(f"The p-value is {p_value:.4f}, which is less than the significance level of {alpha}. Rejecting the null hypothesis that the data follows a log-normal distribution.")



def get_generated_agents(connection: sqlite3.Connection):
    agents = pd.read_sql_query(
    "SELECT * from %s;" % RESULTS_AGENTS, connection)

    return agents


def plot_qq(data: pd.Series, ax: plt.Axes) -> None:
    """
    Plot the Q-Q plot for a given series of data.

    Args:
        data (pd.Series): Series of data.
        ax (plt.Axes): Matplotlib axes to plot on.

    Returns:
        None: This function only creates a plot and does not return any value.
    """
    sm.qqplot(data, line='s', ax=ax)

def plot_qq_plot(db_files: List[str]) -> None:
    """
    Plot the Q-Q plot for the weight data from multiple SQLite databases.

    Args:
        db_files (List[str]): List of paths to SQLite database files.

    Returns:
        None: This function only creates a plot and does not return any value.
    """
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    for idx, db_file in enumerate(db_files):
        # Connect to SQLite database
        conn = sqlite3.connect(db_file)

        data = get_generated_agents(conn)

        # Close the connection
        conn.close()

        # Plot the Q-Q plot for each database
        plot_qq(data['weight'], axs[idx // 2, idx % 2])
        axs[idx // 2, idx % 2].set_title(f'Normalumo (Q-Q) diagrama: {titles[db_file]} scenarijus')
        axs[idx // 2, idx % 2].set_xlabel('Kvantiliai')
        axs[idx // 2, idx % 2].set_ylabel('Stebėjimai')

    plt.tight_layout()
    plt.show()


def plot_weight_distribution(db_files: List[str]) -> None:
    """
    Plot the distribution of weights from multiple SQLite databases.

    Args:
        db_files (List[str]): List of paths to SQLite database files.
        table_name (str): Name of the table in the SQLite database to read the data from.

    Returns:
        None: This function only creates a plot and does not return any value.
    """


    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    for idx, db_file in enumerate(db_files):
        # Connect to SQLite database
        conn = sqlite3.connect(db_file)

        data = get_generated_agents(conn)

        # Close the connection
        conn.close()

        # Plot the distribution of weights for each database
        # Freedman-Diaconis Rule
        # iqr = data['weight'].quantile(0.75) - data['weight'].quantile(0.25)
        # bin_width = 2 * iqr / np.power(len(data['weight']), 1/3)
        # num_bins = int(np.ceil((data['weight'].max() - data['weight'].min()) / bin_width))\
        # Sturges' Rule
        # num_bins = int(np.ceil(np.log2(len(data['weight'])) + 1))
        # Scott's Rule
        num_bins = int(np.ceil(np.log2(len(data['weight'])) + 1))
        # Rice rules
        # num_bins = int(np.ceil(2 * np.power(len(data['weight']), 1/3)))
        sns.histplot(data['weight'], bins=num_bins, kde=True, ax=axs[idx//2, idx%2])
        axs[idx//2, idx%2].set_title(f'Svorio reikšmių pasiskirstymas: {titles[db_file]} scenarijus')
        axs[idx//2, idx%2].set_xlabel('Svoris')
        axs[idx//2, idx%2].set_ylabel('Dažnumas')
                
                # Perform Shapiro-Wilk test for normality
        _, p_value = stats.shapiro(data['weight'])
        alpha = 0.05
        print(f'Shapiro p-value {p_value}: {titles[db_file]} scenarijus')
        check_lognormal_distribution(data)
        if p_value > alpha:
            axs[idx//2, idx%2].text(0.95, 0.95, "Pasiskirstę pagal normalųjį skirstinį", transform=axs[idx//2, idx%2].transAxes,
                                    horizontalalignment='right', verticalalignment='top')
        else:
            axs[idx//2, idx%2].text(0.95, 0.95, "Nepasiskirstę pagal normalųjį skirstinį", transform=axs[idx//2, idx%2].transAxes,
                                    horizontalalignment='right', verticalalignment='top')
            

# db_files = glob.glob("*.db")
    plt.tight_layout()
    plt.show()

def count_lines_of_code():
    folder_path = os.getcwd()  # Get current working directory
    total_lines = 0
    comment_lines = 0
    comment_pattern = r"^\s*#"

    for file in os.listdir(folder_path):
        if file.endswith(".py"):
            file_path = os.path.join(folder_path, file)
            with open(file_path, "r") as f:
                for line in f:
                    total_lines += 1
                    if re.match(comment_pattern, line.strip()):
                        comment_lines += 1

    code_lines = total_lines - comment_lines
    return code_lines, comment_lines

    

code_lines, comment_lines = count_lines_of_code()

print("Total lines of code:", code_lines)
print("Total comment lines:", comment_lines)


files = os.listdir()
databases = ["base_deliveries.db", "florida_deliveries.db", "georgia_deliveries.db", "texas_deliveries.db"]

for i, database in enumerate(databases):
    print(f"{i+1}. {database}")

get_warehouse_deliveries(databases)
compare_seasonality("clean_data_2021.csv", "texas_deliveries.db")
compare_series("clean_data_2021.csv", "texas_deliveries.db")
plot_ship_time_distribution(databases)
plot_weight_distribution(databases)
plot_qq_plot(databases)
# selected_number = int(input("Enter the number of the database you want to select: "))

# if selected_number < 1 or selected_number > len(databases):
#     print("Invalid selection")
# else:
#     # Get the filename of the selected database
#     selected_database = databases[selected_number - 1]

#     # Connect to the selected database
#     connection = sqlite3.connect(selected_database)
    

#     connection.close()