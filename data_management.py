from datetime import datetime
import glob
import os
import time
from typing import Dict, Tuple
import googlemaps
import numpy as np
import pandas as pd
import sqlite3
from fuzzywuzzy import fuzz, process
import concurrent.futures
from scipy.stats import poisson, kstest, ks_2samp
from scipy.stats import gaussian_kde


DATE_FORMAT = '%Y-%m-%d'
BOOK_DATE_FORMAT = '%d-%m-%y'

_API_KEY = '-3sLKmkI'
_SLEEP_TIME = 0.05  # Add a sleep time between API requests to avoid hitting rate limits

def read_deliveries(filename: str):
    """
    Read a CSV file and perform cleaning and formatting operations.

    Args:
        filename (str): The name of the CSV file to read.

    Returns:
        pandas.DataFrame: The cleaned and formatted dataframe.
    """

    # Read the CSV file
    df = pd.read_csv(filename)

    # Convert dates to datetime format
    df['Book Date'] = pd.to_datetime(df['Book Date'], format=BOOK_DATE_FORMAT)
    df['Ship Day??'] = pd.to_datetime(df['Ship Day??'], format=DATE_FORMAT)
    df['Delivery Date'] = pd.to_datetime(
        df['Delivery Date'], format=DATE_FORMAT)

    df = df.rename(columns={'Dest City': 'Destination City'})
    df = df.rename(columns={'Ship Day??': 'Ship Day'})
    df = df.rename(columns={'Delivery Date': 'Delivery Day'})


    # Only use first time because it is slow algorithm to run
    # find_similar_cities(df)

    # Replace known aliases of city names
    city_aliases = {
        'WOODSTOCK': 'Woodstock',
        'GENEVA': 'Geneva',
        'CHATSWORTH': 'Chatsworth',
        'NORTHLAKE': 'Northlake',
        'COLUMBUS': 'Columbus',
        'AUSTIN': 'Austin',
        'FREEPORT': 'Freeport',
        'LAKEWOOD': 'Lakewood',
        'AURORA': 'Aurora',
        'La Porte': 'LA Porte',
        'WINSTON SALEM': 'Winston-Salem',
        'GREENVILLE': 'Greenville',
        'SYRACUSE': 'Syracuse',
        'RICHFIELD': 'Richfield',
        'LaFayette': 'Lafayette',
        'LEWISTON': 'Lewiston',
        'WASHINGTON': 'Washington',
        'LAKE FOREST': 'Lake Forest',
        'MOUNT PLEASANT': 'Mount Pleasant',
        'BATAVIA': 'Batavia',
        'ANTIOCH': 'Antioch',
        'WEST HAVEN': 'West Haven',
        'DARIEN': 'Darien',
        'BLOOMINGDALE': 'Bloomingdale',
        'WOODBRIDGE': 'Woodbridge',
        'WESTON': 'Weston',
        'WESTMINSTER': 'Westminster',
        'HARRISONBURG': 'Harrisonburg',
        'MILLVILLE': 'Millville',
        'DILLON': 'Dillon',
        'NORTHWOOD': 'Northwood',
        'PAYSON': 'Payson',
        'TAYLORSVILLE': 'Taylorsville',
        'ANTHONY': 'Anthony',
        'INDUSTRY': 'Industry',
        'PARK CITY': 'Park City',
        'APEX': 'Apex',
        'SALEM': 'Salem',
        'Mcminnville': 'McMinnville',
        'MIDWAY': 'Midway',
        'CLINTON': 'Clinton',
        'COMMERCE': 'Commerce',
        'City of Commerce': 'Commerce',
        'Commerce City': 'COMMERCE CITY',
        'MANCHESTER': 'Manchester',
        'Valley Center': 'Center Valley',
        'MINERAL WELLS': 'Mineral Wells',
        'SARATOGA SPRINGS': 'Saratoga Springs',
        'MIDLAND': 'Midland',
        'HIGHLAND PARK': 'Highland Park',
        'VAN BUREN': 'Van Buren',
        'ADDISON': 'Addison',
        'Grantsville': 'GRANTSVILLE',
        'ANDOVER': 'Andover',
        'THORNTON': 'Thornton',
        'HURRICANE': 'Hurricane',
        'HUNTSVILLE': 'Huntsville',
        'SALINA': 'Salina',
        'MAPLETON': 'Mapleton',
        'ORION': 'Orion',
        'WADSWORTH': 'Wadsworth',
        'INGLESIDE':'Ingleside',
        'NATIONAL CITY': 'National City',
        'HAWTHORNE': 'Hawthorne',
        'AVONDALE': 'Avondale'
    }

    df['Origin City'] = df['Origin City'].replace(city_aliases)
    df['Destination City'] = df['Destination City'].replace(city_aliases)

    df.to_csv('./data/clean_data_2021.csv', index=False)

    return df


def find_similar_cities(df: pd.DataFrame, threshold: int = 80) -> None:
    """
    Find similar city names in origin and destination cities using fuzzy string matching.

    Args:
        df (pd.DataFrame): A pandas DataFrame containing columns "Origin City" and "Destination City".
        threshold (int): The minimum score (out of 100) for a match to be considered a "similar" match.

    Returns:
        None. Prints a preview of the similar words for each city name.
    """

    # Define a function to apply fuzzy string matching to each pair of city names
    def find_best_match(city, choices):
        return process.extractOne(city, choices, scorer=fuzz.token_sort_ratio, score_cutoff=threshold)

    # Find all unique city names across both origin and destination cities
    all_cities = pd.concat(
        [df['Origin City'], df['Destination City']]).unique()

    city_mapping = {}

    # Use ThreadPoolExecutor to parallelize the fuzzy string matching
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_city_mapping = {executor.submit(
            find_best_match, city, all_cities): city for city in all_cities}
        for future in concurrent.futures.as_completed(future_city_mapping):
            city = future_city_mapping[future]
            match = future.result()
            if match is not None:
                city_mapping[city] = match[0]

    # Print a preview of the similar city names
    print("Similar city names preview:")
    for city, match in city_mapping.items():
        if city != match:
            print(f"{city} -> {match}")


def count_routes(df):
    df_temp = df.copy()
    df_temp['Route'] = df_temp.apply(lambda row: tuple(sorted([(row['Origin City'], row['Origin State']),
                                                               (row['Destination City'], row['Destination State'])])), axis=1)
    route_counts = df_temp.groupby('Route').size().reset_index(name='Count')
    route_counts[['Origin City', 'Origin State']] = pd.DataFrame(route_counts['Route'].tolist())[0].apply(pd.Series)
    route_counts[['Destination City', 'Destination State']] = pd.DataFrame(route_counts['Route'].tolist())[1].apply(pd.Series)
    route_counts = route_counts.drop('Route', axis=1)
    return route_counts

def _request_distances(api_key, routes_df, checkpoint_interval=250, checkpoint_folder='google_api_checkpoints'):
    gmaps = googlemaps.Client(key=api_key)

    # Create a list of origin-destination pairs
    routes = list(zip(routes_df['Origin City'] + ', ' + routes_df['Origin State'],
                      routes_df['Destination City'] + ', ' + routes_df['Destination State']))

    all_distances = []
    all_durations = []
    all_start_coords = []
    all_end_coords = []

    # Check for existing checkpoint files
    if not os.path.exists(checkpoint_folder):
        # Create the folder
        os.makedirs(checkpoint_folder)
        print(f"Folder '{checkpoint_folder}' created successfully.")

    existing_checkpoints = glob.glob(f"{checkpoint_folder}/*.csv")
    last_checkpoint = 0

    if existing_checkpoints:
        last_checkpoint = max([int(os.path.basename(f).split('_')[-1].split('.')[0]) for f in existing_checkpoints])
        checkpoint_df = pd.read_csv(f"{checkpoint_folder}/routes_with_distances_checkpoint_{last_checkpoint}.csv")
        all_distances = checkpoint_df['Distance (km)'].tolist()[:last_checkpoint]
        all_durations = checkpoint_df['Duration (s)'].tolist()[:last_checkpoint]
        all_start_coords = checkpoint_df['Start Coordinates'].tolist()[:last_checkpoint]
        all_end_coords = checkpoint_df['End Coordinates'].tolist()[:last_checkpoint]

    # Iterate over the remaining routes
    for i, (origin, destination) in enumerate(routes[last_checkpoint:], start=last_checkpoint):
        # Make the API request
        try:
            response = gmaps.directions(origin, destination, units='metric')

            # Extract the distance and duration from the response
            distance = sum([leg['distance']['value'] for leg in response[0]['legs']]) / 1000
            duration = sum([leg['duration']['value'] for leg in response[0]['legs']])
            start_coordinates = str(response[0]['legs'][0]['start_location']['lat']) + ',' + str(response[0]['legs'][0]['start_location']['lng'])
            end_coordinates = str(response[0]['legs'][0]['end_location']['lat']) + ',' + str(response[0]['legs'][0]['end_location']['lng'])
            
        except Exception as e:
            print(f"Error at route {i} ({origin} - {destination}): {e}")
            now = datetime.now()
            date_string = now.strftime("%Y-%m-%d %H:%M:%S")

            # Open the file in append mode (creates the file if it doesn't exist)
            with open('date_log.txt', 'a') as file:
                # Write the date to the file
                file.write(f"Error occured at {date_string}\n{i} {origin} - {destination}\n")
                
            distance = -1
            duration = -1
            start_coordinates = "-1,-1"
            end_coordinates = "-1,-1"

        all_distances.append(distance)
        all_durations.append(duration)
        all_start_coords.append(start_coordinates)
        all_end_coords.append(end_coordinates)

        # Sleep between API requests
        time.sleep(_SLEEP_TIME)

        # Save a checkpoint file after every checkpoint_interval routes or if an exception occurred
        if (i + 1) % checkpoint_interval == 0 or distance == -1:
            routes_df.loc[:i, 'Distance (km)'] = all_distances[:i+1]
            routes_df.loc[:i, 'Duration (s)'] = all_durations[:i+1]
            routes_df.loc[:i, 'Start Coordinates'] = all_start_coords[:i+1]
            routes_df.loc[:i, 'End Coordinates'] = all_end_coords[:i+1]
            checkpoint_path = f"{checkpoint_folder}/routes_with_distances_checkpoint_{i+1}.csv"
            routes_df.to_csv(checkpoint_path, index=False)
            print(f'Checkpoint saved at route {i+1}')

    return all_distances, all_durations, all_start_coords, all_end_coords

def get_routes_distances(routes):

    distances, durations, start_coords, end_coords = _request_distances(_API_KEY, routes)

    # Add the distances to the dataframe and save it to a new CSV file
    routes['Distance (km)'] = distances
    routes['Duration (s)'] = durations
    routes['Start Coordinates'] = start_coords
    routes['End Coordinates'] = end_coords

    routes.to_csv('routes_with_distances.csv', index=False)

    return routes

# Add the distances,
def clean_routes(routes):
    
    # Filter out routes with a distance of -1
    routes = routes[routes['Distance (km)'] != -1]

    # Save the filtered data back to the CSV file
    routes.to_csv('filtered_data.csv', index=False)

    return routes

def read_routes_from_file() -> pd.DataFrame:
    return pd.read_csv('./data/routes.csv')

def read_fuel_prices_from_file() -> pd.DataFrame:
    df = pd.read_csv('./data/fuel_prices.csv', delimiter=';')

    for column in df.columns:
        if column != 'Date':
            df[column] = pd.to_numeric(df[column].str.replace(',', '.'))

    column_mapping = {
        'Weekly U.S. No 2 Diesel Retail Prices (Dollars per Gallon)': 'US',
        'Weekly East Coast No 2 Diesel Retail Prices (Dollars per Gallon)': 'East Coast',
        'Weekly New England (PADD 1A) No 2 Diesel Retail Prices (Dollars per Gallon)': 'New England',
        'Weekly Central Atlantic (PADD 1B) No 2 Diesel Retail Prices (Dollars per Gallon)': 'Central Atlantic',
        'Weekly Lower Atlantic (PADD 1C) No 2 Diesel Retail Prices (Dollars per Gallon)': 'Lower Atlantic',
        'Weekly Midwest No 2 Diesel Retail Prices (Dollars per Gallon)': 'Midwest',
        'Weekly Gulf Coast No 2 Diesel Retail Prices (Dollars per Gallon)': 'Gulf Coast',
        'Weekly Rocky Mountain No 2 Diesel Retail Prices (Dollars per Gallon)': 'Rocky Mountain',
        'Weekly West Coast No 2 Diesel Retail Prices (Dollars per Gallon)': 'West Coast',
        'Weekly California No 2 Diesel Retail Prices (Dollars per Gallon)': 'California',
        'Weekly West Coast (PADD 5) Except California No 2 Diesel Retail Prices (Dollars per Gallon)': 'West Coast Except CA'
    }

    # Rename the columns
    df.rename(columns=column_mapping, inplace=True)

    return df


def calculate_weekday_routes(deliveries: pd.DataFrame) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    deliveries = parse_dates(deliveries)

    weekday_routes = {}
    for i in range(7):
        weekday_data = deliveries[deliveries['Ship Day'].dt.weekday ==
                                  i]['Ship Day'].value_counts()

        if not weekday_data.empty:
            kernel = gaussian_kde(weekday_data)
            unique_deliveries = np.arange(1, weekday_data.max() + 1)
            probabilities = kernel(unique_deliveries)
            probabilities /= probabilities.sum()
        else:
            unique_deliveries = np.array([])
            probabilities = np.array([])

        weekday_routes[i] = (unique_deliveries, probabilities)

    return weekday_routes


def sample_num_deliveries(weekday_routes: Dict[int, Tuple[np.ndarray, np.ndarray]], weekday: int) -> int:
    unique_deliveries, probabilities = weekday_routes[weekday]
    if len(unique_deliveries) > 0:
        num_deliveries = np.random.choice(unique_deliveries, p=probabilities)
        return num_deliveries
    else:
        return


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    df['Ship Day'] = pd.to_datetime(df['Ship Day'])
    return df


def validate_weekday_routes(weekday_routes: Dict[int, Tuple[np.ndarray, np.ndarray]], deliveries: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    deliveries = parse_dates(deliveries)

    fig, axs = plt.subplots(2, 4, figsize=(18, 8))
    axs = axs.flatten()

    for weekday in range(7):
        ax = axs[weekday]
        unique_deliveries, probabilities = weekday_routes[weekday]

        if len(unique_deliveries) > 0:
            ax.plot(unique_deliveries, probabilities, label='KDE')
            original_counts = deliveries[deliveries['Ship Day'].dt.weekday ==
                                         weekday]['Ship Day'].value_counts().values
            ax.hist(original_counts, bins=range(
                1, max(unique_deliveries) + 2), alpha=0.5, density=True, label='Original')
            ks_stat, ks_pval = kstest(original_counts, lambda x: np.interp(
                x, unique_deliveries, probabilities))
            ax.set_title(
                f'Weekday {weekday} | KS Stat: {ks_stat:.5f} | p-value: {ks_pval:.5f}')
            if ks_pval < 0.05:
                print(
                    f'Weekday {weekday} distributions are significantly different')
            else:
                print(
                    f'Weekday {weekday} distributions are not significantly different')
        else:
            ax.set_title(f'Weekday {weekday} | No data')

        ax.set_xlabel('Delivery count')
        ax.set_ylabel('Probability')
        ax.legend()

    # Remove the unused subplot
    fig.delaxes(axs[7])

    # Adjust the layout
    fig.tight_layout()

    plt.show()



from scipy.stats import gaussian_kde

def calculate_route_distribution_kde(routes: pd.DataFrame):
    """
    This function calculates the Gaussian KDE for route distributions and adds a 'Probability' column to a copy of the input DataFrame.
    
    Parameters:
    routes (pd.DataFrame): A DataFrame containing route information and 'Count' of each route.

    Returns:
    pd.DataFrame: A copy of the input DataFrame with an added 'Probability' column.
    """
    # Create a copy of the input DataFrame
    routes_copy = routes.copy()

    # Calculate the route frequencies
    route_counts = routes_copy['Count']

    # Calculate the Gaussian KDE
    kernel = gaussian_kde(route_counts)

    # Define a range of possible route counts
    route_counts_range = np.arange(route_counts.min(), route_counts.max() + 1)

    # Calculate the probabilities
    probabilities = kernel(route_counts_range)

    # Normalize the probabilities so they sum to 1
    probabilities /= probabilities.sum()

    # Create a probability distribution mapping
    probability_mapping = dict(zip(route_counts_range, probabilities))

    # Map the probabilities to the corresponding counts in the dataframe
    routes_copy['Probability'] = routes_copy['Count'].map(probability_mapping)

    return routes_copy


def sample_route(deliveries_data: pd.DataFrame) -> pd.Series:
    """
    This function randomly samples a route from the given DataFrame.

    Parameters:
    deliveries_data (pd.DataFrame): A DataFrame with 'Count' and 'Probability' columns,
                                     and other columns with route details.
                                     'Count' column contains route counts and 
                                     'Probability' column contains the corresponding probabilities.

    Returns:
    pd.Series: A randomly sampled route from the DataFrame.
    """
    return deliveries_data.sample(n=1, weights='Probability').iloc[0]

def add_average_duration(deliveries: pd.DataFrame, routes: pd.DataFrame) -> pd.DataFrame:
    """
    Add the average shipping duration from the deliveries DataFrame to the routes DataFrame.
    
    Parameters:
    deliveries (pd.DataFrame): The deliveries DataFrame.
    routes (pd.DataFrame): The routes DataFrame.
    
    Returns:
    pd.DataFrame: The updated routes DataFrame with the average shipping duration added.
    """
    
    # Convert 'Ship Day' and 'Delivery Day' to datetime
    deliveries['Ship Day'] = pd.to_datetime(deliveries['Ship Day'])
    deliveries['Delivery Day'] = pd.to_datetime(deliveries['Delivery Day'])
    
    # Calculate the shipping duration in seconds
    deliveries['Shipping Duration'] = (deliveries['Delivery Day'] - deliveries['Ship Day']).dt.total_seconds()
    
    # Calculate the average shipping duration for each route
    average_duration = deliveries.groupby(['Origin City', 'Origin State', 'Destination City', 'Destination State'])['Shipping Duration'].mean().reset_index()
    average_duration.columns = ['Origin City', 'Origin State', 'Destination City', 'Destination State', 'Average Duration']
    
    # Merge the average_duration DataFrame into the routes DataFrame
    updated_routes = pd.merge(routes, average_duration, how='left',
                              left_on=['Origin City', 'Origin State', 'Destination City', 'Destination State'],
                              right_on=['Origin City', 'Origin State', 'Destination City', 'Destination State'])
    
    # If there are still NaN values in 'Average Duration' after the first merge, try to fill them by swapping origin and destination
    remaining_na = updated_routes['Average Duration'].isna()
    remaining_routes = updated_routes.loc[remaining_na].copy()
    remaining_routes = remaining_routes.rename(columns={"Origin City": "Destination City", "Origin State": "Destination State", 
                                                        "Destination City": "Origin City", "Destination State": "Origin State"})
    remaining_routes = pd.merge(remaining_routes, average_duration, how='left',
                                left_on=['Origin City', 'Origin State', 'Destination City', 'Destination State'],
                                right_on=['Origin City', 'Origin State', 'Destination City', 'Destination State'])
    remaining_routes = remaining_routes.rename(columns={"Origin City": "Destination City", "Origin State": "Destination State", 
                                                        "Destination City": "Origin City", "Destination State": "Origin State"})
    updated_routes.loc[remaining_na, 'Average Duration'] = remaining_routes['Average Duration_y']

    # Return the updated routes DataFrame
    return updated_routes

import pandas as pd
from typing import List, Tuple, Set

def create_new_routes(
    warehouse, 
    routes: pd.DataFrame, 
    catchment_cities: Set[Tuple[str, str]], 
    output_file: str
) -> pd.DataFrame:
    """
    Function to create new routes considering warehouse location.

    Parameters:
    warehouses: warehouse data.
    routes (pd.DataFrame): Dataframe containing routes data.
    catchment_cities (Set[Tuple[str, str]]): Set of tuples with catchment cities and their states.
    warehouse_id (int): Index of the selected warehouse in warehouses dataframe.

    Returns:
    pd.DataFrame: Dataframe with the updated routes.
    """

    if os.path.exists(output_file):
        df = pd.read_csv(output_file)

        existing_columns = [col for col in ['Average Duration','Probability'] if col in df.columns]
        
        # Remove the columns
        
        df = df.drop(columns=existing_columns)
        df = df.drop_duplicates()
        df.to_csv(output_file, index=False)

        # Find duplicate rows
        duplicates = df[df.duplicated(keep=False)]

        print(duplicates)

        return df

    # Find routes with destination city in catchment area
    routes_in_catchment_area = routes[routes.apply(lambda x: (x['Destination City'], x['Destination State']) in catchment_cities, axis=1)]
    
    print(routes_in_catchment_area.head())


    # Create an empty dataframe for new routes
    new_routes = pd.DataFrame()

    for _, row in routes_in_catchment_area.iterrows():
        # Create a copy of the row and update the destination to the warehouse
        row_to_warehouse = row.copy()
        row_to_warehouse['Destination City'] = warehouse['city']
        row_to_warehouse['Destination State'] = warehouse['state']
        row_to_warehouse['End Coordinates'] = f"{warehouse['latitude']},{warehouse['latitude']}"


        # Create a copy of the row and update the origin to the warehouse
        row_from_warehouse = row.copy()
        row_from_warehouse['Origin City'] = warehouse['city']
        row_from_warehouse['Origin State'] = warehouse['state']
        row_to_warehouse['Start Coordinates'] = f"{warehouse['latitude']},{warehouse['latitude']}"

        # Add the new route to the new_routes dataframe
        new_routes = pd.concat([new_routes, pd.DataFrame([row_from_warehouse, row_to_warehouse])], ignore_index=True)

    new_routes = new_routes.drop('Count', axis=1)
    existing_columns = [col for col in ['Average Duration','Probability'] if col in df.columns]
    
    df = df.drop(columns=existing_columns)

    new_routes['Distance (km)'] = -1
    new_routes['Duration (s)'] = -1
    
    df_no_duplicates = new_routes.drop_duplicates(subset=['Origin City', 'Origin State', 'Destination City', 'Destination State'])
    
    df_no_duplicates.reset_index(drop=True, inplace=True)

    df_no_duplicates.to_csv(output_file, index=False)

    return df_no_duplicates