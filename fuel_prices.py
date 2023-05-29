from typing import Dict, Union
import pandas as pd


state_to_padd = {
    'CT': 'New England', 'ME': 'New England', 'MA': 'New England',
    'NH': 'New England', 'RI': 'New England', 'VT': 'New England',
    'DE': 'Central Atlantic', 'DC': 'Central Atlantic', 'MD': 'Central Atlantic',
    'NJ': 'Central Atlantic', 'NY': 'Central Atlantic', 'PA': 'Central Atlantic',
    'FL': 'Lower Atlantic', 'GA': 'Lower Atlantic', 'NC': 'Lower Atlantic',
    'SC': 'Lower Atlantic', 'VA': 'Lower Atlantic', 'WV': 'Lower Atlantic',
    'IL': 'Midwest', 'IN': 'Midwest', 'IA': 'Midwest', 'KS': 'Midwest',
    'KY': 'Midwest', 'MI': 'Midwest', 'MN': 'Midwest', 'MO': 'Midwest',
    'NE': 'Midwest', 'ND': 'Midwest', 'SD': 'Midwest', 'OH': 'Midwest',
    'OK': 'Midwest', 'TN': 'Midwest', 'WI': 'Midwest',
    'AL': 'Gulf Coast', 'AR': 'Gulf Coast', 'LA': 'Gulf Coast',
    'MS': 'Gulf Coast', 'NM': 'Gulf Coast', 'TX': 'Gulf Coast',
    'CO': 'Rocky Mountain', 'ID': 'Rocky Mountain', 'MT': 'Rocky Mountain',
    'UT': 'Rocky Mountain', 'WY': 'Rocky Mountain',
    'CA': 'California',
    'AK': 'West Coast Except CA', 'AZ': 'West Coast Except CA', 'HI': 'West Coast Except CA',
    'NV': 'West Coast Except CA', 'OR': 'West Coast Except CA', 'WA': 'West Coast Except CA',
}

def get_diesel_price_for_route(route: pd.Series, fuel_prices: pd.DataFrame, ship_date: Union[str, pd.Timestamp]) -> float:
    # get the PADD region for the origin state
    padd_region = state_to_padd[route['Origin State']]
    if padd_region is None:
        padd_region = "US"

    # convert the ship_date to pandas Timestamp if it's a string
    if isinstance(ship_date, str):
        ship_date = pd.to_datetime(ship_date)
    
    # convert the 'Date' column in the fuel_prices DataFrame to datetime
    fuel_prices['Date'] = pd.to_datetime(fuel_prices['Date'])
    
    # find the row in the fuel prices data where the date is the closest to the shipping date 
    # but does not exceed it
    closest_date = fuel_prices[fuel_prices['Date'] <= ship_date].iloc[-1]
    
    # get the fuel price for the PADD region at the closest date
    diesel_price = closest_date[padd_region]

    # return the diesel price
    return diesel_price