from datetime import datetime
import sqlite3

import pandas as pd


DATABASE_NAME = "logistics.db"
DELIVERIES_TABLE = "deliveries"
KPI_TABLE = "key_perfomance_indicators"
NETWORK_EDGES_TABLE = "network_edges"
NETWORK_NODES_TABLE = "network_nodes"
RESULTS_AGENTS = "truck_agents"


KPI_CURRENT_STOCK = "current_stock"
KPI_INVENTORY_IN_ROAD = "inventory_in_road"
KPI_INVENTORY_TURNOVER = "inventory_turnover"


def create_connection() -> sqlite3.Connection:
    current_date = datetime.now().strftime("%Y-%m-%d_%H_%M")
    return sqlite3.connect(f"{current_date}_{DATABASE_NAME}", detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)


def initialize_database(conn: sqlite3.Connection):
    cursor = conn.cursor()
    try:
        # cursor.execute(f'DROP TABLE IF EXISTS {DELIVERIES_TABLE}')

        cursor.execute(f'DROP TABLE IF EXISTS {KPI_TABLE}')
        cursor.execute(
            f'CREATE TABLE IF NOT EXISTS {KPI_TABLE} (day INTEGER PRIMARY KEY, {KPI_INVENTORY_IN_ROAD} INT, {KPI_CURRENT_STOCK} INT, {KPI_INVENTORY_TURNOVER} INT)')

        cursor.execute(f'DROP TABLE IF EXISTS {NETWORK_EDGES_TABLE}')
        cursor.execute(
            f'CREATE TABLE IF NOT EXISTS {NETWORK_EDGES_TABLE} (node_x INTEGER, node_y INTEGER, distance REAL)')

        cursor.execute(f'DROP TABLE IF EXISTS {NETWORK_NODES_TABLE}')
        cursor.execute(
            f'CREATE TABLE IF NOT EXISTS {NETWORK_NODES_TABLE} (id INTEGER, name TEXT, type TEXT)')

    except Exception as e:
        if hasattr(e, 'message'):
            print(e.message)
        else:
            print(e)


def save_deliveries(conn: sqlite3.Connection, data: pd.DataFrame):
    """
    Inserts provided data to sqlite database deliveries table.
    :param data: pandas list
    """
    curr = conn.cursor()

    data.to_sql(DELIVERIES_TABLE, conn, if_exists='replace', index=False)

    last_row_id = curr.lastrowid

    return last_row_id


def to_report_KPI(_, connection: sqlite3.Connection, day, kpi_indicator):

    day = day - 1
    indicator = None
    try: 
        connection.execute(f'SELECT {kpi_indicator} FROM {KPI_TABLE} WHERE day = {day}')
        indicator = connection.cursor().fetchall()
        connection.commit()

        if len(indicator) != 0:
            indicator = indicator[0][0]
        elif len(indicator) == 0:
            indicator = 0
        return indicator
    except:    
        print("exception")
    
    return 0


def calculate_KPI(day: int, connection: sqlite3.Connection):

    connection.execute(f'INSERT INTO {KPI_TABLE} (day, {KPI_CURRENT_STOCK}, {KPI_INVENTORY_IN_ROAD}, {KPI_INVENTORY_TURNOVER}) VALUES ({day}, 0, 0, 0)')

def get_generated_agents(connection: sqlite3.Connection):
    agents = pd.read_sql_query(
    "SELECT * from %s;" % RESULTS_AGENTS, connection)

    return agents


def insert_truck_kpi(connection: sqlite3.Connection, data: pd.DataFrame): 
    curr = connection.cursor()

    data.to_sql(RESULTS_AGENTS, connection, if_exists='append', index=False)

    last_row_id = curr.lastrowid

    return last_row_id