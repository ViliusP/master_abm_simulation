
from datetime import timedelta
from typing import Dict, List, Tuple, Union
from mesa import Agent, Model
import mesa
import numpy as np
import pandas as pd
from catchment_area import get_catchment_area_cities
from data_management import add_average_duration, calculate_route_distribution_kde, calculate_weekday_routes, create_new_routes, get_routes_distances, read_deliveries, read_fuel_prices_from_file, read_routes_from_file, sample_num_deliveries, sample_route
import database
import networkx
import seaborn as sns
from fuel_prices import get_diesel_price_for_route
from model_utilties import calculate_weight_distribution_kde, sample_weight

# Fuel consumption in liters per 100 kilometers
FUEL_CONSUMPTION = 32.6

AGENT_TYPE_WAREHOUSE = "agent_type_warehouse"
AGENT_TYPE_TRUCK = "agent_type_truck"
AGENT_TYPE_CUSTOMER = "agent_type_customer"

AGENT_TYPE_OTHER = "agent_type_other"

DATA_FILE = './data/clean_data_2021.csv'


def calc_average_distance_per_shipment(shipments: int, distance):
    if shipments == 0 or distance == 0:
        return 0
    return distance/shipments


def calc_average_shipments_per_day(shipments_total: int, days):
    if shipments_total == 0 or days == 0:
        return 0
    return shipments_total/days


def calc_average_delivery_time(deviveries_total: int, deliveries_time_total):
    if deviveries_total == 0 or deliveries_time_total == 0:
        return 0
    return deliveries_time_total/deviveries_total


def calc_total_cost(total_distance_delivered):
    cost_per_mile = 1.855
    # To convert a cost per mile to cost per kilometer, conversion factor between miles and kilometers can be used.
    # The conversion factor is approximately 1.60934, which means there are 1.60934 kilometers in one mile.
    cost_per_kilometer = cost_per_mile / 1.60934
    return total_distance_delivered * cost_per_kilometer


def calc_greenhouse_gas(total_distance_delivered, total_weight_delivered):
    # total_distance_delivered is in kilometers !
    emission_coeficient = 1700
    # There are 0.621371 miles in one kilometer.
    total_distance_delivered_miles = total_distance_delivered * 0.621371
    # 1 kilogram = 2.20462 pounds
    total_weight_delivered_pounds = kg_to_lb(total_weight_delivered)

    return lb_to_kg(total_distance_delivered_miles * total_weight_delivered_pounds * emission_coeficient)


def calc_average_shipment_fuel_cost(total_fuel_cost, shipments: int):
    if total_fuel_cost == 0 or shipments == 0:
        return 0
    return total_fuel_cost/shipments


def kg_to_lb(kilograms):
    pounds = kilograms * 2.20462
    return pounds


def lb_to_kg(pounds):
    kilograms = pounds / 2.20462
    return kilograms


class LogisticsSimulation(Model):
    """
    The ABM (Agent-Based Model) simulates a delivery system where trucks move products between warehouses and customers. 
    The agents in the model are the trucks, the warehouses, and the customers. 
    The model takes into account the distance between the warehouses and customers, and the capacity of the trucks. 
    The goal of the model is to find the optimal route for the trucks to deliver products to the customers.

    The model is implemented in Python using the MESA (Multi-Agent Evolutionary Algorithm) platform. 
    The MESA platform allows us to define the behavior of the agents, simulate the interaction between the agents, and analyze the results of the simulation.

    By running the simulation, we can analyze the performance of the delivery system, such as the delivery time, the utilization of the trucks, and the cost of the delivery.
    We can also experiment with different scenarios, such as adding new warehouses or customers, changing the capacity of the trucks, or adjusting the behavior of the agents.
    """

    def __init__(self, warehouse_id=1, force_data_read=False, debug_mode=False, connection=None):

        self.debug_mode = debug_mode
        self.force_data_read = force_data_read
        self.connection = connection

        print("--- Reading data from CSV  ---") if debug_mode else 0
        deliveries = pd.read_csv(DATA_FILE)
        print(deliveries.head(10)) if debug_mode else 0
        self.deliveries = deliveries

        # Routes
        routes = read_routes_from_file()
        routes = add_average_duration(deliveries.copy(), routes)
        self.routes_distribution = calculate_route_distribution_kde(routes)
        print("Routes: ") if debug_mode else 0
        print(routes.head(10)) if debug_mode else 0
        print("Routes distribution: ") if debug_mode else 0
        print(self.routes_distribution.head(10)) if debug_mode else 0

        # Fuel prices
        self.fuel_prices = read_fuel_prices_from_file()
        print("Fuel prices: ") if debug_mode else 0
        print(self.fuel_prices.head(10)) if debug_mode else 0

        # Weight
        self.weight_distribution = calculate_weight_distribution_kde(
            deliveries)

        database.save_deliveries(connection, deliveries)

        self.database_connection = connection

        self.weekdays_deliveries_distributions = calculate_weekday_routes(
            deliveries)

        # Set seed for more predictable and predictable results
        self._seed = 80085
        self.random.seed(self._seed)
        np.random.seed(seed=self._seed)

        # Initial model data
        self.trucks_generated = 0
        self.customers_generated = 0
        self.shipments_sent = 0
        self.shipments_delivered_today = 0
        self.shipments_delivered_total = 0
        self.delivery_duration_total = 0
        self.total_weight_delivered = 0
        # self.shipment_received=0
        self.day = 0
        self.current_datetime = pd.to_datetime(
            self.deliveries['Ship Day'], format='%Y-%m-%d').min()
        self.total_distance_delivered = 0
        self.total_fuel_used = 0
        self.total_fuel_cost = 0

        self.G = networkx.DiGraph()
        self.grid = mesa.space.NetworkGrid(self.G)
        self.schedule = mesa.time.BaseScheduler(self)

        self.initialize_warehouse(warehouse_id)

        # agent_reporters = {"Wealth": "wealth"}
        model_reporters = {"Shipments sent": lambda m: self.shipments_sent,
                           "Average delivery time (min)": lambda m: calc_average_delivery_time(self.shipments_delivered_total, self.delivery_duration_total),
                           "Total transportation cost (dollar)": lambda m: calc_total_cost(self.total_distance_delivered),
                           "C02 emission (tonnes)": lambda m: calc_greenhouse_gas(self.total_distance_delivered, self.total_weight_delivered),
                           "Total fuel used (liters)": lambda m: self.total_fuel_used,
                           "Total fuel cost (dollars)": lambda m: self.total_fuel_cost,
                           "Average fuel cost per shipment (dollars)": lambda m: calc_average_shipment_fuel_cost(self.total_fuel_cost, self.shipments_delivered_total),
                           "Average distance per shipment": lambda m: calc_average_distance_per_shipment(self.shipments_sent, self.total_distance_delivered),

                           "Distance travelled in total": lambda m: self.total_distance_delivered,
                           "Average shipments per day": lambda m: calc_average_shipments_per_day(self.shipments_sent, self.day),
                           "Trucks on the road": lambda m: m.count_agents_by_type(AGENT_TYPE_TRUCK),
                           "Shipments delivered on day": lambda m: self.shipments_delivered_today,
                           }

        self.datacollector = mesa.DataCollector(
            model_reporters,
            # agent_reporters
        )

        self.running = True
        self.datacollector.collect(self)

    def step(self):
        # Stop condition
        if self.schedule.steps >= 365:
            self.running = False  # Stop the model after n steps
            return

        self.schedule.step()

        current_weekday = self.current_datetime.weekday()

        # Calculate the daily delivery count distribution with the current day
        num_deliveries = sample_num_deliveries(
            self.weekdays_deliveries_distributions, current_weekday)

        num_deliveries = num_deliveries

        warehouse = None

        for agent in self.schedule.agents:
            # Check if the agent is a customer
            if isinstance(agent, WarehouseAgent):
                warehouse = agent
                break

        for _ in range(num_deliveries):
            self.generate_agents(warehouse)

        self.clean_isolated_nodes()

        if self.debug_mode:
            print(f"Current day: {self.day}")
            print(f"Current date: {self.current_datetime}")
            print(f"Current weekday {current_weekday}")
            print(f"Shipments sent: {num_deliveries}")

        self.shipments_sent += int(num_deliveries)

        self.shipments_delivered_total += self.shipments_delivered_today

        self.datacollector.collect(self)

        # calculate_KPI(self.day, self.database_connection)

        self.day += 1
        self.current_datetime = self.current_datetime + timedelta(days=1)
        self.shipments_delivered_today = 0

    def clean_isolated_nodes(self):
        isolated_nodes = list(networkx.isolates(self.grid.G))
        for node_id in isolated_nodes:
            agent = self.grid.G.nodes[node_id].get("agent")[0]
            if agent is not None and agent.type != AGENT_TYPE_WAREHOUSE:
                self.schedule.remove(agent)
                self.grid.remove_agent(agent)
                self.grid.G.remove_node(agent.unique_id)

    def generate_agents(self, warehouse):
        # Random route
        random_shipment_route = sample_route(self.routes_distribution)
        fuel_price = get_diesel_price_for_route(
            random_shipment_route, self.fuel_prices, self.current_datetime)

        weight = sample_weight(self.weight_distribution)

        # Agent generation
        truck_id = self.trucks_generated
        self.trucks_generated += 1

        truck_agent = None
        destination_in_catchment_area = False

        if warehouse is not None and isinstance(warehouse, WarehouseAgent):
            modified_route = warehouse.modify_route(
                random_shipment_route, truck_id)
            if modified_route is not None:
                destination_in_catchment_area = True
                random_shipment_route = modified_route

        truck_agent = TruckAgent(
            self, truck_id, f"truck_{truck_id}", random_shipment_route, fuel_price, weight, self.current_datetime)

        truck_agent.type = AGENT_TYPE_TRUCK

        if not self.grid.G.has_node(truck_agent.unique_id):
            self.grid.G.add_node(truck_agent.unique_id)
        if not 'agent' in self.grid.G.nodes[truck_agent.unique_id]:
            self.grid.G.nodes[truck_agent.unique_id]["agent"] = mesa.space.NetworkGrid.default_val(
            )

        self.schedule.add(truck_agent)
        self.grid.place_agent(truck_agent, truck_agent.unique_id)

        if destination_in_catchment_area:
            self.grid.G.add_edge(truck_agent.unique_id, warehouse.unique_id,
                                 distance=f"{random_shipment_route['Distance (km)']}")

        if not destination_in_catchment_area:
            # Customer generation
            for agent in self.schedule.agents:
                # Check if the agent is a customer
                if isinstance(agent, CustomerAgent):
                    # Check if the agent has the desired properties
                    if agent.state == random_shipment_route['Destination State'] and agent.city == random_shipment_route['Destination City']:
                        self.grid.G.add_edge(truck_agent.unique_id, agent.unique_id,
                                             distance=f"{random_shipment_route['Distance (km)']}")
                        return

            customer_id = self.customers_generated
            self.customers_generated += 1

            customer_agent = CustomerAgent(
                self, customer_id, f"customer_{customer_id}", random_shipment_route)
            customer_agent.type = AGENT_TYPE_CUSTOMER

            if not self.grid.G.has_node(customer_agent.unique_id):
                self.grid.G.add_node(customer_agent.unique_id)
            if not 'agent' in self.grid.G.nodes[customer_agent.unique_id]:
                self.grid.G.nodes[customer_agent.unique_id]["agent"] = mesa.space.NetworkGrid.default_val(
                )

            self.schedule.add(customer_agent)
            self.grid.place_agent(customer_agent, customer_agent.unique_id)

            self.grid.G.add_edge(truck_agent.unique_id, customer_agent.unique_id,
                                 distance=f"{random_shipment_route['Distance (km)']}")

    def count_agents_by_type(self, agent_type):
        return sum([1 for agent in self.schedule.agents if agent.type == agent_type])

    def initialize_warehouse(self, id: int):
        selected_warehouse = None

        warehouses_data = {
            'state': ['Texas', 'Georgia', 'Florida', 'Pennsylvania'],
            'city': ['Dallas', 'Atlanta', 'Tampa', 'Philadelphia'],
            'latitude': [32.7767, 33.7490, 27.9506, 39.9526],
            'longitude': [-96.7970, -84.3880, -81.4565, -75.1652]
        }

        warehouses = pd.DataFrame(warehouses_data)

        if id == 0 or id >= 4:
            return

        selected_warehouse = warehouses.loc[id-1]

        print("Selected warehouse:")
        print(selected_warehouse)

        catchment_cities = get_catchment_area_cities(
            self.routes_distribution, selected_warehouse["latitude"], selected_warehouse["longitude"], 150)
        
        if self.debug_mode:
            print(catchment_cities)
            
            # Create a new Series that represents the Destination as a tuple
            destination_tuples = pd.Series(list(zip(self.deliveries['Destination City'], self.deliveries['Destination State'])), index=self.deliveries.index)

            # Check if each Destination tuple is in the catchment area
            in_catchment = destination_tuples.apply(lambda x: x in catchment_cities)

            # Count the number of True values in the in_catchment Series
            num_deliveries_in_catchment = in_catchment.sum()

            print(f'Number of deliveries in catchment area: {num_deliveries_in_catchment}')


        output_file = f"./data/{selected_warehouse['state']}_{selected_warehouse['city']}_warehouse_routes.csv"

        warehouse_routes = create_new_routes(
            selected_warehouse, self.routes_distribution, catchment_cities, output_file)

        print(warehouse_routes.head())

        # Check if any row has distance or duration equal to -1
        has_invalid_values = (
            warehouse_routes['Distance (km)'] == -1) | (warehouse_routes['Duration (s)'] == -1)
        if has_invalid_values.any():
            distanced_routes = get_routes_distances(warehouse_routes)
            # distanced_routes.to_csv(output_file)
        else:
            print("No row has distance or duration equal to -1")

        warehouse = WarehouseAgent(
            self, id, f"warehouse_{id}", selected_warehouse, catchment_cities, warehouse_routes)

        warehouse.type = AGENT_TYPE_WAREHOUSE
        if not self.grid.G.has_node(warehouse.unique_id):
            self.grid.G.add_node(warehouse.unique_id)
        if not 'agent' in self.grid.G.nodes[warehouse.unique_id]:
            self.grid.G.nodes[warehouse.unique_id]["agent"] = mesa.space.NetworkGrid.default_val(
            )

        self.grid.place_agent(warehouse, warehouse.unique_id)
        self.schedule.add(warehouse)


class TruckAgent(Agent):

    # Route example:
    # Origin City                    ELIZABETHTON
    # Origin State                             TN
    # Destination City                 Wilmington
    # Destination State                        OH
    # Distance (km)                       541.638
    # Duration (s)                        22403.0
    # Start Coordinates     36.3487032,-82.210657
    # End Coordinates      39.4453424,-83.8285971
    # Average Duration                    86400.0
    # Probability                        0.224614
    def __init__(self, model, id, name, route, fuel_price, weight, ship_day):
        super().__init__(f"9{id}", model)
        self.name = f"9{name}"
        self.trip_distance = route["Distance (km)"]
        self.weight = weight
        self.ship_day = ship_day
        self.origin_state = route['Origin State']
        self.origin_city = route['Origin City']
        self.destination_state = route['Destination State']
        self.destination_city = route['Destination City']
        self.fuel_price = fuel_price
        self.days_on_road = 0
        # self.travel_duration = route['Average Duration'] // 86400
        self.travel_duration = route['Duration (s)']
        self.time_traveled = 0
        self.through_warehouse = route.get("Through warehouse", False)
        self.through_warehouse_original = route.get("Through warehouse", False)
        self.in_warehouse = False

    def step(self):
        if self.in_warehouse:
            return
        if self.time_traveled >= self.travel_duration:
            # Reporting to model
            self.model.total_distance_delivered += self.trip_distance
            self.time_traveled = self.travel_duration
            self.model.shipments_delivered_today += 1
            self.model.delivery_duration_total += (self.travel_duration // 60)
            self.model.total_weight_delivered += self.weight

            # Fuel
            # In liters
            fuel_used = FUEL_CONSUMPTION * (self.trip_distance / 100)
            liters_per_gallon = 3.78541

            self.model.total_fuel_used += fuel_used
            self.model.total_fuel_cost += (fuel_used /
                                           liters_per_gallon) * self.fuel_price

            # Agent removing
            if not self.through_warehouse:
                database.insert_truck_kpi(self.model.connection, self.to_dataframe())
                self.model.grid.remove_agent(self)
                self.model.schedule.remove(self)
                self.model.grid.G.remove_node(self.unique_id)
            if self.through_warehouse and not self.in_warehouse:
                self.in_warehouse = True
            return

        self.time_traveled += 11 * 3600  # Add 11 hours in seconds to the time traveled
        self.days_on_road += 1
        pass

    def reroute(self, route, warehouse_id):
        self.time_traveled = 0
        self.trip_distance = route["Distance (km)"]
        self.travel_duration = route['Duration (s)']
        self.origin_state = route['Origin State']
        self.origin_city = route['Origin City']
        self.destination_state = route['Destination State']
        self.destination_city = route['Destination City']
        self.through_warehouse = route.get("Through warehouse", False)
        self.in_warehouse = False
        self._remove_edges()

        self.model.grid.G.add_edge(self.unique_id, warehouse_id,
                                   distance=route["Distance (km)"])

    def _remove_edges(self):
        edges_to_remove = []
        for edge in self.model.grid.G.edges([self.unique_id]):
            edges_to_remove.append(edge)

        self.model.grid.G.remove_edges_from(edges_to_remove)

    def to_dataframe(self):
        fuel_used = FUEL_CONSUMPTION * (self.trip_distance / 100)
        liters_per_gallon = 3.78541
        fuel_cost = (fuel_used / liters_per_gallon) * self.fuel_price

        agent_attributes = {
            'id': [self.unique_id],
            'name': [self.name],
            'ship_day': [self.ship_day],
            'trip_distance': [self.trip_distance],
            'weight': [self.weight],
            'origin_state': [self.origin_state],
            'origin_city': [self.origin_city],
            'destination_state': [self.destination_state],
            'destination_city': [self.destination_city],
            'fuel_price': [self.fuel_price],
            'fuel_used': [fuel_used],
            'fuel_cost': [fuel_cost],
            'days_on_road': [self.days_on_road],
            'travel_duration': [self.travel_duration],
            'time_traveled': [self.time_traveled],
            'through_warehouse': [self.through_warehouse_original]
        }
        return pd.DataFrame(agent_attributes)


class WarehouseAgent(Agent):

    def __init__(self, model, id, name, data, catchment_cites: set, warehouse_routes: pd.DataFrame):
        super().__init__(id, model)

        self.name = name
        self.state = data['state']
        self.city = data['city']

        self.latitude = data["latitude"]
        self.longitude = data["longitude"]

        self.catchment_cites = catchment_cites
        self.warehouse_routes = warehouse_routes

        self.trucks_to_dispatch = pd.DataFrame()

    def step(self):
        for agent in self.model.schedule.agents:
            if isinstance(agent, TruckAgent) and agent.through_warehouse and agent.in_warehouse:
                next_route = self.trucks_to_dispatch[self.trucks_to_dispatch['Agent ID'] == agent.unique_id].iloc[0].copy(
                )
                agent.reroute(next_route, self.unique_id)
                self.trucks_to_dispatch = self.trucks_to_dispatch[
                    self.trucks_to_dispatch['Agent ID'] != agent.unique_id]

    def modify_route(self, route, truck_id):
        if (route['Destination City'], route['Destination State']) in self.catchment_cites:

            origin_city = route['Origin City']
            destination_city = route['Destination City']

            first_route = self.warehouse_routes[(
                self.warehouse_routes['Origin City'] == origin_city)]
            if first_route.empty:
                first_route = self.warehouse_routes[(
                    self.warehouse_routes['Destination City'] == origin_city)]

            second_route = self.warehouse_routes[(
                self.warehouse_routes['Origin City'] == destination_city)]
            if second_route.empty:
                second_route = self.warehouse_routes[(
                    self.warehouse_routes['Destination City'] == destination_city)]

            if first_route.empty or second_route.empty:
                print("You shouldn't see this")
                return None

            truck_to_dispatch = second_route.iloc[0].copy()
            truck_to_dispatch['Agent ID'] = f"9{truck_id}"
            truck_to_dispatch['Origin State'] = self.state
            truck_to_dispatch['Origin City'] = self.city
            truck_to_dispatch['Destination State'] = route['Destination State']
            truck_to_dispatch['Destination City'] = route['Destination City']

            self.trucks_to_dispatch = pd.concat(
                [self.trucks_to_dispatch, truck_to_dispatch.to_frame().T], ignore_index=True, axis=0)

            new_route = first_route.iloc[0].copy()
            new_route['Destination State'] = self.state
            new_route['Destination City'] = self.city
            new_route['Origin State'] = route['Origin State']
            new_route['Origin City'] = route['Origin City']
            new_route['Through warehouse'] = True

            return new_route

        return None


class CustomerAgent(Agent):

    def __init__(self, model, id, name, shipment):
        super().__init__(f"5{id}", model)
        self.name = f"5{name}"

        self.state = shipment['Destination State']
        self.city = shipment['Destination City']

        self.latitude = shipment["End Coordinates"][0]
        self.longitude = shipment["End Coordinates"][1]

    def step(self):
        pass
