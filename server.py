import mesa
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import UserSettableParameter
from mesa.visualization.modules import ChartModule
from mesa.visualization.modules import NetworkModule
from mesa.visualization.modules import TextElement
from database import create_connection
from model import AGENT_TYPE_CUSTOMER, AGENT_TYPE_TRUCK, AGENT_TYPE_WAREHOUSE, LogisticsSimulation


class CustomTextElement(TextElement):
    """
    Display a text count of how many happy agents there are.
    """

    def __init__(self, label, property):
        self.label = label
        self.property = property
        pass

    def render(self, model):
        return f"{self.label}: {getattr(model, str(self.property))}"


def supply_chain_network(graph):

    def node_color(agent):
        return {
            AGENT_TYPE_TRUCK: '#ff00ff',
            AGENT_TYPE_WAREHOUSE: '#323ca8',
            AGENT_TYPE_CUSTOMER: "#44fcb3"
        }.get(agent.type, '#8a8a8a')

    portrayal = dict()
    portrayal['nodes'] = [{'size': 6,
                           'id': id,
                           'color': node_color(agents[0]),
                           'name': agents[0].name,
                           'tooltip': "id: {}<br>type: {}<br>name: {}<br>Coming entities:".format(agents[0].unique_id, agents[0].type, agents[0].name),
                           }
                          for (id, agents) in graph.nodes.data('agent')]

    id_map = {v['id']: i for i, v in enumerate(portrayal['nodes'])}

    portrayal['edges'] = [{'id': edge_id,
                           'source': id_map[source],
                           'target': id_map[target],
                           'distance': data['distance'],
                           'color': '#000000',
                           }
                          for edge_id, (source, target, data) in enumerate(graph.edges(data=True))]

    for edge in portrayal['edges']:
        source_id = edge['source']
        target_id = edge['target']
        distance = edge['distance']
        portrayal['nodes'][target_id]['tooltip'] = portrayal['nodes'][target_id]['tooltip'] + \
            "<br>{} - {}".format(portrayal['nodes']
                                 [source_id]['name'], distance)

    return portrayal


grid = mesa.visualization.NetworkModule(supply_chain_network, 650, 650)

# 8
distance_travalled_chart = mesa.visualization.ChartModule(
    [{"Label": "Distance travelled in total", "Color": "#32a852"}], data_collector_name="datacollector"
)

# 7,9,1
average_distance_chart = mesa.visualization.ChartModule(
    [{"Label": "Average distance per shipment", "Color": "#32a852"}, {"Label": "Average shipments per day", "Color": "#37f852"}, {"Label": "Average delivery time (min)", "Color": "#f86437"}], data_collector_name="datacollector"
)

# 2
transport_cost_chart = mesa.visualization.ChartModule(
    [{"Label": "Total transportation cost (dollar)", "Color": "#0000FF"},], data_collector_name="datacollector"
)


# 3
greenhouse_gasses_emission = mesa.visualization.ChartModule(
    [{"Label": "C02 emission (tonnes)", "Color": "#0000FF"},], data_collector_name="datacollector"
)

# 4
total_fuel_used = mesa.visualization.ChartModule(
    [{"Label": "Total fuel used (liters)", "Color": "#0000FF"},], data_collector_name="datacollector"
)

# 5 
total_fuel_cost = mesa.visualization.ChartModule(
    [{"Label": "Total fuel cost (dollars)", "Color": "#0000FF"},], data_collector_name="datacollector"
)

# 6 
average_fuel_cost_shipment = mesa.visualization.ChartModule(
    [{"Label": "Average fuel cost per shipment (dollars)", "Color": "#0000FF"},], data_collector_name="datacollector"
)




truck_on_road = mesa.visualization.ChartModule(
    [{"Label": "Trucks on the road", "Color": "#0000FF"}, {"Label": "Shipments delivered on day", "Color": "#32a852"}], data_collector_name="datacollector"
)


watched_properties = {"Current day": "current_datetime",
                      "Shipments sent": "shipments_sent",
                      "Total distance (km)": "total_distance_delivered"
                      }

text_elements = [CustomTextElement(label, property)
                 for label, property in watched_properties.items()]


model_params = {
    "warehouse_id": mesa.visualization.Slider(
        "Warehouse location ID",
        1,
        0,
        3,
        1,
        description="Choose how many agents to include in the model",
    ),

}


def initialize_mesa_server(force_data_read=False, debug_mode=False):

    model_params_plus = model_params
    model_params_plus["force_data_read"] = force_data_read
    model_params_plus["debug_mode"] = debug_mode
    model_params_plus['connection'] = create_connection()

    server = ModularServer(
        LogisticsSimulation, [
            grid, *text_elements, average_distance_chart, distance_travalled_chart, transport_cost_chart, greenhouse_gasses_emission, total_fuel_used, total_fuel_cost, average_fuel_cost_shipment, truck_on_road], "Logistics supply chain simulation", model_params_plus, 8500
    )

    # Run the simulation using Mesa
    server.launch()
