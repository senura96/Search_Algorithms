import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
from geopy.distance import geodesic
import plotly.express as px
import matplotlib.pyplot as plt
import pandas as pd
import plotly.io as pio
import plotly.graph_objects as go
import numpy as np





# Define functions to create different visualizations
# Specify Plotly to use the Streamlit renderer
pio.renderers.default = "plotly_mimetype"



def count_edges_in_subgraph(graph, visited_nodes):
    subgraph = graph.subgraph(visited_nodes)
    return subgraph.number_of_edges()


# Function to compute distance between two locations
def compute_distance(location1, location2):
    return geodesic(location1, location2).kilometers


def calculate_covered_distance(graph, path):
    covered_distance = 0
    for i in range(len(path) - 1):
        source_node = path[i]
        destination_node = path[i + 1]
        # Assuming the graph is a weighted graph and the edge weight represents distance
        edge_weight = graph.get_edge_data(source_node, destination_node)['weight']
        covered_distance += edge_weight
    return covered_distance




def remove_duplicates(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]
    



def create_graph(merge_df):
    # Create an empty graph
    G = nx.Graph()

    # Add nodes to the graph
    for index, row in merge_df.iterrows():
        G.add_node(row["City"], pos=(row["lat"], row["lon"]) ,max_cap=row["po_cap"], cur_vol=row["cur_vol"], tp=row["avg_tp"], prb=row["avg_prb"] )

    # Add edges based on distance threshold (e.g., 1000 km)
    threshold_distance = 5  # kilometers            45 better

    for index1, row1 in merge_df.iterrows():
        for index2, row2 in merge_df.iterrows():
            if index1 != index2:
                distance = compute_distance((row1["lat"], row1["lon"]), (row2["lat"], row2["lon"]))
                if distance <= threshold_distance:
                    G.add_edge(row1["City"], row2["City"], weight=distance)

    
    # Find isolated nodes (nodes without edges)
    isolated_nodes = list(nx.isolates(G))

    # Remove isolated nodes from the graph
    G.remove_nodes_from(isolated_nodes)                
   
    return G



def page_graph(G , source_node , destination_node):

    st.title('Graph Visualization of Network Topology')
    
    # Change the Matplotlib backend to TkAgg for interactivity
    plt.switch_backend('TkAgg')

    # Visualize the graph with improved clarity
    fig = plt.figure(figsize=(60, 60))  # Adjust the figure size
    pos = nx.get_node_attributes(G, 'pos')
    nx.draw(G, pos, with_labels=True, node_size=4000, node_color='skyblue', font_size=40)

    # Highlight source node
    nx.draw_networkx_nodes(G, pos, nodelist=[source_node], node_size=8000, node_color='red')

    # Highlight destination node
    nx.draw_networkx_nodes(G, pos, nodelist=[destination_node], node_size=6000, node_color='green')


    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.title("Graph of Locations")

    

    plt.axis('off')  # Turn off axis

    # Display the plot in Streamlit
    st.pyplot(fig)

    # Add a button to go back to the Home page
    if st.button('Go back to Home'):
        st.experimental_rerun()  # Rerun the app to go back to the Home page






def page_map(merge_df, G , source_node , destination_node):
    st.title('Home')
    st.write('Welcome to the Network Routing Optimization in Private Networks Through Search Algorithms!')
    st.write('Select a page from the sidebar to explore different visualizations.')

    st.title('Map Visualization')
    fig = px.scatter_mapbox(merge_df, lat="lat", lon="lon", hover_name="City",
                            color_discrete_sequence=["white"], zoom=20, height=800 , center={"lat": merge_df['lat'].mean(), "lon": merge_df['lon'].mean()})

    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox_zoom=15,
        mapbox_layers=[
            {
                "below": 'traces',
                "sourcetype": "raster",
                "sourceattribution": "United States Geological Survey",
                "source": [
                    "https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}"
                ]
            }
        ])
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

    # Define positions for nodes using information from merge_df
    pos = {city: (lat, lon) for city, lat, lon in zip(merge_df['City'], merge_df['lat'], merge_df['lon'])}
    
   
    # Plot NetworkX graph nodes on the map
    for node in G.nodes():
        marker_color = 'red' if node == source_node else 'orange' if node == destination_node else 'green'  # Adjust colors as needed
        marker_size  = 15  if node == source_node or node == destination_node else 10
        x, y = pos[node][1], pos[node][0]  # lon, lat for the node
        fig.add_trace(
            go.Scattermapbox(
                mode="markers",
                lon=[x],
                lat=[y],
                marker={'size': marker_size, 'color' : marker_color },  # Adjust size and color as needed
                opacity=1,  # Set opacity to fully visible
                hoverinfo='text',  # Display text on hover
                hovertext=node  # Display node label on hover
            )
        )
    
    # Plot NetworkX graph on the map
    for edge in G.edges():
        x0, y0 = pos[edge[0]][1], pos[edge[0]][0]  # lon, lat for node 1
        x1, y1 = pos[edge[1]][1], pos[edge[1]][0]  # lon, lat for node 2
        fig.add_trace(
            go.Scattermapbox(
                mode="lines",
                lon=[x0, x1],
                lat=[y0, y1],
                marker={'size': 10, 'color': 'black'},
                line=dict(width=1, color='blue'),
                opacity=0.5,
                hoverinfo='none'
            )
        )

    st.plotly_chart(fig)

    # Add a button to go back to the Home page
    if st.button('Go back to Home'):
        st.experimental_rerun()  # Rerun the app to go back to the Home page



#Uninformed Search Algorithms
###############################################################################################BFS           
def bfs_find_path(graph, source_node, destination_nodes):
    # Initialize a set to keep track of visited nodes (CLOSED_LIST)
    visited = set()

    # Initialize a queue to perform BFS traversal (OPEN_LIST)
    queue = [[source_node]]

    # Track the number of nodes visited during BFS
    nodes_visited = 0
    
    # Initialize a list to store the nodes traversed during BFS
    traversed_nodes = []

    while queue:
        # Pop the first path from the queue
        path = queue.pop(0)
        node = path[-1]

        # Increment the visited node count
        nodes_visited += 1
        
        # Add the current node to the list of traversed nodes
        traversed_nodes.append(node)

        if node in destination_nodes:
            # If the current node is one of the destination nodes, return the path
            traversed_nodes.append(node)
            final_travesed_nodes = remove_duplicates(traversed_nodes)
            covered_distance = calculate_covered_distance(graph, path)

            return path, final_travesed_nodes, len(final_travesed_nodes) , covered_distance , len(path)

        if node not in visited:
            visited.add(node)
            # Get neighbors of the current node
            neighbors = graph.neighbors(node)
            for neighbor in neighbors:
                # Create a new path by appending the neighbor to the current path
                new_path = list(path)
                new_path.append(neighbor)
                # Add the new path to the queue for traversal
                queue.append(new_path)

    # If no path is found to any of the destination nodes, return None
    return None, traversed_nodes, nodes_visited


def page_bfs(G , source_node , merge_df ):

    st.title('BFS Algorithm')
    
     # Default values for destination nodes
    default_destination1 = 'CM1889-EXPO_GRAND_KOTAHENA_A_IND@2'

    
    # Text inputs for destination nodes with default values
    destination_node1 = st.text_input("Enter destination node 1", value=default_destination1)

    
    # Convert input values to a list of destination nodes
    destination_nodes = [destination_node1.strip()]
    




    if st.button('Find Path'):
        # Call the bfs_find_path function with user inputs
        path, traversal, nodes_visited , covered_distance , hop_count = bfs_find_path(G, source_node, destination_nodes)
        
        num_of_edges = count_edges_in_subgraph(G, traversal)
        merge_df = merge_df[merge_df['City'].isin(path)]
        merge_df = merge_df[['City' , 'cur_vol' , 'po_cap' , 'avg_tp']]
        merge_df['remain_cap'] = merge_df['po_cap'] - merge_df['cur_vol']
        merge_df.rename(columns ={"po_cap":"max_cap"} , inplace=True)

        # Display the results
        if path is not None:
            st.success(f"Path found: {path}")
            st.success(f"Traversal: {traversal}")
            st.success(f"Number of nodes visited: {nodes_visited}")
            st.success(f"Number of edges visited: {num_of_edges}")
            st.success(f"Hop Count: {hop_count-2}")
            st.success(f"Completeness: Yes")
            st.success(f"Optimal: Yes")
            st.success(f"Time Complexity: O(b^d)")
            st.success(f"Space Complexity: O(b^d+1)")

            # Extract latitude and longitude coordinates from the graph nodes
            node_positions = nx.get_node_attributes(G, 'pos')
            
            # Extract coordinates of nodes in the path
            path_positions = [node_positions[node] for node in path]

            # Unzip the coordinates into separate latitude and longitude lists
            path_latitudes, path_longitudes = zip(*path_positions)

            # Create a scatter plot for the nodes in the path
            fig = go.Figure(go.Scattermapbox(
                mode="markers+lines",
                lon=path_longitudes,
                lat=path_latitudes,
                marker={'size': 10, 'color': 'blue'},
                line=dict(width=2, color='blue'),
                hoverinfo='text',
                hovertext=[f'Node: {node}' for node in path]
            ))

            # Add map layout
            fig.update_layout(
                mapbox_style="open-street-map",
                mapbox_center={"lat": sum(path_latitudes)/len(path_latitudes), "lon": sum(path_longitudes)/len(path_longitudes)},
                mapbox_zoom=10,
                margin={"r": 0, "t": 0, "l": 0, "b": 0},
                height=800
            )

            # Show the plot
            st.plotly_chart(fig)
            st.write(merge_df)
        else:
                st.error("No path found to the destination nodes.")


############################################################################################DFS
def dfs_find_path_indi(graph, source_node, destination_nodes):
    # Initialize a set to keep track of visited nodes
    visited = set()

    # Initialize a stack to perform DFS traversal
    stack = [[source_node]]

    # Track the number of nodes visited during DFS
    nodes_visited = 0
    
    # Initialize a list to store the nodes traversed during DFS
    traversed_nodes = []

    while stack:
        # Pop the last path from the stack
        path = stack.pop()
        node = path[-1]

        # Increment the visited node count
        nodes_visited += 1
        
        # Add the current node to the list of traversed nodes
        traversed_nodes.append(node)

        if node in destination_nodes:
            # If the current node is one of the destination nodes, return the path
            traversed_nodes.append(node)
            final_travesed_nodes = remove_duplicates(traversed_nodes)
            return   path, final_travesed_nodes, len(final_travesed_nodes)  , len(path)

        if node not in visited:
            visited.add(node)
            # Get neighbors of the current node
            neighbors = graph.neighbors(node)
            for neighbor in neighbors:
                # Create a new path by appending the neighbor to the current path
                new_path = list(path)
                new_path.append(neighbor)
                # Add the new path to the stack for traversal
                stack.append(new_path)

    # If no path is found to any of the destination nodes, return None
    return None, traversed_nodes, nodes_visited



def page_dfs_indi(G , source_node , merge_df ):

    st.title('DFS Algorithm')
    
     # Default values for destination nodes
    default_destination1 = 'CM1889-EXPO_GRAND_KOTAHENA_A_IND@2'

    
    # Text inputs for destination nodes with default values
    destination_node1 = st.text_input("Enter destination node 1", value=default_destination1)

    
    # Convert input values to a list of destination nodes
    destination_nodes = [destination_node1.strip()]
    




    if st.button('Find Path'):
        # Call the bfs_find_path function with user inputs
        path, traversal, nodes_visited , hop_count = dfs_find_path_indi(G, source_node, destination_nodes)
        

        num_of_edges = count_edges_in_subgraph(G, traversal)
        merge_df = merge_df[merge_df['City'].isin(path)]
        merge_df = merge_df[['City' , 'cur_vol' , 'po_cap' , 'avg_tp']]
        merge_df['remain_cap'] = merge_df['po_cap'] - merge_df['cur_vol']
        merge_df.rename(columns ={"po_cap":"max_cap"} , inplace=True)
        # Display the results
        if path is not None:
            st.success(f"Path found: {path}")
            st.success(f"Traversal: {traversal}")
            st.success(f"Number of nodes visited: {nodes_visited}")
            st.success(f"Number of edges visited: {num_of_edges}")
            st.success(f"Hop Count: {hop_count-2}")
            st.success(f"Completeness: No")
            st.success(f"Optimal: No")
            st.success(f"Time Complexity: O(b^d)")
            st.success(f"Space Complexity: O(bd)")

                        # Extract latitude and longitude coordinates from the graph nodes
            node_positions = nx.get_node_attributes(G, 'pos')
            
            # Extract coordinates of nodes in the path
            path_positions = [node_positions[node] for node in path]

            # Unzip the coordinates into separate latitude and longitude lists
            path_latitudes, path_longitudes = zip(*path_positions)

            # Create a scatter plot for the nodes in the path
            fig = go.Figure(go.Scattermapbox(
                mode="markers+lines",
                lon=path_longitudes,
                lat=path_latitudes,
                marker={'size': 10, 'color': 'blue'},
                line=dict(width=2, color='blue'),
                hoverinfo='text',
                hovertext=[f'Node: {node}' for node in path]
            ))

            # Add map layout
            fig.update_layout(
                mapbox_style="open-street-map",
                mapbox_center={"lat": sum(path_latitudes)/len(path_latitudes), "lon": sum(path_longitudes)/len(path_longitudes)},
                mapbox_zoom=10,
                margin={"r": 0, "t": 0, "l": 0, "b": 0},
                height=800
            )

            # Show the plot
            st.plotly_chart(fig)
            st.write(merge_df)

        else:
            st.error("No path found to the destination nodes.")



########################################################################################################DFS_LIMIT
def dfs_find_path(graph, source_node, destination_nodes, depth_limit):
    # Initialize a set to keep track of visited nodes
    visited = set()

    # Initialize a stack to perform DFS traversal
    stack = [[source_node]]

    # Track the number of nodes visited during DFS
    nodes_visited = 0

    # Initialize a list to store the nodes traversed during DFS
    traversed_nodes = []

    while stack:
        # Pop the last path from the stack
        path = stack.pop()
        node = path[-1]

        # Increment the visited node count
        nodes_visited += 1

        # Add the current node to the list of traversed nodes
        traversed_nodes.append(node)

        if node in destination_nodes:
            # If the current node is one of the destination nodes, return the path
            traversed_nodes.append(node)
            final_travesed_nodes = remove_duplicates(traversed_nodes)
            return     path, final_travesed_nodes, len(final_travesed_nodes)  , len(path)

        if len(path) <= depth_limit:
            if node not in visited:
                visited.add(node)
                # Get neighbors of the current node
                neighbors = graph.neighbors(node)
                for neighbor in neighbors:
                    # Create a new path by appending the neighbor to the current path
                    new_path = list(path)
                    new_path.append(neighbor)
                    # Add the new path to the stack for traversal
                    stack.append(new_path)

    # If no path is found to any of the destination nodes, return None
    return None, traversed_nodes, nodes_visited , None



def page_dfs_limit(G , source_node , merge_df ):

    st.title('DFS Algorithm with DEPTH LIMIT')
    
     # Default values for destination nodes
    default_destination1 = 'CM1889-EXPO_GRAND_KOTAHENA_A_IND@2'

    
    # Text inputs for destination nodes with default values
    destination_node1 = st.text_input("Enter destination node 1", value=default_destination1)

    
    # Convert input values to a list of destination nodes
    destination_nodes = [destination_node1.strip()]
    
    depth_limit = st.number_input("Set the limit", value=5)



    if st.button('Find Path'):
        # Call the bfs_find_path function with user inputs
        path, traversal, nodes_visited , hop_count = dfs_find_path(G, source_node, destination_nodes , depth_limit)
        
        num_of_edges = count_edges_in_subgraph(G, traversal)

        merge_df = merge_df[merge_df['City'].isin(path)]
        merge_df = merge_df[['City' , 'cur_vol' , 'po_cap' , 'avg_tp']]
        merge_df['remain_cap'] = merge_df['po_cap'] - merge_df['cur_vol']
        merge_df.rename(columns ={"po_cap":"max_cap"} , inplace=True)
        # Display the results
        if path is not None:
            st.success(f"Path found: {path}")
            st.success(f"Traversal: {traversal}")
            st.success(f"Number of nodes visited: {nodes_visited}")
            st.success(f"Number of edges visited: {num_of_edges}")
            st.success(f"Hop Count: {hop_count-2}")
            st.success(f"Completeness: No")
            st.success(f"Optimal: No")
            st.success(f"Time Complexity: O(b^l)")
            st.success(f"Space Complexity: O(bl)")
             # Extract latitude and longitude coordinates from the graph nodes
            node_positions = nx.get_node_attributes(G, 'pos')
            
            # Extract coordinates of nodes in the path
            path_positions = [node_positions[node] for node in path]

            # Unzip the coordinates into separate latitude and longitude lists
            path_latitudes, path_longitudes = zip(*path_positions)

            # Create a scatter plot for the nodes in the path
            fig = go.Figure(go.Scattermapbox(
                mode="markers+lines",
                lon=path_longitudes,
                lat=path_latitudes,
                marker={'size': 10, 'color': 'blue'},
                line=dict(width=2, color='blue'),
                hoverinfo='text',
                hovertext=[f'Node: {node}' for node in path]
            ))

            # Add map layout
            fig.update_layout(
                mapbox_style="open-street-map",
                mapbox_center={"lat": sum(path_latitudes)/len(path_latitudes), "lon": sum(path_longitudes)/len(path_longitudes)},
                mapbox_zoom=10,
                margin={"r": 0, "t": 0, "l": 0, "b": 0},
                height=800
            )

            # Show the plot
            st.plotly_chart(fig)

            st.write(merge_df)
        else:
            st.error("No path found to the destination nodes.")



###################################################################################################################IDS
            
def ids_find_path(graph, source_node, destination_nodes):
    depth = 0
    traversed_nodes_all = []
    nodes_visited = 0
    
    while True:
        path, traversed_nodes, visited_count , hop_count = dfs_find_path(graph, source_node, destination_nodes, depth)
        traversed_nodes_all.extend(traversed_nodes)
        nodes_visited += visited_count
        
        if path:
            return path, traversed_nodes_all, nodes_visited , None
        depth += 1

def page_ids(G , source_node , merge_df ):

    st.title('IDS Algorithm')
    
     # Default values for destination nodes
    default_destination1 = 'CM1889-EXPO_GRAND_KOTAHENA_A_IND@2'
    default_destination2 = 'CM0592-LAKSAPATHIYA@1'
    default_destination3 = 'CM1464-IVY_PARK@3'
    
    # Text inputs for destination nodes with default values
    destination_node1 = st.text_input("Enter destination node 1", value=default_destination1)

    
    # Convert input values to a list of destination nodes
    destination_nodes = [destination_node1.strip()]
    




    if st.button('Find Path'):
        # Call the bfs_find_path function with user inputs
        path, traversal, nodes_visited , hop_count = ids_find_path(G, source_node, destination_nodes )
        num_of_edges = count_edges_in_subgraph(G, traversal)

        merge_df = merge_df[merge_df['City'].isin(path)]
        merge_df = merge_df[['City' , 'cur_vol' , 'po_cap' , 'avg_tp']]
        merge_df['remain_cap'] = merge_df['po_cap'] - merge_df['cur_vol']
        merge_df.rename(columns ={"po_cap":"max_cap"} , inplace=True)

        # Display the results
        if path is not None:
            st.success(f"Path found: {path}")
            st.success(f"Traversal: {traversal}")
            st.success(f"Number of nodes visited: {nodes_visited}")
            st.success(f"Number of edges visited: {num_of_edges}")
            st.success(f"Hop Count: {len(path)-2}")
            st.success(f"Completeness: Yes")
            st.success(f"Optimal: Yes")
            st.success(f"Time Complexity: O(b^m)")
            st.success(f"Space Complexity: O(bm)")
             # Extract latitude and longitude coordinates from the graph nodes
            node_positions = nx.get_node_attributes(G, 'pos')
            
            # Extract coordinates of nodes in the path
            path_positions = [node_positions[node] for node in path]

            # Unzip the coordinates into separate latitude and longitude lists
            path_latitudes, path_longitudes = zip(*path_positions)

            # Create a scatter plot for the nodes in the path
            fig = go.Figure(go.Scattermapbox(
                mode="markers+lines",
                lon=path_longitudes,
                lat=path_latitudes,
                marker={'size': 10, 'color': 'blue'},
                line=dict(width=2, color='blue'),
                hoverinfo='text',
                hovertext=[f'Node: {node}' for node in path]
            ))

            # Add map layout
            fig.update_layout(
                mapbox_style="open-street-map",
                mapbox_center={"lat": sum(path_latitudes)/len(path_latitudes), "lon": sum(path_longitudes)/len(path_longitudes)},
                mapbox_zoom=10,
                margin={"r": 0, "t": 0, "l": 0, "b": 0},
                height=800
            )

            # Show the plot
            st.plotly_chart(fig)
            st.write(merge_df)
        else:
            st.error("No path found to the destination nodes.")



##########################################################################UCS
import heapq

def ucs_find_path(graph, source_node, destination_nodes):
    # Initialize a set to keep track of visited nodes
    visited = set()

    # Initialize a priority queue for the frontier, with elements (cost, node, path)
    frontier = [(0, source_node, [])]

    # Track the number of nodes visited during UCS
    nodes_visited = 0

    # Initialize a list to store the nodes traversed during UCS
    traversed_nodes = []

    while frontier:
        # Pop the node with the lowest cost from the priority queue
        cost, node, path = heapq.heappop(frontier)

        # Increment the number of nodes visited
        nodes_visited += 1

        

        # If the current node is one of the destination nodes, return the path
        if node in destination_nodes:
            path = [source_node] + path 
            traversed_nodes.append(node)
            traversed_nodes = [source_node] + traversed_nodes
            final_travesed_nodes = remove_duplicates(traversed_nodes)
            return path, final_travesed_nodes, len(final_travesed_nodes)
        # Mark the current node as visited
        visited.add(node)

        # Explore neighbors of the current node
        for neighbor, neighbor_attr in graph[node].items():
            neighbor_cost = neighbor_attr['weight']  # Accessing the weight attribute
            if neighbor not in visited:
                # Calculate the total cost to reach the neighbor
                total_cost = cost + neighbor_cost
                # Push the neighbor into the priority queue with its cost and updated path
                heapq.heappush(frontier, (total_cost, neighbor, path + [neighbor]))
                traversed_nodes.append(node)
    # If no path is found to any of the destination nodes, return None
                
    return None, traversed_nodes, nodes_visited

def page_ucs(G , source_node , merge_df ):

    st.title('UCS Algorithm')
    
     # Default values for destination nodes
    default_destination1 = 'CM1889-EXPO_GRAND_KOTAHENA_A_IND@2'

    
    # Text inputs for destination nodes with default values
    destination_node1 = st.text_input("Enter destination node 1", value=default_destination1)

    
    # Convert input values to a list of destination nodes
    destination_nodes = [destination_node1.strip()]
    




    if st.button('Find Path'):
        # Call the bfs_find_path function with user inputs
        path, traversal, nodes_visited = ucs_find_path(G, source_node, destination_nodes )
        num_of_edges = count_edges_in_subgraph(G, traversal)

        merge_df = merge_df[merge_df['City'].isin(path)]
        merge_df = merge_df[['City' , 'cur_vol' , 'po_cap' , 'avg_tp']]
        merge_df['remain_cap'] = merge_df['po_cap'] - merge_df['cur_vol']
        merge_df.rename(columns ={"po_cap":"max_cap"} , inplace=True)

        # Display the results
        if path is not None:
            st.success(f"Path found: {path}")
            st.success(f"Traversal: {traversal}")
            st.success(f"Number of nodes visited: {nodes_visited}")
            st.success(f"Number of edges visited: {num_of_edges}")
            st.success(f"HopCount: {len(path)-2}")
            st.success(f"Completeness: Yes")
            st.success(f"Optimal: Yes")
            st.success(f"Time Complexity: O(b^d)")
            st.success(f"Space Complexity: O(b^d+1)")

             # Extract latitude and longitude coordinates from the graph nodes
            node_positions = nx.get_node_attributes(G, 'pos')
            
            # Extract coordinates of nodes in the path
            path_positions = [node_positions[node] for node in path]

            # Unzip the coordinates into separate latitude and longitude lists
            path_latitudes, path_longitudes = zip(*path_positions)

            # Create a scatter plot for the nodes in the path
            fig = go.Figure(go.Scattermapbox(
                mode="markers+lines",
                lon=path_longitudes,
                lat=path_latitudes,
                marker={'size': 10, 'color': 'blue'},
                line=dict(width=2, color='blue'),
                hoverinfo='text',
                hovertext=[f'Node: {node}' for node in path]
            ))

            # Add map layout
            fig.update_layout(
                mapbox_style="open-street-map",
                mapbox_center={"lat": sum(path_latitudes)/len(path_latitudes), "lon": sum(path_longitudes)/len(path_longitudes)},
                mapbox_zoom=10,
                margin={"r": 0, "t": 0, "l": 0, "b": 0},
                height=800
            )

            # Show the plot
            st.plotly_chart(fig)

            st.write(merge_df)
        else:
            st.error("No path found to the destination nodes.")



#####################################################################################################################################
            ##A* Search ##


import heapq

def heuristic_source_to_node(source_node, graph):
    # Get attributes of source node
    source_attrs = graph.nodes[source_node]
    
    # Initialize a dictionary to store heuristic values for each node
    heuristic_values = {}
    
    # Iterate through all nodes in the graph
    for node in graph.nodes:
        # Get attributes of the current node
        node_attrs = graph.nodes[node]
        
        # Compute heuristic value based on the provided conditions
        max_cap_diff_node = node_attrs["max_cap"] - node_attrs["cur_vol"]
        if max_cap_diff_node > source_attrs["cur_vol"]:
            cost_cap = 10 * max_cap_diff_node
        elif max_cap_diff_node > 0 and  max_cap_diff_node < source_attrs["cur_vol"] :
            cost_cap = 5 * max_cap_diff_node
        else:
            cost_cap = 0
        
        cost_tp = 5 * 3600 / 8000 * node_attrs["tp"] if 5 - node_attrs["tp"] <= 5 - source_attrs["tp"] else 0
        
        
        
        # Combine individual costs
        total_cost = cost_cap + cost_tp 
        
        # Store the total cost as the heuristic value for the current node
        heuristic_values[node] = total_cost
    
    return heuristic_values




def heuristic_node_to_destination(destination_node, graph):
    # Get attributes of source node
    destination_attrs = graph.nodes[destination_node]
    
    # Initialize a dictionary to store heuristic values for each node
    heuristic_values = {}
    
    # Iterate through all nodes in the graph
    for node in graph.nodes:
        # Get attributes of the current node
        node_attrs = graph.nodes[node]
        
        # Compute heuristic value based on the provided conditions
        
        max_cap_diff_destination_node = destination_attrs["max_cap"] - destination_attrs["cur_vol"]
        if max_cap_diff_destination_node > node_attrs["cur_vol"] :
                cost_cap = 10 * max_cap_diff_destination_node
        elif  max_cap_diff_destination_node > 0 and   max_cap_diff_destination_node <  node_attrs["cur_vol"] :
                cost_cap = 5 * max_cap_diff_destination_node
        else:
                cost_cap = 0
                    
        cost_tp = 5 * 3600 / 8000 * node_attrs["tp"] if 5 - node_attrs["tp"] <= 5 - destination_attrs["tp"] else 0
                    
        
     
        
        # Combine individual costs
        total_cost = cost_cap + cost_tp 
        
        # Store the total cost as the heuristic value for the current node
        heuristic_values[node] = total_cost
    
    return heuristic_values




import heapq

def a_star_search(graph, source, destination, heuristic_source_to_node, heuristic_node_to_destination):
    class PriorityQueue:
        def __init__(self):
            self.elements = []
            self.history = {}  # Dictionary to store the history of paths to each node
            self.priority_order = {}  # Dictionary to store the priority order of each node

        def empty(self):
            return len(self.elements) == 0

        def put(self, item, priority, path, selected_node):
            heapq.heappush(self.elements, (-priority, item))
            self.history[item] = path  # Store the path to the node
            self.priority_order[item] = priority  # Store the priority order of the node
            self.selected_node = selected_node  # Store the selected node

        def get(self):
            return heapq.heappop(self.elements)[1], self.selected_node  # Return both node and selected node

    def reconstruct_path(source, destination, selected_node, came_from):
        current = destination
        path = [current]
        while current != source:
            current = came_from[current]
            path.append(current)
        path.reverse()
        if selected_node is not None and selected_node != destination:
            path.append(selected_node)  # Append the selected node to the path if it's not None and not the destination
        elif selected_node == destination:
            path.pop()  # Remove the selected node if it's the destination itself
        return path

    open_set = PriorityQueue()  # Priority queue for open nodes
    open_set.put(source, 0, [source], None)   # Add source node to open set with priority 0 and its path
    
    # Initialize f_score dictionary
    f_score = {node: float('inf') for node in graph.nodes}

    # Update f_score for source node
    f_score[source] = heuristic_source_to_node[source] + heuristic_node_to_destination[source]

    came_from = {}  # Keep track of predecessors
    
    traversal_steps = []  # List to store traversal steps
    
    while not open_set.empty():
        current, selected_node = open_set.get()  # Get node with lowest f-score from open set
        
        traversal_steps.append(current)  # Add current node to traversal steps
        
        if current == destination:
            # Destination reached, return path and traversal steps
            path = reconstruct_path(source, destination, selected_node, came_from)
            return path, traversal_steps
        
        for neighbor in graph.neighbors(current):
            # Use heuristic_source_to_node as tentative g-score
            tentative_g_score = heuristic_source_to_node[neighbor]
            
            # Calculate f-score for neighbor
            f_score_neighbor = tentative_g_score + heuristic_node_to_destination[neighbor]
            
            # Update f-score if tentative f-score is lower
            if f_score_neighbor < f_score[neighbor]:
                came_from[neighbor] = current  # Update came_from for neighbor
                f_score[neighbor] = f_score_neighbor
                # Store the path to the neighbor node
                new_path = list(open_set.history[current])
                new_path.append(neighbor)
                open_set.put(neighbor, f_score_neighbor, new_path, current)  # Update priority in open set with selected node
    
    return None, traversal_steps  # No path found, return traversal steps

# Example usage:
# path, traversal_steps = a_star_search(graph, source, destination, heuristic_source_to_node, heuristic_node_to_destination)




def page_ASearch(G , source_node , merge_df ):

    st.title('A* Search Algorithm')
    
     # Default values for destination nodes
    default_destination1 = 'CM1889-EXPO_GRAND_KOTAHENA_A_IND@2'

    
    # Text inputs for destination nodes with default values
    destination_node1 = st.text_input("Enter destination node 1", value=default_destination1)

    
    # Convert input values to a list of destination nodes
    #destination_nodes = [destination_node1.strip()]
    




    if st.button('Find Path'):
        # Call the bfs_find_path function with user inputs
        heuristic_source_to_node_values      = heuristic_source_to_node(source_node , G)
        heuristic_node_to_destination_values = heuristic_node_to_destination(destination_node1, G)
        path, traversal = a_star_search(G, source_node, destination_node1, heuristic_source_to_node_values, heuristic_node_to_destination_values)
        
        

        num_of_edges = count_edges_in_subgraph(G, traversal)

        merge_df = merge_df[merge_df['City'].isin(path[:-1])]
        merge_df = merge_df[['City' , 'cur_vol' , 'po_cap' , 'avg_tp']]
        merge_df['remain_cap'] = merge_df['po_cap'] - merge_df['cur_vol']
        merge_df.rename(columns ={"po_cap":"max_cap"} , inplace=True)

        # Display the results
        if path is not None:
            st.success(f"Path found: {path[:-1]}")
            st.success(f"Traversal: {traversal}")
            st.success(f"Number of nodes visited: {len(traversal)}")
            st.success(f"Number of edges visited: {num_of_edges}")
            st.success(f"Hop Count: {len(path)-3}")
            st.success(f"Completeness: Yes")
            st.success(f"Optimal: Yes")




             # Extract latitude and longitude coordinates from the graph nodes
            node_positions = nx.get_node_attributes(G, 'pos')
            
            # Extract coordinates of nodes in the path
            path_positions = [node_positions[node] for node in path[:-1] ]

            # Unzip the coordinates into separate latitude and longitude lists
            path_latitudes, path_longitudes = zip(*path_positions)

            # Create a scatter plot for the nodes in the path
            fig = go.Figure(go.Scattermapbox(
                mode="markers+lines",
                lon=path_longitudes,
                lat=path_latitudes,
                marker={'size': 10, 'color': 'blue'},
                line=dict(width=2, color='blue'),
                hoverinfo='text',
                hovertext=[f'Node: {node}' for node in path]
            ))

            # Add map layout
            fig.update_layout(
                mapbox_style="open-street-map",
                mapbox_center={"lat": sum(path_latitudes)/len(path_latitudes), "lon": sum(path_longitudes)/len(path_longitudes)},
                mapbox_zoom=10,
                margin={"r": 0, "t": 0, "l": 0, "b": 0},
                height=800
            )

            # Show the plot
            st.plotly_chart(fig)

            st.write(merge_df)
        else:
            st.error("No path found to the destination nodes.")






############################################################ Greedy Search
def geedy_search(graph, source, destination,  heuristic_node_to_destination):
    class PriorityQueue:
        def __init__(self):
            self.elements = []
            self.history = {}  # Dictionary to store the history of paths to each node
            self.priority_order = {}  # Dictionary to store the priority order of each node

        def empty(self):
            return len(self.elements) == 0

        def put(self, item, priority, path, selected_node):
            heapq.heappush(self.elements, (-priority, item))
            self.history[item] = path  # Store the path to the node
            self.priority_order[item] = priority  # Store the priority order of the node
            self.selected_node = selected_node  # Store the selected node

        def get(self):
            return heapq.heappop(self.elements)[1], self.selected_node  # Return both node and selected node

    def reconstruct_path(source, destination, selected_node, came_from):
        current = destination
        path = [current]
        while current != source:
            current = came_from[current]
            path.append(current)
        path.reverse()
        if selected_node is not None and selected_node != destination:
            path.append(selected_node)  # Append the selected node to the path if it's not None and not the destination
        elif selected_node == destination:
            path.pop()  # Remove the selected node if it's the destination itself
        return path

    open_set = PriorityQueue()  # Priority queue for open nodes
    open_set.put(source, 0, [source], None)   # Add source node to open set with priority 0 and its path
    
    # Initialize f_score dictionary
    f_score = {node: float('inf') for node in graph.nodes}

    # Update f_score for source node
    f_score[source] =  heuristic_node_to_destination[source]

    came_from = {}  # Keep track of predecessors
    
    traversal_steps = []  # List to store traversal steps
    
    while not open_set.empty():
        current, selected_node = open_set.get()  # Get node with lowest f-score from open set
        
        traversal_steps.append(current)  # Add current node to traversal steps
        
        if current == destination:
            # Destination reached, return path and traversal steps
            path = reconstruct_path(source, destination, selected_node, came_from)
            return path, traversal_steps
        
        for neighbor in graph.neighbors(current):
            # Use heuristic_source_to_node as tentative g-score
            
            
            # Calculate f-score for neighbor
            f_score_neighbor =  heuristic_node_to_destination[neighbor]
            
            # Update f-score if tentative f-score is lower
            if f_score_neighbor < f_score[neighbor]:
                came_from[neighbor] = current  # Update came_from for neighbor
                f_score[neighbor] = f_score_neighbor
                # Store the path to the neighbor node
                new_path = list(open_set.history[current])
                new_path.append(neighbor)
                open_set.put(neighbor, f_score_neighbor, new_path, current)  # Update priority in open set with selected node
    
    return None, traversal_steps  # No path found, return traversal steps



def page_greedysearch(G , source_node , merge_df ):

    st.title('Greedy Search Algorithm')
    
     # Default values for destination nodes
    default_destination1 = 'CM1889-EXPO_GRAND_KOTAHENA_A_IND@2'

    
    # Text inputs for destination nodes with default values
    destination_node1 = st.text_input("Enter destination node 1", value=default_destination1)

    
    # Convert input values to a list of destination nodes
    #destination_nodes = [destination_node1.strip()]
    




    if st.button('Find Path'):
        # Call the bfs_find_path function with user inputs
        
        heuristic_node_to_destination_values = heuristic_node_to_destination(destination_node1, G)
        path, traversal = geedy_search(G, source_node, destination_node1, heuristic_node_to_destination_values)
        
        num_of_edges = count_edges_in_subgraph(G, traversal)
        merge_df = merge_df[merge_df['City'].isin(path[:-1])]
        merge_df = merge_df[['City' , 'cur_vol' , 'po_cap' , 'avg_tp']]
        merge_df['remain_cap'] = merge_df['po_cap'] - merge_df['cur_vol']
        merge_df.rename(columns ={"po_cap":"max_cap"} , inplace=True)
        # Display the results
        if path is not None:
            st.success(f"Path found: {path[:-1]}")
            st.success(f"Traversal: {traversal}")
            st.success(f"Number of nodes visited: {len(traversal)}")
            st.success(f"Number of edges visited: {num_of_edges}")
            st.success(f"Hop Count: {len(path)-3}")
            st.success(f"Completeness: No")
            st.success(f"Optimal: No")
            
            

             # Extract latitude and longitude coordinates from the graph nodes
            node_positions = nx.get_node_attributes(G, 'pos')
            
            # Extract coordinates of nodes in the path
            path_positions = [node_positions[node] for node in path[:-1] ]

            # Unzip the coordinates into separate latitude and longitude lists
            path_latitudes, path_longitudes = zip(*path_positions)

            # Create a scatter plot for the nodes in the path
            fig = go.Figure(go.Scattermapbox(
                mode="markers+lines",
                lon=path_longitudes,
                lat=path_latitudes,
                marker={'size': 10, 'color': 'blue'},
                line=dict(width=2, color='blue'),
                hoverinfo='text',
                hovertext=[f'Node: {node}' for node in path]
            ))

            # Add map layout
            fig.update_layout(
                mapbox_style="open-street-map",
                mapbox_center={"lat": sum(path_latitudes)/len(path_latitudes), "lon": sum(path_longitudes)/len(path_longitudes)},
                mapbox_zoom=10,
                margin={"r": 0, "t": 0, "l": 0, "b": 0},
                height=800
            )

            # Show the plot
            st.plotly_chart(fig)

            st.write(merge_df)
        else:
            st.error("No path found to the destination nodes.")









####################### Hill Climbing 
def hill_climbing_find_path(graph, source_node, destination_node , heuristic_source_to_node ,heuristic_node_to_destination):
    # Initialize a set to keep track of visited nodes (CLOSED_LIST)
    visited = set()

    # Initialize the current node to the source node
    current_node = source_node

    # Track the number of nodes visited during the search
    nodes_visited = 0
    
    # Initialize a list to store the nodes traversed during the search
    traversed_nodes = []

    # Initialize the path to an empty list
    path = []

    while True:
        # Add the current node to the list of traversed nodes
        traversed_nodes.append(current_node)

        if current_node == destination_node:
            # If the current node is one of the destination nodes, return the path
            traversed_nodes.append(current_node)
            final_travesed_nodes = remove_duplicates(traversed_nodes)
            covered_distance = calculate_covered_distance(graph, path)

            return path, final_travesed_nodes , covered_distance 

        # Add the current node to the set of visited nodes
        visited.add(current_node)

        # Append the current node to the path
        path.append(current_node)

        # Get neighbors of the current node
        neighbors = graph.neighbors(current_node)

        # Initialize variables to track the best neighbor found so far
        best_neighbor = None
        best_heuristic_value = float('inf')

        # Iterate over the neighbors to find the best neighbor based on heuristic value
        for neighbor in neighbors:
            if neighbor not in visited:
                # Calculate the heuristic value for the neighbor using both source-to-node and node-to-destination heuristics
                heuristic_source_to_neighbor  = heuristic_source_to_node[neighbor]
                heuristic_neighbor_to_destination = heuristic_node_to_destination[neighbor]
                heuristic_value = heuristic_source_to_neighbor + heuristic_neighbor_to_destination 
                
                # Update the best neighbor if a better heuristic value is found
                if heuristic_value > best_heuristic_value:
                    best_neighbor = neighbor
                    best_heuristic_value = heuristic_value

        # If no unvisited neighbors are found, or if the best neighbor does not improve the heuristic value,
        # terminate the search and return the traversed nodes
        if best_neighbor is None or best_heuristic_value >= float('inf'):
            covered_distance = calculate_covered_distance(graph, traversed_nodes)
            return path, traversed_nodes, covered_distance

        # Update the current node to the best neighbor for the next iteration
        current_node = best_neighbor



def page_hillclimbing(G , source_node ):

    st.title('Hill Climbing Search')
    
     # Default values for destination nodes
    default_destination1 = 'CM1889-EXPO_GRAND_KOTAHENA_A_IND@2'

    
    # Text inputs for destination nodes with default values
    destination_node1 = st.text_input("Enter destination node 1", value=default_destination1)

    
    # Convert input values to a list of destination nodes
    #destination_nodes = [destination_node1.strip()]
    




    if st.button('Find Path'):
        # Call the bfs_find_path function with user inputs
        heuristic_source_to_node_values      = heuristic_source_to_node(source_node , G)
        heuristic_node_to_destination_values = heuristic_node_to_destination(destination_node1, G)
        
        path, traversal , covered_distance = hill_climbing_find_path(G, source_node, destination_node1 , heuristic_source_to_node_values ,heuristic_node_to_destination_values)
        
        num_of_edges = count_edges_in_subgraph(G, traversal)
        # Display the results
        if path is not None:
            st.success(f"Path found: {path[:-1]}")
            st.success(f"Traversal: {traversal}")
            st.success(f"Number of nodes visited: {len(traversal)}")
            st.success(f"Number of nodes visited: {num_of_edges}")
            st.success(f"Hop Count: {len(path)-2}")
            st.success(f"Completeness: No")
            st.success(f"Optimal: No")

             # Extract latitude and longitude coordinates from the graph nodes
            node_positions = nx.get_node_attributes(G, 'pos')
            
            # Extract coordinates of nodes in the path
            path_positions = [node_positions[node] for node in path[:-1] ]

            # Unzip the coordinates into separate latitude and longitude lists
            path_latitudes, path_longitudes = zip(*path_positions)

            # Create a scatter plot for the nodes in the path
            fig = go.Figure(go.Scattermapbox(
                mode="markers+lines",
                lon=path_longitudes,
                lat=path_latitudes,
                marker={'size': 10, 'color': 'blue'},
                line=dict(width=2, color='blue'),
                hoverinfo='text',
                hovertext=[f'Node: {node}' for node in path]
            ))

            # Add map layout
            fig.update_layout(
                mapbox_style="open-street-map",
                mapbox_center={"lat": sum(path_latitudes)/len(path_latitudes), "lon": sum(path_longitudes)/len(path_longitudes)},
                mapbox_zoom=10,
                margin={"r": 0, "t": 0, "l": 0, "b": 0},
                height=800
            )

            # Show the plot
            st.plotly_chart(fig)
        else:
            st.error("No path found to the destination nodes.")






def main():

    # Defining Source node and destination nodes
    source_node = 'CM0184-ATHURUGIRIYA_TOWN@1'
    destination_node = 'CM1889-EXPO_GRAND_KOTAHENA_A_IND@2' #'CM0592-LAKSAPATHIYA@1'
 
    destination_nodes = [destination_node]

    #Loading the Dataset
    merge_df = pd.read_csv('D:/MSC/Search/FINAL_BANK_SEARCH.csv')
    merge_df.rename(columns = {"longitude":"lon" , "latitude":"lat" , "sect_id":"City"} , inplace=True)


    # Creating Graph
    G = create_graph(merge_df)

    # Visualizing Graph

     



    st.sidebar.title('Navigation')
    page_options = ['Map' , 'BFS' , 'DFS' , 'DFS_DEPTH' , 'IDS' , 'UCS'  , 'GREEDY' ,  'A*' , 'HILL_CLIMBING'  ]
    page = st.sidebar.radio('Go to', page_options)

  
    if page == 'Graph Visualization':
        page_graph(G  , source_node , destination_node)
    elif page == 'Map':
         page_map(merge_df , G , source_node , destination_node)    
    elif page == 'BFS':
         page_bfs(G , source_node , merge_df)
    elif page == 'DFS':
         page_dfs_indi(G , source_node , merge_df)
    elif page == 'DFS_DEPTH':
          page_dfs_limit(G , source_node , merge_df )     
    elif page == 'IDS':
         page_ids(G , source_node  , merge_df)
    elif page == 'UCS':
         page_ucs(G , source_node , merge_df )
    elif page == 'A*':
        page_ASearch(G , source_node ,merge_df)     
    elif page == 'GREEDY':
        page_greedysearch(G , source_node , merge_df)    
    elif page =='HILL_CLIMBING':
         page_hillclimbing(G , source_node  )    
     

if __name__ == '__main__':
     main()
