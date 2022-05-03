import os
import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from dash.exceptions import PreventUpdate
from plotly.subplots import make_subplots
import plotly.io as pio
import pickle
import json
import numpy as np


class Solution:

    _id = 0

    def __init__(self, representation, objective_values):
        self._solution_id = Solution._id
        Solution._id += 1
        self.representation = representation
        self.objective_values = objective_values

app = dash.Dash(
    __name__,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
    ],
)
app.title = "Pareto fronts of Land Conservation Optimization"


def convert_to_monetary_scale(soil_loss,labor_req, study_area_size):
    # 1. Estimate yield loss in dollars per soil loss
    # crop
    # Yields for cereals(make 88.52% of total production)
    # Teff: 16.38 Qt/Hectare, percentage of agriculture area used for production:  16,55995524%, price is 750 USD/t (year 2018)
    # Maize: 34.76 Qt/Hectare, percentage of agriculture area used for production:  6,622324444%, price is 430 USD/t
    # Sorghum: 28.59 Qt/Hectare,  percentage of agriculture area used for production: 76,81772031, price 500 USD/t
    # 0.1655995524 * 1.638t * 750 USD/t = 203 USD/ha
    # 0.0622324444 * 3.476 * 2200 = 475 USD/ha
    # 0.768 * 2.859 * 500 =  1097 USD/ha
    # --> Avg/ha: 203+ 475 + 1097 = 591.66 USD/ha

    # Computing yield loss for 10 years:
    #  Tonnes of soil loss to mm soil depth erosion conversion:
    #  10t = 1mm (https://www.umweltbundesamt.de/en/topics/soil-agriculture/land-a-precious-resource/erosion#what-are-the-consequences-of-water-erosion)
    #  estimated yield loss per mm soil loss = 0.74% (https://www.gov.scot/publications/developing-method-estimate-costs-soil-erosion-high-risk-scottish-catchments/pages/10/)
    # (t/10) * 0.0074 * 561.55USD/ha * study area size * 10 years
    lost_money_from_yield_loss_in_USD = soil_loss * 0.0074 * 561.55 * 10 * study_area_size

    # 2. Compute labor cost in dollars
    # reference: https://europa.eu/capacity4dev/file/96444/download?token=9vRED4Xj
    # Base wage in 2015: 2.5 USD per day:
    # Wage increase per year: 9%
    # In 2022: 2.5 + (1.09^7) = 4.32
    labor_cost_in_USD = labor_req * 4.32 * study_area_size

    return lost_money_from_yield_loss_in_USD, labor_cost_in_USD



def create_background_map(watersheds):
    lons = []
    lats = []
    for feature in watersheds["features"]:
        for i in feature["geometry"]["coordinates"]:
            for j in i:
                feature_lats = np.array(j)[:, 1].tolist()
                feature_lons = np.array(j)[:, 0].tolist()
                lons = lons + feature_lons + [None]
                lats = lats + feature_lats + [None]
    lats = np.array(lats)
    lons = np.array(lons)
    background_map = go.Scattermapbox(
        fill="toself",
        lat=lats,
        lon=lons,
        fillcolor='rgba(27,158,119,0.3)',
        marker={'size': 0, 'color': "#f9ba00"},
        showlegend=False)
    return background_map


def blank_figure():
    fig = go.Figure(go.Scatter(x=[], y=[]))
    fig.update_layout(template=None)
    fig.update_xaxes(showgrid=False, showticklabels=False, zeroline=False)
    fig.update_yaxes(showgrid=False, showticklabels=False, zeroline=False)

    return fig

def create_layout():
    # Actual layout of the app
    return html.Div(
        id="root",
        children=[
            html.Div(
                id="header",
                children=[
                    html.H4(children="Allocation of the SWC-Measure Bench Terrace"),
                    html.P(
                        id="description",
                        children="The illustrated Pareto Front shows the optimal solutions generated with the optimization algorithm NSGA 2, \
                        The conflicting objectives are to minimize the labor requirements and to minimize soil erosion\
                        Uncertain spatial data inputs are soil properties and precipitation datasets \
                        You may select between a boxplot and scatter plot.",
                    ),
                ],
            ),
            html.Div(
                id="app-container",
                children=[
                    html.Div(
                        id="left-column",
                        children=[
                            dcc.Dropdown(
                                options=[
                                    {
                                        "label": "Gumobila",
                                        "value": "gumobila",
                                    },
                                    {
                                        "label": "Ennerata",
                                        "value": "enerata",
                                    },
                                    {
                                        "label": "Mender",
                                        "value": "mender",
                                    },
                                ],
                                value="gumobila",
                                id="study_area-dropdown",
                            ),
                            dcc.Dropdown(
                                        options=[
                                            {
                                                "label": "Boxplot",
                                                "value": "boxplot",
                                            },
                                            {
                                                "label": "Scatter plot",
                                                "value": "scatter",
                                            },
                                        ],
                                        value="boxplot",
                                        id="plotmode-dropdown",
                                    ),

                                html.Div(
                                id="pareto_front-container",
                                children=[
                                    html.P(
                                        "Pareto front",
                                        id="pareto_front-title",
                                    ),
                                    dcc.Graph(
                                        id="pareto_front",
                                    ),
                                ],
                            ),
                        ],
                    ),
                    html.Div(
                        id="map-container",
                        children=[
                            html.P(id="layer-selector", children="Plot bench terraces"),
                            dcc.Dropdown(
                                options=[
                                    {
                                        "label": "Yes",
                                        "value": "Yes",
                                    },
                                    {
                                        "label": "No",
                                        "value": "No",
                                    },
                                ],
                                value="No",
                                id="layer-dropdown",
                            ),
                            #

                            dcc.Graph(
                                id="selected_data",
                                figure= blank_figure()
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )


def interactiveParetoFront(app,input_data, save_front = None):
    
    def generate_figure_image(study_area, figure, points, layout,opacity):
        figure.add_trace(go.Scatter(
            x=points.iloc[:, 0],
            y=points.iloc[:, 1],

            # z=val["z"],
            showlegend=True,
            legendgroup="scatterpoints",
            textposition="top left",
            mode="markers",
            marker=dict(size=3, symbol="circle", opacity=opacity),

        ),
            secondary_y=False,
        )
        # add second trace in order to have second x and y axis
        x,y = convert_to_monetary_scale(points.iloc[:, 0],points.iloc[:, 1], study_area["study_area_size"])
        figure.add_trace(go.Scatter(
            x=x,
            y=y,

            # z=val["z"],
            showlegend=True,
            legendgroup="scatterpoints",
            textposition="top left",
            mode="markers",
            marker=dict(size=3, symbol="circle", opacity=0),

        ),
            secondary_y=True,
        )


        figure.update_layout(
            title={
                'y': 0.975,
                'x': 0.055,
                'xanchor': 'left',
                'yanchor': 'top'},
            legend={'itemsizing': 'constant'})
        # figure.update_layout(legend_title=labels[np.where(run_folders == run)][0])
        return figure
    @app.callback(
        Output("pareto_front", "figure"),
        [Input("plotmode-dropdown", "value"),Input("study_area-dropdown", "value")]
    )
    
    def display_3d_scatter_plot(illustrationtype, study_area):
        selected_study_area = input_data[study_area]
        
        
        def draw_boxplot(x_position, y_position, min_value, max_value,
                         horizontal=False):
            if min_value != max_value:
                if horizontal is False:
                    shape = {'type': "line",
                             'x0': x_position,
                             'y0': min_value,
                             'x1': x_position,
                             'y1': max_value,
                             'line': dict(color='#d62728', width=1)
                             }
                else:
                    shape = {'type': "line",
                             'x0': min_value,
                             'y0': y_position,
                             'x1': max_value,
                             'y1': y_position,
                             'line': dict(color="RoyalBlue", width=1)
                             }
                return shape
            else:
                return None

        axes = dict(title="", showgrid=True, zeroline=False, showticklabels=False)

        layout = go.Layout(
            # title={'text': "Pairwise Pareto front of objectives {} and {}".format("Yearly soil loss",
            #                                                                       "Required labour requirements"),
            #        'x': 0.2, },
            margin=dict(r=0, l=0, t=0, b=0),
            scene=dict(xaxis=axes, yaxis=axes),
            width=1500, height=700,
            autosize=True,
            xaxis_title="Yearly soil loss in t/ha",
            yaxis_title="Labour requirements in labour days/ha",
            font=dict(
                family="Arial",
                size=14
            )

        )
        figure = go.Figure(layout=layout)
        figure = make_subplots(specs=[[{"secondary_y": True}]],figure=figure)
        figure.update_layout(xaxis2={'anchor': 'y', 'overlaying': 'x', 'side': 'top', 'title': 'Estimated total yield loss from soil loss in US Dollars (10 years)'},
                          yaxis_domain=[0, 0.94])
        i = 0
        obj1_values = []
        obj2_values = []
        solution_ids = []
        
        for solution in selected_study_area["final_pareto_front"]:
            for realization_id in range(len(solution.objective_values[0])):
                obj1_values.append(solution.objective_values[0][realization_id])
                obj2_values.append(solution.objective_values[1][realization_id])
                solution_ids.append(i)
            i += 1
        scattered_points = pd.DataFrame(
            {'obj1': np.array(obj1_values), 'obj2': np.array(obj2_values), 'sol_id': np.array(solution_ids)})
        plot_mode = illustrationtype

        if plot_mode == 'scatter':
            figure = generate_figure_image(selected_study_area,figure, scattered_points, layout, opacity=1)

        elif plot_mode == 'boxplot':
            figure = generate_figure_image(selected_study_area,figure, scattered_points, layout, opacity=0)
            shapes = []
            for solution in selected_study_area["final_pareto_front"]:
                obj1_values = []
                obj2_values = []
                for realization_id in range(len(solution.objective_values[0])):
                    obj1_values.append(solution.objective_values[0][realization_id])
                    obj2_values.append(solution.objective_values[1][realization_id])
                obj1_median = np.median(obj1_values)
                obj2_median = np.median(obj2_values)

                obj1_min = np.min(obj1_values)
                obj1_max = np.max(obj1_values)

                obj2_min = np.min(obj2_values)
                obj2_max = np.max(obj2_values)

                line1 = draw_boxplot(x_position=obj1_median, y_position=obj2_median,
                                     min_value=obj1_min, max_value=obj1_max,
                                     horizontal=True)
                if line1 is not None and line1 != [None, None]:
                    shapes.append(line1)

                line2 = draw_boxplot(x_position=obj1_median, y_position=obj2_median,
                                     min_value=obj2_min, max_value=obj2_max,
                                     horizontal=False)
                if line2 is not None and line2 != [None, None]:
                    shapes.append(line2)
                i += 1
            figure.update_layout(shapes=shapes)

        elif plot_mode == 'mixed':
            figure = generate_figure_image(figure, scattered_points, layout, opacity=0.25)
            shapes = []
            for solution in selected_study_area["final_pareto_front"]:
                obj1_values = []
                obj2_values = []
                for realization_id in range(len(solution.objective_values[0])):
                    obj1_values.append(solution.objective_values[0][realization_id])
                    obj2_values.append(solution.objective_values[1][realization_id])
                obj1_median = np.median(obj1_values)
                obj2_median = np.median(obj2_values)

                obj1_min = np.min(obj1_values)
                obj1_max = np.max(obj1_values)

                obj2_min = np.min(obj2_values)
                obj2_max = np.max(obj2_values)

                line1 = draw_boxplot(x_position=obj1_median, y_position=obj2_median,
                                     min_value=obj1_min, max_value=obj1_max,
                                     horizontal=True)
                if line1 is not None and line1 != [None, None]:
                    shapes.append(line1)

                line2 = draw_boxplot(x_position=obj1_median, y_position=obj2_median,
                                     min_value=obj2_min, max_value=obj2_max,
                                     horizontal=False)
                if line2 is not None and line2 != [None, None]:
                    shapes.append(line2)
                i += 1
            figure.update_layout(shapes=shapes)

        names = set()
        figure.for_each_trace(
            lambda trace:
            trace.update(showlegend=False)
            if (trace.name in names) else names.add(trace.name))

        figure.update_yaxes(title_text="Estimated total labor costs in US dollars", secondary_y=True)
        figure.data[1].update(xaxis='x2')
        #define which trace should be on top. The first needs to be on top as its click event initiate the map figure callback
        figure.data = (figure.data[1], figure.data[0])

        #save figure
        if save_front is not None:
            ppi = 96
            width_cm = 3
            height_cm = 2
            pio.write_image(figure, os.path.join(save_front, "pf.svg"),
                            width=(width_cm * 2.54) * ppi,
                            height=(height_cm * 2.54) * ppi)
        return figure

    @app.callback(
        Output("selected_data", "figure"),
        [
            Input("pareto_front", "clickData"),
            Input("layer-dropdown", "value"),
            Input("study_area-dropdown", "value"),
        ],
    )
    def display_click_image(clickData, show_contours, selected_study_area):
        if clickData:
            #print(clickData)
            clicked_solution = None
            click_point_np = [float(clickData["points"][0][i]) for i in ["x", "y"]]

            filtered_solutions = []
            for solution in input_data[selected_study_area]["final_pareto_front"]:
                filtered_solutions.append(solution)
                if click_point_np[0] in solution.objective_values[0] and \
                        click_point_np[1] in solution.objective_values[1]:
                    clicked_solution = solution
                    patchmap_of_picked_solution = None
            try:
                if clickData and clicked_solution is not None:
                    layout1 = go.Layout(
                        title=f'Corresponding land use Map',
                        yaxis=dict(showticklabels=False),
                        xaxis=dict(showticklabels=False)
                    )

                    ws_lons = []
                    ws_lats = []
                    cl_lons = []
                    cl_lats = []
                    indizes_protected_areas = tuple(np.where(clicked_solution.representation == True)[0]+1)
                    #print(indizes_protected_areas)
                    watersheds = input_data[selected_study_area]["watersheds"]
                    for feature in watersheds["features"]:
                        if feature["properties"]["pos_rank"] in indizes_protected_areas:
                            for i in feature["geometry"]["coordinates"]:
                                for j in i:
                                    ws_feature_lats = np.array(j)[:, 1].tolist()
                                    ws_feature_lons = np.array(j)[:, 0].tolist()
                                    ws_lons = ws_lons + ws_feature_lons + [None]
                                    ws_lats = ws_lats + ws_feature_lats + [None]
                    ws_lats = np.array(ws_lats)
                    ws_lons = np.array(ws_lons)


                    trace1 = go.Scattermapbox(
                        lat=ws_lats,
                        lon=ws_lons,
                        mode="lines",
                        fill="toself",
                        #opacity = 0.7,
                        fillcolor='rgba(31,120,180,0.6)',
                        line=dict(width=1, color = 'rgb(31,120,180)'),
                        showlegend=False

                    )

                    if show_contours == "Yes":
                        terraces = input_data[selected_study_area]["terraces"]
                        flist = []
                        for feature in terraces["features"]:
                            if feature["properties"]["pos_rank"] in indizes_protected_areas:
                                for f in feature["geometry"]["coordinates"]:
                                    flist.append(np.array(f))
                        for f in flist:
                            cl_feature_lats = f[:, 1].tolist()
                            cl_feature_lons = f[:, 0].tolist()
                            cl_lons = cl_lons + cl_feature_lons + [None]
                            cl_lats = cl_lats + cl_feature_lats + [None]
                        cl_lats = np.array(cl_lats)
                        cl_lons = np.array(cl_lons)
                        trace2 =  go.Scattermapbox(
                                    lat=cl_lats,
                                    lon=cl_lons,
                                    mode="lines",
                                    line=dict(width=0.01, color='rgba(5,5,5, 1.0)'),
                                    showlegend=False
                                )

                    fig = go.Figure()
                    #get extent of features to define center
                    lons = []
                    lats = []
                    for feature in watersheds["features"]:
                        feature_lats = np.array(feature["geometry"]["coordinates"][0][0])[:, 1].tolist()
                        feature_lons = np.array(feature["geometry"]["coordinates"][0][0])[:, 0].tolist()
                        lons = lons + feature_lons
                        lats = lats + feature_lats
                    mean_lat = np.mean(np.array(lats))
                    mean_lon = np.mean(np.array(lons))

                    fig.update_layout(
                        margin=dict(t=5, r=5, b=5, l=5),
                        autosize=True,
                        mapbox=go.layout.Mapbox(
                            style="stamen-terrain",
                            zoom=12,
                            center_lat=mean_lat,
                            center_lon=mean_lon,
                        )
                    )
                    #add the background app
                    fig.add_trace(input_data[selected_study_area]["background_map"])
                    fig.add_trace(trace1)
                    #plot the bench terrace contour lines
                    if show_contours == "Yes":
                        fig.add_trace(trace2)
                    fig.update_layout(height=700, width=800)
                    return fig
            #if element is not found directly (mouse slightly off) prevent error message
            except  KeyError as error:
                raise PreventUpdate
        return {}

# inititalize the app
app = dash.Dash(
    __name__,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
    ],
)
#create the layout
app.layout = create_layout()
#define location of output directory

input_data = {}
input_data["gumobila"]={"outputdir": "gumobila",
                       "filename": r"all_gens_gumobila_200_gens_100_popsize_fixed_index.pkl",
                        "study_area_size": 3156.4394804916697}

input_data["enerata"]={"outputdir": "enerata",
                       "filename": r"all_gens_enerata_200_gens_100_popsize_final.pkl",
                        "study_area_size": 4544.096672822011}

input_data["mender"]={"outputdir": r"mender",
                       "filename": r"all_gens_mender_200_gens_100_popsize_final.pkl",
                        "study_area_size": 674.9871122104956}

for study_area in ["gumobila", "enerata", "mender"]:
    with open(os.path.join(input_data[study_area]["outputdir"], input_data[study_area]["filename"]), 'rb') as handle:
            populations = pickle.load(handle)
    
    input_data[study_area]["all_populations"] = populations
    
    final_population =  populations[-1]
    final_population_objective_values = [F for F in final_population[0]]
    
    #correction to soil loss per ha, total size of watershed polygons is 6872.8406 ha
    final_population_objective_values = [[F[0],F[1]] for F in final_population_objective_values]
    final_population_genes = [X for X in final_population[1]]
    #final_population_metadata = [elem.data for elem in final_population_df[0]]
    optimal_solutions = []
    for i in range(len(final_population_objective_values)):
        optimal_solutions.append(Solution(final_population_genes[i],final_population_objective_values[i]))
    
    input_data[study_area]["final_pareto_front"] = optimal_solutions
    
    with open(os.path.join(input_data[study_area]["outputdir"],'watersheds4326.geojson')) as watersheds_json_file:
        watersheds = json.load(watersheds_json_file)
    
    with open(os.path.join(input_data[study_area]["outputdir"],'contourlines4326.geojson')) as terraces_json_file:
        terraces = json.load(terraces_json_file)
        
    input_data[study_area]["watersheds"] = watersheds
    input_data[study_area]["terraces"] = terraces
    input_data[study_area]["background_map"] = create_background_map(watersheds)


interactiveParetoFront(app, input_data, save_front = None)
app.run_server(debug=True)
