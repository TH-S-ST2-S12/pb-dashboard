# api/index.py

import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px
import pandas as pd
import glob
import os

# Specify the directory containing the CSV files
directory_path = 'Trial8'

# Use glob to find all CSV files in the directory
csv_files = glob.glob(f'{directory_path}/*.csv')

# Check if any CSV files were found
if not csv_files:
    raise ValueError(f"No CSV files found in directory: {directory_path}")

# Initialize a list to hold the DataFrames
dataframes = []

# Iterate over the list of CSV files
for file_path in csv_files:
    # Extract the cluster ID from the file name
    # Assuming the file name format is 'clusterX.csv' where X is the cluster ID
    cluster_id = file_path.split('cluster')[-1].split('.')[0]
    
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    # Add a new column to the DataFrame for the cluster ID
    if cluster_id == "0":
        df['ClusterID'] = cluster_id + " - Food"
    if cluster_id == "1":
        df['ClusterID'] = cluster_id + " - Media"
    if cluster_id == "2":
        df['ClusterID'] = cluster_id + " - Travel"
    if cluster_id == "3":
        df['ClusterID'] = cluster_id + " - Culture"
    if cluster_id == "4":
        df['ClusterID'] = cluster_id + " - Beach"
    if cluster_id == "5":
        df['ClusterID'] = cluster_id + " - Politics"
    if cluster_id == "6":
        df['ClusterID'] = cluster_id + " - Foreign Travel"
    if cluster_id == "7":
        df['ClusterID'] = cluster_id + " - Engagement"
    if cluster_id == "8":
        df['ClusterID'] = cluster_id + " - Tour"
    if cluster_id == "9":
        df['ClusterID'] = cluster_id + " - Entertainment"

    # Append the DataFrame to the list
    dataframes.append(df)

all_clusters_df = pd.concat(dataframes, ignore_index=True)

def create_hyperlink(row):
    return html.A(row['Channel Title'], href=f"https://www.youtube.com/channel/{row['Channel Id']}")

# Assuming 'all_clusters_df' is the combined dataframe from both datasets
all_clusters_df['Year'] = pd.to_datetime(all_clusters_df['Publish Date']).dt.year

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, 'https://codepen.io/chriddyp/pen/bWLwgP.css'])
server = app.server

# Create a Plotly figure
fig = go.Figure()
fig2 = go.Figure()
fig3 = go.Figure()

#START: TOPICS OVER TIME
# Add a bar chart to the figure for each year and each cluster
for year in sorted(all_clusters_df['Year'].unique()):
    for cluster in sorted(all_clusters_df['ClusterID'].unique()):
        # Filter data for the current year and cluster
        filtered_df = all_clusters_df[(all_clusters_df['Year'] == year) & (all_clusters_df['ClusterID'] == cluster)]
        count = len(filtered_df)  # Get the count of entries
        
        # Add trace
        fig.add_trace(go.Bar(
            x=[str(cluster)],  # Ensure x is treated as categorical data
            y=[count],
            name=f'Cluster {cluster}',
            visible=(year == sorted(all_clusters_df['Year'].unique())[0])  # Only the first year is visible
        ))

# Define a consistent color palette
colors = px.colors.qualitative.Plotly

cluster_names = {
    0: 'Food', 1: 'Media', 2: 'Travel', 3: 'Culture', 
    4: 'Beach', 5: 'Politics', 6: 'Foreign Travel', 
    7: 'Engagement', 8: 'Tour', 9: 'Entertainment'
}

video_counts = all_clusters_df.groupby(['Year', 'ClusterID']).size().reset_index(name='Count')

data = []
for cluster in range(10):
    cluster_data = video_counts[video_counts['ClusterID'].str[0].astype(int) == cluster]
    trace = go.Scatter(x=cluster_data['Year'], y=cluster_data['Count'], mode='lines', name=f'{cluster_names[cluster]}')
    data.append(trace)

layout = go.Layout(
    xaxis=dict(title='Year'),
    yaxis=dict(title='Number of Videos'),
    legend=dict(x=1.02, y=1, xanchor='left', yanchor='top'),  # Position legend outside the graph
    height=600  # Adjust the height as desired
)

fig3 = go.Figure(data=data, layout=layout)

# START: CHANNEL STATISTICS
grouped_channels = all_clusters_df.groupby(['ClusterID', 'Channel Title'], as_index=False).agg({
    'Video Views': 'mean',
    'Video Likes': 'mean'
})

def get_top_n_per_cluster(df, column, n=10):
    return df.groupby('ClusterID', group_keys=False).apply(lambda x: x.nlargest(n, column)).reset_index(drop=True)

# Get top 10 channels per cluster by video views
top_10_views_per_cluster = get_top_n_per_cluster(grouped_channels, 'Video Views', n=10)

cluster_ids = grouped_channels['ClusterID'].unique()
cluster_ids_list = cluster_ids.tolist()

for cluster_id in cluster_ids_list:
    filtered_df_views = top_10_views_per_cluster[top_10_views_per_cluster['ClusterID'] == cluster_id]

    # Adding a table for views
    fig2.add_trace(go.Table(
        header=dict(values=['Channel Title', 'Video Views', 'Video Likes'], fill_color='paleturquoise'),
        cells=dict(values=[filtered_df_views['Channel Title'], filtered_df_views['Video Views'], filtered_df_views['Video Likes']], fill_color='lavender'),
        visible=(cluster_id == cluster_ids_list[0]),  # Make the first cluster visible by default
        name=f"Views-{cluster_id}"
    ))

dropdown_items = []
for cluster_id in cluster_ids_list:
    visibility = [False] * len(cluster_ids_list)  # Now we have only 1 table (views) for each cluster
    visibility[cluster_ids_list.index(cluster_id)] = True  # Enable visibility for the current cluster's views

    dropdown_items.append(
        {
            "label": f"Cluster {cluster_id}",
            "method": "update",
            "args": [{"visible": visibility},
                     {"title": f"Top 10 Channels by Video Views in Cluster {cluster_id}"}]
        }
    )

most_popular_country_per_cluster = all_clusters_df.groupby(['ClusterID', 'Channel Country']).size().reset_index(name='Count')

def process_cluster_data(df, metric):
    df['Publish Date'] = pd.to_datetime(df['Publish Date'])
    threshold_low = df[metric].quantile(0.025)
    threshold_high = df[metric].quantile(0.975)
    df_trimmed = df[df[metric].between(threshold_low, threshold_high)].copy()
    metric_by_year_trimmed = df_trimmed.groupby(df_trimmed['Publish Date'].dt.year)[metric].mean().reset_index()
    return metric_by_year_trimmed

file_path_pattern = 'Trial8\\cluster{}.csv'

# Load all data into a dictionary of DataFrames
cluster_data = {}
for cluster_id in range(10):
    file_path = file_path_pattern.format(cluster_id)
    cluster_data[cluster_id] = pd.read_csv(file_path)

# Define the layout of the Dash app
app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H1("Pinoy Baiting Dashboard", className="text-center mb-4"), width=12)),
    dbc.Tabs([
        dbc.Tab(label="Statistics Over Time", children=[
            dbc.Row([
                dbc.Col([
                    html.H4("Top Topics Over Time", className="text-center mb-4"),
                    dcc.Dropdown(
                    id='year-dropdown1',  # Update to year-dropdown1
                    options=[{'label': str(year), 'value': str(year)} for year in sorted(all_clusters_df['Year'].unique())],
                    value=str(sorted(all_clusters_df['Year'].unique())[0]),
                    clearable=False,
                    className="mb-4"
                    ),
                    dcc.Graph(
                    id='graph1',  # Update to graph1
                    figure=fig,
                    config={'staticPlot': False, 'responsive': True}
                    ),
                    dcc.Graph(
                        id='videos-per-year-per-cluster',
                        figure=fig3
                    )
                ], width=6),
                dbc.Col([
                    html.Div([
                        html.H4("Views and Likes Over Time", className="text-center mb-4"),
                        dcc.Graph(id='views-graph'),
                        dcc.Graph(id='likes-graph')
                    ])
                ], width=6)
            ])
        ]),
        dbc.Tab(label="Channel & Country Statistics", children=[
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H4("Channel Statistics", className="text-center mb-4"),
                        html.Label('Cluster:', style={'font-family': 'Inter, sans-serif'}),
                        dcc.Dropdown(
                            id='cluster-dropdown',
                            options=[{'label': value, 'value': key} for key, value in cluster_names.items()],
                            value=0,  # Default value
                            style={'font-family': 'Inter, sans-serif'}
                        ),
                        html.Label('Parameter:', style={'font-family': 'Inter, sans-serif'}),
                        dcc.Dropdown(
                            id='parameter-dropdown',
                            options=[{'label': option, 'value': option} for option in ['Upload Count', 'Video Views', 'Video Likes', 'Comment Count']],
                            value='Upload Count',  # Default value
                            style={'font-family': 'Inter, sans-serif'}
                        ),
                        html.Label('Number of Results:', style={'font-family': 'Inter, sans-serif'}),
                        dcc.Slider(
                            id='results-slider',
                            min=1,
                            max=10,
                            step=1,
                            value=1,
                            marks={i: str(i) for i in range(1, 11)}
                        ),
                        dcc.Checklist(
                            id='year-filter-checkbox',
                            options=[{'label': 'Filter by Year', 'value': 'filter_year'}],
                            value=['filter_year'],  # Default value is checked
                            style={'font-family': 'Inter, sans-serif'}
                        ),
                        dcc.RangeSlider(
                            id='year-range-slider',
                            min=cluster_data[0]['Publish Date'].apply(lambda x: pd.to_datetime(x).year).min(),
                            max=cluster_data[0]['Publish Date'].apply(lambda x: pd.to_datetime(x).year).max(),
                            step=1,
                            value=[
                                cluster_data[0]['Publish Date'].apply(lambda x: pd.to_datetime(x).year).min(),
                                cluster_data[0]['Publish Date'].apply(lambda x: pd.to_datetime(x).year).max()
                            ],
                            marks={i: str(i) for i in range(cluster_data[0]['Publish Date'].apply(lambda x: pd.to_datetime(x).year).min(), 
                                                            cluster_data[0]['Publish Date'].apply(lambda x: pd.to_datetime(x).year).max() + 1)}
                        ),
                        html.Div(id='data-table-container', style={'overflowY': 'scroll', 'maxHeight': '200px'})
                    ])
                ], width=6),
                dbc.Col([
                    html.Div([
                        html.Div(style={'margin-top': '20px'}),  # Add space before the H6 element

                        html.H6("Top Country Uploads per Cluster", className="text-center mb-4"),
                        dcc.Dropdown(
                            id='cluster-dropdown-4',
                            options=[{'label': value, 'value': key} for key, value in cluster_names.items()],
                            value=0  # Default value
                        ),
                        dcc.Slider(
                            id='country-slider',
                            min=5,
                            max=most_popular_country_per_cluster['Channel Country'].nunique(),
                            step=5,
                            value=5,  # Default value
                            marks={i: str(i) for i in range(5, most_popular_country_per_cluster['Channel Country'].nunique() + 1, 5)}
                        ),
                        dcc.Graph(
                            id='most-popular-country-per-cluster'
                        )
                    ])
                ], width=6)
            ])
        ])
    ])
], className="container-fluid")

@app.callback(
    Output('results-slider', 'max'),
    Output('results-slider', 'marks'),
    [Input('cluster-dropdown', 'value'), 
     Input('year-filter-checkbox', 'value'),
     Input('year-range-slider', 'value')]
)
def update_slider_max(selected_cluster, filter_year, selected_year_range):
    df = cluster_data[selected_cluster]
    df['Year'] = pd.to_datetime(df['Publish Date']).dt.year

    if 'filter_year' in filter_year:
        df = df[(df['Year'] >= selected_year_range[0]) & (df['Year'] <= selected_year_range[1])]
        max_results = df.groupby('Year')['Channel Title'].nunique().max()
    else:
        max_results = df['Channel Title'].nunique()

    max_results = max_results if max_results else 1
    marks = {i: str(i) for i in range(1, max_results + 1, 10)}
    return max_results, marks

@app.callback(
    [Output('graph1', 'figure'),
     Output('data-table-container', 'children'),
     Output('most-popular-country-per-cluster', 'figure'),
     Output('views-graph', 'figure'),
     Output('likes-graph', 'figure')],
    [Input('year-dropdown1', 'value'),
     Input('cluster-dropdown', 'value'), Input('parameter-dropdown', 'value'), Input('results-slider', 'value'), 
     Input('year-filter-checkbox', 'value'), Input('year-range-slider', 'value'),
     Input('cluster-dropdown-4', 'value'), Input('country-slider', 'value')])  # Add input for the cluster dropdown)

def update_figures(selected_year, selected_cluster, selected_parameter, results_per_year, filter_year, selected_year_range, selected_cluster_4, num_countries):
    # Create new figures to update the graphs based on the selected year
    selected_year = int(selected_year)

    # Update graph1
    new_fig = go.Figure()
    for i, cluster in enumerate(sorted(all_clusters_df['ClusterID'].unique())):
        # Filter data for the selected year and cluster
        filtered_df = all_clusters_df[(all_clusters_df['Year'] == selected_year) & (all_clusters_df['ClusterID'] == cluster)]
        count = len(filtered_df)  # Get the count of entries

        # Add trace for the current cluster using 'i' to access colors array safely
        new_fig.add_trace(go.Bar(
            x=[str(cluster)],  # Ensure x is treated as categorical data
            y=[count],
            name=f'Cluster {cluster}',
            marker=dict(color=colors[i % len(colors)])  # Use colors from the palette
        ))
    # Update layouts
    new_fig.update_layout(
        paper_bgcolor='white',
        plot_bgcolor='white',
        height=400
        # Add other layout properties as needed
    )

    #graph2
    df = cluster_data[selected_cluster]
    df['Year'] = pd.to_datetime(df['Publish Date']).dt.year
    
    if 'filter_year' in filter_year:
        df = df[(df['Year'] >= selected_year_range[0]) & (df['Year'] <= selected_year_range[1])]
        if selected_parameter == 'Upload Count':
            top_channels = df.groupby(['Year', 'Channel Title']).size().reset_index(name='Upload Count').groupby('Year').apply(lambda x: x.nlargest(results_per_year, 'Upload Count')).reset_index(drop=True)
        else:
            top_channels = df.groupby(['Year', 'Channel Title']).agg({selected_parameter: 'sum'}).reset_index()
            top_channels = top_channels.groupby('Year').apply(lambda x: x.nlargest(results_per_year, selected_parameter)).reset_index(drop=True)
    else:
        if selected_parameter == 'Upload Count':
            top_channels = df.groupby('Channel Title').size().reset_index(name='Upload Count').nlargest(results_per_year, 'Upload Count')
        else:
            top_channels = df.groupby('Channel Title').agg({selected_parameter: 'sum'}).reset_index().nlargest(results_per_year, selected_parameter)

    # Generate a table figure using Plotly
    # table_fig = ff.create_table(top_channels[['Year', 'Channel Title', selected_parameter]])
    columns = ['Year', 'Channel Title', selected_parameter] if 'filter_year' in filter_year else ['Channel Title', selected_parameter]
    table_fig = go.Figure(data=[go.Table(
        header=dict(values=columns, fill_color='paleturquoise', align='left'),
        cells=dict(values=[top_channels[col] for col in columns], fill_color='lavender', align='left')
    )])

    #graph4
    filtered_df = most_popular_country_per_cluster[most_popular_country_per_cluster['ClusterID'].str[0].astype(int) == selected_cluster_4]
    filtered_df = filtered_df.sort_values(by='Count', ascending=False).head(num_countries)
    
    fig4 = go.Figure(data=[go.Table(
        header=dict(values=['Channel Country', 'Count'],
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[filtered_df['Channel Country'], filtered_df['Count']],
                   fill_color='lavender',
                   align='left'))
    ])
    
    fig4.update_layout(height=400)  # Set specific height

    #fig 5 6
    fig_views = go.Figure()
    fig_likes = go.Figure()

    # Loop through each cluster and add to the plots
    for cluster_id in range(10):  # Assuming cluster IDs go from 0 to 9
        # Load the data for the current cluster
        file_path = file_path_pattern.format(cluster_id)
        cluster_df = pd.read_csv(file_path)
        
        # Process the data
        cluster_data_views = process_cluster_data(cluster_df, 'Video Views')
        cluster_data_likes = process_cluster_data(cluster_df, 'Video Likes')
        
        # Add traces to the figures
        fig_views.add_trace(go.Scatter(x=cluster_data_views['Publish Date'], y=cluster_data_views['Video Views'],
                                       mode='lines+markers', name=f'{cluster_names[cluster_id]}',
                                       marker=dict(size=8)))
        
        fig_likes.add_trace(go.Scatter(x=cluster_data_likes['Publish Date'], y=cluster_data_likes['Video Likes'],
                                       mode='lines+markers', name=f'{cluster_names[cluster_id]}',
                                       marker=dict(size=8)))

    # Update layout for views graph
    fig_views.update_layout(
        title='Video Views',
        xaxis_title='Year',
        yaxis_title='Trimmed Video Views',
        legend_title='Cluster ID'
    )

    # Update layout for likes graph
    fig_likes.update_layout(
        title='Video Likes',
        xaxis_title='Year',
        yaxis_title='Trimmed Video Likes',
        legend_title='Cluster ID'
    )


    return new_fig, dcc.Graph(figure=table_fig), fig4, fig_views, fig_likes

# Run the app
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8050))

    app.run_server(debug=True, dev_tools_ui=False, dev_tools_props_check=False, host="0.0.0.0", port=port)