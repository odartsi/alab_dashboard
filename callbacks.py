import json
from dash.dependencies import Input, Output, State, ALL
from data import fetch_data, fetch_data2
from utils import create_clickable_list, calculate_experiment_similarity, get_samples_with_same_precursors, get_samples_with_same_target, get_samples_with_same_powder, periodic_table_layout, element_categories
import plotly.graph_objects as go
from dash import dcc, html, Input, Output
from utils import convert_time_and_sync_temperature, get_number, get_phase_weights, get_powder_data, get_sample_correlations, create_color_mapping, create_correlation_plot
import numpy as np
import dash


def register_callbacks(app):
    df = fetch_data()  # Fetch the cached data or query MongoDB if necessary
    df2 = fetch_data2()  # Fetch the cached data or query MongoDB if necessary
    
    
    @app.callback(
        [Output('precursors-plot', 'figure'),
         Output('target-box', 'children'),
         Output('similar-experiments-box', 'children')],
        [Input('sample-name-dropdown', 'value')]
    )
    def update_precursors_plot(selected_samples):
        if not selected_samples:
            return {}, "No data available", []
        print("In precursonrs: ", selected_samples)
        selected_sample = selected_samples[0]
        # df = fetch_data()
        print("i GOT THE DF: ", df)
        if df.empty or not selected_sample:
            return {}, "No data available", []

        data = df[df['name'] == selected_sample].iloc[0]
        sample_precursors = data['metadata'].get('elements_present', [])
        sample_target = data['metadata'].get('target', [])

        powder_dosing = data['metadata'].get('powderdosing_results', {})
        selected_precursors_powders = powder_dosing.get('Powders', [])
        sample_powder_names = [p['PowderName'] for p in selected_precursors_powders]
        
        # Get all precursors in the database
        all_precursors_in_db = set()
        for index, row in df.iterrows():
            all_precursors_in_db.update(row['metadata'].get('elements_present', []))

        # Prepare data for periodic table layout
        x_coords = []
        y_coords = []
        text_e = []
        fill_colors = []
        outline_colors = []

        for element, (x, y) in periodic_table_layout.items():
            x_coords.append(x)
            y_coords.append(y)
            text_e.append(element)
            category = element_categories.get(element, 'unknown')
            if element not in sample_precursors and element not in all_precursors_in_db:
                fill_colors.append('rgba(240, 128, 128, 0.5)')
                outline_colors.append('lightgrey')
            elif element in sample_precursors:
                fill_colors.append('rgba(135, 206, 250, 0.5)')
                outline_colors.append('lightgrey')
            elif element in all_precursors_in_db:
                fill_colors.append('rgba(144, 238, 144, 0.5)')
                outline_colors.append('lightgrey')

        fig = go.Figure()
        for i, fill_color in enumerate(fill_colors):
            fig.add_shape(type="rect",
                          x0=x_coords[i] - 0.5, y0=y_coords[i] - 0.5,
                          x1=x_coords[i] + 0.5, y1=y_coords[i] + 0.5,
                          line=dict(color=outline_colors[i]),
                          fillcolor=fill_color,
                          opacity=0.6
                          )

        fig.add_trace(go.Scatter(
            x=x_coords,
            y=y_coords,
            text=text_e,
            mode='text',
            textfont=dict(color='black', size=14, family='Arial Black'),
            textposition='middle center',  # Center the text
            showlegend=False,
            hoverinfo="skip"  # Disable hover text
        ))

        fig.update_layout(
            title='Available Precursors in Selected Sample',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )

        # Get samples with the same precursors
        matching_samples = get_samples_with_same_precursors(sample_precursors,selected_sample)
        sample_names_list = create_clickable_list(matching_samples)

        # Get target for this sample
        target = data['metadata'].get('target', 'Unknown')
        similar_targets = get_samples_with_same_target(sample_target, selected_sample)
        # sample_targets_list = create_clickable_list(similar_targets)
        sample_targets_list = f" {', '.join(similar_targets)}"

        similar_powder = get_samples_with_same_powder(sample_powder_names)
        # sample_powder_list = create_clickable_list(similar_powder)
        sample_powder_list = f" {', '.join(similar_powder)}"

        similarity = calculate_experiment_similarity(data, selected_sample)
        formatted_similarity = [
            f'{experiment}: similarity score = {score * 100:.0f}%' for experiment, score in similarity
        ]
        # similarity_text = create_clickable_list(formatted_similarity)
        # Join the formatted elements into a single string with line breaks
        # similarity_text = '\n\t- '.join(formatted_similarity)
        
        similarity_text = [
            dcc.Link(f"{experiment}: similarity score = {score * 100:.0f}%",
                    href=f"/select/{experiment}")
            for experiment, score in similarity
        ]

        print('similarity_text', similarity_text)
        
        # Create the target box content
        target_box_content = html.Div([
            html.H4('Target:'),
            html.P(target)
        ], style={'background-color': '#e5f5f1', 'padding': '10px'})

        # Create the similar experiments box content
        similar_experiments_box_content = html.Div([
            html.H4(f'{len(similarity)} Similar Experiments:'),
            # html.Div(similarity_text),
            html.Div([html.Div(link) for link in similarity_text]),
            html.H4(f'{len(similar_targets)} Experiments with same target:'),
            html.Div(sample_targets_list),
            html.H4(f'{len(similar_powder)} Experiments with same powders:'),
            html.Div(sample_powder_list)
        ], style={'background-color': '#C8EDDC', 'padding': '10px'})

        return fig, target_box_content, similar_experiments_box_content

    @app.callback(
    Output('sample-name-dropdown', 'value'),
    [Input('url', 'pathname')],
    [State('sample-name-dropdown', 'options')]
    )
    def update_selected_sample_from_link(pathname, options):
        # Extract the experiment name from the URL
        if pathname and pathname.startswith('/select/'):
            selected_experiment = pathname.split('/select/')[1]
            # Check if the selected experiment is in the dropdown options
            if any(option['value'] == selected_experiment for option in options):
                return selected_experiment
        return dash.no_update

    @app.callback(
        Output('temperature-plot', 'figure'),
        [Input('sample-name-dropdown', 'value')]
    )
    def update_temperature_plot(selected_samples):
        if df.empty or not selected_samples:
            return go.Figure() 

        fig = go.Figure()

        if not isinstance(selected_samples, list):
            selected_samples = [selected_samples]

        for selected_sample in selected_samples:
            data = df[df['name'] == selected_sample].iloc[0]
            target = data['metadata'].get('target', 'Unknown')
            try:
                temperature_log = data['metadata'].get('heating_results', {}).get('temperature_log', {})
                time_minutes = temperature_log.get('time_minutes', [])
                temperature_celsius = temperature_log.get('temperature_celsius', [])
            except:
                return go.Figure()

            if time_minutes and temperature_celsius:
                time_minutes, temperature_celsius = convert_time_and_sync_temperature(time_minutes, temperature_celsius)

                # Determine if the sample hit the target
                if '_' in selected_sample:
                    base_name, sample_number = selected_sample.split('_', 1)
                else:
                    base_name = selected_sample
                    sample_number = None

                matching_samples = df2[df2['name'].str.contains(base_name)]
                if sample_number:
                    matching_samples = matching_samples[matching_samples['name'].str.contains(sample_number)]

                hit_target = False
                if not matching_samples.empty:
                    for _, doc in matching_samples.iterrows():
                        output = doc.get('output')
                        if isinstance(output, str):
                            try:
                                output = json.loads(output)
                            except json.JSONDecodeError:
                                print(f"Error decoding output for document {doc['name']}")
                                continue  # Skip this document

                        if isinstance(output, dict):
                            final_result = output.get('final_result')
                            if isinstance(final_result, str):
                                try:
                                    final_result = json.loads(final_result)
                                except json.JSONDecodeError:
                                    print(f"Error decoding final_result for document {doc['name']}")
                                    continue  # Skip this document

                            if final_result:
                                lst_data = final_result.get('lst_data')
                                if lst_data:
                                    phases_results = lst_data.get('phases_results')
                                    if phases_results:
                                        phases = list(phases_results)  # Safely convert to list if not None
                                        # Checking if the target is in the phases
                                        if any(target in phase.split()[0] for phase in phases):
                                            hit_target = True
                                            break

                line_style = 'solid' if hit_target else 'dash'

                fig.add_trace(go.Scatter(
                    x=time_minutes,
                    y=temperature_celsius,
                    mode='lines',
                    name=f'{selected_sample}, hit target = {hit_target}',
                    line=dict(width=4, dash=line_style)
                ))
        fig.update_layout(
            title='Temperature vs Time',
            xaxis=dict(title='Time (hours)'),
            yaxis=dict(title='Temperature (°C)'),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )

        return fig

    @app.callback(
    [Output('xrd-plot', 'figure'),
     Output('best_rwp_box', 'children'),
     Output('results_box', 'children')],
    [Input('sample-name-dropdown', 'value')]    
    )


    def update_xrd_plot(selected_samples):
        if not selected_samples:
            return go.Figure(), "No data available", "No data available"  # Return empty outputs if no sample is selected

        selected_sample = selected_samples[0]
        fig = go.Figure()
        if df2.empty or not selected_sample:
            return go.Figure(), "No data available", "No data available"  # Return empty outputs if no data is available

        if '_' in selected_sample:
            base_name, sample_number = selected_sample.split('_', 1)
        else:
            base_name = selected_sample
            sample_number = None

        matching_samples = df2[df2['name'].str.contains(base_name, na=False)]
        if sample_number:
            matching_samples = matching_samples[matching_samples['name'].str.contains(sample_number, na=False)]

        if matching_samples.empty:
            return go.Figure(), "No data available", "No data available"  # Return empty outputs if no matching samples are found

        data = {}
        fig = go.Figure()
        for _, doc in matching_samples.iterrows():
            name = doc['name']
            fw_id = doc.get('fw_id') or doc.get('uuid')  # Update field access based on actual data structure
            if name not in data or fw_id > data[name]['fw_id']:
                data[name] = {'doc': doc, 'fw_id': fw_id}

        for doc_info in data.values():
            doc = doc_info['doc']
            output = doc.get('output') or doc.get('some_output_field')  # Replace with actual field

            # Check if output is a string and needs to be deserialized
            if isinstance(output, str):
                try:
                    output = json.loads(output)
                except json.JSONDecodeError as e:
                    print(f"Error decoding output for document {doc['name']}: {e}")
                    continue  # Skip this document
                except Exception as e:
                    print(f"Unexpected error decoding output for document {doc['name']}: {e}")
                    continue  # Skip this document

            # Ensure output is now a dictionary
            if isinstance(output, dict):
                final_result = output.get('final_result') or output.get('some_final_result_field')  # Update field

                # Check if final_result is a string and needs to be deserialized
                if isinstance(final_result, str):
                    try:
                        final_result = json.loads(final_result)
                    except json.JSONDecodeError as e:
                        print(f"Error decoding final_result for document {doc['name']}: {e}")
                        continue  # Skip this document
                    except Exception as e:
                        print(f"Unexpected error decoding final_result for document {doc['name']}: {e}")
                        continue  # Skip this document

                # Process the valid final_result
                if final_result:
                    fig = go.Figure()
                    plot_data = final_result.get('plot_data', {})
                    best_rwp = None
                    lst_data = final_result.get('lst_data')
                    if lst_data:
                        best_rwp = lst_data.get('rwp')
                        phases_results = lst_data.get('phases_results')
                        if phases_results:
                            phases = list(phases_results)  # Safely convert to list if not None

                    if plot_data:
                        x = plot_data.get('x', [])
                        y_obs = plot_data.get('y_obs', [])
                        y_calc = plot_data.get('y_calc', [])
                        y_bkg = plot_data.get('y_bkg', [])
                        structs = plot_data.get('structs', {})

                        diff = np.array(y_obs) - np.array(y_calc)
                        diff_offset_val = 0

                        fig.add_trace(go.Scatter(x=x, y=y_obs, mode='lines', name='Observed', showlegend=True))
                        fig.add_trace(go.Scatter(x=x, y=y_calc, mode='lines', name='Calculated', showlegend=True))
                        fig.add_trace(go.Scatter(x=x, y=y_bkg, mode='lines', name='Background', showlegend=True))
                        fig.add_trace(go.Scatter(x=x, y=diff - diff_offset_val, mode='lines', name='Difference', showlegend=True))

                        weight_fractions = get_phase_weights(final_result)
                        colormap = [
                            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
                            "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
                        ]

                        for i, (phase_name, phase) in enumerate(structs.items()):
                            if i >= len(colormap):
                                i = i % len(colormap)

                            phase_display_name = f"{phase_name} ({weight_fractions[phase_name] * 100:.2f} %)" if len(weight_fractions) > 1 else phase_name
                            fig.add_trace(
                                go.Scatter(
                                    x=x,
                                    y=y_bkg,
                                    mode="lines",
                                    line=dict(color=colormap[i], width=0),
                                    fill=None,
                                    showlegend=False,
                                    hoverinfo="none",
                                    legendgroup=phase_name,
                                )
                            )
                            fig.add_trace(
                                go.Scatter(
                                    x=x,
                                    y=np.array(phase) + np.array(y_bkg),
                                    mode="lines",
                                    line=dict(color=colormap[i], width=1.5),
                                    fill="tonexty",
                                    name=phase_display_name,
                                    visible="legendonly",
                                    legendgroup=phase_name,
                                )
                            )
                        fig.update_layout(
                            #title='XRD Characterization',
                            xaxis_title='2θ [°]',
                            yaxis_title='Intensity',
                            height=500,
                            width=1000,
                            plot_bgcolor='white',
                            paper_bgcolor='white',
                            legend=dict(
                                orientation="h",
                                x=0,
                                y=-0.2
                            )
                        )

        best_rwp_box = html.Div([
            html.H4('Best rwp:', style={'text-align': 'right'}),
            html.P(best_rwp, style={'text-align': 'right'})
        ], style={'background-color': '#D5A6BD', 'padding': '10px'})

        results_box = html.Div([
            html.H4('Phases:', style={'text-align': 'right'}),
            html.Ul([
                html.Li(phase, style={'text-align': 'right'}) for phase in phases
            ])
        ], style={'background-color': '#d6efd8', 'padding': '10px'})

        return fig, best_rwp_box, results_box
    
    # def update_xrd_plot(selected_samples):
    #     print("In xrd: ", selected_samples)
    #     if not selected_samples:
    #         return go.Figure(), "No data available", "No data available"  # Return empty outputs if no sample is selected

    #     selected_sample = selected_samples[0]
    #     fig = go.Figure()
    #     print(f"Selected sample in XRD plot: {selected_sample}")
    #     df2 = fetch_data2()  # Fetch the cached data or query MongoDB if necessary
    #     if df2.empty or not selected_sample:
    #         return go.Figure(), "No data available", "No data available"  # Return empty outputs if no data is available

    #     if '_' in selected_sample:
    #         base_name, sample_number = selected_sample.split('_', 1)
    #     else:
    #         base_name = selected_sample
    #         sample_number = None

    #     matching_samples = df2[df2['name'].str.contains(base_name)]
    #     if sample_number:
    #         matching_samples = matching_samples[matching_samples['name'].str.contains(sample_number)]

    #     if matching_samples.empty:
    #         return go.Figure(), "No data available", "No data available"  # Return empty outputs if no matching samples are found

    #     data = {}
    #     for _, doc in matching_samples.iterrows():
    #         name = doc['name']
    #         fw_id = doc['metadata'].get('fw_id')
    #         if name not in data or fw_id > data[name]['fw_id']:
    #             data[name] = {'doc': doc, 'fw_id': fw_id}

    #     for index, doc_info in data.items():
    #         doc = doc_info['doc']
    #         output = doc.get('output')
    #         final_result = output.get('final_result')

    #         if final_result:
    #             plot_data = final_result.get('plot_data', {})
    #             best_rwp = None
    #             if output:
    #                 final_result = output.get('final_result')
    #                 results = output.get('results')
    #                 if final_result:
    #                     lst_data = final_result.get('lst_data')
    #                     if lst_data:
    #                         best_rwp = lst_data.get('rwp')
    #                         phases_results = lst_data.get('phases_results')
    #                         if phases_results:
    #                             phases = list(phases_results)  # Safely convert to list if not None

    #             if plot_data:
    #                 x = plot_data.get('x', [])
    #                 y_obs = plot_data.get('y_obs', [])
    #                 y_calc = plot_data.get('y_calc', [])
    #                 y_bkg = plot_data.get('y_bkg', [])
    #                 structs = plot_data.get('structs', {})
                    
    #                 diff = np.array(y_obs) - np.array(y_calc)
    #                 diff_offset_val = 0
                    
    #                 fig.add_trace(go.Scatter(x=x, y=y_obs, mode='lines', name='Observed',showlegend=True))
    #                 fig.add_trace(go.Scatter(x=x, y=y_calc, mode='lines', name='Calculated',showlegend=True))
    #                 fig.add_trace(go.Scatter(x=x, y=y_bkg, mode='lines', name='Background', showlegend=True))
    #                 fig.add_trace(go.Scatter(x=x, y=diff - diff_offset_val, mode='lines', name='Difference', showlegend=True))

    #                 weight_fractions = get_phase_weights(final_result)
    #                 colormap = [
    #                     "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
    #                     "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    #                 ]

    #                 for i, (phase_name, phase) in enumerate(structs.items()):
    #                     if i >= len(colormap):
    #                         i = i % len(colormap)

    #                     phase_display_name = f"{phase_name} ({weight_fractions[phase_name] * 100:.2f} %)" if len(weight_fractions) > 1 else phase_name
    #                     fig.add_trace(
    #                         go.Scatter(
    #                             x=x,
    #                             y=y_bkg,
    #                             mode="lines",
    #                             line=dict(color=colormap[i], width=0),
    #                             fill=None,
    #                             showlegend=False,
    #                             hoverinfo="none",
    #                             legendgroup=phase_name,
    #                         )
    #                     )
    #                     fig.add_trace(
    #                         go.Scatter(
    #                             x=x,
    #                             y=np.array(phase) + np.array(y_bkg),
    #                             mode="lines",
    #                             line=dict(color=colormap[i], width=1.5),
    #                             fill="tonexty",
    #                             name=phase_display_name,
    #                             visible="legendonly",
    #                             legendgroup=phase_name,
    #                         )
    #                     )
    #                 fig.update_layout(
    #                     title='XRD Characterization',
    #                     xaxis_title='2θ [°]',
    #                     yaxis_title='Intensity',
    #                     height=500,
    #                     width=1000,
    #                     plot_bgcolor='white',
    #                     paper_bgcolor='white',
    #                     legend=dict(
    #                         orientation="h",  
    #                         x=0, 
    #                         y=-0.2  
    #                     )
    #                 )

    #     best_rwp_box = html.Div([
    #         html.H4('Best rwp:', style={'text-align': 'right'}),
    #         html.P(best_rwp, style={'text-align': 'right'})
    #     ], style={'background-color': '#D5A6BD', 'padding': '10px'})

    #     results_box = html.Div([
    #         html.H4('Phases:', style={'text-align': 'right'}),
    #         html.Ul([
    #             html.Li(phase, style={'text-align': 'right'}) for phase in phases
    #         ])
    #     ], style={'background-color': '#d6efd8', 'padding': '10px'})

    #     return fig, best_rwp_box, results_box
    @app.callback(
        Output('correlation-plot', 'figure'),
        [Input('sample-name-dropdown', 'value')]
    )
    def update_correlation_plot(selected_samples):
        print("In correlation: ", selected_samples)
        if not selected_samples:
            return go.Figure()  # Return an empty figure if no samples are selected
        return create_correlation_plot(selected_samples)

    @app.callback(
        Output('pie-plot', 'figure'),
        [Input('sample-name-dropdown', 'value')]
    )
    def update_pie_plot(selected_samples):
        # df = fetch_data()  # Fetch the cached data or query MongoDB if necessary
        if df.empty or not selected_samples:
            return go.Figure()

        selected_sample = selected_samples[0]  # Take the first selected sample for the pie chart
        data = df[df['name'] == selected_sample].iloc[0]
        metadata = data.get('metadata', {})
        powder_names, target_masses = get_powder_data(metadata)
        
        # Define custom colors
        custom_colors = ['#daf8e3', '#97ebdb', '#00c2c7', '#0086ad', '#005582', '#008870', '#c1da87']

        fig = go.Figure(data=[go.Pie(labels=powder_names, values=target_masses, 
                                    hoverinfo='label+value+percent', 
                                    textinfo='value', 
                                    texttemplate='%{value:.2f}g',
                                    marker=dict(colors=custom_colors))])

        fig.update_layout(
            title=f'Powder Composition for {selected_sample}',
        )

        return fig
    @app.callback(
    [Output('image1', 'src'), Output('image2', 'src')],
    [Input('sample-name-dropdown', 'value')]
    )
    def display_images(selected_samples):
        if not selected_samples:
            # Return default images or placeholders if no sample is selected
            return "/images/Spectrum.png", "/images/DefaultImage.png"

        # selected_sample = selected_samples[0]
        # Assuming the images are stored with names corresponding to the sample names
        # image_filename1 = f"{selected_sample}_image1.png"  # Example: "sample_name_image1.png"
        # image_filename2 = f"{selected_sample}_image2.png"  # Example: "sample_name_image2.png"
        image_path1 = "/assets/favicon_io/Spectrum.png"  # Image path within the assets directory for image1
        image_path2 = "/assets/favicon_io/ImageAdjustment_0_324_540.png"  # Image path within the assets directory for image2

        return image_path1, image_path2