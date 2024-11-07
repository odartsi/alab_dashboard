import json
from dash.dependencies import Input, Output, State 
from data import fetch_data, fetch_data2
from utils import calculate_ratios, calculate_experiment_similarity, element_composition_in_moles, get_samples_with_same_target, get_samples_with_same_powder, periodic_table_layout, element_categories
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from dash import dcc, html
from utils import convert_time_and_sync_temperature, get_phase_weights, get_powder_data, create_correlation_plot, grams_to_moles, create_correlation_plot_with_metric
import numpy as np
import dash


def register_callbacks(app):
    df = fetch_data() 
    df2 = fetch_data2()  

    @app.callback(
    [Output('target-box', 'children'),
     Output('similar-experiments-box', 'children')],
    [Input('sample-name-dropdown', 'value')]
    )
    def update_target_and_similar_experiments(selected_samples):
        if not selected_samples:
            return "No data available", "No data available"
        
        selected_sample = selected_samples[0]
        if df.empty or not selected_sample:
            return "No data available", "No data available"

        data = df[df['name'] == selected_sample].iloc[0]
        sample_target = data['metadata'].get('target', [])

        powder_dosing = data['metadata'].get('powderdosing_results', {})
        selected_precursors_powders = powder_dosing.get('Powders', [])
        sample_powder_names = [p['PowderName'] for p in selected_precursors_powders]

        # Get target for this sample
        target_of_sample = data['metadata'].get('target', 'Unknown')
        similar_targets = get_samples_with_same_target(sample_target, selected_sample)
        print("The target isssssssssss: ", target_of_sample)
        print("similar targets : " , similar_targets)
        # sample_targets_list = f" {', '.join(similar_targets)}"
        flattened_targets = set()  # Use a set to keep only unique targets
        for item in similar_targets:
            targets = item.split(',')
            for target in targets:
                flattened_targets.add(target.strip())

        # Convert the set back to a sorted list for display
        sample_targets_list = sorted(flattened_targets)

        similar_powder = get_samples_with_same_powder(sample_powder_names)
        # sample_powder_list = f" {', '.join(similar_powder)}"
        # sample_powder_list = [target.strip() for target in similar_powder]
        flattened_powders = set()  # Use a set to keep only unique targets
        for item in similar_powder:
            targets = item.split(',')
            for target in targets:
                flattened_powders.add(target.strip())

        # Convert the set back to a sorted list for display
        sample_powder_list = sorted(flattened_powders)

        similarity = calculate_experiment_similarity(data, selected_sample)
        similarity_text = [
            html.Div([
                dcc.Link(experiment, href=f"/select/{experiment}", style={'margin-right': '5px'}),
                html.Span(f": similarity score = {score * 100:.0f}%")
            ])
            for experiment, score in similarity
        ]
        print("The target isssssssssss: ", target_of_sample)
        # Create the target box content
        target_box_content = html.Div([
            html.H4('Target:'),
            html.P(target_of_sample)
        ], style={'background-color': '#e5f5f1', 'padding': '10px'})

   
        # Create the similar experiments box content
        similar_experiments_box_content = html.Div([
            html.H4(f'{len(similarity)} Similar Experiments:'),
            html.Div([html.Div(link) for link in similarity_text]),
            
            html.H4(f'{len(similar_targets)} Experiments with same target:'),
            html.Div([
                html.Span([
                    dcc.Link(target, href=f"/select/{target}", style={'margin-right': '5px', 'display': 'inline'}),
                    html.Span(", ") if i < len(sample_targets_list) - 1 else None  # Add comma except after the last item
                ]) for i, target in enumerate(sample_targets_list)
            ], style={'whiteSpace': 'normal', 'overflow': 'hidden', 'textOverflow': 'ellipsis'}),
            
            html.H4(f'{len(similar_powder)} Experiments with same powders:'),
            html.Div([
                html.Span([
                    dcc.Link(target, href=f"/select/{target}", style={'margin-right': '5px', 'display': 'inline'}),
                    html.Span(", ") if i < len(sample_powder_list) - 1 else None  # Add comma except after the last item
                ]) for i, target in enumerate(sample_powder_list)
            ], style={'whiteSpace': 'normal', 'overflow': 'hidden', 'textOverflow': 'ellipsis'})
            
        ], style={
            'background-color': '#C8EDDC',
            'padding': '10px',
            'maxHeight': '300px',  # Limit height of the box
            'overflowY': 'auto',   # Add vertical scroll if content exceeds height
            'border': '1px solid #ccc',
            'borderRadius': '5px'
        })


        return target_box_content, similar_experiments_box_content
    
    @app.callback(
    Output('precursors-plot', 'figure'),
    [Input('sample-name-dropdown', 'value')]
    )
    def update_precursors_plot(selected_samples):
        if not selected_samples:
            return {}

        selected_sample = selected_samples[0]
        if df.empty or not selected_sample:
            return {}

        data = df[df['name'] == selected_sample].iloc[0]
        sample_precursors = data['metadata'].get('elements_present', [])
        
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
            textposition='middle center', 
            showlegend=False,
            hoverinfo="skip"
        ))
        # Add legend entries using scatter traces with markers
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=15, color='rgba(240, 128, 128, 0.5)'),
            legendgroup='not_in_use',
            showlegend=True,
            name='Not in use'
        ))

        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=15, color='rgba(135, 206, 250, 0.5)'),
            legendgroup='in_sample',
            showlegend=True,
            name='Used in sample'
        ))

        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=15, color='rgba(144, 238, 144, 0.5)'),
            legendgroup='in_db',
            showlegend=True,
            name='Used in Alab'
        ))

        # Update layout with titles for the plot and legend
        fig.update_layout(
            title='Available Precursors in Selected Sample',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            paper_bgcolor='white',
            legend=dict(
                title="",
                orientation="h",
                yanchor="top",
                y=1.15,
                xanchor="center",
                x=0.5
            )
        )


        # fig.update_layout(
        #     title='Available Precursors in Selected Sample',
        #     xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        #     yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        #     plot_bgcolor='white',
        #     paper_bgcolor='white'
        # )

        return fig
    

    @app.callback(
    Output('sample-name-dropdown', 'value'),
    [Input('url', 'pathname')],
    [State('sample-name-dropdown', 'options')]
    )
    def update_selected_sample_from_link(pathname, options):
        if pathname and pathname.startswith('/select/'):
            selected_experiment = pathname.split('/select/')[1]
            if any(option['value'] == selected_experiment for option in options):
                return [str(selected_experiment)]  # Ensure it returns a list
        return dash.no_update

    @app.callback(
        Output('temperature-plot', 'figure'),
        [Input('sample-name-dropdown', 'value')]
    )
    def update_temperature_plot(selected_samples):
        print(f"Selected samples in temperature plot callback: {selected_samples}")
        if df.empty or not selected_samples:
            return go.Figure() 

        fig = go.Figure()

        if not isinstance(selected_samples, list):
            selected_samples = [selected_samples]

        for selected_sample in selected_samples:
            data = df[df['name'] == selected_sample].iloc[0]
            metadata = data['metadata']
            target = metadata.get('target', 'Unknown')
            try:
                temperature_log = metadata.get('heating_results', {}).get('temperature_log', {})
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
                                        phases = list(phases_results)
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
    [Output('xrd-plot', 'figure')],
    #  Output('best_rwp_box', 'children'),
    #  Output('results_box', 'children')],
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
            fw_id = doc.get('fw_id') or doc.get('uuid') 
            if name not in data or fw_id > data[name]['fw_id']:
                data[name] = {'doc': doc, 'fw_id': fw_id}
        data2 = df[df['name'] == selected_sample].iloc[0]
        total_mass_dispensed = data2.get('metadata', {}).get('diffraction_results', []).get('total_mass_dispensed_mg')
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
                final_result = output.get('final_result') or output.get('some_final_result_field')

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
                        
                        # fig.update_layout(
                        #     #title='XRD Characterization',
                        #     xaxis_title='2θ [°]',
                        #     yaxis_title='Intensity',
                        #     height=500,
                        #     width=1300,
                        #     plot_bgcolor='white',
                        #     paper_bgcolor='white',
                        #     legend=dict(
                        #         orientation="v",  # Set vertical orientation
                        #         x=1.05,           # Move the legend to the right of the plot
                        #         y=1,              # Align the legend to the top
                        #         yanchor="top",
                        #         xanchor="left"
                        #     )
                           
                        # )
                        fig.update_layout(
                            autosize=True,  # Enable auto-sizing based on container size
                            xaxis_title='2θ [°]',
                            yaxis_title='Intensity',
                            plot_bgcolor='white',
                            paper_bgcolor='white',
                            margin=dict(l=5, r=5, t=5, b=5),  # Set minimal margins to use all space
                            height=500,
                            width=1800,
                            legend=dict(
                                orientation="v",
                                x=1.05,
                                y=1,
                                yanchor="top",
                                xanchor="left"
                            )
                        )
                        fig.update_layout(
                            annotations=[
                                dict(
                                    x=0.5,  # Center the text horizontally
                                    y=1.15,  # Position above the plot
                                    xref="paper", 
                                    yref="paper",
                                    text=f"Best RWP: {best_rwp}% | Total Mass Dispensed: {total_mass_dispensed} mg",
                                    showarrow=False,
                                    font=dict(size=14, color="black")
                                )
                            ]
                        )
        # data = df[df['name'] == selected_sample].iloc[0]
        # total_mass_dispensed = data.get('metadata', {}).get('diffraction_results', []).get('total_mass_dispensed_mg')
        # best_rwp_box = html.Div([
        #     html.H4('Best rwp:', style={'text-align': 'right'}),
        #     html.P(best_rwp, style={'text-align': 'right'}),
        #     html.H4('Total mass dispensed:', style={'text-align': 'right'}),
        #     html.P(total_mass_dispensed, style={'text-align': 'right'}),
        # ], style={'background-color': '#D5A6BD', 'padding': '10px'})

        # results_box = html.Div([
        #     html.H4('Phases:', style={'text-align': 'right'}),
        #     html.Ul([
        #         html.Li(phase, style={'text-align': 'right'}) for phase in phases
        #     ])
        # ], style={'background-color': '#d6efd8', 'padding': '10px'})

        return fig,# best_rwp_box, results_box
    
    @app.callback(
    Output('correlation-plot', 'figure'),
    [Input('sample-name-dropdown', 'value'),
     Input('comparable-sample-dropdown', 'value')]
    )
    def update_correlation_plot(selected_sample, comparable_sample):
        # Check if both samples are selected
        if not selected_sample or not comparable_sample:
            return go.Figure()  # Return an empty figure if either sample is not selected
        
        # Generate the correlation plot with the two selected samples
        return create_correlation_plot_with_metric(selected_sample, comparable_sample)
    # @app.callback(
    #     Output('correlation-plot', 'figure'),
    #     [Input('sample-name-dropdown', 'value')]
    # )
    # def update_correlation_plot(selected_samples):
    #     if not selected_samples:
    #         return go.Figure()  # Return an empty figure if no samples are selected
    #     return create_correlation_plot(selected_samples)

    @app.callback(
        Output('pie-plot', 'figure'),
        [Input('sample-name-dropdown', 'value')]
    )
    # def update_pie_plot(selected_samples):
    #     if df.empty or not selected_samples:
    #         return go.Figure()

    #     selected_sample = selected_samples[0]  
    #     data = df[df['name'] == selected_sample].iloc[0]
    #     metadata = data.get('metadata', {})
    #     powder_names, target_masses = get_powder_data(metadata)
    #     masses_in_moles = grams_to_moles(powder_names, target_masses)
        
    #     # Define custom colors
    #     custom_colors = ['#daf8e3', '#97ebdb', '#00c2c7', '#0086ad', '#005582', '#008870', '#c1da87']

    #     fig = go.Figure(data=[go.Pie(labels=powder_names, values=masses_in_moles, 
    #                                 hoverinfo='label+value+percent', 
    #                                 textinfo='value', 
    #                                 texttemplate='%{value:.5f} mol',
    #                                 marker=dict(colors=custom_colors))])

    #     fig.update_layout(
    #         title=f'Powder Composition for {selected_sample}',
    #     )

    #     return fig

    # def update_pie_plot(selected_samples):
    #     if df.empty or not selected_samples:
    #         return go.Figure()

    #     selected_sample = selected_samples[0]  
    #     data = df[df['name'] == selected_sample].iloc[0]
    #     metadata = data.get('metadata', {})
    #     powder_names, target_masses = get_powder_data(metadata)
    #     masses_in_moles = grams_to_moles(powder_names, target_masses)
    #     elemental_composition = element_composition_in_moles(powder_names, target_masses)
        
    #     # Define custom colors
    #     custom_colors = ['#daf8e3', '#97ebdb', '#00c2c7', '#0086ad', '#005582', '#008870', '#c1da87']
    #     custom_colors_2 = [	'#cae7b9', '#eb9486' ,'#f3de8a' ,'#7e7f9a', '#d387ab']

    #     # Create a subplot with two columns
    #     fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])

    #     # Add the first pie chart (Powder Composition)
    #     fig.add_trace(go.Pie(labels=powder_names, values=masses_in_moles, 
    #                         hoverinfo='label+value+percent', 
    #                         textinfo='value', 
    #                         texttemplate='%{value:.5f} mol',
    #                         marker=dict(colors=custom_colors)),
    #                 row=1, col=1)

    #     # Add the second pie chart (Elemental Composition)
    #     print("hoho",masses_in_moles, elemental_composition )
    #     print("hahhahahha ",list(elemental_composition.keys()),list(elemental_composition.values()))
    #     elements = list(elemental_composition.keys())
    #     elements_in_moles = list(elemental_composition.values())

    #     fig.add_trace(go.Pie(labels=elements, 
    #                         values=elements_in_moles, 
    #                         hoverinfo='label+value+percent', 
    #                         textinfo='value', 
    #                         texttemplate='%{value:.5f} mol',
    #                         marker=dict(colors=custom_colors_2)),
    #                 row=1, col=2)

    #     # Update layout with titles for each subplot
       
    #     fig.update_layout(
    #         annotations=[
    #             dict(text='Powder Composition', x=0.20, y=1.1, font_size=15, showarrow=False, xanchor='center'),
    #             dict(text='Elemental Composition', x=0.80, y=1.1, font_size=15, showarrow=False, xanchor='center')
    #         ],
    #         showlegend=True,
    #         # legend=dict(
    #         #     x=0.02,  # Position legend closer to the first pie
    #         #     y=0.5,
    #         #     traceorder='normal',
    #         #     font=dict(size=12),
    #         #     xanchor="left"
    #         # )
    #     )

    #     return fig
    def update_pie_plot(selected_samples):
        if df.empty or not selected_samples:
            return go.Figure()

        selected_sample = selected_samples[0]  
        data = df[df['name'] == selected_sample].iloc[0]
        metadata = data.get('metadata', {})
        powder_names, target_masses = get_powder_data(metadata)
        masses_in_moles = grams_to_moles(powder_names, target_masses)
        elemental_composition = element_composition_in_moles(powder_names, target_masses)
        
        # Calculate ratios for the elemental composition
        ratios = calculate_ratios(elemental_composition)
        
        # Define custom colors
        custom_colors = ['#daf8e3', '#97ebdb', '#00c2c7', '#0086ad', '#005582', '#008870', '#c1da87']
        custom_colors_2 = ['#cae7b9', '#eb9486', '#f3de8a', '#7e7f9a', '#d387ab']

        # Create a subplot with two columns
        fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])

        # Add the first pie chart (Powder Composition)
        fig.add_trace(go.Pie(labels=powder_names, values=masses_in_moles, 
                     customdata=masses_in_moles,  # Pass the values as customdata for use in hovertemplate
                     hoverinfo='label+value',            # Disable default hoverinfo
                     textinfo='label+percent',    # Display label and percent on the chart
                     texttemplate='%{label}: %{percent}',  # Display label and percent on the chart
                     hovertemplate='%{label}: %{customdata:.5f} mol',  # Customize the hover text
                     marker=dict(colors=custom_colors),
                     name=""),
              row=1, col=1)

        # Add the second pie chart (Elemental Composition with Ratios)
        elements = list(ratios.keys())
        element_ratios = list(ratios.values())

        fig.add_trace(go.Pie(labels=elements, 
                            values=element_ratios, 
                            hoverinfo='label+percent',             # Disable default hoverinfo
                            textinfo='label+text',        # Display label and text (value) on the chart
                            texttemplate='%{label}: %{value} parts',  # Customize the text on the chart
                            hovertemplate='%{label}: %{percent}',     # Customize the hover text
                            marker=dict(colors=custom_colors_2),
                            name=""),
                    row=1, col=2)

        # Update layout with titles for each subplot
        fig.update_layout(
            annotations=[
                dict(text='Powder Composition', x=0.20, y=1.1, font_size=15, showarrow=False, xanchor='center'),
                dict(text='Elemental Composition (Ratios)', x=0.80, y=1.1, font_size=15, showarrow=False, xanchor='center')
            ],
            showlegend=True,
        )

        return fig

    @app.callback(
    [Output('image1', 'src'), Output('image2', 'src')],
    [Input('sample-name-dropdown', 'value')]
    )
    def display_images(selected_samples):
        if not selected_samples:
            return "/images/Spectrum.png", "/images/DefaultImage.png"

        # selected_sample = selected_samples[0]
        # Assuming the images are stored with names corresponding to the sample names
        # image_filename1 = f"{selected_sample}_image1.png"  # Example: "sample_name_image1.png"
        # image_filename2 = f"{selected_sample}_image2.png"  # Example: "sample_name_image2.png"
        image_path1 = "/assets/favicon_io/Spectrum.png"  # Image path within the assets directory for image1
        image_path2 = "/assets/favicon_io/ImageAdjustment_0_324_540.png"  # Image path within the assets directory for image2

        return image_path1, image_path2
    import re

    def natural_sort_key(name):
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', name)]


    df = fetch_data()
    sorted_names = sorted(df['name'].unique(), key=natural_sort_key)
    @app.callback(
    Output('tab-content', 'children'),
    Input('tabs', 'value'),
    State('saved-plots', 'data')
    )
    

    def render_tab_content(tab, saved_plots):
        
        
        if tab == 'main-tab':
            return html.Div([
                # Dropdown for sample selection
                dcc.Dropdown(
                    id='sample-name-dropdown',
                    options=[{'label': name, 'value': name} for name in sorted_names],
                    multi=True,
                    style={'width': '100%', 'margin-bottom': '15px'}
                ),
                
                # Row 1: Target and Similar Experiments
                html.Div([
                    html.Div(id='target-box', style={'padding': '10px', 'margin': '10px', 'display': 'inline-block', 'width': '15%'}),
                    html.Div(id='similar-experiments-box', style={'padding': '10px', 'margin': '10px', 'display': 'inline-block', 'width': '45%'}),
                ], style={'display': 'flex', 'justify-content': 'space-between'}),

                # Row 2: Precursors and Pie plots side by side with save buttons
                html.Div([
                    html.Div([
                        dcc.Graph(id='precursors-plot', style={'padding': '10px', 'margin': '10px', 'width': '100%'}),
                        html.Button("Save Precursors Plot", id="save-precursors-plot-btn", style={'margin-top': '5px'}),
                    ], style={'display': 'inline-block', 'width': '45%'}),

                    html.Div([
                        dcc.Graph(id='pie-plot', style={'padding': '10px', 'margin': '10px', 'width': '100%'}),
                        html.Button("Save Pie Plot", id="save-pie-plot-btn", style={'margin-top': '5px'}),
                    ], style={'display': 'inline-block', 'width': '45%'}),
                ], style={'display': 'flex', 'justify-content': 'space-between', 'flex-wrap': 'wrap'}),

                # Row 3: Temperature and Correlation plots side by side with save buttons
                html.Div([
                    html.Div([
                        dcc.Graph(id='temperature-plot', style={'padding': '10px', 'margin': '10px', 'width': '100%'}),
                        html.Button("Save Temperature Plot", id="save-temperature-plot-btn", style={'margin-top': '5px'}),
                    ], style={'display': 'inline-block', 'width': '45%'}),

                    html.Div([
                        dcc.Dropdown(
                            id='comparable-sample-dropdown',
                            options=[{'label': name, 'value': name} for name in sorted_names],
                            placeholder="Select a comparable sample",
                            style={'width': '100%', 'font-size': '14px', 'margin': '10px'}
                        ),
                        dcc.Graph(id='correlation-plot', style={'padding': '10px', 'margin': '10px', 'width': '100%'}),
                        html.Button("Save Correlation Plot", id="save-correlation-plot-btn", style={'margin-top': '5px'}),
                    ], style={'display': 'inline-block', 'width': '45%'}),
                ], style={'display': 'flex', 'justify-content': 'space-between', 'flex-wrap': 'wrap'}),

                # Row 4: XRD plot with save button
                html.H2("XRD characterization", style={'text-align': 'left'}),
                html.Div([
                    html.Div([
                        dcc.Graph(id='xrd-plot', style={'padding': '10px', 'margin': '10px', 'width': '100%'}),
                        html.Button("Save XRD Plot", id="save-xrd-plot-btn", style={'margin-top': '5px'}),
                    ], style={'width': '40%', 'display': 'inline-block'}),
                    html.Div([
                        html.Div(id='best_rwp_box', style={'padding': '10px', 'margin': '10px'}),
                        html.Div(id='results_box', style={'padding': '10px', 'margin': '10px'})
                    ], style={'display': 'flex', 'flex-direction': 'column', 'height': '80%', 'width': '80%'})
                ], style={'display': 'flex'}),

                # Row 5: SEM characterization images side by side
                html.H2("SEM characterization", style={'text-align': 'left'}),
                html.Div([
                    html.Div([
                        html.Img(id='image1', style={'width': '100%', 'display': 'inline-block'}),
                    ], style={'width': '45%', 'display': 'inline-block', 'padding': '10px', 'margin': '10px'}),
                    
                    html.Div([
                        html.Img(id='image2', style={'width': '100%', 'display': 'inline-block'}),
                    ], style={'width': '45%', 'display': 'inline-block', 'padding': '10px', 'margin': '10px'}),
                ], style={'display': 'flex', 'justify-content': 'space-between'}),

            ])
        
        elif tab == 'saved-plots-tab':
            if not saved_plots:
                return html.Div("No plots saved.")
            # return html.Div([
            #     # Button to download saved plots as PDF
            #     html.Button("Download Saved Plots as PDF", id="download-pdf-btn", style={'margin-bottom': '10px'}),
                
            #     # Download component to trigger the download
            #     dcc.Download(id="download-pdf"),

            #     # Display each saved plot as a graph
            #     *[dcc.Graph(figure=plot) for plot in saved_plots]
            # ])
            
            return html.Div([
                dcc.Graph(figure=plot) for plot in saved_plots
            ])
    # def render_tab_content(tab, saved_plots):
    #     df = fetch_data()
    #     sorted_names = sorted(df['name'].unique(), key=natural_sort_key)
    #     if tab == 'main-tab':
    #         # Original dashboard content
    #         return html.Div([
    #             dcc.Dropdown(
    #                 id='sample-name-dropdown',
    #                 options=[{'label': name, 'value': name} for name in sorted_names],
    #                 multi=True
    #             ),
    #             html.Div([
    #                 html.Div(id='target-box', style={'padding': '10px', 'margin': '10px', 'display': 'inline-block', 'width': '15%'}),
    #                 html.Div(id='similar-experiments-box', style={'padding': '10px', 'margin': '10px', 'display': 'inline-block', 'width': '45%'}),
    #             ], style={'display': 'flex', 'justify-content': 'space-between'}),
    #             html.Div([
    #                 dcc.Graph(id='precursors-plot', style={'padding': '10px', 'margin': '10px','width': '45%', 'display': 'inline-block' }),
    #                 dcc.Graph(id='pie-plot', style={'padding': '10px', 'margin': '10px','width': '45%', 'display': 'inline-block'}),
    #             ], style={'display': 'flex', 'flex-wrap': 'wrap','justify-content': 'space-between'}),
    #             html.Div([
    #                 dcc.Graph(id='temperature-plot', style={'padding': '10px', 'margin': '10px','width': '45%', 'display': 'inline-block'}),
    #                 dcc.Dropdown(
    #                     id='comparable-sample-dropdown',
    #                     options=[{'label': name, 'value': name} for name in df['name'].unique()],
    #                     placeholder="Select a comparable sample",
    #                     style={'width': '5%', 'height': '5%', 'font-size': '14px'}
    #                 ),
    #                 dcc.Graph(id='correlation-plot', style={'padding': '10px', 'margin': '10px','width': '45%', 'display': 'inline-block'}),
    #             ], style={'display': 'flex', 'flex-wrap': 'wrap','justify-content': 'space-between'}),
    #             html.Button("Save correlation Plot", id="save-plot-btn"),
    #             html.H2("XRD characterization", style={'text-align': 'left'}),
    #             html.Div([
    #                 html.Div([
    #                     dcc.Graph(id='xrd-plot', style={'padding': '10px', 'margin': '10px', 'width': '100%'})
    #                 ], style={'width': '40%', 'display': 'inline-block'}),
    #                 html.Div([
    #                     html.Div(id='best_rwp_box', style={'padding': '10px', 'margin': '10px'}),
    #                     html.Div(id='results_box', style={'padding': '10px', 'margin': '10px'})
    #                 ], style={'display': 'flex', 'flex-direction': 'column', 'height': '80%', 'width': '80%'})
    #             ], style={'display': 'flex'}),
    #             html.Button("Save XRD Plot", id="save-plot-btn"),
    #             html.H2("SEM characterization", style={'text-align': 'left'}),
    #             html.Div([
    #                 html.Img(id='image1', style={'width': '35%', 'display': 'inline-block'}),
    #                 html.Img(id='image2', style={'width': '35%', 'display': 'inline-block'}),
    #             ], style={'display': 'flex', 'flex-wrap': 'wrap', 'justify-content': 'space-between'}),
                
    #         ])
    #     elif tab == 'saved-plots-tab':
    #         # Display saved plots
    #         if not saved_plots:
    #             return html.Div("No plots saved.")
            
    #         return html.Div([
    #             dcc.Graph(figure=plot) for plot in saved_plots  # Display each saved plot
    #         ])


    # @app.callback(
    # Output('saved-plots', 'data'),
    # Input('save-plot-btn', 'n_clicks'),
    # State('xrd-plot', 'figure'),
    # # State('pie-plot', 'figure'),
    # State('correlation-plot','figure'),
    # # State('heating-plot','figure'),
    # State('saved-plots', 'data')
    # )
    # def save_plot(n_clicks, figure, saved_plots):
    #     if n_clicks is None:
    #         return saved_plots  # No clicks yet, so return existing saved plots
    #     if saved_plots is None:
    #         saved_plots = []  # Initialize if empty
    #     saved_plots.append(figure)  # Add the new figure to saved plots
    #     return saved_plots


    # Callback to render the tab content

           

    # Individual callbacks for saving each plot
    @app.callback(
        Output('saved-plots', 'data', allow_duplicate=True),
        Input('save-precursors-plot-btn', 'n_clicks'),
        State('precursors-plot', 'figure'),
        State('saved-plots', 'data'),
        prevent_initial_call=True
    )
    def save_precursors_plot(n_clicks, figure, saved_plots):
        if n_clicks:
            if saved_plots is None:
                saved_plots = []
            saved_plots.append(figure)
        return saved_plots

    
    @app.callback(
        Output('saved-plots', 'data', allow_duplicate=True),
        Input('save-xrd-plot-btn', 'n_clicks'),
        State('xrd-plot', 'figure'),
        State('saved-plots', 'data'),
        prevent_initial_call=True
    )
    def save_xrd_plot(n_clicks, figure, saved_plots):
        if n_clicks:
            if saved_plots is None:
                saved_plots = []
            saved_plots.append(figure)
        return saved_plots
    

    # Callback for saving the pie plot
    @app.callback(
        Output('saved-plots', 'data', allow_duplicate=True),
        Input('save-pie-plot-btn', 'n_clicks'),
        State('pie-plot', 'figure'),
        State('saved-plots', 'data'),
        prevent_initial_call=True
    )
    def save_pie_plot(n_clicks, figure, saved_plots):
        if n_clicks:
            if saved_plots is None:
                saved_plots = []
            saved_plots.append(figure)
        return saved_plots

    # Callback for saving the correlation plot
    @app.callback(
        Output('saved-plots', 'data', allow_duplicate=True),
        Input('save-correlation-plot-btn', 'n_clicks'),
        State('correlation-plot', 'figure'),
        State('saved-plots', 'data'),
        prevent_initial_call=True
    )
    def save_correlation_plot(n_clicks, figure, saved_plots):
        if n_clicks:
            if saved_plots is None:
                saved_plots = []
            saved_plots.append(figure)
        return saved_plots

    # Callback for saving the temperature plot
    @app.callback(
        Output('saved-plots', 'data', allow_duplicate=True),
        Input('save-temperature-plot-btn', 'n_clicks'),
        State('temperature-plot', 'figure'),
        State('saved-plots', 'data'),
        prevent_initial_call=True
    )
    def save_temperature_plot(n_clicks, figure, saved_plots):
        if n_clicks:
            if saved_plots is None:
                saved_plots = []
            saved_plots.append(figure)
        return saved_plots

    import pdfkit
    import tempfile
    import plotly.io as pio
   

    @app.callback(
        Output("download-pdf", "data"),
        Input("download-pdf-btn", "n_clicks"),
        State("saved-plots", "data"),
        prevent_initial_call=True
    )
    def download_pdf(n_clicks, saved_plots):
        if not saved_plots:
            return None

        # HTML content initialization
        html_content = "<html><head><style>body { font-family: Arial; }</style></head><body>"

        # Save each Plotly figure as a temporary PNG file and add it to the HTML
        image_paths = []
        for i, plot in enumerate(saved_plots):
            # Save each figure as a PNG file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_image:
                pio.write_image(plot, tmp_image.name)
                image_paths.append(tmp_image.name)
                # Add image path to HTML content
                html_content += f'<h3>Plot {i+1}</h3><img src="{tmp_image.name}" style="width:600px;"><br><br>'

        html_content += "</body></html>"

        # Save HTML content to a temporary file and convert it to PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_html:
            tmp_html.write(html_content.encode("utf-8"))
            tmp_html.flush()
            pdf_path = tempfile.mktemp(suffix=".pdf")
            pdfkit.from_file(tmp_html.name, pdf_path)

        # Serve the PDF file for download
        return dcc.send_file(pdf_path)