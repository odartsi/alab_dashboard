from dash import html
import math
from data import fetch_data, fetch_data2
import plotly
import matplotlib.colors as mc
import colorsys
import numpy as np
import plotly.graph_objs as go
import json

periodic_table_layout = {
    'H': (1, 7), 'He': (18, 7),
    'Li': (1, 6), 'Be': (2, 6), 'B': (13, 6), 'C': (14, 6), 'N': (15, 6), 'O': (16, 6), 'F': (17, 6), 'Ne': (18, 6),
    'Na': (1, 5), 'Mg': (2, 5), 'Al': (13, 5), 'Si': (14, 5), 'P': (15, 5), 'S': (16, 5), 'Cl': (17, 5), 'Ar': (18, 5),
    'K': (1, 4), 'Ca': (2, 4), 'Sc': (3, 4), 'Ti': (4, 4), 'V': (5, 4), 'Cr': (6, 4), 'Mn': (7, 4), 'Fe': (8, 4), 
    'Co': (9, 4), 'Ni': (10, 4), 'Cu': (11, 4), 'Zn': (12, 4), 'Ga': (13, 4), 'Ge': (14, 4), 'As': (15, 4), 
    'Se': (16, 4), 'Br': (17, 4), 'Kr': (18, 4),
    'Rb': (1, 3), 'Sr': (2, 3), 'Y': (3, 3), 'Zr': (4, 3), 'Nb': (5, 3), 'Mo': (6, 3), 'Tc': (7, 3), 'Ru': (8, 3), 
    'Rh': (9, 3), 'Pd': (10, 3), 'Ag': (11, 3), 'Cd': (12, 3), 'In': (13, 3), 'Sn': (14, 3), 'Sb': (15, 3), 
    'Te': (16, 3), 'I': (17, 3), 'Xe': (18, 3),
    'Cs': (1, 2), 'Ba': (2, 2), 'La': (3, 2), 'Hf': (4, 2), 'Ta': (5, 2), 'W': (6, 2), 'Re': (7, 2), 'Os': (8, 2), 
    'Ir': (9, 2), 'Pt': (10, 2), 'Au': (11, 2), 'Hg': (12, 2), 'Tl': (13, 2), 'Pb': (14, 2), 'Bi': (15, 2), 
    'Po': (16, 2), 'At': (17, 2), 'Rn': (18, 2),
    'Fr': (1, 1), 'Ra': (2, 1), 'Ac': (3, 1), 'Rf': (4, 1), 'Db': (5, 1), 'Sg': (6, 1), 'Bh': (7, 1), 'Hs': (8, 1), 
    'Mt': (9, 1), 'Ds': (10, 1), 'Rg': (11, 1), 'Cn': (12, 1), 'Nh': (13, 1), 'Fl': (14, 1), 'Mc': (15, 1), 
    'Lv': (16, 1), 'Ts': (17, 1), 'Og': (18, 1),
    'Ce': (4, -1),  'Pr': (5, -1), 'Nd': (6, -1), 'Pm': (7, -1), 'Sm': (8, -1), 'Eu': (9, -1), 'Gd': (10, -1), 'Tb': (11, -1), 'Dy': (12, -1), 'Ho': (13, -1), 'Er': (14, -1), 'Tm': (15, -1), 'Yb': (16, -1), 'Lu': (17, -1),
    'Th': (4,  -2), 'Pa': (5,  -2), 'U': (6,  -2), 'Np': (7,  -2), 'Pu': (8,  -2), 'Am': (9,  -2), 'Cm': (10,  -2), 'Bk': (11,  -2), 'Cf': (12,  -2), 'Es': (13,  -2), 'Fm': (14,  -2), 'Md': (15,  -2), 'No': (16,  -2), 'Lr': (17, -2),
}

element_categories = {
    'H': 'nonmetal', 'He': 'noble gas',
    'Li': 'alkali metal', 'Be': 'alkaline earth metal', 'B': 'metalloid', 'C': 'nonmetal', 'N': 'nonmetal', 'O': 'nonmetal', 'F': 'halogen', 'Ne': 'noble gas',
    'Na': 'alkali metal', 'Mg': 'alkaline earth metal', 'Al': 'post-transition metal', 'Si': 'metalloid', 'P': 'nonmetal', 'S': 'nonmetal', 'Cl': 'halogen', 'Ar': 'noble gas',
    'K': 'alkali metal', 'Ca': 'alkaline earth metal', 'Sc': 'transition metal', 'Ti': 'transition metal', 'V': 'transition metal', 'Cr': 'transition metal', 'Mn': 'transition metal', 'Fe': 'transition metal', 
    'Co': 'transition metal', 'Ni': 'transition metal', 'Cu': 'transition metal', 'Zn': 'transition metal', 'Ga': 'post-transition metal', 'Ge': 'metalloid', 'As': 'metalloid', 
    'Se': 'nonmetal', 'Br': 'halogen', 'Kr': 'noble gas',
    'Rb': 'alkali metal', 'Sr': 'alkaline earth metal', 'Y': 'transition metal', 'Zr': 'transition metal', 'Nb': 'transition metal', 'Mo': 'transition metal', 'Tc': 'transition metal', 'Ru': 'transition metal', 
    'Rh': 'transition metal', 'Pd': 'transition metal', 'Ag': 'transition metal', 'Cd': 'transition metal', 'In': 'post-transition metal', 'Sn': 'post-transition metal', 'Sb': 'metalloid', 
    'Te': 'metalloid', 'I': 'halogen', 'Xe': 'noble gas',
    'Cs': 'alkali metal', 'Ba': 'alkaline earth metal', 'La': 'lanthanide', 'Hf': 'transition metal', 'Ta': 'transition metal', 'W': 'transition metal', 'Re': 'transition metal', 'Os': 'transition metal', 
    'Ir': 'transition metal', 'Pt': 'transition metal', 'Au': 'transition metal', 'Hg': 'transition metal', 'Tl': 'post-transition metal', 'Pb': 'post-transition metal', 'Bi': 'post-transition metal', 
    'Po': 'metalloid', 'At': 'halogen', 'Rn': 'noble gas',
    'Fr': 'alkali metal', 'Ra': 'alkaline earth metal', 'Ac': 'actinide', 'Rf': 'transition metal', 'Db': 'transition metal', 'Sg': 'transition metal', 'Bh': 'transition metal', 'Hs': 'transition metal', 
    'Mt': 'transition metal', 'Ds': 'transition metal', 'Rg': 'transition metal', 'Cn': 'transition metal', 'Nh': 'post-transition metal', 'Fl': 'post-transition metal', 'Mc': 'post-transition metal', 
    'Lv': 'post-transition metal', 'Ts': 'halogen', 'Og': 'noble gas',
    'Ce': 'lanthanide',  'Pr': 'lanthanide','Nd' :'lanthanide' ,'Pm' :'lanthanide', 'Sm' :'lanthanide', 'Eu'  :'lanthanide','Gd' :'lanthanide', 'Tb' :'lanthanide','Dy' :'lanthanide', 'Ho' :'lanthanide', 'Er' :'lanthanide', 'Tm' :'lanthanide', 'Yb' :'lanthanide', 'Lu' : 'lanthanide' ,
    'Th': 'actinide', 'Pa' : 'actinide','U': 'actinide', 'Np': 'actinide','Pu': 'actinide', 'Am': 'actinide', 'Cm': 'actinide', 'Bk': 'actinide', 'Cf': 'actinide', 'Es': 'actinide', 'Fm': 'actinide', 'Md': 'actinide', 'No': 'actinide', 'Lr': 'actinide',

}

category_colors = {
    'alkali metal': 'rgba(255, 182, 193, 0.5)',  # light pink
    'alkaline earth metal': 'rgba(135, 206, 250, 0.5)',  # light blue
    'transition metal': 'rgba(144, 238, 144, 0.5)',  # light green
    'post-transition metal': 'rgba(221, 160, 221, 0.5)',  # light purple
    'metalloid': 'rgba(255, 228, 196, 0.5)',  # light beige
    'nonmetal': 'rgba(255, 255, 224, 0.5)',  # light yellow
    'halogen': 'rgba(255, 228, 225, 0.5)',  # light pinkish
    'noble gas': 'rgba(230, 230, 250, 0.5)',  # light lavender
    'lanthanide': 'rgba(176, 224, 230, 0.5)',  # light cyan
    'actinide': 'rgba(240, 230, 140, 0.5)',  # light khaki
}

def get_samples_with_same_precursors(selected_precursors,selected_sample):
    df = fetch_data()  # Fetch the cached data or query MongoDB if necessary
    matching_samples = []
    for index, row in df.iterrows():
        sample_precursors = row['metadata'].get('elements_present', [])
        if set(sample_precursors) == set(selected_precursors) and selected_sample != row['name']:
            matching_samples.append(row['name'])
    return matching_samples

def get_samples_with_same_target(target, selected_sample):
    df = fetch_data()  # Fetch the cached data or query MongoDB if necessary
    matching_samples = []
    for index, row in df.iterrows():
        sample_precursors = row['metadata'].get('target', 'Unknown')
        if set(sample_precursors) == set(target) and selected_sample != row['name']:
            matching_samples.append(row['name'])
    return matching_samples

def get_samples_with_same_powder(selected_precursors_powders):
    df = fetch_data()  # Fetch the cached data or query MongoDB if necessary
    matching_samples = []
    
    for index, row in df.iterrows():
        metadata = row.get('metadata', {})
        powder_dosing_results = metadata.get('powderdosing_results', [])

         # Get powder names from the current sample
        if isinstance(powder_dosing_results, dict):
            powders = powder_dosing_results.get('Powders', [])
        else:
        # Handle the case where powder_dosing_results is not a dictionary (optional)
            powders = []
      
        if len(powders)> 0:
            sample_powder_names  = [p['PowderName'] for p in powders]
        # sample_powder_names = [p['PowderName'] for p in powder_dosing_results.get('Powders', [])]
        else:
            sample_powder_names=[]
        # Check for exact match between selected and sample powder names
        if set(selected_precursors_powders) == set(sample_powder_names):
            matching_samples.append(row['name'])

    return matching_samples

# Utility function to create a list of clickable links
def create_clickable_list(items):
    return [html.A(item, href=f'/{item}', style={'display': 'block'}) for item in items]

# def create_clickable_list(items, show_score=False):
#     clickable_items = []
#     for item in items:
#         if isinstance(item, tuple):
#             experiment, score = item
#             clickable_items.append(
#                 html.Div([
#                     html.A(experiment, href=f'/{experiment}', style={'display': 'inline-block'}),
#                     html.Span(f' {score:.2f}' if show_score else '', style={'display': 'inline-block'})
#                 ])
#             )
#         else:
#             clickable_items.append(
#                 html.A(item, href=f'/{item}', style={'display': 'block'})
#             )
#     return clickable_items

# def create_clickable_list(similarity):
#     clickable_items = []
#     for experiment, _ in similarity:
#         link_id = f"link-{experiment}"
#         clickable_items.append(
#             html.A(
#                 experiment,
#                 href="#",  # Prevents default link behavior
#                 id={'type': 'clickable-link', 'index': link_id},
#                 style={'display': 'block'}
#             )
#         )
#     return clickable_items

# Calculate the similarity between two experiments
def calculate_individual_similarity(selected_temperature, selected_time, selected_target, selected_powder_names, selected_phases,
                                     sample_temperature, sample_time, sample_target, sample_powder_names, sample_phases,
                                     weight_temperature=0.6, weight_time=0.2, weight_target=0.1, weight_powders=0.1):
            
    # Similarity calculation logic
    try:
        temperature_distance = math.sqrt(sum((selected_temperature[t] - sample_temperature[t])**2 for t in selected_temperature))
        temperature_similarity = 1 - (temperature_distance / max(max(selected_temperature.values()), max(sample_temperature.values())))
    except:
        temperature_similarity = 0

    time_similarity = 1 - abs(selected_time - sample_time) / max(selected_time, sample_time)
    target_similarity = 1 if selected_target == sample_target else 0
    powder_similarity = 1 if set(selected_powder_names) == set(sample_powder_names) else 0
    phase_similarity = 1 if set(selected_phases) == set(sample_phases) else 0

    # print(f'phase_similarity =  {phase_similarity}', selected_phases,sample_phases)

    overall_similarity = (weight_temperature * temperature_similarity +
                          weight_time * time_similarity +
                          weight_target * target_similarity +
                          weight_powders * powder_similarity +
                          weight_powders * phase_similarity)

    similarity_score = overall_similarity / (weight_powders + weight_powders + weight_target + weight_time + weight_temperature)
    return similarity_score


def calculate_experiment_similarity(selected_data, selected_sample):
    df = fetch_data()  # Fetch the cached data or query MongoDB if necessary
    df2 = fetch_data2()  # Fetch the cached data or query MongoDB if necessary

    # Check if 'name' column exists in df2
    if 'name' not in df2.columns:
        print("Error: 'name' column not found in df2")
        return []

    # Precompute and cache information for all samples in df
    sample_info_cache = {}

    for index, row in df.iterrows():
        sample_data = row.to_dict()  # Convert DataFrame row to dictionary
        sample_name = sample_data.get("name")
        
        # Ensure 'metadata' is properly deserialized
        metadata = sample_data.get('metadata')
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                print(f"Error decoding metadata for sample {sample_name}")
                continue  # Skip this sample

        sample_info_cache[sample_name] = {
            "temperature": metadata.get('heating_results', {}).get('heating_temperature', {}),
            "time": metadata.get('heating_results', {}).get('heating_time', {}),
            "target": metadata.get('target', "Unknown"),
            "powder_names": [p['PowderName'] for p in metadata.get('powderdosing_results', {}).get('Powders', [])],
            "phases": []
        }

        # Fetch matching samples from df2
        if '_' in sample_name:
            base_name, sample_number = sample_name.split('_', 1)
        else:
            base_name = sample_name
            sample_number = None

        matching_samples = df2[df2['name'].str.contains(base_name, na=False)]
        if sample_number:
            matching_samples = matching_samples[matching_samples['name'].str.contains(sample_number, na=False)]

        data = {}
        for _, doc in matching_samples.iterrows():
            # Ensure 'metadata' is properly deserialized
            doc_metadata = doc['metadata']
            if isinstance(doc_metadata, str):
                try:
                    doc_metadata = json.loads(doc_metadata)
                except json.JSONDecodeError:
                    print(f"Error decoding metadata for document {doc['name']}")
                    continue  # Skip this document
            
            name = doc['name']
            fw_id = doc_metadata.get('fw_id')
            if name not in data or fw_id > data[name]['fw_id']:
                data[name] = {'doc': doc, 'fw_id': fw_id}

        for doc_info in data.values():
            doc = doc_info['doc']
            output = doc.get('output')
            # Check if output is a string and needs parsing
            if isinstance(output, str):
                try:
                    output = json.loads(output)
                except json.JSONDecodeError:
                    print(f"Error decoding output for document {doc['name']}")
                    continue  # Skip this document

            final_result = output.get('final_result', {})
            if isinstance(final_result, str):
                try:
                    final_result = json.loads(final_result)
                except json.JSONDecodeError:
                    print(f"Error decoding final_result for document {doc['name']}")
                    continue  # Skip this document
            
            if final_result:
                lst_data = final_result.get('lst_data', {})
                if isinstance(lst_data, str):
                    try:
                        lst_data = json.loads(lst_data)
                    except json.JSONDecodeError:
                        print(f"Error decoding lst_data for document {doc['name']}")
                        continue  # Skip this document
                
                phases_results = lst_data.get('phases_results', [])
                if phases_results:
                    sample_info_cache[sample_name]["phases"] = list(phases_results)

    # Extract information from the selected data
    metadata = selected_data.get('metadata', {})
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            print(f"Error decoding metadata for selected sample {selected_sample}")
            return []  # Return an empty list

    selected_temperature = metadata.get('heating_results', {}).get('heating_temperature', {})
    selected_time = metadata.get('heating_results', {}).get('heating_time', {})
    selected_target = metadata.get('target', "Unknown")
    selected_powder_names = [p['PowderName'] for p in metadata.get('powderdosing_results', {}).get('Powders', [])]
    
    # Fetch phases for the selected sample
    selected_phases = []
    if selected_sample in sample_info_cache:
        selected_phases = sample_info_cache[selected_sample]["phases"]
    print("Selected phases: ", selected_phases)

    all_similarities = {}  # Dictionary to store sample name and similarity score pairs

    # Iterate through the precomputed sample information
    for sample_name, info in sample_info_cache.items():
        if sample_name == selected_sample:
            continue  # Exclude the selected sample from similarity calculation

        # Ensure the times are numeric
        selected_time_value = selected_time if isinstance(selected_time, (int, float)) else 0
        sample_time_value = info["time"] if isinstance(info["time"], (int, float)) else 0

        # Calculate phase similarity (Jaccard similarity)
        selected_phases_set = set(selected_phases)
        sample_phases_set = set(info["phases"])
        
        intersection = selected_phases_set & sample_phases_set
        union = selected_phases_set | sample_phases_set
        phase_similarity = len(intersection) / len(union) if union else 0

        # Weights for similarity calculation
        weight_temperature = 0.2
        weight_time = 0.2
        weight_target = 0.1
        weight_powders = 0.1
        weight_phases = 0.4

        try:
            temperature_distance = math.sqrt(sum((selected_temperature[t] - info["temperature"][t])**2 for t in selected_temperature))
            temperature_similarity = 1 - (temperature_distance / max(max(selected_temperature.values()), max(info["temperature"].values())))
        except:
            temperature_similarity = 0

        time_similarity = (1 - abs(selected_time_value - sample_time_value) / (selected_time_value + sample_time_value + 1))
        target_similarity = 1 if selected_target == info['target'] else 0
        powder_similarity = 1 if set(selected_powder_names) == set(info["powder_names"]) else 0
        

        # print(f'phase_similarity =  {phase_similarity}', selected_phases,sample_phases)

        overall_similarity = (weight_temperature * temperature_similarity +
                            weight_time * time_similarity +
                            weight_target * target_similarity +
                            weight_powders * powder_similarity +
                            weight_phases * phase_similarity)

        similarity_score = overall_similarity / (weight_powders + weight_powders + weight_target + weight_time + weight_temperature)
        

        # Add sample name and similarity score to the dictionary
        all_similarities[sample_name] = similarity_score

    # Assuming all_similarities is a dictionary with sample names as keys and similarity scores as values
    top_n = 3  # Number of most similar samples to display

    # Sort samples by similarity score in descending order (most similar first)
    sorted_similarities = sorted(all_similarities.items(), key=lambda x: x[1], reverse=True)

    print(f"Top {top_n} Most Similar Samples:")
    for i in range(min(top_n, len(sorted_similarities))):
        sample, score = sorted_similarities[i]
        print(f"\t- {sample}: {score:.2f}")  # Format score with 2 decimal places

    return sorted_similarities[:top_n]
    

def calculate_experiment_similarity_old(selected_data, selected_sample):
    df = fetch_data()  # Fetch the cached data or query MongoDB if necessary
    df2 = fetch_data2()  # Fetch the cached data or query MongoDB if necessary

    # Check if 'name' column exists in df2
    if 'name' not in df2.columns:
        print("Error: 'name' column not found in df2")
        return []

    # Precompute and cache information for all samples in df
    sample_info_cache = {}

    for index, row in df.iterrows():
        sample_data = row.to_dict()  # Convert DataFrame row to dictionary
        sample_name = sample_data.get("name")
        
        # Ensure 'metadata' is properly deserialized
        metadata = sample_data.get('metadata')
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                print(f"Error decoding metadata for sample {sample_name}")
                continue  # Skip this sample

        sample_info_cache[sample_name] = {
            "temperature": metadata.get('heating_results', {}).get('heating_temperature', {}),
            "time": metadata.get('heating_results', {}).get('heating_time', {}),
            "target": metadata.get('target', "Unknown"),
            "powder_names": [p['PowderName'] for p in metadata.get('powderdosing_results', {}).get('Powders', [])],
            "phases": []
        }

        # Fetch matching samples from df2
        if '_' in sample_name:
            base_name, sample_number = sample_name.split('_', 1)
        else:
            base_name = sample_name
            sample_number = None

        matching_samples = df2[df2['name'].str.contains(base_name, na=False)]
        if sample_number:
            matching_samples = matching_samples[matching_samples['name'].str.contains(sample_number, na=False)]

        data = {}
        for _, doc in matching_samples.iterrows():
            # Ensure 'metadata' is properly deserialized
            doc_metadata = doc['metadata']
            if isinstance(doc_metadata, str):
                try:
                    doc_metadata = json.loads(doc_metadata)
                except json.JSONDecodeError:
                    print(f"Error decoding metadata for document {doc['name']}")
                    continue  # Skip this document
            
            name = doc['name']
            fw_id = doc_metadata.get('fw_id')
            if name not in data or fw_id > data[name]['fw_id']:
                data[name] = {'doc': doc, 'fw_id': fw_id}

        for doc_info in data.values():
            doc = doc_info['doc']
            output = doc.get('output')
            # Check if output is a string and needs parsing
            if isinstance(output, str):
                try:
                    output = json.loads(output)
                except json.JSONDecodeError:
                    print(f"Error decoding output for document {doc['name']}")
                    continue  # Skip this document

            final_result = output.get('final_result', {})
            if isinstance(final_result, str):
                try:
                    final_result = json.loads(final_result)
                except json.JSONDecodeError:
                    print(f"Error decoding final_result for document {doc['name']}")
                    continue  # Skip this document
            
            if final_result:
                lst_data = final_result.get('lst_data', {})
                if isinstance(lst_data, str):
                    try:
                        lst_data = json.loads(lst_data)
                    except json.JSONDecodeError:
                        print(f"Error decoding lst_data for document {doc['name']}")
                        continue  # Skip this document
                
                phases_results = lst_data.get('phases_results', [])
                if phases_results:
                    sample_info_cache[sample_name]["phases"] = list(phases_results)

    # Extract information from the selected data
    metadata = selected_data.get('metadata', {})
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            print(f"Error decoding metadata for selected sample {selected_sample}")
            return []  # Return an empty list

    selected_temperature = metadata.get('heating_results', {}).get('heating_temperature', {})
    selected_time = metadata.get('heating_results', {}).get('heating_time', {})
    selected_target = metadata.get('target', "Unknown")
    selected_powder_names = [p['PowderName'] for p in metadata.get('powderdosing_results', {}).get('Powders', [])]
    
    # Fetch phases for the selected sample
    selected_phases = []
    if selected_sample in sample_info_cache:
        selected_phases = sample_info_cache[selected_sample]["phases"]
    print("in similarity :", selected_phases)

    all_similarities = {}  # Dictionary to store sample name and similarity score pairs

    # Iterate through the precomputed sample information
    for sample_name, info in sample_info_cache.items():
        if sample_name == selected_sample:
            continue  # Exclude the selected sample from similarity calculation

        # Ensure the times are numeric
        selected_time_value = selected_time if isinstance(selected_time, (int, float)) else 0
        sample_time_value = info["time"] if isinstance(info["time"], (int, float)) else 0

        similarity_score = calculate_individual_similarity(
            selected_temperature, selected_time_value, selected_target, selected_powder_names, selected_phases,
            info["temperature"], sample_time_value, info["target"], info["powder_names"], info["phases"]
        )

        # Add sample name and similarity score to the dictionary
        all_similarities[sample_name] = similarity_score

    # Assuming all_similarities is a dictionary with sample names as keys and similarity scores as values
    top_n = 3  # Number of most similar samples to display

    # Sort samples by similarity score in descending order (most similar first)
    sorted_similarities = sorted(all_similarities.items(), key=lambda x: x[1], reverse=True)

    print(f"Top {top_n} Most Similar Samples:")
    for i in range(min(top_n, len(sorted_similarities))):
        sample, score = sorted_similarities[i]
        print(f"\t- {sample}: {score:.2f}")  # Format score with 2 decimal places

    return sorted_similarities[:top_n]


def darken_color(color, factor=0.9):
    if 'rgba' in color:
        color = color.replace('rgba', '').replace('(', '').replace(')', '').split(',')
        color = [int(c) for c in color[:3]]  # Take only the RGB part
        color = tuple([c/255 for c in color])  # Normalize RGB values
    elif 'rgb' in color:
        color = color.replace('rgb', '').replace('(', '').replace(')', '').split(',')
        color = [int(c) for c in color[:3]]  # Take only the RGB part
        color = tuple([c/255 for c in color])  # Normalize RGB values
    else:
        try:
            color = mc.cnames[color]
        except:
            color = color
        color = mc.to_rgb(color)
        
    color = colorsys.rgb_to_hls(*color)
    darkened_color = colorsys.hls_to_rgb(color[0], max(0, min(1, factor * color[1])), color[2])
    darkened_color = tuple(int(c * 255) for c in darkened_color)  # Denormalize RGB values
    return 'rgb' + str(darkened_color)

def convert_time_and_sync_temperature(time_minutes, temperature_celsius):
    # Convert time_minutes to time_hours
    time_hours = [round(t / 60, 2) for t in time_minutes]
    
    # Filter the lists to include only whole hours
    filtered_time_hours = []
    filtered_temperature_celsius = []
    
    for i, t in enumerate(time_hours):
        filtered_time_hours.append(t)
        filtered_temperature_celsius.append(temperature_celsius[i])
    
    return filtered_time_hours, filtered_temperature_celsius

def get_number(s):
    """Get the number from a float or tuple of floats."""
    if isinstance(s, (tuple, list)):
        return s[0]
    else:
        return s

def get_phase_weights(result, normalize=True):
    """Return the weights for each phase. Default is to normalize and return weight fractions."""
    weights = {}
    for phase, data in result["lst_data"]["phases_results"].items():
        weights[phase] = get_number(data["gewicht"])
    if normalize:
        tot = np.sum([v for v in weights.values() if isinstance(v, (int, float))])
        weights = {k: v / tot for k, v in weights.items()}
    return dict(sorted(weights.items(), key=lambda item: item[1], reverse=True))

def get_powder_data(metadata):
    powder_dosing_results = metadata.get('powderdosing_results',[])
    powders = powder_dosing_results.get('Powders', [])
    powder_names = [p['PowderName'] for p in powders]
    target_masses = [p['TargetMass'] for p in powders]
    
    return powder_names, target_masses

def get_sample_correlations(df, selected_samples):
    correlations = []
    for selected_sample in selected_samples:
        selected_sample_data = df[df['name'] == selected_sample].iloc[0]
        selected_elements = set(selected_sample_data['metadata'].get('elements_present', []))

        for _, row in df.iterrows():
            sample_elements = set(row['metadata'].get('elements_present', []))
            if selected_elements == sample_elements:
                correlations.append((row['name'], row['metadata'].get('target', 'Unknown')))
    
    return correlations


# Function to create color mapping
def create_color_mapping(targets):
    unique_targets = list(set(targets))
    color_map = {}
    colors = ['#26DBDB', '#b2d8d8', '#008080', '#eecbff', '#004c4c', '#005073', '#189ad3', '#1ebbd7'] #b2d8d8
    for i, target in enumerate(unique_targets):
        color_map[target] = colors[i % len(colors)]
    return color_map

# Function to create the correlation plot for multiple samples
def create_correlation_plot(selected_samples):
    df = fetch_data()  # Fetch the cached data or query MongoDB if necessary
    if df.empty or not selected_samples:
        return go.Figure()

    correlations = get_sample_correlations(df, selected_samples)
    
    if not correlations:
        return go.Figure()

    sample_names, targets = zip(*correlations)
    color_map = create_color_mapping(targets)
    colors = [color_map[target] for target in targets]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=targets,
        y=sample_names,
        mode='markers+text',
        marker=dict(color=colors, size=10),
        text=sample_names,
        textposition='top center',
        name='Sample Correlations'
    ))
    
    fig.update_layout(
        title='Correlation Plot',
        xaxis=dict(title='Target'),
        yaxis=dict(title='Sample Name'),
        # plot_bgcolor='rgb(245, 245, 240)',
        # paper_bgcolor='rgb(255, 255, 255)'
    )
    return fig