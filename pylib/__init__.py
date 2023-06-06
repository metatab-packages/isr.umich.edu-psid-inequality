import seaborn as sns
import metapack as mp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display

import re
import xml.etree.ElementTree as ET
import pandas as pd

def xml_to_dataframe(xml_file):
    """Extract variable metadata from the XML version of the codebook"""
    # Parse the XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()

    rows = []

    # Define the DataFrame columns
    columns = ["YEAR", "TYPE_ID", "NAME", "LABEL", "QTEXT", "ETEXT"]
    df = pd.DataFrame(columns=columns)

    # Iterate through each VARIABLE tag in the XML data
    for variable in root.iter('VARIABLE'):
        data = []
        # Extract the values of the child tags
        for column in columns:
            node = variable.find(column)
            if node is not None:
                data.append(node.text)
            else:
                data.append(None)
        # Append the data to the DataFrame
        rows.append(pd.Series(data, index=columns))

    return pd.DataFrame(rows)

def extract_code_labels(path):
    """Extract the value labels from the XML version of the codebook"""

    import xml.etree.ElementTree as ET
    import pandas as pd

    # Parse the XML file
    root = ET.parse(path).getroot()

    # Initialize an empty list for storing dictionaries
    data_list = []

    # Loop through the VARIABLES
    for variable in root.findall('.//VARIABLE'):
        var_name = variable.find('NAME').text
        # For each variable, loop through the codes
        for code in variable.findall('.//CODE'):
            # Create a dictionary with variable name, code value, and code text
            data_dict = {
                'name': var_name,
                'value': code.find('VALUE').text,
                'label': code.find('TEXT').text
            }
            # Append the dictionary to the list
            data_list.append(data_dict)

    # Create a DataFrame from the list of dictionaries
    return  pd.DataFrame(data_list)

def clean_text(s):
    """Clean up the texts and turn them into paths. """

    # Remove numbers
    s = re.sub(r'\d+', '', s)
    # Remove newlines
    s = s.replace('\n', '')
    # Convert to lower case
    s = s.lower()
    # Split by '>'
    parts = s.split('>')
    parts = [e.strip() for e in parts]
    # Join with '/' and append to the formatted list
    return '/'.join(parts)


#
# Split the codes that are actually ranges
#

def clean_code(v):
    if not isinstance(v, str):
        return v

    code = v.replace(',', '').strip()

    try:
        c = int(float(code))
        return c
    except ValueError:
        pass

    try:
        c = float(code)
        return c
    except ValueError:
        pass

    if '-' in code:
        l, h = [float(e) for e in code.split(' - ')]
        return [l, h]

    assert False


def null_map(title):
    cols = mdf[mdf.category == title].name.unique()
    t = df[cols]

    fig, ax = plt.subplots(figsize=(15, .25 * len(t.columns)))
    sns.heatmap(t.isnull().T, cbar=False, xticklabels=False, cmap='viridis', ax=ax)

    plt.suptitle(title + ' Nullmap', fontsize=20)
    plt.title('Yellow indicates missing data records')
    plt.tight_layout()