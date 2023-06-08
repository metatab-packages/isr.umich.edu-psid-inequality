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

def codebook_path(pkg):
    """Return the URL for the codebook XML file"""
    surl = pkg.reference('source').resolved_url
    base_name = surl.resource_file.split('.')[0]
    return surl.set_target_file(base_name+'_codebook.xml').get_target().fspath

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

def update_schema(pkg, schema_term, mdf_sel, df):
    """

    Update the metadata schema for a datafile, based on the metadata for the selected columns.

    :param schema_term: Schema term for the datafiel to update
    :type schema_term: Term
    :param mdf_sel: metadata for the set of columns
    :type mdf_sel: Dataframe
    :param df: Source dataframe
    :type df: Dataframe
    :return: Re-ordered dataframe
    :rtype: Dataframe
    """

    ss = pkg['Schema']
    ss.args = ['DataType',  'Year', 'Label', 'Category', 'Description', 'Has_codes', 'Labels']

    # Clears out all of the schema children.
    for c in schema_term.children:
        ss.remove_term(c)

    schema_term.children = []

    schema_term.new_child('Table.Column','pid', DataType='integer',Description='Constructed person id')

    for idx, c in mdf_sel.iterrows():
        t = schema_term.new_child('Table.Column', c['name'].lower())
        t['DataType'] = 'number' if df[c['name']].dtype == 'float64' else 'integer'
        t['Year'] = c.year
        t['Label'] = c.label
        t['Category'] = c.category
        t['Description'] = (c.qtext+": "+c.etext).strip()

    pkg.write()

    final_cols = ['pid'] + list(mdf_sel.name)

    return df[final_cols]


def null_map(title):
    cols = mdf[mdf.category == title].name.unique()
    t = df[cols]

    fig, ax = plt.subplots(figsize=(15, .25 * len(t.columns)))
    sns.heatmap(t.isnull().T, cbar=False, xticklabels=False, cmap='viridis', ax=ax)

    plt.suptitle(title + ' Nullmap', fontsize=20)
    plt.title('Yellow indicates missing data records')
    plt.tight_layout()


def extract_metadata(pkg):

    # Open up the ZIP file and get the codebook

    # The Path has some useful information about the variable.
    varis = pkg.reference('variables').dataframe()

    varis['group'] = varis.index.tolist()

    varis = varis.set_index(['TYPE', 'CATEGORY', 'TEXT', 'HEAD_WIFE', 'VAR_COUNT', 'group'])
    varis = varis.stack().to_frame('NAME')

    varis = varis.reset_index()
    varis.columns = ['type','category','text','head_wife','var_count','group', 'year','name']
    varis['name'] = varis.name.replace('',None)
    varis = varis.dropna(subset=['name'])
    varis['path'] = varis.text.apply(clean_text)
    varis['year'] = varis.year.apply(lambda v: int(v[1:]) )

    # XML Codebook is the main source of variable information

    mdf = xml_to_dataframe(codebook_path(pkg))
    mdf['YEAR'] = pd.to_numeric(mdf.YEAR)
    mdf.columns = [c.lower() for c in mdf.columns]
    mdf = mdf.merge(varis, on= ('year','name' ))

    mdf['qtext'] = mdf.qtext.apply(lambda v: v.replace('\n',' ') )
    mdf['etext'] = mdf.etext.apply(lambda v: v.replace('\n',' ') )

    return varis, mdf


def extract_labels(pkg):

    labels = extract_code_labels(codebook_path(pkg))
    labels = labels.rename(columns={'name':'column','value':'code'})
    labels['column'] = labels['column'].str.lower()

    rows = []
    for idx, r in labels.iterrows():
        code = clean_code(r.code)

        if isinstance(code, list):

            r['code'] = np.nan
            r['low'], r['high'] = code
        elif isinstance(code, int):
            r['code'] = code
        else:
            assert False # Looks like this never actually happens
            r['value'] = code

        rows.append(r)

    labels = pd.DataFrame(rows)
    labels = labels[['column','code', 'low','high','label']]

    #
    # for columns where there is a mix of codes and high/low, move everything
    # over to value/high/low
    #

    mixed_cols = []

    for gn, g in labels.groupby('column'):

        # Either all of the codes are Nan, or all of the High/Low are nan,
        # but there should never be some of each

        code_nans = g.code.isnull().sum()
        low_nans = g.low.isnull().sum()
        high_nans = g.high.isnull().sum()

        assert low_nans == high_nans

        if code_nans > 0 and low_nans > 0:
            mixed_cols.append(gn)

    mc_idx = labels.column.isin(mixed_cols)& (~labels.code.isnull())


    labels.loc[mc_idx,'low' ] = labels.code
    labels.loc[mc_idx,'high' ] = labels.code
    labels.loc[mc_idx,'code' ] = np.nan

    return labels


def find_year(varis, var_name, year=None):
    """Find equivalent variables in other years"""

    name_rec = varis[varis['name'] == var_name.upper()].iloc[0]

    t = varis[varis.group == name_rec.group]
    if isinstance(year, (list, tuple)):
        df = t[t.year.isin(year)]
    elif isinstance(year, int):
        df = t[t.year == year]
    else:
        df =  t

    return df

def translate_year(vars, year):
    out = []

    for v in vars:
        out.append( find_year(varis, v, year).name.values[0] )

    return out

def lookup_vars(varis, vars):
    return varis[varis['name'].isin([e.upper() for e in vars])]

def vars_for_year(varis, resource, year, upper=False):
    all_cols = [e['name'] for e in resource.columns()]
    meta = lookup_vars(varis, all_cols)

    v = [e for e in meta[meta.year == year]['name'].values]

    if upper:
        return [e.upper() for e in v]
    else:
        return v


def reorder_schema(r, df):
    """Reorder the schema  for a resource"""

    mdf_sel = mdf[mdf.name.isin(df.columns)].sort_values(['category', 'label', 'year'])

    # We're only using the 1968 variables for family and person numbers, so move them to the top

    init_cols = vars_for_year(varis, r, 1968, upper=True)

    is_init = mdf_sel['name'].isin(init_cols)
    is_si = mdf_sel.category == 'SURVEY INFORMATION'

    a = mdf_sel.loc[ is_init ]
    b = mdf_sel.loc[~is_init &  is_si]
    c = mdf_sel.loc[~is_init & ~is_si]

    mdf_sel = pd.concat([a,b,c])

    display(mdf_sel)

    pkg = mp.jupyter.open_source_package()
    st = r.schema_term

    return update_schema(pkg, st, mdf_sel, df)

r = pkg.resource('psid_ineq')
df = reorder_schema(r, df)
