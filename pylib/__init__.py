import seaborn as sns
import metapack as mp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display

from typing import List, Dict, Tuple, Any, Optional, Union

import re
import xml.etree.ElementTree as ET
import pandas as pd

def codebook_path(reference):
    """Return the URL for the codebook XML file"""
    surl = reference.resolved_url
    base_name = surl.resource_file.split('.')[0]
    return surl.set_target_file(base_name + '_codebook.xml').get_target().fspath


def variables(resource, doc, env, *args, **kwargs):
    """ Row generator function for extracting variables.

    Reference this function in a Metatab file as the value of a Datafile:

            Datafile: python:pylib#row_generator

    The function must yield rows, with the first being headers, and subsequenct rows being data.

    :param resource: The Datafile term being processed
    :param doc: The Metatab document that contains the term being processed
    :param args: Positional arguments passed to the generator
    :param kwargs: Keyword arguments passed to the generator
    :return:


    The env argument is a dict with these environmental keys:

    * CACHE_DIR
    * RESOURCE_NAME
    * RESOLVED_URL
    * WORKING_DIR
    * METATAB_DOC
    * METATAB_WORKING_DIR
    * METATAB_PACKAGE

    It also contains key/value pairs for all of the properties of the resource.

    """

    return extract_varis(doc)

def extract_varis(pkg):
    # Open up the ZIP file and get the codebook

    # The Path has some useful information about the variable.

    varis = pkg.reference('variables').dataframe()

    varis['group'] = varis.index.tolist()

    varis = varis.set_index(['TYPE', 'CATEGORY', 'TEXT', 'HEAD_WIFE', 'VAR_COUNT', 'group'])
    varis = varis.stack().to_frame('NAME')

    varis = varis.reset_index()
    varis.columns = ['type', 'category', 'text', 'head_wife', 'var_count', 'group', 'year', 'name']
    varis['name'] = varis.name.replace('', None)
    varis = varis.dropna(subset=['name'])
    varis['path'] = varis.text.apply(clean_text)
    varis['year'] = varis.year.apply(lambda v: int(v[1:]))

    return varis

race_merges = {
    9:np.nan, # DK; NA; Refused
    8:np.nan,
    0:np.nan,
    6:7,
}

race_map = {
    1: 'white',
    2: 'black',
    3: 'aian',
    4: 'aapi',
    5: 'hisp',
    7: 'other'
}

def cmp9919_labels(resource, doc, env, *args, **kwargs):
    return extract_code_labels(codebook_path(doc.reference('cmp9919_source')))

def long_wealth_labels(resource, doc, env, *args, **kwargs):
    df = extract_code_labels(codebook_path(doc.reference('long_wealth_source')))

    # Add conversion for a constructed race variable. Using this requires mapping race codes first,
    # using race_merges
    rows = [{'column': 'race', 'code': k, 'category': v} for k, v in race_map.items()]
    df = pd.concat([df, pd.DataFrame(rows)])

    return df

def cmp9919_dd(resource, doc, env, *args, **kwargs):
    varis = extract_varis(doc)
    return extract_metadata(varis, doc.reference('cmp9919_source'))

def long_wealth_dd(resource, doc, env, *args, **kwargs):
    varis = extract_varis(doc)
    return extract_metadata(varis, doc.reference('long_wealth_source'))


def open_dbf(reference):

    surl   = reference.resolved_url
    base_name = surl.resource_file.split('.')[0]
    database_url = surl.set_target_file(base_name+'.dbf')

    from simpledbf import Dbf5
    dbf = Dbf5(database_url.get_target().fspath)
    df = dbf.to_dataframe()

    # Make everything as numerical as possible
    for c in df.columns:
        df[c] = pd.to_numeric(df[c])

    return df.copy()

def xml_to_dataframe(xml_file):
    """Extract variable metadata from the XML version of the codebook"""
    # Parse the XML file

    try:
        tree = ET.parse(xml_file)
    except ET.ParseError:
        print(f'Failed to parse {xml_file}')
        raise

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

    def  number_maybe(v):

        try:
            v = v.replace(',','')
        except AttributeError:
            return v # Either None or a number

        try:
            return int(v)
        except ValueError:
            pass

        try:
            return float(v)
        except ValueError:
            pass

        return v

    def split_codes(code:str):
        if ' - ' in code:
            low, high = code.split(' - ')
            code = None
        else:
            low, high = None, None

        return {
            'code': number_maybe(code),
            'low_code': number_maybe(low),
            'high_code': number_maybe(high)
        }

    # Loop through the VARIABLES
    for variable in root.findall('.//VARIABLE'):
        var_name = variable.find('NAME').text
        # For each variable, loop through the codes
        for code in variable.findall('.//CODE'):
            # Create a dictionary with variable name, code value, and code text
            data_dict = {
                'column': var_name.lower(),
                **split_codes(code.find('VALUE').text),
                'category': code.find('TEXT').text,
            }
            # Append the dictionary to the list
            data_list.append(data_dict)

    # Create a DataFrame from the list of dictionaries
    return pd.DataFrame(data_list)




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


def update_schema(resource, mdf_sel, df, extra_terms= []):
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

    pkg = resource.doc
    schema_term = resource.schema_term

    ss = pkg['Schema']
    ss.args = ['DataType', 'Year', 'Label', 'Category', 'Description', 'Has_codes', 'Labels']

    # Clears out all of the schema children.

    for c in schema_term.children:
        ss.remove_term(c)

    schema_term.children = []

    for t in extra_terms:
        schema_term.new_child('Table.Column', **t)


    for idx, c in mdf_sel.iterrows():

        t = schema_term.new_child('Table.Column', c['name'].lower())
        t['DataType'] = 'number' if df[c['name']].dtype == 'float64' else 'integer'
        t['Year'] = c.year
        t['Label'] = c.label
        t['Category'] = c.category
        t['Description'] = (c.qtext + ": " + c.etext).strip()

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




def extract_metadata(varis, reference):
    # XML Codebook is the main source of variable information

    mdf = xml_to_dataframe(codebook_path(reference))
    mdf['YEAR'] = pd.to_numeric(mdf.YEAR)
    mdf.columns = [c.lower() for c in mdf.columns]
    mdf = mdf.merge(varis, on=('year', 'name'))

    mdf['qtext'] = mdf.qtext.apply(lambda v: v.replace('\n', ' '))
    mdf['etext'] = mdf.etext.apply(lambda v: v.replace('\n', ' '))

    return mdf

def extract_labels(reference):

    labels = extract_code_labels(codebook_path(reference))
    labels = labels.rename(columns={'name': 'column', 'value': 'code'})
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
            assert False  # Looks like this never actually happens
            r['value'] = code

        rows.append(r)

    labels = pd.DataFrame(rows)
    labels = labels[['column', 'code', 'low', 'high', 'label']]

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

    mc_idx = labels.column.isin(mixed_cols) & (~labels.code.isnull())

    labels.loc[mc_idx, 'low'] = labels.code
    labels.loc[mc_idx, 'high'] = labels.code
    labels.loc[mc_idx, 'code'] = np.nan

    return labels

def reorder_schema(resource, mdf, df):
    """Reorder the schema  for a resource.
    mdf can be either varis or mdf from extract_metadata"""

    pkg = resource.doc

    mdf_sel = mdf[mdf.name.isin(df.columns)].sort_values(['category', 'label', 'year'])

    init_cols = mdf_sel[mdf_sel.year == 1968].name.to_list()

    is_init = mdf_sel['name'].isin(init_cols)
    is_si = mdf_sel.category == 'SURVEY INFORMATION'

    a = mdf_sel.loc[is_init]
    b = mdf_sel.loc[~is_init & is_si]
    c = mdf_sel.loc[~is_init & ~is_si]

    mdf_sel = pd.concat([a, b, c])

    return mdf_sel



class Metadata(object):

    def __init__(self,  resource=None, varis=None, mdf=None):

        self.pkg = resource.doc
        self.resource = resource

        try:
            self.pkg['Sources'] # Source packages don't have this section
            is_build = True
        except KeyError:
            is_build = False

        self.varis = varis if varis is not None else (
                self.pkg.resource('variables').dataframe() if is_build
                else extract_varis(self.pkg) )

        self.mdf = mdf if mdf is not None else (
            self.pkg.resource(self.resource.name+'_dd').dataframe() if is_build
            else extract_metadata(self.varis, self.pkg.reference(self.resource.name+'_source')) )

    def find_year(self, var_name, year=None):
        """Find equivalent variables in other years"""

        varis = self.varis

        name_rec = varis[varis['name'] == var_name.upper()]

        if len(name_rec):
            name_rec = name_rec.iloc[0]
        else:
            return None

        t = varis[varis.group == name_rec.group]

        if isinstance(year, (list, tuple)):
            df = t[t.year.isin(year)]
        elif isinstance(year, int):
            df = t[t.year == year]
        else:
            df = t

        return df

    def translate_year(self, vrs, year):
        out = []

        if not isinstance(vrs, list):
            vrs = [vrs]

        for v in vrs:
            r = self.find_year(v, year)
            if r is not None and len(r) > 0:
                out.append(r.name.values[0])

        return out

    def lookup(self, vrs: List[str]):
        """Lookup variables in the varis. Less information, but returns
        info for every variable in PSID"""

        if not isinstance(vrs, list):
            vrs = [vrs]

        vrs_u = [e.upper() for e in vrs]
        return self.varis[self.varis['name'].isin(vrs_u)]

    def codebook(self, vrs: List[str]):
        """Like lookup, but returns info from the  more detailed metadata for this extract,
        based on the XML codebook"""

        if not isinstance(vrs, list):
            vrs = [vrs]

        vrs_u = [e.upper() for e in vrs]
        return self.mdf[self.mdf['name'].isin(vrs_u)]

    def search_label(self, v, year = None):
        """Search for a variable by label"""

        v = v.lower()

        df=  self.mdf[self.mdf.label.str.lower().str.contains(v)]

        if year is not None:
            df = df[df.year == year]

        return df


    def vars_for_year(self, year, upper=False):

        assert self.resource is not None

        all_cols = [e['name'] for e in self.resource.columns()]
        meta = self.lookup(all_cols)

        v = [e for e in meta[meta.year == year]['name'].values]

        if upper:
            return [e.upper() for e in v]
        else:
            return v


