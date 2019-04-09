# ahrf_analysis.py


import numpy as np
import pandas as pd
import plotly.plotly as py
import plotly.figure_factory as ff
import plotly.graph_objs as go


# Setup global variables
DATA_DIR = '../data'
ASC_FILE = '{}/ahrf2018.asc'.format(DATA_DIR)
SAS_FILE = '{}/AHRF2017-18.sas'.format(DATA_DIR)

NAME = 'State Name'
STATE = 'FIPS State Code'
COUNTY = 'FIPS County Code'
FIPS = 'FIPS'
NE = ('Connecticut', 'Maine', 'Massachusetts', 'New Hampshire', 'Rhode Island', 'Vermont')


class AHRFDataAnalyzer(object):

    def __init__(self, asc_file, sas_file, normalize_data=False):
        self._asc_file = asc_file
        self._sas_file = sas_file
        self._normalize_data = normalize_data

    def parse_data(self):
        """
        Read fixed-width data file and parse into respective CSV files.
        """
        # Get fixed-width data widths
        widths = self._get_col_widths()
        # Get data labels
        labels = self._get_col_labels()
        # Load dataframe
        df = pd.read_fwf(self._asc_file, widths=widths, header=None)
        # Replace "." placeholder character in numeric data with NaN
        df = df.replace(r'^\.*$', np.nan, regex=True)
        # Category label and start columns (continuous)
        cats = [('info', 0),  # Meta info category
                ('codes', 8),  # Codes category
                ('profs', 76),  # Professions category
                ('facils', 3345),  # Facilities category
                ('util', 4098),  # Utilization category
                ('exps', 4233),  # Expenses category
                ('pop', 4373),  # Population category
                ('env', 7219)]  # Environmental category
        for i in range(len(cats)):
            cat = cats[i][0]
            start = cats[i][1]
            end = cats[i + 1][1] if i < len(cats) - 1 else 7278
            # Get columns for category from dataframe
            cat_df = df.iloc[:, start:end]
            if cat not in ('info', 'codes'):
                if self._normalize_data:
                    # Normalize category data
                    cat_norm = self._normalize(cat_df.values)
                    cat_df = pd.DataFrame(cat_norm, columns=cat_df.columns)
            # Save category data to CSV file
            cat_df.to_csv('{}/ahrf2018_{}.csv'.format(DATA_DIR, cat), header=labels[start:end], index=False)

    def get_environmental_data(self, column='Population Density per Sq Mile 2010', states=NE, normalize=False):
        """
        Plot environmental <column> data for selected <states>.
        """
        geocodes, results = self._collect_data('env', column, states, normalize)
        return geocodes, results

    def get_expenses_data(self, column='Total Actual Medicare Costs Fee for Service 2015', states=NE, normalize=False):
        """
        Plot expenses <column> data for selected <states>.
        """
        geocodes, results = self._collect_data('exps', column, states, normalize)
        return geocodes, results

    def get_facilities_data(self, column='Total Number Hospitals 2016', states=NE, normalize=False):
        """
        Plot facilities <column> data for selected <states>.
        """
        geocodes, results = self._collect_data('facils', column, states, normalize)
        return geocodes, results

    def get_population_data(self, column='Census Population 2010', states=NE, normalize=False):
        """
        Plot population <column> data for selected <states>.
        """
        geocodes, results = self._collect_data('pop', column, states, normalize)
        return geocodes, results

    def get_professions_data(self, column='Total Active M.D.s Non-Federal 2016', states=NE, normalize=False):
        """
        Plot professions <column> data for selected <states>.
        """
        geocodes, results = self._collect_data('profs', column, states, normalize)
        return geocodes, results

    def get_utilization_data(self, column='Inpatient Days Incl Nurs Home;Total Hosp 2016', states=NE, normalize=False):
        """
        Plot utilization <column> data for selected <states>.
        """
        geocodes, results = self._collect_data('util', column, states, normalize)
        return geocodes, results

    def _collect_data(self, category, column, states, normalize=False):
        """
        Get data from <category> CSV file for specified <column> and <states>. Plots and returns FIPS geocodes and
        <column> data.
        """
        # Get state names
        info = pd.read_csv('{}/ahrf2018_info.csv'.format(DATA_DIR))
        info = info[NAME]
        # Get FIPS codes for state and county
        codes = pd.read_csv('{}/ahrf2018_codes.csv'.format(DATA_DIR))
        codes = codes[[STATE, COUNTY]]
        codes[STATE] = codes[STATE].astype(str).str.zfill(3).values
        codes[COUNTY] = codes[COUNTY].astype(str).str.zfill(3).values
        # Concatenate state and county FIPS codes
        codes[FIPS] = codes[STATE].astype(str).str.zfill(3).values + codes[COUNTY].astype(str).str.zfill(3).values
        codes = codes[FIPS]
        # Get selected column data
        data = pd.read_csv('{}/ahrf2018_{}.csv'.format(DATA_DIR, category))
        data = data[column]
        # Concatenate state name, FIPS codes and column data into single DataFrame
        data = pd.concat([info, codes, data], axis=1)
        # Select only data rows for specified states
        data = data.loc[data[NAME].isin(states)]
        # Bin data by FIPS geocode
        results = data.groupby([FIPS]).sum()
        geocodes = results.index.values
        if normalize:
            # Normalize data
            results = self._normalize(results.values).flatten()
        else:
            results = results.values.flatten()
        # Build choropleth map
        return geocodes, results

    def _get_col_widths(self):
        """
        Get .asc file fixed column widths from .sas file.
        """
        widths = pd.read_fwf(self._sas_file, skiprows=6, nrows=7277, header=None)[3].astype(int).values.tolist()
        return widths

    def _get_col_labels(self):
        """
        Get .asc file column names from .sas file.
        """
        labels = pd.read_fwf(self._sas_file, skiprows=7285, header=None)[1].dropna().values.tolist()
        labels = [r.split('\"')[1] for r in labels]
        return labels

    @staticmethod
    def plot_data(fips, values, states, column, category):
        """
        Build geocoded choropleth map of <column> data by county for selected <states>.
        """
        fig = ff.create_choropleth(fips=fips,
                                   values=values,
                                   scope=states,
                                   show_state_data=True,
                                   round_legend_values=True,
                                   legend_title=column,
                                   exponent_format=True)
        py.plot(fig, filename='choropleth_map_{}'.format(category))

    @staticmethod
    def _normalize(data):
        """
        Normalize dataset by column.
        """
        return (data - np.mean(data, axis=0)) / np.std(data, axis=0)


if __name__ == '__main__':
    analyzer = AHRFDataAnalyzer(ASC_FILE, SAS_FILE)
    analyzer.get_environmental_data()
