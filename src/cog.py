
import os
import sys
import yaml
import pandas as pd
import numpy as np

import folium


class CenterOfGravity:
    """
        Center of Gravity object

        Key internal variables include:
        params: Dict of key parameters
        df_demand: DataFrame with demand info. Key (default) fields are lat, lng, demand
        df_supply: DataFrame with supply info. Key (default) fields are lat, lng, supply, supply_group

        distance_mode: Linear or haversine

    """

    def __init__(self, params, df_demand_data, df_supply_data=None,
                 dmd_fld='demand', supply_fld='supply', supply_grp_fld='supply_group',
                 x_fld='lng', y_fld='lat', units='degrees', distance_units='mi', distance_mode='haversine'):
        """

        Sets initial values for parameters and input datq

        Note: If units are degrees or radians,
        init sets up x, y fields with hard-coded names that contain x, y in radians: lat_rad and lng_rad

        :param params: Dict of parameter values
        :param df_demand_data: DataFrame of demand data, with lat/lng fields
        :param df_supply_data: DataFrame of supply data, with lat/lng fields; optional
        :param dmd_fld:
        :param supply_fld:
        :param supply_grp_fld:
        :param x_fld:
        :param y_fld:
        :param units: Coordinate unit of measure. Can be: mi, km, m, radians, degrees. Latter 2 are latitude/longitude
        :param distance_units: Units used for distance. Doesn't have to match units
        :param distance_mode: haversine or linear

        """

        self.cog_params = params
        self.df_cog_demand_data = df_demand_data
        self.df_cog_supply_data = df_supply_data

        self.flds = dict()
        self.flds['dmd'] = dmd_fld
        self.flds['supply'] = supply_fld
        self.flds['supply_grp'] = supply_grp_fld
        self.flds['x'] = x_fld
        self.flds['y'] = y_fld

        self.bbox_demand = dict()
        self.calc_bounding_box_demand()
        self.bbox_supply = dict()
        self.calc_bounding_box_supply()

        self.units = units
        self.distance_units = distance_units
        self.distance_mode = distance_mode

        # Convert coordinates if necessary:

        self.earth_radius = 0
        if units in ['degrees', 'radians']:
            self.compute_radians_demand(units)
            self.compute_radians_supply(units)
            self.set_earth_radius(distance_units)

    def calc_bounding_box(self, df):
        # Convenience
        colx = self.flds['x']
        coly = self.flds['y']

        min_x = df[colx].min()
        min_y = df[coly].min()
        max_x = df[colx].max()
        max_y = df[coly].max()
        diff_x = max_x - min_x
        diff_y = max_y - min_y

        bbox = {'min_x': min_x, 'max_x': max_x, 'min_y': min_y, 'max_y': max_y, 'diff_x': diff_x, 'diff_y': diff_y}

        return bbox

    def calc_bounding_box_demand(self):
        bbox = self.calc_bounding_box(self.df_cog_demand_data)
        self.bbox_demand = bbox

    def calc_bounding_box_supply(self):
        if self.cog_params['flag_supply']:
            bbox = self.calc_bounding_box(self.df_cog_supply_data)
            self.bbox_supply = bbox
        else:
            self.bbox_supply = None

    def compute_radians(self, df, units):
        if units == 'degrees':
            df['lat_rad'] = np.radians(df[self.flds['y']])
            df['lng_rad'] = np.radians(df[self.flds['x']])
        elif units == 'radians':
            # No conversion necessary
            df['lat_rad'] = df[self.flds['y']]
            df['lng_rad'] = df[self.flds['x']]
        else:
            df['lat_rad'] = np.nan()
            df['lng_rad'] = np.nan()

        return df

    def compute_radians_demand(self, units):
        self.df_cog_demand_data = self.compute_radians(self.df_cog_demand_data, units)

    def compute_radians_supply(self, units):
        if not self.df_cog_supply_data is None:
            self.df_cog_supply_data = self.compute_radians(self.df_cog_supply_data, units)

    def set_earth_radius(self, units):
        if units == 'mi':
            self.earth_radius = 3959.87
        elif units == 'km':
            self.earth_radius = 6372.8
        elif units == 'm':
            self.earth_radius = 6372800

    def init_cog(self, n=1, method='circles_1'):
        """

        Initializes CoG run with initial CoGs

        Methods include:
        - circles_1: Pick a spot, draw an exclusivity circle, pick the next spot, repeat

        :param n: Number of centers of gravity
        :param method: Method to use to initialize
        :return:
        """

        # For simplicity we don't worry about whether x, y coordinates are linear or angular for initialization

        if method == 'circles_1':
            for i in range(1,n+1):
                print(i)

    def get_df_distances(self, df_o, df_d, mode='haversine'):
        """

        Compute distances between each combination in 2 sets of coordinates

        Assumes each DataFrame has x, y coordinates as indicated by flds['x'] and flds['y']

        :param df_o:
        :param df_d:
        :param mode:
        :param units: mi, km, or m
        :return:
        """

        df_o['cartesianjoinkey'] = 0
        df_d['cartesianjoinkey'] = 0

        df_join = df_o.merge(df_d, on='cartesianjoinkey', how='outer', suffixes=['_o', '_d'])
        # We keep the key column for speed; otherwise we could drop it for hygiene

        col_x_o = self.flds['x'] + '_o'
        col_y_o = self.flds['y'] + '_o'
        col_x_d = self.flds['x'] + '_d'
        col_y_d = self.flds['y'] + '_d'

        if mode == 'linear':
            # Unit-invariant
            df_join['distance'] = np.sqrt(np.power(df_join[col_x_o] - df_join[col_x_d], 2) +
                                          np.power(df_join[col_y_o] - df_join[col_y_d], 2))

        elif mode == 'haversine':
            dlon = df_join['lon_rad_d'] - df_join['lon_rad_o']
            dlat = df_join['lat_rad_d'] - df_join['lat_rad_o']
            a = (np.power(np.sin(dlat/2.0), 2) + np.cos(df_join['lat_rad_o'])
                 * np.cos(df_join['lat_rad_d']) * np.power(np.sin(dlon/2.0)), 2)
            c = 2.0 * np.arcsin(np.sqrt(a))
            df_join['distance'] = self.earth_radius * c

    # TESTS

    def test_dmd_bbox(self):
        return self.bbox_demand

    def test_supply_bbox(self):
        return self.bbox_supply

    def test_dmd_rows(self):
        return self.df_cog_demand_data.shape[0]

    def test_dmd_pct(self):
        return self.df_cog_demand_data['demand_pct'].sum()


def get_parameters(file_config, path='../cfg'):
    """
    :param file_config: Name of config file
    :param path: Path for config file
    :return: Dict of configuration parameters
    """

    # Load YAML file...
    with open(os.path.join(path, file_config)) as file:
        params = yaml.load(file, Loader=yaml.SafeLoader)

    return params


def load_demand_data(file_data, path='../data',
                     country_fld='country', geo_fld='geo_id',
                     demand_fld='demand', demand_fld_norm='demand_pct'):
    """

    Loads demand data, which consists of a geo ID (typically postal code) and a demand weight.

    :param file_data: Name of data file
    :param path: Path for data file
    :param geo_fld: Name of field with geographical location in returned df
    :param demand_fld: Name of field with demand in returned df
    :param demand_fld_norm: Name of field with normalized demand info
    :return: df with demand data
    """

    df = pd.read_csv(os.path.join(path, file_data))

    # Replace field names based on a number of possibilities

    # Country field
    list_of_country_fields = ['Country']
    for f in list_of_country_fields:
        df.rename(inplace=True, columns={f: country_fld})

    # Demand field
    list_of_demand_fields = ['2010 Census Population', 'Demand', 'demand', 'dmd']
    for f in list_of_demand_fields:
        df.rename(inplace=True, columns={f: demand_fld})

    # We convert to formatted strings so that we can handle multiple types of postal codes in one column

    df_columns = [x.lower() for x in df.columns]
    df.columns = df_columns

    # ZIP5
    list_of_geo_fields = ['zip code zcta', 'zip', 'zip5']
    for f in list_of_geo_fields:
        if f in df_columns:
            df[geo_fld] = df[f].astype(str).str.zfill(5)

    # ZIP3
    list_of_geo_fields = ['zip3']
    for f in list_of_geo_fields:
        if f in df_columns:
            df[geo_fld] = df[f].astype(str).str.zfill(3)

    # Canadian FSA
    list_of_geo_fields = ['fsa']
    for f in list_of_geo_fields:
        if f in df_columns:
            df[geo_fld] = df[f].astype(str).str.zfill(3)

    # Normalize demand
    df[demand_fld_norm] = df[demand_fld] / df[demand_fld].sum()

    return df


def load_supply_data(file_data, path='../data', country_fld='country', geo_fld='geo_id',
                     supply_fld='supply',
                     supply_fld_norm='supply_pct',
                     supply_grp_fld='supply_group'):
    """

    :param file_data: Name of file with supply data
    :param path: Path to supply data file
    :param country_fld: Name of country column in returned df
    :param geo_fld: Name of geo ID column in returned df
    :param supply_fld: Name of supply amount column in returned df
    :param supply_fld_norm: Name of normalized supply amount column in returned df
    :param supply_grp_fld: Name of supply group fied in returned df
    :return: df with supply info
    """

    # TODO: Add renaming and normalizing

    df = pd.read_csv(os.path.join(path, file_data))

    return df


def load_geo_id_data(list_of_files, path='../data/geo', country_fld='country',
                     geo_fld='geo_id', geo_type_fld='geo_type',
                     lat_fld='lat', lng_fld='lng'):
    """

    Loads and cleans a bunch of files with geo IDs and coordinates, then appends them

    The field names are passed as arguments for flexibility, but the main codebase assumes these values

    :param list_of_files: list of files with geographic IDs to load
    :param path: Path of files
    :param country_fld:
    :param geo_fld: Geo ID field
    :param geo_type_fld:
    :param lat_fld:
    :param lng_fld:
    :return: DataFrame with concatenated geoID data; columns are geo_fld, lat_fld, lng_fld
    """

    list_df = []

    for f in list_of_files:
        df = pd.read_csv(os.path.join(path, f))

        # Format various geoIDs. We do this to remove ambiguity between ID types, such as ZIP3 and ZIP5,
        # which can conflict if treated as ints.
        # Order of operations matters here;
        # we start with the coarsest and end with the most granular in case the input file has multiple levels
        # (e.g. ZIP3 and ZIP5)

        initial_columns = [x.lower() for x in df.columns]
        df.columns = initial_columns

        # Canadian FSA
        if 'fsa' in initial_columns:
            df[geo_fld] = df['fsa'].astype(str)
            df[geo_type_fld] = 'FSA'
        elif 'zip3' in initial_columns:
            df[geo_fld] = df['zip3'].astype(str).str.zfill(3)
            df[geo_type_fld] = 'ZIP3'
        elif 'zip5' in initial_columns:
            df[geo_fld] = df['zip5'].astype(str).str.zfill(5)
            df[geo_type_fld] = 'ZIP5'

        # Rename lat/lng fields
        list_flds_latitude = ['latitude', 'lat']
        for fld in list_flds_latitude:
            df.rename(inplace=True, columns={fld: lat_fld})

        list_flds_longitude = ['longitude', 'long', 'lon', 'lng']
        for fld in list_flds_longitude:
            df.rename(inplace=True, columns={fld: lng_fld})

        list_flds_country = ['country']
        for fld in list_flds_country:
            df.rename(inplace=True, columns={fld: country_fld})

        df_clean = df[[country_fld, geo_fld, geo_type_fld, lat_fld, lng_fld]].copy()

        list_df.append(df_clean)

        df_concat = pd.concat(list_df)

        return df_concat


def add_geo_coords(df, df_geo,
                   country_fld='country', geo_fld='geo_id',
                   lat_fld='lat', lng_fld='lng'):

    df_joined = pd.merge(df, df_geo, how='left', on=[country_fld, geo_fld], suffixes=['_data', '_geo'])

    # Separate rows with and without nulls
    df_nulls = df_joined[df_joined.isnull().any(axis=1)].copy()
    df_good = df_joined.dropna(axis=0)

    return [df_good, df_nulls]


def renorm_demand_data(df, qty_fld='demand', qty_norm_fld='demand_pct'):
    """

    Use for renormalizing demand data if we dropped some due to e.g. geo ID mismatches

    :param df: DataFrame with demand data
    :param qty_fld: Name of demand field
    :param qty_norm_fld: Name of normalized demand field
    :return:
    """
    return renorm_data(df, qty_fld, qty_norm_fld)


def renorm_supply_data(df, qty_fld='supply', qty_norm_fld='supply_pct'):
    """

    Use for renormalizing supply data if we dropped some due to e.g. geo ID mismatches

    :param df:
    :param qty_fld:
    :param qty_norm_fld:
    :return:
    """
    return renorm_data(df, qty_fld, qty_norm_fld)


def renorm_data(df, qty_fld, qty_norm_fld):
    """

    General purpose data normalization function

    :param df: DataFrame
    :param qty_fld: Column with demand or supply data
    :param qty_norm_fld: Column for normalized data
    :return:
    """

    df[qty_norm_fld] = df[qty_fld] / df[qty_fld].sum()

    return df


if __name__ == "__main__":

    # If no command line arguments, use a default config file;
    # otherwise load the file specified on the command line

    if len(sys.argv) == 1:
        cfg = 'demo-demand.yaml'
    else:
        cfg = sys.argv[1]

    # Setup: Get parameters and input data

    dict_params = get_parameters(cfg)

    # TODO: Check to see if there's cached data

    # This is a little ugly but lets us respect default values in the function
    if 'path_data' in dict_params.keys():
        data_demand = load_demand_data(dict_params['file_demand'], path=dict_params['path_data'])
    else:
        data_demand = load_demand_data(dict_params['file_demand'])

    if 'flag_supply' in dict_params.keys():
        if dict_params['flag_supply']:
            if 'path_data' in dict_params.keys():
                data_supply = load_supply_data(dict_params['file_demand'], path=dict_params['path_data'])
            else:
                data_supply = load_supply_data(dict_params['file_demand'])

    else:
        dict_params['flag_supply'] = False

    # Load geoID to lat/lng conversion data
    if 'path_geo' in dict_params.keys():
        df_geo = load_geo_id_data(dict_params['files_geo'], dict_params['path_geo'])
    else:
        df_geo = load_geo_id_data(dict_params['files_geo'])

    # Join geoID and lat/lng data
    [df_demand, df_demand_bad] = add_geo_coords(data_demand, df_geo)
    df_demand = renorm_demand_data(df_demand)

    # TODO: Do the same for supply data

    # TODO: Save the results, and create an additional cfg file that contains references to the cached data

    # Insert params and input data into a Cog object

    cog = CenterOfGravity(dict_params, df_demand, None)

    print('Valid demand rows: {}'.format(cog.test_dmd_rows()))
    print('Valid demand pct: {}'.format(cog.test_dmd_pct()))

    print('Demand Bounding Box {}'.format(cog.test_dmd_bbox()))
    print('Supply Bounding Box {}'.format(cog.test_supply_bbox()))

    print('Bad: {}'.format(df_demand_bad.shape[0]))
    print(df_demand_bad['demand_pct'].sum())

