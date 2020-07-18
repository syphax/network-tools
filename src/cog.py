
import os
import sys
import yaml
import pandas as pd
import folium


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


def load_demand_data(file_data, path='../data', geo_fld='zip5', dmd_fld='dmd', dmd_fld_norm='dmd_pct'):
    """

    Loads demand data, which consists of a geo ID (typically postal code) and a demand weight

    :param file_data: Name of data file
    :param path: Path for data file
    :param geo_fld: Name of field with geographical location in returned df
    :param dmd_fld: Name of field with demand in returned df
    :param dmd_fld_norm: Name of field with normalized demand info
    :return: df with demand data
    """

    df = pd.read_csv(os.path.join(path, file_data))

    # Replace field names based on any number of possibilities
    list_of_zip_fields = ['Zip Code ZCTA']
    for f in list_of_zip_fields:
        df.rename(inplace=True, columns={f: geo_fld})

    list_of_dmd_fields = ['2010 Census Population']
    for f in list_of_dmd_fields:
        df.rename(inplace=True, columns={f: dmd_fld})

    # Normalize demand
    df[dmd_fld_norm] = df[dmd_fld] / df[dmd_fld].sum()

    return df


def load_supply_data(file_data, path='../data', geo_fld='zip5', dmd_fld='dmd', dmd_fld_norm='dmd_pct'):
    """

    :param file_data:
    :param path:
    :param geo_fld:
    :param dmd_fld:
    :param dmd_fld_norm:
    :return:
    """

    df = pd.read_csv(os.path.join(path, file_data))

    return df

if __name__ == "__main__":

    # If no command line arguments, use a default config file;
    # otherwise load the file specified on the command line

    if len(sys.argv) == 1:
        cfg = 'demo.yaml'
    else:
        cfg = sys.argv[1]

    dict_params = get_parameters(cfg)

    # This is a little ugly but lets us respect default values in the function
    if 'path_data' in dict_params.keys():
        data_dmd = load_demand_data(dict_params['file_demand'], path=dict_params['path_data'])
    else:
        data_dmd = load_demand_data(dict_params['file_demand'])

    if dict_params['flag_supply']:
        if 'path_data' in dict_params.keys():
            data_supply = load_supply_data(dict_params['file_demand'], path=dict_params['path_data'])
        else:
            data_supply = load_supply_data(dict_params['file_demand'])
