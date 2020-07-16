
import sys
import yaml
import pandas
import folium

def get_parameters(file_config, path='../cfg'):
    """
    :param file_config: Name of config file
    :param path: Path for config file
    :return: Dict of configuration parameters
    """

    # Load YAML file...

    return params

def load_demand_data(file_data, path='../data'):
    """

    :param file_data: Name of data file
    :param path: Path for data file
    :return: df with demand data
    """


if __name__ == "__main__":

    # If no command line arguments, use a default config file;
    # otherwise load the file specified on the command line

    if len(sys.argv) == 1:
        cfg = 'demo.yaml'
    else:
        cfg = sys.argv[1]

    dict_params = get_parameters(cfg)

    data_dmd = load_demand_data(dict_params['file_demand'])

