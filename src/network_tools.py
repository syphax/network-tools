import math
import pandas as pd


def get_df_distances(df_o, df_d, lat_fld='lat', lng_fld='lng'):
    """

    Computes haversine distances between all rows in df_o and all rows in df_d
    Math is not directional; origin and destination don't have real meaning here.

    Requires both df_o and df_d to have the same names for latitude & longitude

    Result is a df that is the Cartesian of df_o and df_d, with the addition of fields dist_mi and dist_km

    :param df_o: DataFrame with origins
    :param df_d: DataFrame with destinations
    :param lat_fld: Name of latitude field in BOTH DataFrames
    "param lng_fild: Name of longitude field in BOTH DataFrames
    :return: df that is the Cartesian of all origins and destinations

    """

    # Cartesian join
    df = pd.merge(df_o.assign(key=0), df_d.assign(key=0), on='key', suffixes=['_o', '_d']).drop('key', axis=1)

    return df


if __name__ == "__main__":
    print('Network Tools')
