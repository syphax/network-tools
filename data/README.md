# Data

## Base Datasets

### US Population by ZCTA
This data is actually hard to extract from https://data.census.gov/; I found a dataset [here.](https://blog.splitwise.com/2013/09/18/the-2010-us-census-population-by-zip-code-totally-free/) 
It's fine for our purposes.

File: `2010-pop-by-zcta.csv`

### US Zip Code Lat/Lon
Lightly edited for ease of use (added headers, removed a couple columns)

File: `US-zip-lat-lon.csv`

Source: http://download.geonames.org/export/zip/

(Yes, we're mixing ZIP and ZCTA data here...)

# Supply and Demand Data

### Demand Data

Demand data must have columns for:
* Country ID (ISO 2-letter)
* Geographical ID
* Demand data

Currently, the only supported Geo ID type is ZIP5 for the U.S.
The units for the demand column are immaterial; they just need to be internally consistent. 
The demand gets summed and normalized. 

Column names are flexible for convenience. Valid names are:
* Country ID: `Country`
* Geographical ID: `Zip Code ZCTA`, `ZIP`, `ZIP5`, `zip`, `zip5`,
                          `Postal Code`, `postal_code`
* Demand: `2010 Census Population`, `Demand`, `demand`, `dmd`