# Data

## Base Datasets

### US Population by ZCTA
This data is actually hard to extract from https://data.census.gov/; I found a dataset [here.](https://blog.splitwise.com/2013/09/18/the-2010-us-census-population-by-zip-code-totally-free/) 
It's fine for our purposes.

File: `2010-pop-by-zcta.csv`

# Supply and Demand Data

Both supply and demand data 

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
* Geographical ID: `Zip Code ZCTA`, `ZIP`, `ZIP5`, `zip`, `zip5` (all ZIP5 format)
* Demand: `2010 Census Population`, `Demand`, `demand`, `dmd`

### Supply Data

Supply data files are similar to demand data, except that "demand" is replaced by "supply", and there's one additional field:
* Supply Group ID

Supply Groups allow us to differentiate between a fixed amount of supply at a fixed point,
and a fixed amount of supply that can come from multiple sources. The logic is this: 
if a Supply Group ID shows up once, that amount of supply must come from that source.
But if there are multiple rows with the same supply group ID, then the total supply across those rows is fungible across those sites. 
So if you have 3 sites with the same Supply Group ID, with a total supply of 6 
(it does not matter how the supply is distributed across the sites in the data file; it could be 6, 0, 0 or 2, 2, 2), 
then 6 units of supply will ultimately be sourced in some combination from those 3 sources.

Supply Group IDs can be text or numeric; are treated as strings in the model.

