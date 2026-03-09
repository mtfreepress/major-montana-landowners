# Major Montana landowners analysis

This is a set of Python scripts for taking [Montana Cadastral](https://svc.mt.gov/msl/cadastral/) parcel data and processing it to identify Montana's largest landowners by geographic expanse.

## Data notes

The Cadastral system, maintained by the Montana State Library, presents property data maintained for tax purposes by the Montana Department of Revenue. 

The data we're using is a large (~700 MB) file that lives here: https://ftpgeoinfo.msl.mt.gov/Data/Spatial/MSDI/Cadastral/

We use the `MontanaCadastral_SHP.zip`, which is a GIS shapefile. I've downloaded it manually via a web browser and put it in the `raw` folder.

The unzipped version of the includes a few different data layers. We're interested in `Montana_Cadastral/OWNERPARCEL.shp`

That shapefile reflects every or nearly every parcel in the state with land attached to it but seems to exclude properties that aren't attached to a geographic area (e.g., condos). This means it shouldn't be used for property tax base analyses (request data from DOR directly for that) but it's fine for what we care about — who owns Montana's geographic landscape.

The state does have a system for letting people request exclusions from the Cadastral system (it's intended for individuals like police officers and judges who are worried about their home addresses being searchable) but it seems to be rarely used, meaning this data is the easiest way to explore broad Montana land ownership patterns.

Key fields in the parcel data include the following:
- `PARCELID` -- The state Geocode
- `CountyName` -- the county the property is in
- `TotalAcres` - The acreage associated with the property. There is also a GISAcres field but "TotalAcres" seems to be the value that matches what's presented in the public-facing Cadastral database so it's what we use for the analysis.
- `TotalBuild` / `TotalLandV` / `TotalValue` -- Fields that represent the valuations assigned to the parcel by DOR for tax purporses. These are generally market value for residential/commercial properties but are instead productive value (generally much lower than market value) for ag properties. As a result, we generally CANNOT sum the "TotalValue" for a group of properties to produce an accurate "properties worth XXX" figure. Don't be tempted to do this.
- `OwnerName` -- The listed owner name.
- `OwnerAddre`, `OwnerAdd_1`, `OwnerCity`, etc. etc. - Fields that describe the owner's address -- i.e. where the tax bill is sent. This is the address field we're focused on
- `AddressLin`, `AddressL_1`, etc. -- Fields that describe the property's physical address
- `PropType` -- Residential, Commercial, etc. (This can be a little deciving because parcels can actually include components with different property types )

# Analysis tools

This analysis is done with iPython notebooks using Python's pandas/geopandas libraries. Eric was running code inside Microsoft VS Code, using a Python 3.13.0 kernal.

The `geodataframe.explore()` command we're using to generate interactive "slippy" maps may take some extra configuration depending on how you're running Python. Documentation on that [is here](https://geopandas.org/en/stable/docs/user_guide/interactive_mapping.html) — I had to do a bit of troubleshooting to get the Folium library the interactives use running without issue on my system.

# Analysis notes

Goal: 1) Identify Montana's 10 largest landowners by acreage and 2) produce Geodata (GeoJson) files with the bounds of their holdings to allow easy mapping in MTFP coverage products.

General process: We're using the Cadastral data described above to look for ownership groups.

**Analysis files**

- `major-landowners-naive-analysis.ipynb` -- This was our 'first pass' analysis, which simply grouped parcels by `OwnerName`, sums the acreage for each unique private owner, presenting the results as a ranked list. It naively assumes that each unique OwnerName in the dataset is a unique "owner." It's very clear that isn't the best way to do this — Ted Turner, for example, owns some of his properties as `TURNER ENTERPRISES INC` and others as `TURNER LAND HOLDINGS LLC`. Public entities listed in `known-public-landowners.json` are excluded.
- `shared-address-identification.ipynb` -- A workflow for identifying owner name variants based on ownership addresses. IDs places where owner name variants share mailing addresses, indicating they belong to the same real-world owner. Output of this was placed manually in `owner-name-cleaning.json`.
- `major-landowners-final-analysis.ipynb` - Second/final pass analysis that also accounts for owner name variations, using name variations specified in `owner-name-groupings.json`. This also exports geodata files for each of the top 20 landowners to `geodata-outputs/` and a version of the data with owner groups appended to `cleaned/`
- `explore.ipynb` -- A scratchpad notebook for one-off analysis, e.g. mapping the parcels associated with specific owners.

**Config files** — these are manually curated in a JSON format guide the analysis scripts
- `known-public-landowners.json` -- Owner names that represent public entities, i.e. federal and state agencies, as well as tribes. Eric produced this by inspection but probably missed some small public landowners. There are hundreds of items on this list  because the naming conventions for public agencies appear to be wildly inconsistent.
- `owner-name-cleaning.json` -- Groups of owner names that should be classified as a group (i.e. they're really the same owner). This is focused on 

**Output files** 
- `naive-top-10.json` -- output from "naive" analysis that doesn't account for owner name variation
- `final-top-20.json` -- ouutput from final analysis as a JSON object
- `final-top-20.txt` -- easier-to-read output from final analysis, also includes breakdowns for ownership by owner name variants.