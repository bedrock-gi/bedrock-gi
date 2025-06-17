# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "bedrock-ge==0.2.3",
#     "chardet==5.2.0",
#     "folium==0.19.5",
#     "geopandas==1.0.1",
#     "mapclassify==2.8.1",
#     "marimo",
#     "matplotlib==3.10.1",
#     "pandas==2.2.3",
#     "pyproj==3.7.1",
#     "requests==2.32.3",
#     "shapely==2.1.0",
#     "pygef"==0.11.1"
# ]
# ///

import marimo

__generated_with = "0.13.11"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    # GEF Data for A2 Tunnel Maastricht

    This notebook demonstrates how to 

    1. Read in Ground Investigation (GI) data from [GEF files]() using [pygef](https://cemsbv.github.io/pygef/)
    1. Use `bedrock-ge` to convert that data into a standardized GI database using `bedrock-ge`
    1. Transform the GI data into 3D spatial features with proper coordinates and geometry ([OGC Simple Feature](https://en.wikipedia.org/wiki/Simple_Features))
    1. Explore and analyze the GI data using interactive filtering with Pandas DataFrames and interactive visualization on a map with GeoPandas.
    1. Export the processed GI database to a GeoPackage file for use in other software, like QGIS.

    <details>
        <summary>What are GEF files?</summary>
        <p>
            <abbr>Geotechnical Exchange Format (GEF)</abbr> is a standardized, text-based format designed to facilitate the reliable exchange and archiving of geotechnical investigation data, particularly CPT results, across different organizations and software platforms. GEF can also be used for other types of soil tests and borehole data. It is widely used in the Netherlands in ground investigration.
        </p>
    </details>

    <details>
        <summary>What is a DataFrame?</summary>
        <p>
          A DataFrame is like a spreadsheet, it is a two-dimensional data structure that holds data like a table with rows and columns.
        </p>
    </details>

    ## Context

    The Koning Willem-Alexander Tunnel is a double-deck tunnel for motorized traffic in the Dutch city of Maastricht. The tunnel has a length of 2.5 kilometers (lower tunnel tubes) and 2.3 kilometers (upper tunnel tubes).

    The tunnel has moved the old A2 highway underground. This highway previously formed a barrier for the city and slowed traffic.

    ### Geology

    The uppermost layer consists of topsoil, clay, and loam, with a thickness of about 2 to 4 meters. These soft Holocene deposits are attributed to the Boxtel Formation, laid down by the Meuse River, as the tunnel is situated in a former river arm.

    Beneath the surface layer lies an approximately 8-meter-thick gravel deposit. This gravel acts as a significant aquifer and was a key factor in the groundwater management strategies required for the tunnel construction.

    Below the gravel lies a fissured limestone layer belonging to the Maastricht Formation (mergel). This layer is a very weak, porous, sandy, shallow marine limestone, often weathered, and includes chalk and calcarenite components.

    The limestone is relatively young and shallow, resulting in low compaction and cementation. Its mechanical strength is highly variable and generally low, especially when saturated with groundwater.

    Extensive geophysical surveys and borehole investigations were conducted to map the subsurface, identify faults, flint layers, and assess the risk of cavities within the limestone. While faults were detected, no significant cavities were found.

    The stability of the excavation pit was monitored in real-time, with groundwater levels and pressures carefully controlled to prevent collapse or excessive deformation of the pit walls.

    Due to the high permeability of the gravel and fissured limestone, groundwater management was a major challenge. Over 500 wells were drilled to depths of up to 32 meters for dewatering, and a reinfiltration system was implemented to return nearly all pumped water to the ground, protecting local buildings and ecosystems.

    #### Sources

    * [Tunnel A2 Maastricht: Groundwater Management with DSI System](https://www.tunnel-online.info/en/artikel/tunnel-a2-maastricht-groundwater-management-with-dsi-system-1564115.html)
    * [Geotechniek-en-Risicos bij A2 Maastricht](https://www.cob.nl/magazines-brochures-en-nieuws/verdieping/verdieping-sept2012/geotechniek-en-risicos-bij-a2-maastricht/)
    * [Laboratory Tests on Dutch Limestone (Mergel) ](https://onepetro.org/ISRMEUROCK/proceedings-abstract/EUROCK15/All-EUROCK15/ISRM-EUROCK-2015-072/43534)
    * [Eduard van Herk en Bjorn Vink](https://a2maastricht.nl/application/files/3315/2060/1222/Interview_Eduard_van_Herk_en_Bjorn_Vink.pdf)

    ## Ground Investigation Data

    The GI data was downloaded from [Dinoloket](https://www.dinoloket.nl/ondergrondgegevens), a platform where you can view and request data and models from TNO and BRO about the subsurface of the Netherlands.
    """
    )
    return


@app.cell
def _(Path, pygef):
    folder_path = Path("./gefs")
    gef_files = list(folder_path.glob("*.gef"))
    boreholes = [pygef.read_bore(gef_file) for gef_file in gef_files]
    return (boreholes,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("pygef uses [polars](https://pola.rs/) for DataFrames, for consistency we will convert them to [Pandas](http://pandas.pydata.org/) DataFrames in this notebook.").callout("warn")
    return


@app.cell(hide_code=True)
def _(boreholes, mo):
    options = {d.alias: i for i, d in enumerate(boreholes)}
    multiselect = mo.ui.dropdown(options, label="Select borehole")
    multiselect
    return (multiselect,)


@app.cell(hide_code=True)
def _(multiselect):
    index = multiselect.value or 0
    return (index,)


@app.cell(hide_code=True)
def _(boreholes, index):
    boreholes[index].data.to_pandas().dropna(axis=1, how='all') # drop empty columns for display
    return


@app.cell
def _(boreholes, index, pygef):
    pygef.plotting.plot_bore(boreholes[index])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Converting multiple GEF files to a relational database

    Rather than dealing with a folder of files, we would like to combine all files into a single database with spatial information. This is where Bedrock comes in.

    ### Relational Databases

    A [relational database](https://observablehq.com/blog/databases-101-basics-data-analysts#what-are-relational-databases) is a database with multiple tables that are linked to each other with relations. This type of database is ideal for storing GI data, given its [hierarchical structure](https://bedrock.engineer/docs/#hierarchical-nature-of-gi-data).

    In Python it's convenient to represent a relational database as a dictionary of DataFrame's.

    ### Coordinated Reference System (CRS)

    First, let's check in which projected coordinate system the provided data was recorded.
    """
    )
    return


@app.cell
def _(boreholes):
    code = {bore.delivered_location.srs_name for bore in boreholes}.pop()
    orig_epsg_code = code.split("EPSG::")[-1]
    orig_crs = f"EPSG:{orig_epsg_code}"
    orig_crs
    return


@app.cell
def _():
    crs = "EPSG:7415"
    return (crs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The data is in EPSG:28992, which is the [Rijksdriehoekscoördinaten (NL)](https://nl.wikipedia.org/wiki/Rijksdriehoeksco%C3%B6rdinaten) system, also called "Amersfoort / RD New". This reference system does not include elevation.

    To represent GI data spatially in 3D geometry we need a CRS with elevation. That's why we will use
    EPSG:7415 Amersfoort / RD New + NAP height.
    """
    )
    return


@app.cell
def _():
    wgs = "EPSG:4326"
    return


@app.cell
def _():
    project_uid = "Maastricht A2 tunnel"
    return (project_uid,)


@app.cell
def _(CRS, crs, pd, project_uid):
    project = pd.DataFrame({
        "project_uid": [project_uid], # primary key
        "crs_wkt": CRS(crs).to_wkt()
    })
    return (project,)


@app.cell
def _(insitu_geo, locations, project):
    brgi_db = {"Project": project, "Location": locations.drop(columns=["data"]), "InSitu_GEOL": insitu_geo }
    return (brgi_db,)


@app.function
def process_data(bore):
    df = bore.data.to_pandas().dropna(axis=1, how='all').rename(columns=
    {
        'upperBoundary': 'depth_to_top',
        'lowerBoundary': 'depth_to_base',
        'upperBoundaryOffset': 'elevation_at_top',
        'lowerBoundaryOffset': 'elevation_at_base'
    })

    return df


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Here, we create a new DataFrame for locations and remap the GEF keys to follow Bedrock's conventions. 
    We need to map `alias` to `location_source_id` and `delivered_vertical_position_offset` to `ground_level_elevation` for example.
    We also need to create a **unique identifier** and add `project_uid` as a key to relate it the project.
    """
    )
    return


@app.cell
def _(boreholes, pd, project_uid):
    locations_df = pd.DataFrame([
        {
            "location_uid": f"{borehole.alias} {project_uid}", # primary key
            "project_uid": project_uid, # foreign key
            "data": process_data(borehole),
            "location_source_id": borehole.alias,
            "date": borehole.research_report_date,
            "location_type": "Hole",
            "easting": borehole.delivered_location.x,
            "northing": borehole.delivered_location.y,
            "depth_to_base": min(borehole.data["lowerBoundaryOffset"]),
            "ground_level_elevation": borehole.delivered_vertical_position_offset,
            "elevation_at_base": borehole.delivered_vertical_position_offset - min(borehole.data["lowerBoundaryOffset"]),
        }
        for borehole in boreholes
    ])
    return (locations_df,)


@app.cell
def _(calculate_location_gis_geometry, crs, locations_df):
    locations = calculate_location_gis_geometry(locations_df, crs=crs)
    return (locations,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Displaying the GI locations on a map

    Rather than multiple tables (DataFrames) and soil profiles, we would like see an overview of what this ground investigation covers. It's **spatial** data after all, let's view it in a spatial context.

    ### Web Mapping Caveats

    Web-mapping tools are rarely capable of handling geometry in non-WGS84 coordinates. Additionally, vertical lines are not visible when looking at a map from straight above. That's why use `create_lon_lat_height_table` to create points in the WGS84 CRS so we can view the locations of the boreholes.
    """
    )
    return


@app.cell
def _(create_lon_lat_height_table, crs, locations):
    create_lon_lat_height_table(locations, crs).explore(marker_kwds={"radius":5})
    return


@app.cell
def _(mo):
    mo.md(r"""Here we create a DataFrame for the In-Situ data of all locations. To relate to locations and the project we add foreign keys.""")
    return


@app.cell
def _(locations, pd):
    insitu = pd.DataFrame([
        {
            **layer,
            "location_uid": location["location_uid"], # foreignkey
            "project_uid": location["project_uid"], # foreignkey
        }
        # Outer loop: iterate through each location
        for location in locations.to_dict('records')
        # Inner loop: iterate through each layer in the location's data dataframe
        for layer in location["data"].to_dict('records')
    ])
    return (insitu,)


@app.cell
def _(insitu):
    insitu
    return


@app.cell
def _(mo):
    mo.md(r"""In-situ data is also spatial data. It has a location, a depth and a height. We can also turn it into spatial data using Bedrock's `calculate_in_situ_gis_geometry` """)
    return


@app.cell
def _(calculate_in_situ_gis_geometry, crs, insitu, locations):
    insitu_geo = calculate_in_situ_gis_geometry(insitu, locations, crs)
    return (insitu_geo,)


@app.cell
def _(insitu_geo):
    insitu_geo
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Saving the GI geospatial database as a GeoPackage (.gpkg)

    Finally, we'll write it to an actual geospatial database file, a GeoPackage, so we can share our GI data with others, for example, to reuse it in other computational notebooks, create dashboards, access the GI data in QGIS or ArcGIS, and more...

    A GeoPackage is an <abbr title="Open Geospatial Consortium">OGC-standardized</abbr> extension of SQLite (a relational database in a single file, .sqlite or .db) that allows you to store any type of GIS data (both raster as well as vector data) in a single file that has the .gpkg extension. Therefore, many (open-source) GIS software packages support GeoPackage!
    """
    )
    return


@app.cell
def _(brgi_db, check_brgi_database):
    check_brgi_database(brgi_db)
    return


@app.cell
def _(brgi_db, write_gi_db_to_gpkg):
    write_gi_db_to_gpkg(brgi_db, gpkg_path="./output/A2_Maastricht.gpkg")
    return


@app.cell
def _():
    import marimo as mo
    import pygef
    import os
    from pathlib import Path
    import pandas as pd
    import geopandas as gpd
    import matplotlib
    import pyarrow
    import folium
    import mapclassify
    from shapely.geometry import Point, LineString
    from bedrock_ge.gi.gis_geometry import calculate_wgs84_coordinates, calculate_in_situ_gis_geometry, calculate_gis_geometry, calculate_location_gis_geometry, create_lon_lat_height_table
    from bedrock_ge.gi.write import write_gi_db_to_gpkg
    from bedrock_ge.gi.validate import check_brgi_database
    from pyproj import CRS
    return (
        CRS,
        Path,
        calculate_in_situ_gis_geometry,
        calculate_location_gis_geometry,
        check_brgi_database,
        create_lon_lat_height_table,
        mo,
        pd,
        pygef,
        write_gi_db_to_gpkg,
    )


if __name__ == "__main__":
    app.run()
