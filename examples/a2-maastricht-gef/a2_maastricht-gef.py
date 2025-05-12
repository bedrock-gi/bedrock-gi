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

__generated_with = "0.13.2"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        """
        # GEF Data for A2 Tunnel Maastricht

        This notebook demonstrates how to 

        1. Read in GEF files using [pygef](https://cemsbv.github.io/pygef/)
        1. Use `bedrock-ge` to load Ground Investigation (GI) data from these GEF files
        1. Convert that data into a standardized GI database using `bedrock-ge`
        1. Transform the GI data into 3D GIS features with proper coordinates and geometry ([OGC Simple Feature Access](https://en.wikipedia.org/wiki/Simple_Features))
        1. Explore and analyze the GI data using:
          * Interactive filtering with Pandas dataframes
          * Visualization on interactive maps with GeoPandas
        1. Export the processed GI database to a GeoPackage file for use in GIS software

        ## Context

        The Koning Willem-Alexander Tunnel is a double-deck tunnel for motorized traffic in the Dutch city of Maastricht. The tunnel has a length of 2.5 kilometers (lower tunnel tubes) and 2.3 kilometers (upper tunnel tubes).

        The tunnel has moved the old A2 highway underground. It previously formed a barrier for the city and slowed traffic.

        ### Geology


        The soil here consists of a coarse gravel layer with limestone limestone with karst phenomena underneath. There are also faults.
        Zwerfkeien

        [Geotechniek-en-Risicos bij A2 Maastricht](https://www.cob.nl/magazines-brochures-en-nieuws/verdieping/verdieping-sept2012/geotechniek-en-risicos-bij-a2-maastricht/)
        [Source](https://archisarchief.cultureelerfgoed.nl/Archis2/Archeorapporten/24/AR26905/RAP%202709_4130060%20Maastricht%20A2-traverse.pdf)

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

        First, let's check in which projected coordinate system the provided data was recorded
        """
    )
    return


@app.cell
def _(boreholes):
    boreholes[1].delivered_location.srs_name
    return


@app.cell
def _(boreholes):
    code = {bore.delivered_location.srs_name for bore in boreholes}.pop()
    # epsg_code = code.split("EPSG::")[-1]
    # crs = f"EPSG:{epsg_code}"
    crs = "EPSG:7415"
    return (crs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        EPSG:28992 is [Rijksdriehoekscoördinaten](https://nl.wikipedia.org/wiki/Rijksdriehoeksco%C3%B6rdinaten), also called Amersfoort / RD New.

        EPSG:7415 is with Amersfoort / RD New + NAP height. ie with elevation which we need for 3D geometry
        """
    )
    return


@app.cell
def _():
    wgs = "EPSG:4326"
    return


@app.cell
def _():
    project_uid = "Maastricht A2"
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


@app.cell
def _(create_lon_lat_height_table, crs, locations):
    create_lon_lat_height_table(locations, crs).explore()
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

        Finally, lets write it to an actual geospatial database file, so we can share our GI data with others. For example, to reuse it in other notebooks, create dashboards, access the GI data in QGIS or ArcGIS, and more...

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
