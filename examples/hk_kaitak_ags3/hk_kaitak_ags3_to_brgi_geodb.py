# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "bedrock-ge==0.2.4",
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
# ]
# ///

import marimo

__generated_with = "0.13.15"
app = marimo.App(
    app_title="Kai Tak, HK AGS 3 data to bedrock_ge.gi geodatabase",
)


@app.cell
def _():
    # %pip install bedrock-ge geopandas folium mapclassify marimo --quiet

    import io
    import platform
    import sys
    import zipfile
    from pathlib import Path

    import folium
    import geopandas as gpd
    import mapclassify
    import marimo as mo
    import matplotlib
    import numpy as np
    import pandas as pd
    import requests
    from pyproj import CRS, Transformer
    from pyproj.crs import CompoundCRS
    from shapely import wkt

    from bedrock_ge.gi.ags.read import ags_to_dfs
    from bedrock_ge.gi.ags.transform import ags3_db_to_no_gis_brgi_db
    from bedrock_ge.gi.concatenate import concatenate_databases
    from bedrock_ge.gi.gis_geometry import calculate_gis_geometry
    from bedrock_ge.gi.validate import check_brgi_database, check_no_gis_brgi_database
    from bedrock_ge.gi.write import write_gi_db_to_gpkg

    print(platform.system())
    print(sys.version)
    # print(sys.executable)
    return (
        CRS,
        ags3_db_to_no_gis_brgi_db,
        ags_to_dfs,
        calculate_gis_geometry,
        check_brgi_database,
        check_no_gis_brgi_database,
        concatenate_databases,
        gpd,
        io,
        mo,
        pd,
        platform,
        requests,
        write_gi_db_to_gpkg,
        zipfile,
    )


@app.cell
def _(
    ags3_db_to_no_gis_brgi_db,
    ags_to_dfs,
    check_no_gis_brgi_database,
    concatenate_databases,
    zipfile,
):
    def zip_of_ags3s_to_bedrock_gi_database(zip, crs):
        """Read AGS 3 files from a ZIP archive and convert them to a dictionary of pandas dataframes."""
        brgi_db = {}
        with zipfile.ZipFile(zip) as zip_ref:
            # Iterate over files and directories in the .zip archive
            for file_name in zip_ref.namelist():
                # Only process files that have an .ags or .AGS extension
                if file_name.lower().endswith(".ags"):
                    print(f"\n🖥️ Processing {file_name} ...")
                    with zip_ref.open(file_name) as ags3_file:
                        # Convert content of a single AGS 3 file to a Dictionary of pandas dataframes (a database)
                        ags3_db = ags_to_dfs(ags3_file)
                    report_no = file_name.split("/")[0]
                    ags3_db["PROJ"]["REPORT_NO"] = int(report_no)
                    project_uid = f"{ags3_db['PROJ']['PROJ_ID'].iloc[0]}_{file_name}"
                    ags3_db["PROJ"]["project_uid"] = project_uid
                    # Remove (Static) CPT AGS 3 group 'STCN' from brgi_db, because CPT data processing needs to be reviewed.
                    # Not efficient to create a GIS point for every point where a CPT measures a value.
                    if "STCN" in ags3_db.keys():
                        del ags3_db["STCN"]
                    # Create GI data tables with bedrock-ge names and add columns (project_uid, location_uid, sample_uid),
                    # such that data from multiple AGS files can be combined
                    brgi_db_from_1_ags3_file = ags3_db_to_no_gis_brgi_db(ags3_db, crs)
                    print(
                        f"🧐 Validating the Bedrock GI database from AGS file {file_name}..."
                    )
                    check_no_gis_brgi_database(brgi_db_from_1_ags3_file)
                    print(
                        f"\n✅ Successfully converted {file_name} to Bedrock GI database and validated!\n"
                    )
                    print(
                        f"🧵 Concatenating Bedrock GI database for {file_name} to existing Bedrock GI database...\n"
                    )
                    brgi_db = concatenate_databases(brgi_db, brgi_db_from_1_ags3_file)

                    # Drop all rows that have completely duplicate rows in the Project table
                    brgi_db["Project"] = brgi_db["Project"].drop_duplicates()
                    # Then drop all that unfortunately still have a duplicate project_uid
                    brgi_db["Project"] = brgi_db["Project"].drop_duplicates(
                        subset="project_uid", keep="first"
                    )
        return brgi_db

    return (zip_of_ags3s_to_bedrock_gi_database,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    # AGS 3 Data in Kai Tak, Hong Kong

    This notebook demonstrates how to:

    1. Use `bedrock-ge` to load Ground Investigation (GI) data from AGS 3 files (a common GI data format in Hong Kong)
    2. Convert the AGS 3 data into a standardized GI database using `bedrock-ge`
    3. Transform the GI data into 3D GIS features with proper coordinates and geometry ([OGC Simple Feature Access](https://en.wikipedia.org/wiki/Simple_Features))
    4. Explore and analyze the GI data using:
       - Interactive filtering with Pandas dataframes
       - Visualization on interactive maps with GeoPandas
    5. Export the processed GI database to a GeoPackage file for use in GIS software

    We'll work with real GI data from the Kai Tak neighborhood in Hong Kong.

    ## Context

    Kai Tak is a neighborhood in Kowloon, Hong Kong. One of the highlights of Kai Tak used to be its airport. It holds a special place in aviation history due to its unique and challenging approach, which involved pilots making a steep descent over a densely populated area while making a sharp turn at the same time and then landing on a single runway that jutted out into Victoria Harbor. [Landing at Kai Tak Airport | YouTube](https://www.youtube.com/watch?v=OtnL4KYVtDE)

    In 1998, the new Hong Kong International Airport opened, and operations at Kai Tak Airport were ceased. After the closure, the former Kai Tak Airport and surrounding neighborhood underwent a massive redevelopment project to transform it into a new residential and commercial district, which is still continuing today.

    Have a look at the [Kai Tak Speckle Project](https://app.speckle.systems/projects/013aaf06e7/models/0e43d1f003,a739490298) to get an idea what Kai Tak looks like now. (Developments are going fast, so [Google Maps 3D](https://www.google.com/maps/@22.3065043,114.2020499,462a,35y,343.1h,75.5t/data=!3m1!1e3?entry=ttu) is a bit outdated.)

    ## The Kai Tak AGS 3 ground investigation data

    Ground Investigation Data for all of Hong Kong can be found here:  
    [GEO Data for Public Use](https://www.ginfo.cedd.gov.hk/GEOOpenData/eng/Default.aspx) → [Ground Investigation (GI) and Laboratory Test (LT) Records](https://www.ginfo.cedd.gov.hk/GEOOpenData/eng/GI.aspx)

    The Ground Investigation data specific to the Kai Tak neighborhood in Hong Kong can be found in the `bedrock-ge` GitHub repository:  
    [`github.com/bedrock-engineer/bedrock-ge/examples/hk_kaitak_ags3/kaitak_ags3.zip`](https://github.com/bedrock-engineer/bedrock-ge/blob/main/examples/hk_kaitak_ags3/kaitak_ags3.zip).  
    This archive contains GI data from 88 AGS 3 files, with a total of 834 locations (boreholes and Cone Penetration Tests).

    One of the AGS 3 files with GI data was left outside the ZIP archive, such that you can have a look at the structure of an AGS 3 file:  
    [`github.com/bedrock-engineer/bedrock-ge/examples/hk_kaitak_ags3/ASD012162 AGS.ags`](https://github.com/bedrock-engineer/bedrock-ge/blob/main/examples/hk_kaitak_ags3/64475_ASD012162%20AGS.ags)

    ### Getting the AGS 3 files

    To make it easy to run this notebook on your computer (locally) in the browser (remotely) in [marimo.app](https://marimo.app/) or [Google Colab](https://colab.research.google.com/), the code below requests the ZIP archive from GitHub and directly processes it. However, you can also download the ZIP from GitHub (link above) or directly from this notebook [by clicking this raw.githubusercontent.com raw url [ ↓ ]](http://raw.githubusercontent.com/bedrock-engineer/bedrock-ge/main/examples/hk_kaitak_ags3/kaitak_ags3.zip). 

    The cell below works as is, but has a commented line 2, to help you in case you have downloaded the ZIP, and want to use that downloaded ZIP in this notebook.
    """
    )
    return


@app.cell
def _(io, requests):
    # Read ZIP from disk after downloading manually
    # zip = Path.home() / "Downloads" / "kaitak_ags3.zip"

    # Request ZIP from GitHub
    raw_githubusercontent_url = "https://raw.githubusercontent.com/bedrock-engineer/bedrock-ge/main/examples/hk_kaitak_ags3/kaitak_ags3.zip"
    zip = io.BytesIO(requests.get(raw_githubusercontent_url).content)
    return (zip,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## Converting the AGS 3 files to a relational database

    A relational database is a database with multiple tables that are linked to each other with relations. This type of database is ideal for storing  GI data, given its hierarchical structure:

    ```
    Project
     └───Location
          ├───InSitu_TEST
          └───Sample
              └───Lab_TEST
    ```

    Where `Project`, `Location`, `InSitu_TEST`, `Sample` and `Lab_TEST` are all tables that are linked to each other with the hierarchical structure shown above, meaning that all relations are many-to-one:

    - Each GI location (many) is related to one project.
    - Each sample or in-situ test (many) is related to one GI location.
    - Each lab test is related to one sample.

    In Python it's convenient to represent a relational database as a dictionary of dataframe's.

    ### Converting AGS 3 files to a dictionary of dataframes

    The AGS 3 files can be converted to a dictionary of dataframes using the function `list_of_ags3s_to_bedrock_gi_database(ags3_file_paths, CRS)`. The result is shown below. Have a look at the different tables and the data in those tables. Make sure to use the search and filter functionality to explore the data if you're using marimo to run this notebook!

    Notice the additional columns that were added to the tables by `bedrock-ge`:

    - To make sure that the primary keys of the GI data tables are unique when putting data from multiple AGS files together:  
        `project_uid`, `location_uid`, `sample_uid`
    - To make it possible to generate 3D GIS geometry for the `Location`, `Sample` and `InSitu_TEST` tables:  
        In the `Location` table: `easting`, `northing`, `ground_level_elevation`, `depth_to_base`  
      In the `Sample` and `InSitu_TEST` tables: `depth_to_top` and, in case the test or sample is taken over a depth interval, `depth_to_base`.
    """
    )
    return


@app.cell
def _():
    from bedrock_ge.gi.ags3 import ags3_to_brgi_db_mapping
    from bedrock_ge.gi.ags_parser import ags_to_brgi_db_mapping
    from bedrock_ge.gi.db_operations import merge_databases
    from bedrock_ge.gi.mapper import map_to_brgi_db
    from bedrock_ge.gi.geospatial import create_brgi_geospatial_database
    from bedrock_ge.gi.schemas import InSituTestSchema
    return (
        ags_to_brgi_db_mapping,
        create_brgi_geospatial_database,
        map_to_brgi_db,
        merge_databases,
    )


@app.cell
def _(
    CRS,
    ags_to_brgi_db_mapping,
    map_to_brgi_db,
    merge_databases,
    zip,
    zipfile,
):
    projected_crs = CRS("EPSG:2326")
    vertrical_crs = CRS("EPSG:5738")
    ags3_file_brgi_dbs = []
    with zipfile.ZipFile(zip) as zip_ref:
        # Iterate over files and directories in the .zip archive
        for i, file_name in enumerate(zip_ref.namelist()):
            # Only process files that have an .ags or .AGS extension
            if file_name.lower().endswith(".ags"):
                print(f"\n🖥️ Processing {file_name} ...")
                with zip_ref.open(file_name) as ags3_file:
                    # 1. Convert content of a single AGS 3 file to a Bedrock GI Mapping.
                    # 2. Map the mapping object to a Bedrock GI Database.
                    # 3. Append the Bedrock GI Database to the list of Bedrock GI
                    #    Databases, that were created from single AGS 3 files.
                    ags3_file_brgi_dbs.append(
                        map_to_brgi_db(
                            ags_to_brgi_db_mapping(
                                ags3_file, projected_crs, vertrical_crs
                            )
                        )
                    )

    brgi_db = merge_databases(ags3_file_brgi_dbs)
    return (brgi_db,)


@app.cell
def _(brgi_db):
    brgi_db.Project
    return


@app.cell
def _(CRS, pd, zip, zip_of_ags3s_to_bedrock_gi_database):
    brgi_db_old = zip_of_ags3s_to_bedrock_gi_database(zip, CRS("EPSG:2326"))

    # Some ISPT_NVAL (SPT count) are not numeric, e.g. "100/0.29"
    # When converting to numeric, these non-numeric values are converted to NaN
    brgi_db_old["InSitu_ISPT"]["ISPT_NVAL"] = pd.to_numeric(
        brgi_db_old["InSitu_ISPT"]["ISPT_NVAL"], errors="coerce"
    )
    return (brgi_db_old,)


@app.cell(hide_code=True)
def _(brgi_db_old, mo):
    sel_brgi_table = mo.ui.dropdown(brgi_db_old, value="Project")
    mo.md(f"Select the Bedrock GI table you want to explore: {sel_brgi_table}")
    return (sel_brgi_table,)


@app.cell(hide_code=True)
def _(sel_brgi_table):
    sel_brgi_table.value
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Relational database to 3D geospatial database
    A  database is a relational database that has been enhanced to store  data. There are two broad categories of  data:

    1. [Raster data](https://en.wikipedia.org/wiki/GIS_file_format#Raster_formats): geographic information as a grid of pixels (cells), where each pixel stores a value corresponding to a specific location and attribute, such as elevation, temperature, or land cover. So, a Digital Elevation Model (DEM) is an example of GIS raster data.
    2. [Vector data](https://en.wikipedia.org/wiki/GIS_file_format#Vector_formats): tables in which each row contains:
        - [Simple feature GIS geometry](https://en.wikipedia.org/wiki/Simple_Features), represented as [Well-Known Text](https://en.wikipedia.org/wiki/Well-known_text_representation_of_geometry). For example in the `InSitu_GEOL` and `InSitu_ISPT` tables:  
            `InSitu_GEOL`: a depth interval in a borehole where sand was found.  
            `InSitu_ISPT`: a point in a borehole where an SPT test was performed.
        - Attributes that describe the GIS geometry. For example in the `InSitu_GEOL` and `InSitu_ISPT` tables:  
            `InSitu_GEOL`: the geology code (`GEOL_GEOL`), general description of stratum (`GEOL_DESC`), etc.  
            `InSitu_ISPT`: the SPT N-value (`ISPT_NVAL`), energy ratio of the hammer (`ISPT_ERAT`), etc.

    So, when representing GI data as 3D GIS features, we are talking about GIS vector data.

    ### From GI dataframe to `geopandas.GeoDataFrame` 

    In order to construct the 3D simple feature GIS geometry of the `Location`s, `Sample`s and `InSitu_TEST`s, a few more columns have to be calculated for each of these tables: `elevation_at_top` and `elevation_at_base` if the in-situ test or sample was taken over a depth interval.

    The 3D simple feature GIS geometry as [WKT](https://en.wikipedia.org/wiki/Well-known_text_representation_of_geometry) for point tests and samples:  
    `POINT (easting northing elevation_at_top)`

    The 3D simple feature GIS geometry as WKT for in-situ tests and samples taken over a depth interval:  
    `LINESTRING (easting northing elevation_at_top, easting northing elevation_at_base)`

    Additionally, a `LonLatHeight` table is created which contains the GI locations at ground level in WGS84 - World Geodetic System 1984 - EPSG:4326 coordinates (Longitude, Latitude, Ellipsoidal Height), which in WKT looks like:  
    `POINT (longitude latitude wgs84_ground_level_height)`

    The reason for creating the `LonLatHeight` table is that vertical lines in projected Coordinate Reference Systems (CRS) are often not rendered nicely by default in all web-mapping software. Vertical lines are often not visible when looking at a map from above, and not all web-mapping software is capable of handling geometry in non-WGS84, i.e. (Lon, Lat) coordinates.
    """
    )
    return


@app.cell
def _(brgi_db, brgi_db_old):
    # Check whether the correct number of rows was removed when de-duplicating rows

    from collections import Counter

    from bedrock_ge.gi.io_utils import convert_dtypes_object_to_string

    new_df = brgi_db.Sample
    print(new_df.columns, end="\n\n")
    df = convert_dtypes_object_to_string(
        brgi_db_old["Sample"].copy().convert_dtypes()
    )
    print(df.columns, end="\n\n")
    df = df.drop(
        ["project_uid", "location_uid", "sample_uid"],
        axis=1,
    )
    df = df.drop_duplicates()

    compare_cols = [
        "depth_to_top",
        "HOLE_ID",
        "SAMP_TOP",
        "SAMP_REF",
        "SAMP_TYPE",
        "SAMP_DIA",
        "SAMP_BASE",
        "SAMP_DESC",
        "SAMP_UBLO",
        "SAMP_REM",
        "SAMP_DATE",
    ]
    if len(new_df) != len(df):
        raise ValueError(
            "The new and old Bedrock GI database tables don't have the same length."
        )

    for col in compare_cols:
        has_same_vals = Counter(new_df[col]) == Counter(df[col])
        if has_same_vals:
            print(f"✅ {col} is the same in the new and old Bedrock GI databases.")
        else:
            print(f"❌ {col} is NOT the same after de-duplication.")
            print("New:")
            print(new_df[col])
            print("Old:")
            print(df[col])

    new_df
    return


@app.cell
def _(brgi_db, create_brgi_geospatial_database):
    brgi_geodb = create_brgi_geospatial_database(brgi_db)
    return (brgi_geodb,)


@app.cell
def _(brgi_geodb):
    brgi_geodb.LonLatHeight.explore()
    return


@app.cell
def _(brgi_db_old, calculate_gis_geometry, check_brgi_database):
    brgi_geodb_old = calculate_gis_geometry(brgi_db_old)
    check_brgi_database(brgi_geodb_old)
    return (brgi_geodb_old,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Ground Investigation data exploration

    After creating the Bedrock GI 3D  Database `brgi_geodb` - which is a dictionary of [`geopandas.GeoDataFrame`](https://geopandas.org/en/stable/docs/reference/api/geopandas.GeoDataFrame.html#geopandas.GeoDataFrame)s - you can explore the Kai Tak Ground Investigation data on an interactive map by applying the [`geopandas.GeoDataFrame.explore()`](https://geopandas.org/en/stable/docs/reference/api/geopandas.GeoDataFrame.explore.html#geopandas.GeoDataFrame.explore) method to the different tables in the `brgi_geodb`.

    Do note that this works best on the tables with `POINT` GIS geometry such as `LonLatHeight` or `InSitu_ISPT`. Tables with vertical `LINESTRING` GIS geometry, such as `Location`, `InSitu_GEOL` or `InSitu_WETH`, display very small on the `gdf.explore()` `leaflet`-based interactive map, and don't show at all on the `matplotlib`-based `gdf.plot()`.
    """
    )
    return


@app.cell
def _(brgi_geodb_old):
    brgi_geodb_old["LonLatHeight"].explore()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    With marimo's built-in data exploration tables and dataframes, it's also really easy to filter and visualize the GI data.

    For example, in the `InSitu_ISPT` table (SPT data) you could apply a filter to the `ISPT_NVAL` (SPT N-value) of e.g. 1 - 10. When you then select those rows and then scroll to the map below, you'll see all the locations where soft soils were encountered.
    """
    )
    return


@app.cell(hide_code=True)
def _(brgi_db_old, mo):
    explore_brgi_table = mo.ui.dropdown(brgi_db_old, value="InSitu_ISPT")
    mo.md(f"Select the GI table you want to explore: {explore_brgi_table}")
    return (explore_brgi_table,)


@app.cell(hide_code=True)
def _(explore_brgi_table, mo):
    filtered_table = mo.ui.table(explore_brgi_table.value)
    filtered_table
    return (filtered_table,)


@app.cell(hide_code=True)
def _(brgi_geodb_old, filtered_table, gpd, mo):
    def gi_exploration_map(filtered_brgi_table):
        if "location_uid" not in filtered_brgi_table.value.columns:
            output = mo.md(
                "No interactive map with the data selected in the table above can be shown, because the you're exploring isn't linked to the `LonLatHeight` table with a `location_uid` column, i.e. doesn't have `location_uid` as a foreign key."
            ).callout("warn")
        else:
            filtered_df = filtered_brgi_table.value.merge(
                brgi_geodb_old["LonLatHeight"], on="location_uid", how="inner"
            )
            filtered_gdf = gpd.GeoDataFrame(
                filtered_df,
                geometry=filtered_df["geometry"],
                crs="EPSG:4326",  # 4326 is the WGS84 (lon, lat) EPSG code
            )
            output = filtered_gdf.explore()
        return output

    gi_exploration_map(filtered_table)
    return (gi_exploration_map,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Something else you might be interested in, is where the weathering grade of the soil or rock is low. Weathering grades range from `I` (Fresh Rock) to `VI` (Residual Soil). All rock with a weathering grade of `III` (Moderately Decomposed) or better is still considered competent rock.

    The weathering grades can be found in the `WETH_GRAD` column in the `InSitu_WETH` table (Weathering data). Therefore, to find all competent rock, we need to filter out all the rows that contain a `V`, which you can do in the widget below.

    That widget also shows the Python code that creates the filter:

    ```python
    df_next = df
    df_next = df_next[~((df_next["WETH_GRAD"].str.contains("V")))]
    ```
    """
    )
    return


@app.cell(hide_code=True)
def _(brgi_db_old, mo):
    explore_brgi_df = mo.ui.dropdown(brgi_db_old, value="InSitu_WETH")
    mo.md(f"Select the GI table you want to explore: {explore_brgi_df}")
    return (explore_brgi_df,)


@app.cell
def _(explore_brgi_df, mo):
    filtered_df = mo.ui.dataframe(explore_brgi_df.value)
    filtered_df
    return (filtered_df,)


@app.cell
def _(filtered_df, gi_exploration_map):
    gi_exploration_map(filtered_df)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Saving the GI  database as a GeoPackage (.gpkg)

    Finally, lets write, i.e. persist `brgi_geodb` - a Python dictionary of `geopandas.GeoDataFrames` - to an actual  database file, so we can share our GI data with others.
    For example, to reuse it in other notebooks, create dashboards, access the GI data in QGIS or ArcGIS, and more...

    A GeoPackage is an OGC-standardized extension of SQLite (a relational database in a single file, .sqlite or .db) that allows you to store any type of GIS data (both raster as well as vector data) in a single file that has the .gpkg extension. Therefore, many (open-source) GIS software packages support GeoPackage!

    > [What about Shapefile and GeoJSON?](#what-about-shapefile-and-geojson)
    """
    )
    return


@app.cell
def _(brgi_geodb_old, mo, platform, write_gi_db_to_gpkg):
    output = None
    if platform.system() != "Emscripten":
        write_gi_db_to_gpkg(brgi_geodb_old, mo.notebook_dir() / "kaitak_gi.gpkg")
    else:
        output = mo.md(
            "Writing a GeoPackage from WebAssembly (marimo playground) causes geopandas to think that the GeoDataFrames in the `brgi_geodb` don't have a geometry column. You can [download the GeoPackage from GitHub](https://github.com/bedrock-engineer/bedrock-ge/blob/main/examples/hk_kaitak_ags3/kaitak_gi.gpkg)"
        ).callout("warn")
    output
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## What's next?

    As mentioned above, the `kaitak_gi.gpkg` GeoPackage can be loaded into QGIS or ArcGIS. QGIS and ArcGIS have [connectors for the Speckle platform](https://www.speckle.systems/connectors), which allows you to publish GIS data to Speckle.

    With the Speckle viewer you can visualize the GI data in context with data from other AEC software such as Civil3D (Click the balloon!):

    <iframe title="Speckle" src="https://app.speckle.systems/projects/013aaf06e7/models/1cbe68ed69,44c8d1ecae,9535541c2b,a739490298,ff81bfa02b#embed=%7B%22isEnabled%22%3Atrue%7D" width="100%" height="400" frameborder="0"></iframe>

    Additionally, you can load the GI data in other software that Speckle has a connector for, such as Rhino / Grasshopper to enable parametric geotechnical engineering workflows.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## What about Shapefile and GeoJSON?

    ### Shapefile

    Bluntly put, Shapefile is a bad format.

    Among other problems, Shapefile isn't just a single file. One has to at least share three files [(\*.shp, \*.dbf, \*.shx)](https://en.wikipedia.org/wiki/Shapefile#Mandatory_files), which doesn't include the definition of a CRS. In case that doesn't sound terrible enough to you yet, please have a look at the fantastic website [switchfromshapefile.org](http://switchfromshapefile.org/).

    ### GeoJSON

    GeoJSON is a nice, human readable file format for GIS vector data, which is especially useful for web services, but has a few drawbacks:

    - Although it is technically possible to use GeoJSON with more CRSs, the [specification states clearly](https://tools.ietf.org/html/rfc7946#section-4) that WGS84, with EPSG:4326 and coordinates (Lon, Lat, Height), is the only CRS that should be used in GeoJSON (see [switchfromshapefile.org](http://switchfromshapefile.org/#geojson)).
    - GeoJSON support in ArcGIS isn't fantastic. You have to go through [Geoprocessing - JSON to Features conversion tool](https://pro.arcgis.com/en/pro-app/latest/tool-reference/conversion/json-to-features.htm) to add a GeoJSON to your ArcGIS project, which is a bit cumbersome.
    """
    )
    return


if __name__ == "__main__":
    app.run()
