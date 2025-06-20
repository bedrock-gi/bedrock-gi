import marimo

__generated_with = "0.13.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from pyproj import CRS, Transformer
    from pyproj.crs.crs import CompoundCRS
    return CRS, CompoundCRS, Transformer


@app.cell
def _(CRS, Transformer):
    # https://pyproj4.github.io/pyproj/stable/advanced_examples.html#promote-crs-to-3d
    wgs84 = CRS("EPSG:4326")
    swiss_proj = CRS("EPSG:2056")
    transformer = Transformer.from_crs(wgs84, swiss_proj, always_xy=True)
    # 2D Transformation
    print(f"2D transform of point {(8.37909, 47.01987, 1000)} from {wgs84} to {swiss_proj} gives:")
    print(transformer.transform(8.37909, 47.01987, 1000))

    wgs84_3d = wgs84.to_3d()
    swiss_proj_ellipsoidal_3d = swiss_proj.to_3d()
    transformer_ellipsoidal_3d = Transformer.from_crs(
        wgs84_3d,
        swiss_proj_ellipsoidal_3d,
        always_xy=True,
    )
    # 3D Transformation
    print(f"3D transform of point {(8.37909, 47.01987, 1000)} from {wgs84_3d.to_string()} to {swiss_proj_ellipsoidal_3d} gives:")
    print(transformer_ellipsoidal_3d.transform(8.37909, 47.01987, 1000))

    return swiss_proj, transformer_ellipsoidal_3d, wgs84_3d


@app.cell
def _(
    CRS,
    CompoundCRS,
    Transformer,
    swiss_proj,
    transformer_ellipsoidal_3d,
    wgs84_3d,
):
    # https://pyproj4.github.io/pyproj/stable/build_crs.html#compound-crs
    swiss_lhn95_height = CRS("EPSG:5729")
    swiss_compound = CompoundCRS(
        name="CH1903+ / LV95 + LHN95 height",
        components=[swiss_proj, swiss_lhn95_height]
    )
    transformer_wgs84_3d_to_swiss_compound = Transformer.from_crs(
        wgs84_3d,
        swiss_compound,
        always_xy=True,
    )
    print(transformer_ellipsoidal_3d.transform(8.37909, 47.01987, 1000))
    return


@app.cell
def _(CRS):
    uk_grid_27700 = CRS(27700)
    uk_3d = uk_grid_27700.to_3d()
    swiss_2056 = CRS("EPSG:2056")
    swiss_3d = swiss_2056.to_3d()
    egm2008_3855 = CRS(3855)
    rdnew_nap_7415 = CRS(7415)
    wgs84_egm2008_9518 = CRS(9518)
    crs_components = extract_crs_components(swiss_3d)
    crs_components
    return


@app.function
def extract_crs_components(compound_crs):
    """Extract horizontal and vertical CRS from a compound CRS"""
    if not compound_crs.is_compound:
        return compound_crs, None

    horizontal_crs = None
    vertical_crs = None

    for sub_crs in compound_crs.sub_crs_list:
        if sub_crs.is_projected or sub_crs.is_geographic:
            print(f"Horizontal CRS {sub_crs.name} is a {sub_crs.type_name} and has EPSG:{sub_crs.to_epsg()}.")
            horizontal_crs = sub_crs
        elif sub_crs.is_vertical:
            print(f"Vertical CRS {sub_crs.name} has EPSG:{sub_crs.to_epsg()}.")
            vertical_crs = sub_crs
        else:
            print(f"This CRS is not horizontal (projected or geographic) nor vertical: {sub_crs.type_name}")

    return horizontal_crs, vertical_crs


if __name__ == "__main__":
    app.run()
