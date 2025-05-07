import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import transform
import contextily as ctx

def shift_longitude(geom):
    """
    Shift longitudes to ensure that the data stays within [-180, 180] longitude range.
    This ensures that geometries crossing the dateline appear correctly on a map.
    """
    def wrap(lon, lat):
        return (lon + 360 if lon < 0 else lon, lat)

    return transform(lambda x, y: wrap(x, y), geom)

def plot_eez_with_basemap(shapefile_path):
    try:
        print(f"Reading shapefile: {shapefile_path}")
        gdf = gpd.read_file(shapefile_path)

        if len(gdf) == 0:
            print("Shapefile loaded, but it's empty.")
            return

        # Ensure it's in lat/lon (EPSG:4326)
        if gdf.crs is None:
            gdf.set_crs('EPSG:4326', inplace=True)
        else:
            gdf = gdf.to_crs('EPSG:4326')

        # Apply longitude shifting to handle dateline
        gdf['geometry'] = gdf['geometry'].apply(shift_longitude)

        # Convert to Web Mercator (EPSG:3857) for contextily basemap
        gdf_web_mercator = gdf.to_crs('EPSG:3857')

        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        gdf_web_mercator.plot(ax=ax, edgecolor='blue', facecolor='lightcyan', alpha=0.5)
        
        # Add basemap
        ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik,ssl_verify=False)
        
        ax.set_title("EEZ Plot with Basemap")
        plt.grid(True)
        plt.show()

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

plot_eez_with_basemap("shapefile/country_eez/FJI.shp")