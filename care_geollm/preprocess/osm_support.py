import numpy as np

def fetch_osm_support(place_or_polygon, network_type='walk'):
    """Optional OSM-derived supporting layers. Requires osmnx.

    The manuscript treats street/service locations as derived supporting layers rather than one
    of the three benchmark datasets. This helper records them separately for provenance.
    """
    import osmnx as ox
    if isinstance(place_or_polygon, str):
        G = ox.graph_from_place(place_or_polygon, network_type=network_type, simplify=True)
        pois = ox.features_from_place(place_or_polygon, tags={
            'amenity': ['hospital','clinic','pharmacy','school','community_centre'],
            'shop': ['supermarket'], 'public_transport': True
        })
    else:
        G = ox.graph_from_polygon(place_or_polygon, network_type=network_type, simplify=True)
        pois = ox.features_from_polygon(place_or_polygon, tags={'amenity':True})
    return G, pois
