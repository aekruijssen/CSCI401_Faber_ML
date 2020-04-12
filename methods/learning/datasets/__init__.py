def get_data_loader_by_name(name):
    if name == 'maps':
        from .maps.maps_data_loader import MapsDataLoader
        return MapsDataLoader
    else:
        raise ValueError("Data loader named '{}' not found!".format(name))