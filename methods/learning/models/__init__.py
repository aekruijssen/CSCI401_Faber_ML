def get_model_by_name(name):
    if name == 'mlp':
        from .mlp_model import MlpModel
        return MlpModel
    else:
        raise ValueError("Model named '{}' doesn't exist!".format(name))
