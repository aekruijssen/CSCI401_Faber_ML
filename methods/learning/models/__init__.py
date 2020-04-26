def get_model_by_name(name):
    if name == 'mlp':
        from .mlp_model import MlpModel
        return MlpModel
    elif name == 'lstm':
        from .lstm_model import LSTMModel
        return LSTMModel
    else:
        raise ValueError("Model named '{}' doesn't exist!".format(name))
