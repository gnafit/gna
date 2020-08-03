import ROOT

class GNATypeError(Exception):
    """TransformationTypes.TypeError wrapper"""
    pass

class GNABindError(Exception):
    """TransformationTypes.BindError wrapper"""
    pass

def load_user_exceptions():
    ROOT.UserExceptions.update({
        "KeyError": KeyError,
        "IndexError": IndexError,
        "TransformationTypes::TypeError": GNATypeError,
        "TransformationTypes::BindError": GNABindError,
    })

