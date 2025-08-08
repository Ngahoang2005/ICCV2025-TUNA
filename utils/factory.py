def get_model(model_name, args):
    name = model_name.lower()
    if name == "tuna":
        from models.tuna import Learner
    elif name == "tuna2":
        from models.tuna2 import Learner
    else:
        assert 0
    return Learner(args)
