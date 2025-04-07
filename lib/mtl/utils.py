def convert_state_dict_keys(state_dict):
    """
    Converts keys in the state_dict so that they are consistent with what TLModel expects.
    Specifically, if a key starts with "shared_backbone" or "heads", we prepend "mtl_net.".
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("shared_backbone") or key.startswith("heads"):
            new_key = "mtl_net." + key
        else:
            new_key = key
        new_state_dict[new_key] = value
    return new_state_dict
