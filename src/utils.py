import json


def read_json(file_path: str) -> dict:
    """Reads json file from the given path.

    Args:
        file_path (str): Location of the file

    Returns:
        dict: Json content formatted as python dictionary in most cases
    """
    with open(file_path, "r") as f:
        return json.load(f)
