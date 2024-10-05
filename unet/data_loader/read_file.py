import json
from pathlib import Path
from typing import Any, Dict, Union


def read_json(
    file_path: Union[str, Path],
) -> Dict[str, Any]:
    """Read json file.

    Examples
    --------
    >>> from unet.data_loader.read_file import read_json
    """
    with open(file_path, 'r') as file:
        return json.load(file)
