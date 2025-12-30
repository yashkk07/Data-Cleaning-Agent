from typing import Any, Dict


def detect_json_shape(obj: Any) -> str:
    """
    Detects JSON structure shape.

    Returns:
    - records_list
    - column_dict
    - nested_records
    - single_object
    """

    if isinstance(obj, list):
        if obj and isinstance(obj[0], dict):
            return "records_list"
        return "unknown"

    if isinstance(obj, dict):
        values = list(obj.values())

        # Dict of lists → column-oriented
        if values and all(isinstance(v, list) for v in values):
            return "column_dict"

        # Nested records (first level)
        for v in values:
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return "nested_records"

        return "single_object"

    return "unknown"
