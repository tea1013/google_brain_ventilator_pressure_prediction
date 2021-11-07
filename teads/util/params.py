import pickle
from typing import Dict


class Params:
    def dump_dict(d: Dict, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(d, f)

    def load_dict(path: str) -> Dict:
        with open(path, "rb") as f:
            d = pickle.load(f)

        return d
