# https://github.com/e-mon/lish-moa/blob/master/encode.py

import base64
import gzip
import sys
from pathlib import Path
from typing import List

import git

template = """
import gzip
import base64
import os
from pathlib import Path
from typing import Dict
# this is base64 encoded source code
file_data: Dict = {file_data}
for path, encoded in file_data.items():
    path = Path(path)
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(gzip.decompress(base64.b64decode(encoded)))
# output current commit hash
print('{commit_hash}')
"""


def get_current_commit_hash():
    repo = git.Repo(search_parent_directories=True)
    return repo.head.object.hexsha


def encode_file(path: Path) -> str:
    compressed = gzip.compress(path.read_bytes(), compresslevel=9)
    return base64.b64encode(compressed).decode("utf-8")


def build_script(modules: List[str]):
    all_data = {}
    for module in modules:
        to_encode = list(Path(module).glob("**/*.py"))
        file_data = {str(path).replace("\\", "/"): encode_file(path) for path in to_encode}
        all_data.update(file_data)

    output_path = Path(f"build/build.py")
    output_path.parent.mkdir(exist_ok=True)
    output_path.write_text(template.replace("{file_data}", str(all_data)).replace("{commit_hash}", get_current_commit_hash()), encoding="utf8")


if __name__ == "__main__":
    args = sys.argv
    roots = args[1:]

    build_script(roots)
