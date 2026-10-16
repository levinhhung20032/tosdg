"""JSON-schema loading/validation. Schemas live in <repo>/schemas/<name>.schema.json.

Contents:
  repo_root
  class SchemaError
  schema_errors
  validate_instance
  load_json
  (+ 1 private helper)
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import jsonschema


def repo_root() -> Path:
    env = os.environ.get("TOSDG_ROOT")
    return Path(env) if env else Path(__file__).resolve().parents[2]


class SchemaError(ValueError):
    pass


@lru_cache(maxsize=None)
def _validator(name: str) -> jsonschema.Draft202012Validator:
    path = repo_root() / "schemas" / f"{name}.schema.json"
    schema = json.loads(path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    return jsonschema.Draft202012Validator(schema)


def schema_errors(instance, name: str) -> list[str]:
    v = _validator(name)
    return [f"{'/'.join(map(str, e.absolute_path)) or '<root>'}: {e.message}"
            for e in sorted(v.iter_errors(instance), key=lambda e: list(map(str, e.absolute_path)))]


def validate_instance(instance, name: str) -> None:
    errs = schema_errors(instance, name)
    if errs:
        raise SchemaError(f"{name} schema violation:\n  " + "\n  ".join(errs[:20]))


def load_json(path: str | Path, schema: str | None = None):
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:          # a ValueError: the CLI prints it as one line, now with the file it is about
        raise ValueError(f"{path}: not valid JSON ({e})") from None
    if schema:
        try:
            validate_instance(data, schema)
        except SchemaError as e:
            raise SchemaError(f"{path}: {e}") from None
    return data
