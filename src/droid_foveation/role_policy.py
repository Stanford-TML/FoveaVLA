from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import yaml


@dataclasses.dataclass(frozen=True)
class RolePolicy:
    """Maps RoboInter primitive skills to required semantic roles."""

    version: str
    role_fields: dict[str, str]
    primitive_roles: dict[str, tuple[str, ...]]
    unsupported_primitive: str = "invalid"

    @classmethod
    def load(cls, path: str | Path) -> "RolePolicy":
        with Path(path).open() as stream:
            config: dict[str, Any] = yaml.safe_load(stream)

        role_fields = {str(role): str(field) for role, field in config["roles"].items()}
        primitive_roles = {
            _normalize_primitive(primitive): tuple(spec["required_roles"])
            for primitive, spec in config["primitives"].items()
        }

        for primitive, roles in primitive_roles.items():
            unknown = sorted(set(roles) - set(role_fields))
            if unknown:
                raise ValueError(f"Primitive {primitive!r} references unknown roles: {unknown}")

        return cls(
            version=str(config["version"]),
            role_fields=role_fields,
            primitive_roles=primitive_roles,
            unsupported_primitive=str(config.get("unsupported_primitive", "invalid")),
        )

    def required_roles(self, primitive_skill: object) -> tuple[str, ...] | None:
        return self.primitive_roles.get(_normalize_primitive(primitive_skill))


def _normalize_primitive(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()
