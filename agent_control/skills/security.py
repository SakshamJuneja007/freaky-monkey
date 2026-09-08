"""Security primitives shared by the DEIMOS skill system.

The capability model intentionally describes what a skill can do at the
operating-system boundary rather than trusting a natural-language description.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Capability(str, Enum):
    """Security-relevant capabilities a skill may require."""

    FILESYSTEM_READ = "filesystem_read"
    FILESYSTEM_WRITE = "filesystem_write"
    NETWORK = "network"
    PROCESS = "process"
    BROWSER = "browser"
    ENVIRONMENT = "environment"
    DYNAMIC_CODE = "dynamic_code"
    SUBPROCESS = "subprocess"

    #: Startup/registry/scheduled-task/autorun-style persistence mechanisms.
    PERSISTENCE = "persistence"

    #: Access patterns aimed at credentials, tokens, cookies, or saved
    #: browser/password-manager secrets.
    CREDENTIAL_ACCESS = "credential_access"


class FindingSeverity(str, Enum):
    """Severity assigned to an audit finding."""

    INFO = "INFO"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass(frozen=True)
class SecurityFinding:
    """One concrete observation made during skill inspection."""

    code: str
    message: str
    severity: FindingSeverity
    capability: Capability | None = None
    location: str = ""
    detail: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity.value,
            "capability": (
                self.capability.value
                if self.capability is not None
                else None
            ),
            "location": self.location,
            "detail": dict(self.detail),
        }