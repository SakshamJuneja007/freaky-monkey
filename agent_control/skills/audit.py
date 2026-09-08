"""Static capability audit for DEIMOS skills.

This is deliberately a deterministic source inspector, not an LLM deciding
whether code "looks safe".

The auditor answers a narrower question:

    What security-relevant capabilities are visibly requested by this source?

It then compares those observations with the capabilities declared by the
skill manifest.

Static inspection is not a sandbox and must not be described as complete
malware detection. Its purpose is to detect capability mismatches before a
skill is exposed to the planner.
"""

from __future__ import annotations

import ast
import inspect
from dataclasses import dataclass, field
from typing import Iterable

from .manifest import SkillManifest
from .security import (
    Capability,
    FindingSeverity,
    SecurityFinding,
)
from .status import SkillStatus


@dataclass(frozen=True)
class SkillAuditReport:
    """Result of inspecting one skill."""

    skill_name: str
    declared_capabilities: frozenset[Capability]
    observed_capabilities: frozenset[Capability]
    findings: tuple[SecurityFinding, ...] = ()
    status: SkillStatus = SkillStatus.DISCOVERED
    source_available: bool = True

    @property
    def unexpected_capabilities(
        self,
    ) -> frozenset[Capability]:
        """Capabilities observed but not declared."""
        return (
            self.observed_capabilities
            - self.declared_capabilities
        )

    def to_json(self) -> dict[str, object]:
        return {
            "skill_name": self.skill_name,
            "declared_capabilities": sorted(
                capability.value
                for capability
                in self.declared_capabilities
            ),
            "observed_capabilities": sorted(
                capability.value
                for capability
                in self.observed_capabilities
            ),
            "unexpected_capabilities": sorted(
                capability.value
                for capability
                in self.unexpected_capabilities
            ),
            "findings": [
                finding.to_json()
                for finding in self.findings
            ],
            "status": self.status.value,
            "source_available": self.source_available,
        }


class _CapabilityVisitor(ast.NodeVisitor):
    """Extract obvious security-relevant operations from Python AST."""

    IMPORT_CAPABILITIES = {
        "os": Capability.ENVIRONMENT,
        "pathlib": Capability.FILESYSTEM_READ,
        "shutil": Capability.FILESYSTEM_WRITE,
        "tempfile": Capability.FILESYSTEM_WRITE,
        "subprocess": Capability.SUBPROCESS,
        "socket": Capability.NETWORK,
        "http": Capability.NETWORK,
        "urllib": Capability.NETWORK,
        "requests": Capability.NETWORK,
        "httpx": Capability.NETWORK,
        "webbrowser": Capability.BROWSER,
    }

    DYNAMIC_FUNCTIONS = {
        "eval",
        "exec",
        "compile",
        "__import__",
    }

    FILE_READ_METHODS = {
        "read",
        "read_bytes",
        "read_text",
        "iterdir",
        "glob",
        "rglob",
    }

    FILE_WRITE_METHODS = {
        "write",
        "write_bytes",
        "write_text",
        "mkdir",
        "unlink",
        "rename",
        "replace",
        "rmdir",
        "touch",
    }

    def __init__(self) -> None:
        self.capabilities: set[Capability] = set()
        self.findings: list[SecurityFinding] = []

    def _add(
        self,
        capability: Capability,
        *,
        code: str,
        message: str,
        severity: FindingSeverity,
        node: ast.AST,
    ) -> None:
        self.capabilities.add(capability)

        self.findings.append(
            SecurityFinding(
                code=code,
                message=message,
                severity=severity,
                capability=capability,
                location=(
                    f"line {getattr(node, 'lineno', '?')}"
                ),
            )
        )

    def visit_Import(
        self,
        node: ast.Import,
    ) -> None:
        for alias in node.names:
            root = alias.name.split(".", 1)[0]

            capability = self.IMPORT_CAPABILITIES.get(
                root
            )

            if capability is not None:
                self._add(
                    capability,
                    code="IMPORT_CAPABILITY",
                    message=(
                        f"imports module {alias.name!r}, "
                        f"which may require "
                        f"{capability.value!r}"
                    ),
                    severity=FindingSeverity.LOW,
                    node=node,
                )

        self.generic_visit(node)

    def visit_ImportFrom(
        self,
        node: ast.ImportFrom,
    ) -> None:
        module = node.module or ""
        root = module.split(".", 1)[0]

        capability = self.IMPORT_CAPABILITIES.get(
            root
        )

        if capability is not None:
            self._add(
                capability,
                code="IMPORT_CAPABILITY",
                message=(
                    f"imports from module {module!r}, "
                    f"which may require "
                    f"{capability.value!r}"
                ),
                severity=FindingSeverity.LOW,
                node=node,
            )

        self.generic_visit(node)

    def visit_Call(
        self,
        node: ast.Call,
    ) -> None:
        name = self._call_name(node.func)

        if name in self.DYNAMIC_FUNCTIONS:
            self._add(
                Capability.DYNAMIC_CODE,
                code="DYNAMIC_CODE",
                message=(
                    f"calls dynamic execution function "
                    f"{name!r}"
                ),
                severity=FindingSeverity.HIGH,
                node=node,
            )

        if (
            name == "subprocess.run"
            or name == "subprocess.Popen"
            or name == "subprocess.call"
            or name == "subprocess.check_call"
            or name == "subprocess.check_output"
        ):
            self._add(
                Capability.SUBPROCESS,
                code="SUBPROCESS_EXECUTION",
                message=(
                    f"calls process execution API "
                    f"{name!r}"
                ),
                severity=FindingSeverity.HIGH,
                node=node,
            )

        method_name = self._attribute_name(
            node.func
        )

        if method_name in self.FILE_READ_METHODS:
            self._add(
                Capability.FILESYSTEM_READ,
                code="FILESYSTEM_READ",
                message=(
                    f"calls possible filesystem read "
                    f"method {method_name!r}"
                ),
                severity=FindingSeverity.LOW,
                node=node,
            )

        if method_name in self.FILE_WRITE_METHODS and self._looks_like_filesystem_receiver(node.func):
            self._add(
                Capability.FILESYSTEM_WRITE,
                code="FILESYSTEM_WRITE",
                message=(
                    f"calls possible filesystem write "
                    f"method {method_name!r}"
                ),
                severity=FindingSeverity.MEDIUM,
                node=node,
            )

        self.generic_visit(node)

    @staticmethod
    def _call_name(
        node: ast.AST,
    ) -> str:
        if isinstance(node, ast.Name):
            return node.id

        if isinstance(node, ast.Attribute):
            parts: list[str] = []

            current: ast.AST = node

            while isinstance(
                current,
                ast.Attribute,
            ):
                parts.append(current.attr)
                current = current.value

            if isinstance(
                current,
                ast.Name,
            ):
                parts.append(current.id)

            return ".".join(
                reversed(parts)
            )

        return ""

    @staticmethod
    def _attribute_name(
        node: ast.AST,
    ) -> str:
        if isinstance(node, ast.Attribute):
            return node.attr

        return ""

    @staticmethod
    def _looks_like_filesystem_receiver(node: ast.AST) -> bool:
        """Avoid treating ordinary string helpers such as ``kind.replace`` as
        filesystem operations while retaining detection for Path-like calls.

        The auditor is intentionally conservative, but method-name-only
        detection creates false positives for ubiquitous string methods.
        Explicit ``Path(...)`` receivers and conventional path variable names
        remain covered.
        """

        if not isinstance(node, ast.Attribute):
            return False

        receiver = node.value

        if isinstance(receiver, ast.Call):
            name = _CapabilityVisitor._call_name(receiver.func)
            if name in {"Path", "pathlib.Path", "PurePath", "pathlib.PurePath"}:
                return True

        if isinstance(receiver, ast.Name):
            name = receiver.id.lower()
            return name in {
                "path", "file", "filepath", "file_path", "directory",
                "dir", "target", "source", "destination", "root",
            } or name.endswith(("_path", "_file", "_dir", "_directory"))

        return False


def _status_for(
    findings: Iterable[SecurityFinding],
    unexpected: frozenset[Capability],
    *,
    source_available: bool,
) -> SkillStatus:
    """Classify a skill conservatively."""

    if not source_available:
        return SkillStatus.RESTRICTED

    findings = tuple(findings)

    if any(
        finding.capability
        is Capability.DYNAMIC_CODE
        for finding in findings
    ):
        return SkillStatus.QUARANTINED

    if unexpected:
        return SkillStatus.RESTRICTED

    if any(
        finding.severity
        is FindingSeverity.CRITICAL
        for finding in findings
    ):
        return SkillStatus.BLOCKED

    return SkillStatus.APPROVED


def audit_skill(
    skill: object,
    manifest: SkillManifest | None = None,
) -> SkillAuditReport:
    """Inspect a skill implementation and classify its exposure status."""

    info = getattr(
        skill,
        "info",
        None,
    )

    skill_name = getattr(
        info,
        "name",
        type(skill).__name__,
    )

    if manifest is None:
        manifest = getattr(
            info,
            "manifest",
            None,
        )

    if not isinstance(
        manifest,
        SkillManifest,
    ):
        manifest = SkillManifest()

    try:
        source = inspect.getsource(
            type(skill)
        )
    except (
        OSError,
        TypeError,
    ):
        finding = SecurityFinding(
            code="SOURCE_UNAVAILABLE",
            message=(
                "skill implementation source could "
                "not be inspected"
            ),
            severity=FindingSeverity.MEDIUM,
        )

        return SkillAuditReport(
            skill_name=str(skill_name),
            declared_capabilities=(
                manifest.capabilities
            ),
            observed_capabilities=frozenset(),
            findings=(finding,),
            status=SkillStatus.RESTRICTED,
            source_available=False,
        )

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        finding = SecurityFinding(
            code="SOURCE_PARSE_FAILED",
            message=(
                "skill implementation could not "
                f"be parsed: {exc}"
            ),
            severity=FindingSeverity.HIGH,
        )

        return SkillAuditReport(
            skill_name=str(skill_name),
            declared_capabilities=(
                manifest.capabilities
            ),
            observed_capabilities=frozenset(),
            findings=(finding,),
            status=SkillStatus.RESTRICTED,
            source_available=True,
        )

    visitor = _CapabilityVisitor()
    visitor.visit(tree)

    observed = frozenset(
        visitor.capabilities
    )

    unexpected = (
        observed
        - manifest.capabilities
    )

    findings = list(visitor.findings)

    for capability in sorted(
        unexpected,
        key=lambda item: item.value,
    ):
        findings.append(
            SecurityFinding(
                code="UNDECLARED_CAPABILITY",
                message=(
                    f"skill uses or requests "
                    f"{capability.value!r} without "
                    f"declaring it"
                ),
                severity=FindingSeverity.HIGH,
                capability=capability,
            )
        )

    status = _status_for(
        findings,
        unexpected,
        source_available=True,
    )

    return SkillAuditReport(
        skill_name=str(skill_name),
        declared_capabilities=(
            manifest.capabilities
        ),
        observed_capabilities=observed,
        findings=tuple(findings),
        status=status,
        source_available=True,
    )