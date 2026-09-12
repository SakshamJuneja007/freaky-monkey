"""Compatibility import for the canonical browser verifier.

The browser skill historically exposed two verifier modules.  Keeping this
shim avoids breaking imports while ensuring every caller gets the same
independent, PASS/FAIL/UNKNOWN verification semantics.
"""

from .browser_verifiers import (
    BrowserVerificationResult,
    BrowserVerifier,
    BrowserVerifierBackend,
)

__all__ = [
    "BrowserVerificationResult",
    "BrowserVerifier",
    "BrowserVerifierBackend",
]
