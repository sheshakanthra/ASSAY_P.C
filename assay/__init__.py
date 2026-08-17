"""ASSAY — defensive, pre-deployment security scanner for AI model weight files.

Inspects model artifacts for steganographic malware, embedded payloads, and
backdoor indicators statistically (no training pipeline required) and returns
an explainable Model Risk Score. See CLAUDE.md for the full architecture.
"""

__version__ = "0.1.0"
