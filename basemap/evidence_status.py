"""Append-only evidence status registry and fail-closed discovery.

Decision consumers must use :func:`discover_evidence` instead of globbing an
evidence directory.  The default policy returns only independently accepted
artifacts; provisional, invalidated, and diagnostic-only records require an
explicit opt-in.
"""
from __future__ import annotations

import argparse
import json
import os
from .artifact_identity import sha256_file

STATUSES = frozenset({"accepted", "provisional", "invalidated"})
SCHEMA = "evidence_status.v1"


def _resolve_path(registry_path: str, path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.realpath(os.path.join(os.path.dirname(registry_path), path))


def validate_registry(registry: dict, registry_path: str, *, verify_sources: bool = True) -> dict:
    if registry.get("schema") != SCHEMA:
        raise ValueError(f"evidence registry schema must be {SCHEMA!r}")
    events = registry.get("events")
    if not isinstance(events, list) or not events:
        raise ValueError("evidence registry must contain a nonempty append-only events list")

    seen_event_ids = set()
    last_key = None
    latest = {}
    for event in events:
        required = {"event_id", "evidence_id", "recorded_utc", "status", "reason",
                    "reviewer_source", "source_signatures", "replaces", "superseded_by"}
        missing = sorted(required - set(event))
        if missing:
            raise ValueError(f"status event missing fields {missing}: {event.get('event_id')}")
        event_id = str(event["event_id"])
        if event_id in seen_event_ids:
            raise ValueError(f"duplicate evidence status event_id {event_id}")
        seen_event_ids.add(event_id)
        order_key = (str(event["recorded_utc"]), event_id)
        if last_key is not None and order_key <= last_key:
            raise ValueError("evidence status events are not in append order")
        last_key = order_key
        if event["status"] not in STATUSES:
            raise ValueError(f"unknown evidence status {event['status']!r}")
        if not str(event["reason"]).strip():
            raise ValueError(f"event {event_id} has no reason")
        reviewer = event["reviewer_source"]
        if not isinstance(reviewer, dict) or not reviewer.get("uri") or not reviewer.get("sha256"):
            raise ValueError(f"event {event_id} lacks content-bound reviewer_source")
        sources = event["source_signatures"]
        if not isinstance(sources, list) or not sources:
            raise ValueError(f"event {event_id} has no source signatures")
        for source in sources:
            if not source.get("path") or not source.get("sha256"):
                raise ValueError(f"event {event_id} has an incomplete source signature")
            if verify_sources:
                path = _resolve_path(registry_path, source["path"])
                if not os.path.isfile(path):
                    raise FileNotFoundError(f"registered evidence source missing: {path}")
                got = sha256_file(path)
                if got != source["sha256"]:
                    raise ValueError(f"registered evidence source changed: {path}: {got} != "
                                     f"{source['sha256']}")
        latest[event["evidence_id"]] = event
    return latest


def load_registry(path: str, *, verify_sources: bool = True) -> tuple[dict, dict]:
    path = os.path.realpath(path)
    with open(path, encoding="utf-8") as handle:
        registry = json.load(handle)
    return registry, validate_registry(registry, path, verify_sources=verify_sources)


def resolve_evidence(path: str, evidence_id: str, *, allowed_status=("accepted",),
                     allow_diagnostic: bool = False, verify_sources: bool = True) -> dict:
    _, latest = load_registry(path, verify_sources=verify_sources)
    if evidence_id not in latest:
        raise KeyError(f"evidence {evidence_id!r} is not registered")
    event = latest[evidence_id]
    allowed = set(allowed_status)
    if event["status"] not in allowed:
        raise PermissionError(f"evidence {evidence_id!r} is {event['status']}; allowed={sorted(allowed)}")
    if event.get("diagnostic_only") and not allow_diagnostic:
        raise PermissionError(f"evidence {evidence_id!r} is diagnostic-only")
    return event


def discover_evidence(path: str, *, allowed_status=("accepted",),
                      allow_diagnostic: bool = False, verify_sources: bool = True) -> list[dict]:
    """Discover decision evidence.  Accepted-only is intentionally the default."""
    _, latest = load_registry(path, verify_sources=verify_sources)
    allowed = set(allowed_status)
    return [event for evidence_id, event in sorted(latest.items())
            if event["status"] in allowed
            and (allow_diagnostic or not event.get("diagnostic_only"))]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Resolve content-bound evidence status")
    parser.add_argument("--registry", default="experiments/evidence/status-registry.json")
    parser.add_argument("--id", dest="evidence_id")
    parser.add_argument("--allow-status", action="append", choices=sorted(STATUSES))
    parser.add_argument("--allow-diagnostic", action="store_true")
    parser.add_argument("--no-verify-sources", action="store_true")
    args = parser.parse_args(argv)
    allowed = tuple(args.allow_status or ["accepted"])
    if args.evidence_id:
        result = resolve_evidence(args.registry, args.evidence_id, allowed_status=allowed,
                                  allow_diagnostic=args.allow_diagnostic,
                                  verify_sources=not args.no_verify_sources)
    else:
        result = discover_evidence(args.registry, allowed_status=allowed,
                                   allow_diagnostic=args.allow_diagnostic,
                                   verify_sources=not args.no_verify_sources)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
