"""CPU-only CLI for fresh Round 0005 map and regular-file model staging."""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.round0005_staging import (
    ROUND0005_MODEL_ID, ROUND0005_MODEL_REVISION, build_round0005_testbed_seal,
    stage_regular_model_snapshot, stage_round0005_maps,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    testbed = sub.add_parser("testbed")
    testbed.add_argument("--testbed", required=True)
    testbed.add_argument("--source-embeddings", required=True)
    testbed.add_argument("--source-texts", required=True)
    testbed.add_argument("--seal", required=True)
    maps = sub.add_parser("maps")
    maps.add_argument("--sources-json", required=True,
                      help="JSON object mapping the exact nine canonical labels to source dirs")
    maps.add_argument("--destination-root", required=True)
    maps.add_argument("--seal", required=True)
    maps.add_argument("--testbed-seal", required=True)
    maps.add_argument("--corpus-identity",
                      help="optional redundant check; must be the sealed sample_indices file")
    maps.add_argument("--namespace-name", default="jina-en-2m/coordinate-row-id")
    model = sub.add_parser("model")
    model.add_argument("--source-snapshot", required=True)
    model.add_argument("--destination-root", required=True)
    model.add_argument("--seal", required=True)
    model.add_argument("--testbed-seal", required=True)
    model.add_argument("--model-id", default=ROUND0005_MODEL_ID)
    model.add_argument("--revision", default=ROUND0005_MODEL_REVISION)
    args = parser.parse_args(argv)
    if args.command == "testbed":
        report = build_round0005_testbed_seal(
            testbed=args.testbed, source_embeddings=args.source_embeddings,
            source_texts=args.source_texts, seal_path=args.seal)
    elif args.command == "maps":
        with open(args.sources_json, encoding="utf-8") as handle:
            sources = json.load(handle)
        report = stage_round0005_maps(
            sources=sources, destination_root=args.destination_root,
            seal_path=args.seal, corpus_identity_path=args.corpus_identity,
            namespace_name=args.namespace_name,
            testbed_seal_path=args.testbed_seal, production_contract=True)
    else:
        report = stage_regular_model_snapshot(
            source_snapshot=args.source_snapshot, destination_root=args.destination_root,
            seal_path=args.seal, model_id=args.model_id,
            expected_revision=args.revision, testbed_seal_path=args.testbed_seal,
            production_contract=True)
    print(json.dumps({
        "schema": report["schema"],
        "identity_sha256": report["identity_sha256"],
        "seal_path": report["seal_path"],
        "seal_signature": report["seal_signature"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
