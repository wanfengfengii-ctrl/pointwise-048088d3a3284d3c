from argparse import ArgumentParser
from contextlib import closing
from fractions import Fraction as F
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

from archive.catalog import list_artifacts
from archive.codec import canonical_json
from archive.pickup import pickup_artifact
from archive.publish import PublishAborted
from demo import open_stores
from freeze.assemble import freeze_contour
from freeze.field_view import inspect_cell
from handover.history import load_handover
from handover.register import ack_handover
from intake.revisions import ingest_grid
from intake.source_state import disable_source
from review.report import audit_contour


CHECK = unittest.TestCase()
HALF = F(1, 2)
HORIZONTAL = ((0, HALF), (1, HALF))
DIAGONAL = ((0, 0), (1, 1))
SAMPLES = {
    "first": ([[1, 0], [2, 0]], "1/2"),
    "rescan": ([[1, 0], [5, -1]], "1/2"),
    "two_branches": ([[2, 0], [0, 2]], "3/2"),
    "four_arms": ([[1, 0], [0, 1]], "1/2"),
    "seam": ([[1, 0, 1], [0, 1, 0]], "1/2"),
    "tangent": ([[2, 0], [0, 0]], "1/2"),
}
LOWER, UPPER = (8, -8, 1, 0), (8, -8, 1, 1)
QUERIES = (
    ("first", HORIZONTAL, ((F(2, 3), F(2, 3), 0),)),
    ("rescan", HORIZONTAL, ((F(5, 7), F(5, 7), 0),)),
    ("two_branches", DIAGONAL, ((LOWER, LOWER, 0), (UPPER, UPPER, 1))),
    ("two_branches", DIAGONAL[::-1], ((LOWER, LOWER, 1), (UPPER, UPPER, 0))),
    ("four_arms", DIAGONAL, ((HALF, HALF, 0),)),
    ("tangent", ((0, 1), (1, 0)), ((HALF, HALF, 0),)),
    ("seam", ((0, HALF), (2, HALF)), ((F(0), F(1), 0),)),
    ("seam", ((0, F(1, 4)), (2, F(3, 4))),
     ((F(1, 4), F(1, 4), 0), (HALF, HALF, 0), (F(3, 4), F(3, 4), 0))),
    ("seam", ((1, HALF), (1, 1)), ((F(0), F(0), 0),)),
    ("seam", ((0, 0), (0, F(1, 4))), ()),
)


def baseline():
    carriers = []
    with closing(sqlite3.connect(":memory:")) as connection:
        intake, archive, _ = open_stores(connection)
        for name in ("first", "rescan"):
            vertices, threshold = SAMPLES[name]
            revision = ingest_grid(intake, "kiln-4", vertices)
            artifact = freeze_contour(intake, archive, revision, threshold)
            disable_source(intake, revision)
            carriers.append(pickup_artifact(archive, artifact))

    first, second = (audit_contour(carrier).components for carrier in carriers)
    CHECK.assertEqual(first, second)
    CHECK.assertEqual(json.loads(carriers[0])["body"]["sections"],
                      json.loads(carriers[1])["body"]["sections"])
    (component,) = first
    CHECK.assertEqual(tuple(ref.sides for ref in component.sections), (("N", "S"),))
    points = tuple(port.point for port in component.boundary)
    CHECK.assertEqual(points, ((HALF, 0), (F(3, 4), 1)))
    print("旧包离线核账，两次复扫的边交点均为", [(str(x), str(y)) for x, y in points])


def verify_probe():
    from freeze.assemble import freeze_probe_contour
    from review.report import audit_probe

    with TemporaryDirectory(prefix="kiln-probe-") as temporary:
        database = Path(temporary) / "kiln.sqlite3"
        with closing(sqlite3.connect(database)) as connection:
            intake, archive, handovers = open_stores(connection)
            ids, original, revisions = {}, {}, {}
            for name, (vertices, threshold) in SAMPLES.items():
                batch_id = "kiln-4" if name in ("first", "rescan") else name
                revision = ingest_grid(intake, batch_id, vertices)
                revisions[name] = revision
                ids[name] = freeze_probe_contour(intake, archive, revision, threshold)
                original[name] = pickup_artifact(archive, ids[name])
            first_revision = revisions["first"]
            old_id = freeze_contour(intake, archive, first_revision, "1/2")
            old_carrier = pickup_artifact(archive, old_id)
            old_report = audit_contour(old_carrier)
            receipt = ack_handover(archive, handovers, "night", old_report, old_id)
            CHECK.assertEqual(inspect_cell(archive, ids["first"], (0, 0)),
                              inspect_cell(archive, old_id, (0, 0)))
            published = list_artifacts(archive)
            with CHECK.assertRaises(PublishAborted) as aborted:
                freeze_probe_contour(intake, archive, first_revision, "1/2", fail_before_commit=True)
            for name in ("first", "rescan"):
                disable_source(intake, revisions[name])
            with CHECK.assertRaises(ValueError):
                freeze_probe_contour(intake, archive, first_revision, "1/2")

        with closing(sqlite3.connect(database)) as connection:
            _, archive, handovers = open_stores(connection)
            CHECK.assertEqual(list_artifacts(archive), published)
            with CHECK.assertRaises(ValueError):
                pickup_artifact(archive, aborted.exception.artifact_id)
            CHECK.assertEqual(load_handover(handovers, archive, receipt.receipt_id),
                              (receipt, old_carrier))
            reloaded = {name: pickup_artifact(archive, identity) for name, identity in ids.items()}

    CHECK.assertEqual(reloaded, original)
    CHECK.assertEqual(audit_contour(reloaded["first"]).components, old_report.components)
    CHECK.assertEqual(audit_contour(reloaded["rescan"]).components, old_report.components)
    for name, (start, end), expected in QUERIES:
        actual = audit_probe(reloaded[name], start, end)
        CHECK.assertEqual(actual, expected, (name, start, end))
        print(f"{name} {start!r} -> {end!r}: {actual!r}")
    for carrier, endpoints in ((old_carrier, HORIZONTAL), (reloaded["first"], ((-1, 0), (1, 0)))):
        with CHECK.assertRaises(ValueError):
            audit_probe(carrier, *endpoints)

    # The modified body is canonical and has a fresh checksum, but its segment table is false.
    body = json.loads(reloaded["first"])["body"]
    body["sections"] = []
    invalid = canonical_json({"body": body, "sha256": sha256(canonical_json(body)).hexdigest()})
    with CHECK.assertRaises(ValueError):
        audit_probe(invalid, *HORIZONTAL)


def main():
    parser = ArgumentParser()
    parser.add_argument("mode", choices=("baseline", "verify", "demo"))
    mode = parser.parse_args().mode
    if mode == "baseline":
        baseline()
        return
    if mode == "verify":
        suite = unittest.defaultTestLoader.discover("tests")
        if not unittest.TextTestRunner().run(suite).wasSuccessful():
            raise SystemExit(1)
    verify_probe()


if __name__ == "__main__":
    main()
