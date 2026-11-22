import json
import sqlite3
import unittest
from contextlib import closing
from fractions import Fraction
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

from archive.catalog import list_artifacts
from archive.codec import canonical_json, decode_artifact
from archive.pickup import pickup_artifact
from archive.publish import PublishAborted
from freeze.assemble import freeze_contour, freeze_probe_contour
from freeze.field_view import inspect_cell
from intake.revisions import ingest_grid
from intake.source_state import disable_source
from review.report import audit_contour, audit_probe

HALF = Fraction(1, 2)
HORIZONTAL = ((0, HALF), (1, HALF))
DIAGONAL = ((0, 0), (1, 1))


class ProbeTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.database = Path(temporary.name) / 'probe.sqlite3'
        self.connection = sqlite3.connect(self.database)
        self.addCleanup(self.connection.close)
        from demo import open_stores

        self.intake, self.archive, _ = open_stores(self.connection)

    def seal(self, vertices, threshold):
        revision = ingest_grid(self.intake, 'kiln', vertices)
        artifact = freeze_probe_contour(self.intake, self.archive, revision, threshold)
        return revision, artifact, pickup_artifact(self.archive, artifact)

    def test_sealed_package_matches_plain_contour(self):
        for vertices, threshold in (
            ([[1, 0], [2, 0]], '1/2'), ([[2, 0], [0, 2]], '3/2'), ([[0, 0, 0], [0, 1, 0], [0, 0, 0]], '1/2')
        ):
            with self.subTest(vertices=vertices):
                revision, artifact, carrier = self.seal(vertices, threshold)
                body = json.loads(carrier)['body']
                self.assertEqual(body['version'], 2)
                self.assertEqual(sha256(canonical_json(body)).hexdigest(), json.loads(carrier)['sha256'])
                self.assertEqual(decode_artifact(carrier).model.vertices,
                                 tuple(tuple(row) for row in vertices))
                plain = freeze_contour(self.intake, self.archive, revision, threshold)
                self.assertEqual(audit_contour(carrier).components,
                                 audit_contour(pickup_artifact(self.archive, plain)).components)
                self.assertEqual(inspect_cell(self.archive, artifact, (0, 0)),
                                 inspect_cell(self.archive, plain, (0, 0)))

    def test_hand_examples(self):
        cases = (
            ([[1, 0], [2, 0]], '1/2', HORIZONTAL, ((Fraction(2, 3),) * 2 + (0,),)),
            ([[1, 0], [5, -1]], '1/2', HORIZONTAL, ((Fraction(5, 7),) * 2 + (0,),)),
            ([[2, 0], [0, 2]], '3/2', DIAGONAL,
             (((8, -8, 1, 0),) * 2 + (0,), ((8, -8, 1, 1),) * 2 + (1,))),
            ([[1, 0], [0, 1]], '1/2', DIAGONAL, ((HALF, HALF, 0),)),
            ([[2, 0], [0, 0]], '1/2', ((0, 1), (1, 0)), ((HALF, HALF, 0),)),
        )
        for vertices, threshold, endpoints, expected in cases:
            with self.subTest(vertices=vertices):
                _, _, carrier = self.seal(vertices, threshold)
                self.assertEqual(audit_probe(carrier, *endpoints), expected)

    def test_reversed_probe_reorders_by_new_t(self):
        _, _, carrier = self.seal([[2, 0], [0, 2]], '3/2')
        self.assertEqual(audit_probe(carrier, (1, 1), (0, 0)), (
            ((8, -8, 1, 0), (8, -8, 1, 0), 1),
            ((8, -8, 1, 1), (8, -8, 1, 1), 0),
        ))

    def test_seam_deduplicates_shared_edge(self):
        _, _, carrier = self.seal([[1, 0, 1], [0, 1, 0]], '1/2')
        self.assertEqual(audit_probe(carrier, (0, HALF), (2, HALF)),
                         ((Fraction(0), Fraction(1), 0),))
        self.assertEqual(audit_probe(carrier, (0, Fraction(1, 4)), (2, Fraction(3, 4))), (
            (Fraction(1, 4),) * 2 + (0,),
            (HALF,) * 2 + (0,),
            (Fraction(3, 4),) * 2 + (0,),
        ))
        self.assertEqual(audit_probe(carrier, (1, HALF), (1, 1)), ((Fraction(0),) * 2 + (0,),))
        self.assertEqual(audit_probe(carrier, (0, 0), (0, Fraction(1, 4))), ())

    def test_degenerate_full_line_coincidence(self):
        # 该格内零集是一整条水平线；检尺沿该线时整段重合只报一个区间。
        _, _, carrier = self.seal([[-5, -5], [6, 6]], '1/2')
        self.assertEqual(audit_probe(carrier, (0, HALF), (1, HALF)),
                         ((Fraction(0), Fraction(1), 0),))
        self.assertEqual(audit_probe(carrier, (HALF, 0), (HALF, 1)), ((HALF, HALF, 0),))

    def test_same_branch_can_be_contacted_twice(self):
        # 反对角尺两次穿过同一双曲分支，均由原段出处归到分量 0。
        _, _, carrier = self.seal([[11, 0], [0, 3]], '5/2')
        self.assertEqual(audit_probe(carrier, (0, 1), (1, 0)), (
            ((28, -28, 5, 0),) * 2 + (0,),
            ((28, -28, 5, 1),) * 2 + (0,),
        ))
        # 对角尺分别穿两条分支，归各自原段。
        self.assertEqual(audit_probe(carrier, (0, 0), (1, 1)), (
            ((28, -44, 17, 0),) * 2 + (0,),
            ((28, -44, 17, 1),) * 2 + (1,),
        ))

    def test_invalid_endpoints_and_old_packages_raise(self):
        revision, _, carrier = self.seal([[1, 0], [2, 0]], '1/2')
        for endpoints in (((-1, 0), (1, 0)), ((0, 0), (0, 0)), ((True, 0), (1, 0)),
                          ((0.5, 0), (1, 0)), ((0, 0), (1, 2)), ((0,), (1, 0))):
            with self.subTest(endpoints=endpoints), self.assertRaises(ValueError):
                audit_probe(carrier, *endpoints)
        old = freeze_contour(self.intake, self.archive, revision, '1/2')
        with self.assertRaises(ValueError):
            audit_probe(pickup_artifact(self.archive, old), *HORIZONTAL)

    def test_inconsistent_evidence_raises(self):
        _, _, carrier = self.seal([[2, 0], [0, 2]], '3/2')
        body = json.loads(carrier)['body']
        body['sections'] = []
        with self.assertRaises(ValueError):
            audit_probe(canonical_json(
                {'body': body, 'sha256': sha256(canonical_json(body)).hexdigest()}), *DIAGONAL)
        body = json.loads(carrier)['body']
        body['model']['vertices'][0][0] = 5
        with self.assertRaises(ValueError):
            audit_probe(canonical_json(
                {'body': body, 'sha256': sha256(canonical_json(body)).hexdigest()}), *DIAGONAL)

    def test_abort_and_disabled_source_keep_records(self):
        revision, artifact, carrier = self.seal([[2, 0], [0, 2]], '3/2')
        before = list_artifacts(self.archive)
        with self.assertRaises(PublishAborted) as aborted:
            freeze_probe_contour(self.intake, self.archive, revision, '3/2', fail_before_commit=True)
        self.assertEqual(list_artifacts(self.archive), before)
        with self.assertRaises(ValueError):
            pickup_artifact(self.archive, aborted.exception.artifact_id)
        disable_source(self.intake, revision)
        with self.assertRaises(ValueError):
            freeze_probe_contour(self.intake, self.archive, revision, '3/2')
        self.assertEqual(pickup_artifact(self.archive, artifact), carrier)

    def test_reopen_keeps_sealed_bytes_and_offline_probe(self):
        revision, artifact, carrier = self.seal([[2, 0], [0, 2]], '3/2')
        disable_source(self.intake, revision)
        self.connection.close()
        with closing(sqlite3.connect(self.database)) as connection:
            from demo import open_stores

            intake, archive, _ = open_stores(connection)
            self.assertEqual(pickup_artifact(archive, artifact), carrier)
            ingest_grid(intake, 'kiln', [[0, 0], [0, 0]])
            self.assertEqual(audit_probe(carrier, *DIAGONAL), (
                ((8, -8, 1, 0),) * 2 + (0,),
                ((8, -8, 1, 1),) * 2 + (1,),
            ))


if __name__ == '__main__':
    unittest.main()
