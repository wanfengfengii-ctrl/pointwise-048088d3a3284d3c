import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass

from freeze.bilinear import BilinearModel, model_record, parse_model
from freeze.cell_topology import classify_cell
from freeze.edge_crossings import crossing_from_record, crossing_to_record, grid_edge
from freeze.evidence import EDGE_ORDER, CellSection, ContourEvidence, section_sort_key


@dataclass(frozen=True)
class ArtifactPayload:
    artifact_id: str
    evidence: ContourEvidence
    model: BilinearModel | None = None


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False
    ).encode('ascii')


def _section_records(sections: tuple[CellSection, ...]) -> list:
    return [
        {
            'cell': [section.row, section.col],
            'ports': [[side, crossing_to_record(crossing)] for side, crossing in section.ports],
        }
        for section in sections
    ]


def encode_artifact(payload: ArtifactPayload) -> bytes:
    evidence = payload.evidence
    body = {
        'version': 1,
        'artifact_id': payload.artifact_id,
        'revision_id': evidence.revision_id,
        'threshold2': evidence.threshold2,
        'rows': evidence.rows,
        'cols': evidence.cols,
        'sections': _section_records(evidence.sections),
    }
    if payload.model is not None:
        body['version'] = 2
        body['model'] = model_record(payload.model)
    return canonical_json({'body': body, 'sha256': hashlib.sha256(canonical_json(body)).hexdigest()})


def regenerate_sections(model: BilinearModel, threshold2: int) -> tuple[CellSection, ...]:
    """按原加工规则从模型重算整张段表。"""
    sections = []
    for row in range(model.rows):
        for col in range(model.cols):
            sections.extend(classify_cell(row, col, model.cell_corners(row, col), threshold2))
    return tuple(sections)


def _read_section(record: object, rows: int, cols: int) -> CellSection:
    if not isinstance(record, dict) or record.keys() != {'cell', 'ports'}:
        raise ValueError('invalid section record')
    cell = record['cell']
    if not isinstance(cell, list) or len(cell) != 2 or any(type(value) is not int for value in cell):
        raise ValueError('invalid section cell')
    row, col = cell
    if not (0 <= row < rows and 0 <= col < cols):
        raise ValueError('section lies outside the grid')

    ports = []
    entries = record['ports']
    if not isinstance(entries, list) or len(entries) not in (2, 4):
        raise ValueError('section requires two or four crossings')
    for entry in entries:
        if not isinstance(entry, list) or len(entry) != 2 or entry[0] not in EDGE_ORDER:
            raise ValueError('invalid section port')
        side, raw_crossing = entry
        crossing = crossing_from_record(raw_crossing)
        if crossing.edge != grid_edge(row, col, side):
            raise ValueError('port does not lie on the stated cell side')
        ports.append((side, crossing))
    sides = tuple(side for side, _ in ports)
    if sides != tuple(side for side in EDGE_ORDER if side in sides):
        raise ValueError('section sides must be unique and ordered')
    return CellSection(row, col, tuple(ports))


def _read_sections(entries: object, rows: int, cols: int) -> tuple[CellSection, ...]:
    if not isinstance(entries, list):
        raise ValueError('invalid section table')
    sections = tuple(_read_section(entry, rows, cols) for entry in entries)
    if sections != tuple(sorted(sections, key=section_sort_key)):
        raise ValueError('sections are out of order')

    groups_by_cell = defaultdict(list)
    for section in sections:
        groups_by_cell[section.row, section.col].append(section.sides)
    for groups in groups_by_cell.values():
        if len(groups) == 1:
            continue
        if tuple(groups) not in ((('N', 'W'), ('E', 'S')), (('N', 'E'), ('S', 'W'))):
            raise ValueError('invalid partition of cell crossings')
    return sections


def decode_artifact(carrier: bytes) -> ArtifactPayload:
    if not isinstance(carrier, bytes):
        raise ValueError('artifact must be bytes')
    try:
        envelope = json.loads(carrier)
        normalized = canonical_json(envelope)
    except (ValueError, RecursionError) as error:
        raise ValueError('invalid artifact JSON') from error
    if not isinstance(envelope, dict) or envelope.keys() != {'body', 'sha256'}:
        raise ValueError('invalid artifact envelope')
    if normalized != carrier:
        raise ValueError('artifact encoding is not canonical')
    body = envelope['body']
    if hashlib.sha256(canonical_json(body)).hexdigest() != envelope['sha256']:
        raise ValueError('artifact checksum mismatch')
    if not isinstance(body, dict):
        raise ValueError('invalid artifact body')
    common = {'version', 'artifact_id', 'revision_id', 'threshold2', 'rows', 'cols', 'sections'}
    if body.keys() not in (common, common | {'model'}):
        raise ValueError('invalid artifact body')
    if type(body['version']) is not int or body['version'] not in (1, 2):
        raise ValueError('unsupported artifact version')
    if (body['version'] == 2) != ('model' in body):
        raise ValueError('artifact version does not match its body fields')

    artifact_id, revision_id = body['artifact_id'], body['revision_id']
    if any(not isinstance(value, str) or not value for value in (artifact_id, revision_id)):
        raise ValueError('invalid artifact identity')
    rows, cols = body['rows'], body['cols']
    if any(type(value) is not int or value < 1 for value in (rows, cols)):
        raise ValueError('invalid artifact dimensions')
    threshold2 = body['threshold2']
    if type(threshold2) is not int or threshold2 % 2 != 1:
        raise ValueError('invalid artifact threshold')

    sections = _read_sections(body['sections'], rows, cols)
    evidence = ContourEvidence(revision_id, threshold2, rows, cols, sections)
    if body['version'] == 1:
        return ArtifactPayload(artifact_id, evidence)

    model = parse_model(body['model'], rows, cols)
    if regenerate_sections(model, threshold2) != sections:
        raise ValueError('segment table does not match the sealed model')
    return ArtifactPayload(artifact_id, evidence, model)
