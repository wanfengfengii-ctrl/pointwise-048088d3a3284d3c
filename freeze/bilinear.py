"""精确的双线性温度面模型。

每个格内的温度面由四个整数顶点温度唯一确定。成果中保存整张共享顶点
采样（而非逐格重算的边交点），因此相邻格在共同边上的采样天然一致。
"""

from dataclasses import dataclass

from intake.grid import Grid


@dataclass(frozen=True)
class BilinearModel:
    rows: int
    cols: int
    vertices: tuple[tuple[int, ...], ...]

    def cell_corners(self, row: int, col: int) -> tuple[int, int, int, int]:
        if type(row) is not int or type(col) is not int:
            raise ValueError('cell coordinates must be integers')
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise ValueError('cell is outside the model')
        north, south = self.vertices[row], self.vertices[row + 1]
        return north[col], north[col + 1], south[col], south[col + 1]


def model_from_grid(grid: Grid) -> BilinearModel:
    rows, cols = grid.cell_shape
    return BilinearModel(rows, cols, grid.vertices)


def parse_model(record: object, rows: int, cols: int) -> BilinearModel:
    """校验并读取成果正文中的模型记录。"""
    if not isinstance(record, dict) or record.keys() != {'kind', 'vertices'}:
        raise ValueError('invalid bilinear model record')
    if record['kind'] != 'bilinear-corners':
        raise ValueError('unsupported bilinear model kind')
    raw = record['vertices']
    if not isinstance(raw, list) or len(raw) != rows + 1:
        raise ValueError('bilinear model has the wrong row count')
    vertices = []
    for line in raw:
        if not isinstance(line, list) or len(line) != cols + 1:
            raise ValueError('bilinear model has the wrong column count')
        if any(type(value) is not int for value in line):
            raise ValueError('bilinear model samples must be integers')
        vertices.append(tuple(line))
    return BilinearModel(rows, cols, tuple(vertices))


def model_record(model: BilinearModel) -> dict:
    return {
        'kind': 'bilinear-corners',
        'vertices': [list(line) for line in model.vertices],
    }
