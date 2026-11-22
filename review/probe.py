"""凭固定 v2 成果字节做离线检尺。

只使用成果正文中的整数顶点采样与倍增阈值。每个格内的温度面按四角
双线性插值，阈值方程在检尺 P(t) 上化为一个关于 t 的有理系数二次式。
有理根以 ``Fraction`` 交付；非有理根以 ``(a, b, c, k)`` 四元组交付，
表示归一化整系数方程 ``a*t*t + b*t + c = 0`` 的第 ``k`` 个实根。

实数比较全部落在整数与有理数上：两个二次无理根比大小用高精度有理
夹逼（不同二次无理根之间的间隔有正下界），根与有理数比大小直接用
多项式在该有理点的符号，均不依赖浮点。
"""

from dataclasses import dataclass
from fractions import Fraction
from functools import cmp_to_key
from math import gcd, isqrt

from archive.codec import decode_artifact


@dataclass(frozen=True)
class _Surd:
    """二次无理根；a > 0、整系数、公因数 1、k 为 0（较小）或 1（较大）。"""

    a: int
    b: int
    c: int
    k: int

    @property
    def key(self) -> tuple[int, int, int, int]:
        return self.a, self.b, self.c, self.k

    @property
    def vertex(self) -> Fraction:
        return Fraction(-self.b, 2 * self.a)

    def polynomial_sign(self, t: Fraction) -> int:
        """a*t*t + b*t + c 在有理点 t 的符号。"""
        value = self.a * t * t + self.b * t + self.c
        return (value > 0) - (value < 0)

    def record(self) -> tuple[int, int, int, int]:
        return self.a, self.b, self.c, self.k


def _surd_sign_vs_rational(surd: _Surd, t: Fraction) -> int:
    """sign(root - t)：根在 t 右侧 +1，相等 0，左侧 -1。"""
    sign = surd.polynomial_sign(t)
    if sign == 0:
        return 0
    left_of_vertex = t < surd.vertex
    if surd.k == 0:  # 较小根位于顶点左侧，f 在根外为正、两根间为负。
        if left_of_vertex:
            return 1 if sign > 0 else -1
        return -1
    if left_of_vertex:
        return 1
    return 1 if sign < 0 else -1


def _surd_bounds(surd: _Surd, steps: int) -> tuple[Fraction, Fraction]:
    a, b, c, k = surd.a, surd.b, surd.c, surd.k
    vertex = surd.vertex
    span = Fraction(abs(b) + abs(c), a) + 2
    low, high = (-span, vertex) if k == 0 else (vertex, span)
    for _ in range(steps):
        middle = (low + high) / 2
        value = a * middle * middle + b * middle + c
        if (value > 0) == (k == 0):
            low = middle
        else:
            high = middle
    return low, high


def compare(left, right) -> int:
    """按精确实数比较两个端点，返回 -1、0 或 1。"""
    if isinstance(left, Fraction) and isinstance(right, Fraction):
        return (left > right) - (left < right)
    if isinstance(left, Fraction):
        return -_surd_sign_vs_rational(right, left)
    if isinstance(right, Fraction):
        return _surd_sign_vs_rational(left, right)
    if left.key == right.key:
        return 0
    if (left.a, left.b, left.c) == (right.a, right.b, right.c):
        return (left.k > right.k) - (left.k < right.k)
    steps = 8
    while steps <= 1024:
        low_left, high_left = _surd_bounds(left, steps)
        low_right, high_right = _surd_bounds(right, steps)
        if high_left < low_right:
            return -1
        if high_right < low_left:
            return 1
        steps *= 2
    # 两个不同二次无理数的间隔有正下界，1024 次二分足以区分。
    raise AssertionError('could not separate distinct quadratic roots')


def _sign(value) -> int:
    return (value > 0) - (value < 0)


def _sqrt_fraction(value: Fraction) -> Fraction | None:
    if value < 0:
        return None
    numerator, denominator = value.numerator, value.denominator
    high_n, high_d = isqrt(numerator), isqrt(denominator)
    if high_n * high_n == numerator and high_d * high_d == denominator:
        return Fraction(high_n, high_d)
    return None


def _integer_polynomial(q2: Fraction, q1: Fraction, q0: Fraction) -> tuple[int, int, int]:
    common = 1
    for value in (q2, q1, q0):
        common = common * value.denominator // gcd(common, value.denominator)
    a, b, c = (int(value * common) for value in (q2, q1, q0))
    divisor = gcd(gcd(abs(a), abs(b)), abs(c))
    a, b, c = a // divisor, b // divisor, c // divisor
    if a < 0:
        a, b, c = -a, -b, -c
    return a, b, c


def _solve(q2: Fraction, q1: Fraction, q0: Fraction) -> list:
    if q2 == 0:
        if q1 == 0:
            return []
        return [Fraction(-q0, q1)]
    discriminant = q1 * q1 - 4 * q2 * q0
    if discriminant < 0:
        return []
    if discriminant == 0:
        return [Fraction(-q1, 2 * q2)]
    root = _sqrt_fraction(discriminant)
    if root is not None:
        values = [(-q1 - root) / (2 * q2), (-q1 + root) / (2 * q2)]
        values.sort()
        return values
    a, b, c = _integer_polynomial(q2, q1, q0)
    return [_Surd(a, b, c, 0), _Surd(a, b, c, 1)]


def _axis_window(base: Fraction, step: Fraction) -> tuple[Fraction, Fraction] | None:
    if step == 0:
        return (Fraction(0), Fraction(1)) if 0 <= base <= 1 else None
    first = -base / step
    second = (1 - base) / step
    if first > second:
        first, second = second, first
    return first, second


def _cell_window(row: int, col: int, start, end):
    sx, sy = start
    dx, dy = end[0] - sx, end[1] - sy
    low, high = Fraction(0), Fraction(1)
    for base, step in ((Fraction(sx - col), dx), (Fraction(sy - row), dy)):
        axis = _axis_window(base, step)
        if axis is None:
            return None
        low, high = max(low, axis[0]), min(high, axis[1])
    if low > high:
        return None
    return (low, high), Fraction(sx - col), dx, Fraction(sy - row), dy


def _linear_sign(const: Fraction, coefficient: Fraction, t) -> int:
    if coefficient == 0:
        return _sign(const)
    bound = -const / coefficient
    relation = compare(t, bound)
    return relation if coefficient > 0 else -relation


def _at_const(const: Fraction, coefficient: Fraction, target: int, t) -> bool:
    if coefficient == 0:
        return const == target
    return compare(t, (target - const) / coefficient) == 0


def _port_signature(side: str, offset: Fraction, u_star: Fraction, v_star: Fraction):
    if side == 'N':
        u, v = offset, Fraction(0)
    elif side == 'E':
        u, v = Fraction(1), offset
    elif side == 'S':
        u, v = offset, Fraction(1)
    else:
        u, v = Fraction(0), offset
    return _sign(u - u_star), _sign(v - v_star)


def _merge(intervals: list, position: int) -> list:
    intervals.sort(key=cmp_to_key(lambda x, y: compare(x[0], y[0]) or compare(x[1], y[1])))
    merged = []
    for low, high in intervals:
        if merged and compare(low, merged[-1][1]) <= 0:
            if compare(high, merged[-1][1]) > 0:
                merged[-1] = (merged[-1][0], high)
        else:
            merged.append((low, high))
    return [(low, high, position) for low, high in merged]


def _encode(value):
    return value.record() if isinstance(value, _Surd) else value


def _parse_point(raw, rows: int, cols: int):
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ValueError('probe endpoint must be an (x, y) pair')
    for coordinate in raw:
        if isinstance(coordinate, bool) or not isinstance(coordinate, (int, Fraction)):
            raise ValueError('probe coordinates must be int or Fraction')
    x, y = (Fraction(coordinate) for coordinate in raw)
    if not (0 <= x <= cols and 0 <= y <= rows):
        raise ValueError('probe endpoint lies outside the grid domain')
    return x, y


def audit_probe(carrier: bytes, start, end) -> tuple:
    """按两个有理端点复核检尺与各分量的全部接触。"""
    from review.report import audit_contour

    payload = decode_artifact(carrier)
    evidence = payload.evidence
    if payload.model is None:
        raise ValueError('probe requires a sealed model artifact')

    left = _parse_point(start, evidence.rows, evidence.cols)
    right = _parse_point(end, evidence.rows, evidence.cols)
    if left == right:
        raise ValueError('probe endpoints must be distinct')

    report = audit_contour(carrier)
    component_of = {}
    for position, component in enumerate(report.components):
        for ref in component.sections:
            component_of[(ref.row, ref.col, ref.sides)] = position

    cells = {}
    for section in evidence.sections:
        cells.setdefault((section.row, section.col), []).append(
            (section.sides, component_of[(section.row, section.col, section.sides)], section)
        )

    threshold2 = evidence.threshold2
    model = payload.model
    contacts = {}
    for (row, col), group in cells.items():
        nw, ne, sw, se = model.cell_corners(row, col)
        a, b, c, d = (2 * value - threshold2 for value in (nw, ne, sw, se))
        window = _cell_window(row, col, left, right)
        if window is None:
            continue
        (low_t, high_t), u0, du, v0, dv = window

        delta_b, delta_c = Fraction(b - a), Fraction(c - a)
        mixed = Fraction(a - b - c + d)
        q0 = a + delta_b * u0 + delta_c * v0 + mixed * u0 * v0
        q1 = delta_b * du + delta_c * dv + mixed * (u0 * dv + v0 * du)
        q2 = mixed * du * dv

        if q2 == 0 and q1 == 0 and q0 == 0:
            # 整段检尺落在双线性零集上；只有单一分支时归属明确。
            if len(group) == 1:
                contacts.setdefault(group[0][1], []).append((low_t, high_t))
            continue

        u_star = -delta_c / mixed if mixed else None
        v_star = -delta_b / mixed if mixed else None
        signatures = None
        if len(group) == 2 and mixed:
            signatures = {
                sides: {
                    _port_signature(side, crossing.offset, u_star, v_star)
                    for side, crossing in single.ports
                }
                for sides, _, single in group
            }

        def assign(t):
            if len(group) == 1:
                return group[0][1]
            # 命中点恰在原段端口时，直接按端口出处归属。
            for sides, position, single in group:
                for side, crossing in single.ports:
                    edge_const = (v0, dv) if side in 'NS' else (u0, du)
                    target = 0 if side in 'NW' else 1
                    if _at_const(edge_const[0], edge_const[1], target, t):
                        return position
            if signatures is None:
                return None
            signature = (_linear_sign(u0 - u_star, du, t), _linear_sign(v0 - v_star, dv, t))
            if signature == (0, 0):
                # 检尺穿过鞍点，两条分支都在该点相接，各记一次。
                return [position for _, position, _ in group]
            for sides, position, _ in group:
                if signature in signatures[sides]:
                    return position
            return None

        for value in _solve(q2, q1, q0):
            if compare(value, low_t) < 0 or compare(value, high_t) > 0:
                continue
            positions = assign(value)
            if positions is None:
                continue
            if not isinstance(positions, list):
                positions = [positions]
            for position in positions:
                contacts.setdefault(position, []).append((value, value))

    answer = []
    for position in sorted(contacts):
        answer.extend(_merge(contacts[position], position))
    answer.sort(key=cmp_to_key(
        lambda x, y: compare(x[0], y[0]) or compare(x[1], y[1]) or (x[2] > y[2]) - (x[2] < y[2])
    ))
    return tuple((_encode(low), _encode(high), position) for low, high, position in answer)
