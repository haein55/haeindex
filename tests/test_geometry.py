from collections import Counter

from haeindex.geometry import column_of, find_gutters, stable_columns


def spans(*pairs: tuple[float, float]) -> list[tuple[float, float]]:
    return list(pairs)


def two_columns(gutter_at: float = 180.0, gap: float = 7.0) -> list[tuple[float, float]]:
    left = [(20.0 + i, 25.0 + i) for i in range(0, 150, 5)]
    right = [(gutter_at + gap + i, gutter_at + gap + 5.0 + i) for i in range(0, 140, 5)]
    return left + right


def test_두_단_사이의_빈_구간을_찾는다() -> None:
    g = find_gutters(two_columns(), extent=362.0, bins=160)
    assert len(g) == 1
    assert 170 < g[0][0] < 190


def test_칸이_굵으면_gutter를_놓친다() -> None:
    cols = two_columns(gap=7.0)
    assert len(find_gutters(cols, extent=362.0, bins=160)) == 1
    assert find_gutters(cols, extent=362.0, bins=20) == []


def test_전폭_요소_하나가_gutter를_덮는다() -> None:
    cols = two_columns()
    header = [(20.0, 340.0)]
    assert find_gutters(cols, extent=362.0, bins=160) != []
    assert find_gutters(cols + header, extent=362.0, bins=160) == []


def test_가장자리_빈_구간은_여백이라_버린다() -> None:
    g = find_gutters(spans((100.0, 200.0)), extent=362.0, bins=160)
    assert g == []


def test_column_of_는_gutter_를_경계로_센다() -> None:
    gutters = [(180.0, 187.0)]
    assert column_of(100.0, gutters) == 0
    assert column_of(250.0, gutters) == 1
    assert column_of(50.0, []) == 0


def test_과반인_단_수를_고른다() -> None:
    assert stable_columns(Counter({1: 2, 2: 8, 3: 2})) == 2


def test_과반이_없으면_1단으로_본다() -> None:
    assert stable_columns(Counter({1: 9, 4: 2, 8: 1})) == 1
    assert stable_columns(Counter({2: 4, 3: 4, 4: 4})) == 1
    assert stable_columns(Counter()) == 1
