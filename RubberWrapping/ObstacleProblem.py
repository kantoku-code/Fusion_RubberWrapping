# Fusion 360 API Python script
"""ボディ上面を下限とする膜の障害物問題(Obstacle problem)を解き、ラバーの Form を作る。"""

import math
import time

import adsk
import adsk.core as core
import adsk.fusion as fusion

DIV = 30  # 平面の分割数(X/Y)
MARGIN = 0.5  # BBoxからの片側余白[cm]
THICKNESS = 0.05  # ラバーの厚み(Bodyからの浮き)[cm]
MEMBRANE_ITERS = 2000
SOR = 1.8
SMOOTH_ITERS = 20
FIT_ITERS = 3

Axis = tuple[float, float, float]
Axes = tuple[Axis, Axis, Axis]  # 重力座標系の (ex, ey, up)
Grid = list[list[float]]
Body = fusion.BRepBody | fusion.MeshBody


def _same_body(left: Body, right: Body) -> bool:
    """同じボディの同じ occurrence だけを同一視する。

    Args:
        left: 比較するボディ。
        right: 比較するボディ。

    Returns:
        同じ occurrence 上の同じボディなら True。
    """
    left_occ, right_occ = left.assemblyContext, right.assemblyContext
    left_path = left_occ.fullPathName if left_occ else None
    right_path = right_occ.fullPathName if right_occ else None
    return left_path == right_path and (left.nativeObject or left) == (
        right.nativeObject or right
    )


def _dot(point: core.Point3D | core.Vector3D, axis: Axis) -> float:
    """点(またはベクトル)を軸へ投影した値を返す。

    Args:
        point: ワールド座標の点またはベクトル。
        axis: 単位ベクトルの軸。

    Returns:
        内積の値。
    """
    return point.x * axis[0] + point.y * axis[1] + point.z * axis[2]


def _world_point(x: float, y: float, z: float, axes: Axes) -> core.Point3D:
    """重力座標系の座標をワールド座標の点に変換する。

    Args:
        x: ex 方向の座標 [cm]。
        y: ey 方向の座標 [cm]。
        z: up 方向の座標 [cm]。
        axes: 重力座標系の (ex, ey, up)。

    Returns:
        ワールド座標の点。
    """
    ex, ey, up = axes
    return core.Point3D.create(
        x * ex[0] + y * ey[0] + z * up[0],
        x * ex[1] + y * ey[1] + z * up[1],
        x * ex[2] + y * ey[2] + z * up[2],
    )


def _world_vector(axis: Axis, length: float) -> core.Vector3D:
    """軸方向に指定した長さのベクトルを作る。

    Args:
        axis: 単位ベクトルの軸。
        length: ベクトルの長さ [cm]。

    Returns:
        ワールド座標のベクトル。
    """
    return core.Vector3D.create(axis[0] * length, axis[1] * length, axis[2] * length)


def _basis(gravity: core.Vector3D) -> Axes:
    """重力方向から重力座標系を作る。up は重力の逆向き。-Z ではワールド XYZ と一致させる。

    Args:
        gravity: 重力方向の単位ベクトル。

    Returns:
        正規直交基底 (ex, ey, up)。
    """
    up = (-gravity.x, -gravity.y, -gravity.z)
    helper = (1.0, 0.0, 0.0) if abs(up[0]) < 0.9 else (0.0, 1.0, 0.0)
    projection = sum(helper[i] * up[i] for i in range(3))
    raw_ex = tuple(helper[i] - projection * up[i] for i in range(3))
    length = math.sqrt(sum(value * value for value in raw_ex))
    ex = tuple(value / length for value in raw_ex)
    ey = (
        up[1] * ex[2] - up[2] * ex[1],
        up[2] * ex[0] - up[0] * ex[2],
        up[0] * ex[1] - up[1] * ex[0],
    )
    return ex, ey, up


def _bounds(bodies: list[Body], axes: Axes) -> tuple[list[float], list[float]]:
    """全ボディの BBox の８隅を重力座標系へ投影する。

    Args:
        bodies: 対象ボディ。
        axes: 重力座標系の (ex, ey, up)。

    Returns:
        各軸の最小値と最大値 (lows, highs)。
    """
    lows, highs = [float("inf")] * 3, [float("-inf")] * 3
    for body in bodies:
        box = body.boundingBox
        for x in (box.minPoint.x, box.maxPoint.x):
            for y in (box.minPoint.y, box.maxPoint.y):
                for z in (box.minPoint.z, box.maxPoint.z):
                    point = core.Point3D.create(x, y, z)
                    for index, axis in enumerate(axes):
                        value = _dot(point, axis)
                        lows[index] = min(lows[index], value)
                        highs[index] = max(highs[index], value)
    return lows, highs


class _Progress:
    """進捗ダイアログ。長い計算中も Fusion に再描画の機会を与える。

    Attributes:
        total: 全体の作業量。
        completed: 完了した作業量。
        last_percent: 最後に表示した進捗率 [%]。
        phase: 最後に表示した作業名。
        dialog: Fusion の進捗ダイアログ。
    """

    def __init__(self, total: int) -> None:
        """進捗ダイアログを表示する。

        Args:
            total: 全体の作業量。
        """
        self.total = total
        self.completed = 0
        self.last_percent = -1
        self.phase: str | None = None
        self.dialog: core.ProgressDialog = (
            core.Application.get().userInterface.createProgressDialog()
        )
        self.dialog.isCancelButtonShown = False
        self.dialog.show("RubberWrapping", "ラバーを生成中 (%p%)", 0, total, 0)
        adsk.doEvents()

    def advance(self, phase: str, amount: int = 1) -> None:
        """作業を進め、進捗率か作業名が変わったときだけ表示を更新する。

        Args:
            phase: 作業名。
            amount: 進める作業量。
        """
        self.completed = min(self.total, self.completed + amount)
        percent = self.completed * 100 // self.total
        if percent != self.last_percent or phase != self.phase:
            self.phase = phase
            self.last_percent = percent
            self.dialog.message = "{} (%p%)".format(phase)
            self.dialog.progressValue = self.completed
            adsk.doEvents()

    def hide(self) -> None:
        """進捗ダイアログを閉じる。"""
        self.dialog.hide()

    def complete(self) -> None:
        """進捗を 100% にする。"""
        self.advance("完了", self.total - self.completed)


def create_rubber(
    bodies: list[Body], gravity: core.Vector3D
) -> tuple[fusion.FormFeature, float]:
    """ボディに重力方向からラバーをかけた Form を作る。

    Args:
        bodies: ラバーをかけるボディ（ソリッド／サーフェス／メッシュ）。
        gravity: 重力方向のベクトル。長さは問わない。

    Returns:
        作成した Form と処理時間 [秒]。

    Raises:
        ValueError: デザインが開かれていない、または gravity の長さが 0 のとき。
    """
    start_time = time.perf_counter()
    design = fusion.Design.cast(core.Application.get().activeProduct)
    if design is None:
        raise ValueError("Fusion のデザインを開いてください")
    root = design.rootComponent
    gravity = gravity.copy()
    if not gravity.normalize():
        raise ValueError("重力方向の長さが 0 です")
    count = (DIV + 1) ** 2
    progress = _Progress(
        2 + len(bodies) + MEMBRANE_ITERS + SMOOTH_ITERS + count * (2 + FIT_ITERS)
    )
    try:
        form = _create_form(design, root, bodies, gravity, progress)
        progress.complete()
    finally:
        progress.hide()
    return form, time.perf_counter() - start_time


def _create_form(
    design: fusion.Design,
    root: fusion.Component,
    bodies: list[Body],
    gravity: core.Vector3D,
    progress: _Progress,
) -> fusion.FormFeature:
    """T スプライン平面を作り、障害物問題の解の高さへ制御点を動かす。

    Args:
        design: 対象のデザイン。
        root: Form を追加するルートコンポーネント。
        bodies: ラバーをかけるボディ（ソリッド／サーフェス／メッシュ）。
        gravity: 重力方向の単位ベクトル。
        progress: 進捗ダイアログ。

    Returns:
        作成した Form。失敗したときは Form を削除して例外を送出する。
    """
    axes = _basis(gravity)
    ex, ey, up = axes
    lows, highs = _bounds(bodies, axes)
    origin_x, origin_y = lows[0] - MARGIN, lows[1] - MARGIN
    floor_z, ray_z = lows[2], highs[2] + 1.0
    # メッシュボディにはレイが当たらないので、レイは BRep ボディだけに飛ばす
    brep_bodies = [
        body for body in bodies if body.objectType != fusion.MeshBody.classType()
    ]
    size_x = highs[0] - lows[0] + 2 * MARGIN
    size_y = highs[1] - lows[1] + 2 * MARGIN
    origin = _world_point(origin_x, origin_y, floor_z, axes)

    design.designType = fusion.DesignTypes.ParametricDesignType
    form = root.features.formFeatures.add()
    form.startEdit()
    try:
        ts_bodies = form.tSplineBodies
        plane_input = ts_bodies.createPlaneInput(
            origin, _world_vector(ex, 1), _world_vector(ey, 1), size_x, size_y
        )
        plane_input.xFaceCount = DIV
        plane_input.yFaceCount = DIV
        ts_body = ts_bodies.addByShapeInput(plane_input)
        progress.advance("ラバーの平面を作成", 1)

        step_x, step_y = size_x / DIV, size_y / DIV
        n = DIV + 1
        h = [[floor_z] * n for _ in range(n)]
        hit = [[False] * n for _ in range(n)]
        vertices = []
        for vertex in list(ts_body.vertices):
            geo = vertex.geometry
            x, y = _dot(geo, ex), _dot(geo, ey)
            i = min(max(int(round((x - origin_x) / step_x)), 0), DIV)
            j = min(max(int(round((y - origin_y) / step_y)), 0), DIV)
            vertices.append((vertex, i, j))
            top = (
                get_top_height(root, brep_bodies, x, y, ray_z, axes)
                if brep_bodies
                else None
            )
            if top is not None:
                h[i][j] = top + THICKNESS
                hit[i][j] = True
            progress.advance("ボディとの交点を計算")

        for body in bodies:
            if body.objectType == fusion.MeshBody.classType():
                apply_mesh_body(body, h, hit, origin_x, origin_y, step_x, step_y, axes)
            else:
                apply_mesh_nodes(body, h, hit, origin_x, origin_y, step_x, step_y, axes)
            progress.advance("メッシュの突起を補正")

        heights = solve_drape(h, progress)
        for vertex, i, j in vertices:
            delta = heights[i][j] - _dot(vertex.geometry, up)
            if abs(delta) > 1e-9:
                vertex.move(_world_vector(up, delta))
            progress.advance("ラバーの頂点を移動")

        for _ in range(FIT_ITERS):
            for vertex, i, j in vertices:
                if hit[i][j]:
                    error = h[i][j] - _dot(vertex.smoothGeometry, up)
                    if error > 1e-6:
                        vertex.move(_world_vector(up, error))
                progress.advance("スムース面を補正")
    except Exception:
        form.finishEdit()
        form.deleteMe()
        raise
    form.finishEdit()
    progress.advance("Form を完成", 1)
    return form


def solve_drape(h: Grid, progress: _Progress | None = None) -> Grid:
    """障害物 h と外周固定の下で膜エネルギー最小の高さを緩和法で求める。

    Args:
        h: 格子点ごとの下限高さ(障害物)[cm]。外周の値は固定値として使う。
        progress: 進捗ダイアログ。None なら表示しない。

    Returns:
        格子点ごとのラバーの高さ [cm]。h は変更しない。
    """
    n = len(h)
    z = [row[:] for row in h]
    nodes = [(i, j) for i in range(1, n - 1) for j in range(1, n - 1)]

    def average(i: int, j: int) -> float:
        """上下左右４点の平均の高さを返す。"""
        return (z[i - 1][j] + z[i + 1][j] + z[i][j - 1] + z[i][j + 1]) / 4

    for _ in range(MEMBRANE_ITERS):
        for i, j in nodes:
            v = z[i][j] + SOR * (average(i, j) - z[i][j])
            z[i][j] = max(v, h[i][j])
        if progress is not None:
            progress.advance("ラバーの膜を緩和")
    for _ in range(SMOOTH_ITERS):
        nz = [row[:] for row in z]
        for i, j in nodes:
            nz[i][j] = max(0.5 * z[i][j] + 0.5 * average(i, j), h[i][j])
        z = nz
        if progress is not None:
            progress.advance("ラバーの表面を平滑化")
    return z


def apply_mesh_nodes(
    body: fusion.BRepBody,
    h: Grid,
    hit: list[list[bool]],
    origin_x: float,
    origin_y: float,
    step_x: float,
    step_y: float,
    axes: Axes,
) -> None:
    """メッシュ節点を含むセルの４隅の障害物高さを補う。

    格子点のレイだけでは拾えない細い突起を、メッシュ節点の高さで補う。

    Args:
        body: 対象ボディ。
        h: 格子点ごとの下限高さ [cm]。その場で更新する。
        hit: 格子点がボディに載っているか。その場で更新する。
        origin_x: 格子原点の X [cm]。
        origin_y: 格子原点の Y [cm]。
        step_x: X 方向の格子間隔 [cm]。
        step_y: Y 方向の格子間隔 [cm]。
        axes: 重力座標系の (ex, ey, up)。
    """
    calc = body.meshManager.createMeshCalculator()
    calc.setQuality(fusion.TriangleMeshQualityOptions.NormalQualityTriangleMesh)
    mesh = calc.calculate()
    _apply_nodes(
        mesh.nodeCoordinatesAsDouble, h, hit, origin_x, origin_y, step_x, step_y, axes
    )


def apply_mesh_body(
    body: fusion.MeshBody,
    h: Grid,
    hit: list[list[bool]],
    origin_x: float,
    origin_y: float,
    step_x: float,
    step_y: float,
    axes: Axes,
) -> None:
    """メッシュボディの三角形を格子へ投影し、障害物高さを求める。

    メッシュボディにはレイが当たらないため、三角形ごとに内側の格子点を探して
    高さを補間する。最後に節点の高さでも補う。

    Args:
        body: 対象のメッシュボディ。
        h: 格子点ごとの下限高さ [cm]。その場で更新する。
        hit: 格子点がボディに載っているか。その場で更新する。
        origin_x: 格子原点の X [cm]。
        origin_y: 格子原点の Y [cm]。
        step_x: X 方向の格子間隔 [cm]。
        step_y: Y 方向の格子間隔 [cm]。
        axes: 重力座標系の (ex, ey, up)。
    """
    ex, ey, up = axes
    mesh = body.displayMesh
    coords = mesh.nodeCoordinatesAsDouble
    indices = mesh.nodeIndices
    # 節点を格子単位の (u, v) と高さへ投影しておく
    us, vs, ws = [], [], []
    for k in range(0, len(coords), 3):
        x, y, z = coords[k], coords[k + 1], coords[k + 2]
        us.append((x * ex[0] + y * ex[1] + z * ex[2] - origin_x) / step_x)
        vs.append((x * ey[0] + y * ey[1] + z * ey[2] - origin_y) / step_y)
        ws.append(x * up[0] + y * up[1] + z * up[2] + THICKNESS)
    eps = 1e-9
    for k in range(0, len(indices), 3):
        a, b, c = indices[k], indices[k + 1], indices[k + 2]
        ua, ub, uc = us[a], us[b], us[c]
        va, vb, vc = vs[a], vs[b], vs[c]
        det = (ub - ua) * (vc - va) - (uc - ua) * (vb - va)
        if abs(det) < 1e-12:
            continue  # 重力と平行な三角形は節点の補正に任せる
        i0 = max(math.ceil(min(ua, ub, uc)), 0)
        i1 = min(math.floor(max(ua, ub, uc)), DIV)
        j0 = max(math.ceil(min(va, vb, vc)), 0)
        j1 = min(math.floor(max(va, vb, vc)), DIV)
        wa, wb, wc = ws[a], ws[b], ws[c]
        for i in range(i0, i1 + 1):
            for j in range(j0, j1 + 1):
                s = ((i - ua) * (vc - va) - (uc - ua) * (j - va)) / det
                t = ((ub - ua) * (j - va) - (i - ua) * (vb - va)) / det
                if s < -eps or t < -eps or s + t > 1 + eps:
                    continue
                height = wa + s * (wb - wa) + t * (wc - wa)
                if height > h[i][j]:
                    h[i][j] = height
                    hit[i][j] = True
    _apply_nodes(coords, h, hit, origin_x, origin_y, step_x, step_y, axes)


def _apply_nodes(
    coords: list[float],
    h: Grid,
    hit: list[list[bool]],
    origin_x: float,
    origin_y: float,
    step_x: float,
    step_y: float,
    axes: Axes,
) -> None:
    """節点を含むセルの４隅の障害物高さを節点の高さで補う。

    Args:
        coords: 節点の座標列 [x, y, z, ...] [cm]。
        h: 格子点ごとの下限高さ [cm]。その場で更新する。
        hit: 格子点がボディに載っているか。その場で更新する。
        origin_x: 格子原点の X [cm]。
        origin_y: 格子原点の Y [cm]。
        step_x: X 方向の格子間隔 [cm]。
        step_y: Y 方向の格子間隔 [cm]。
        axes: 重力座標系の (ex, ey, up)。
    """
    ex, ey, up = axes
    for k in range(0, len(coords), 3):
        x, y, z = coords[k], coords[k + 1], coords[k + 2]
        px = x * ex[0] + y * ex[1] + z * ex[2]
        py = x * ey[0] + y * ey[1] + z * ey[2]
        height = x * up[0] + y * up[1] + z * up[2] + THICKNESS
        ci = min(max(int((px - origin_x) // step_x), 0), DIV - 1)
        cj = min(max(int((py - origin_y) // step_y), 0), DIV - 1)
        for i in (ci, ci + 1):
            for j in (cj, cj + 1):
                if height > h[i][j]:
                    h[i][j] = height
                    hit[i][j] = True


def get_top_height(
    root: fusion.Component,
    bodies: list[Body],
    x: float,
    y: float,
    ray_z: float,
    axes: Axes,
) -> float | None:
    """選択ボディへのレイ交点のうち、重力と逆方向で最も高い点を返す。

    Args:
        root: レイを飛ばすルートコンポーネント。
        bodies: 対象ボディ。これ以外のボディへの交点は無視する。
        x: 重力座標系の X [cm]。
        y: 重力座標系の Y [cm]。
        ray_z: レイの始点の高さ [cm]。
        axes: 重力座標系の (ex, ey, up)。

    Returns:
        最も高い交点の高さ [cm]。交点が無ければ None。
    """
    hit_points = core.ObjectCollection.create()
    up = axes[2]
    ents = root.findBRepUsingRay(
        _world_point(x, y, ray_z, axes),
        _world_vector(up, -1),
        fusion.BRepEntityTypes.BRepFaceEntityType,
        -1.0,
        False,
        hit_points,
    )
    heights = [
        _dot(hit_points.item(i), up)
        for i in range(ents.count)
        if any(_same_body(ents.item(i).body, body) for body in bodies)
    ]
    return max(heights) if heights else None
