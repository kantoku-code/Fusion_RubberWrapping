# Fusion 360 API Python script
"""選択した複数のボディに、指定した重力方向から１枚のラバーをかける。"""

import math
import importlib.util
import os
import traceback
from typing import Any

import adsk
import adsk.core as core
import adsk.fusion as fusion

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_OBSTACLE_PATH = os.path.join(_SCRIPT_DIR, "ObstacleProblem.py")
_OBSTACLE_SPEC = importlib.util.spec_from_file_location(
    "rubber_wrapping_obstacle", _OBSTACLE_PATH)
if _OBSTACLE_SPEC is None or _OBSTACLE_SPEC.loader is None:
    raise ImportError("ラバーの計算モジュールを読み込めません: " + _OBSTACLE_PATH)
obstacle = importlib.util.module_from_spec(_OBSTACLE_SPEC)
_OBSTACLE_SPEC.loader.exec_module(obstacle)

_COMMAND_ID = "rubber_wrapping_run_command"
_BODY_ID = "rubber_wrapping_bodies"
_GRAVITY_ID = "rubber_wrapping_gravity"
_REVERSE_ID = "rubber_wrapping_reverse"
_SETTINGS_ID = "rubber_wrapping_settings"
_DIV_ID = "rubber_wrapping_div"
_MARGIN_ID = "rubber_wrapping_margin"
_THICKNESS_ID = "rubber_wrapping_thickness"
_MEMBRANE_ITERS_ID = "rubber_wrapping_membrane_iters"
_SOR_ID = "rubber_wrapping_sor"
_SMOOTH_ITERS_ID = "rubber_wrapping_smooth_iters"
_FIT_ITERS_ID = "rubber_wrapping_fit_iters"
_PREVIEW_NAME = "RubberWrapping gravity preview v1"
_handlers: list[object] = []  # イベントハンドラを GC から守るための参照


def _selected_bodies(
        selection_input: core.SelectionCommandInput
) -> list[fusion.BRepBody | fusion.MeshBody]:
    """選択入力から、重複を除いた有効なボディを取り出す。

    Args:
        selection_input: 対象ボディの選択入力。

    Returns:
        選択順のボディ。
    """
    bodies = []
    for index in range(selection_input.selectionCount):
        body = selection_input.selection(index).entity
        if body.isValid and not any(
            obstacle._same_body(body, old) for old in bodies
        ):
            bodies.append(body)
    return bodies


def _gravity_vector(gravity_input: core.SelectionCommandInput,
                    reverse_input: core.BoolValueCommandInput) -> core.Vector3D:
    """WaterPool と同じ規則で重力方向の単位ベクトルを返す。

    Args:
        gravity_input: 軸・直線の選択入力。未選択なら -Z とする。
        reverse_input: 方向反転のチェックボックス。

    Returns:
        重力方向の単位ベクトル。

    Raises:
        ValueError: 軸・直線以外を選択したとき、または方向の長さが 0 のとき。
    """
    if gravity_input.selectionCount == 0:
        direction = core.Vector3D.create(0, 0, -1)
    else:
        entity = gravity_input.selection(0).entity
        if entity.objectType == fusion.SketchLine.classType():
            line = entity.worldGeometry
            direction = line.startPoint.vectorTo(line.endPoint)
        elif entity.objectType == fusion.ConstructionAxis.classType():
            direction = entity.geometry.direction.copy()
        elif entity.objectType == fusion.BRepEdge.classType():
            direction = entity.startVertex.geometry.vectorTo(
                entity.endVertex.geometry
            )
        else:
            raise ValueError("重力方向には軸または直線を選択してください")
    if not direction.normalize():
        raise ValueError("重力方向の長さが 0 です")
    if reverse_input.value:
        direction.scaleBy(-1)
    return direction


def _add_settings(inputs: core.CommandInputs, length_units: str) -> None:
    """計算パラメータの入力欄を、閉じた状態の「詳細設定」グループに追加する。

    Args:
        inputs: コマンドの入力一覧。
        length_units: 余白・厚みの表示に使う長さ単位。
    """
    group = inputs.addGroupCommandInput(_SETTINGS_ID, "詳細設定")
    group.isExpanded = False
    children = group.children
    children.addIntegerSpinnerCommandInput(
        _DIV_ID, "分割数", 2, 200, 1, obstacle.DIV)
    children.addValueInput(
        _MARGIN_ID, "余白", length_units,
        core.ValueInput.createByReal(obstacle.MARGIN))
    children.addValueInput(
        _THICKNESS_ID, "厚み", length_units,
        core.ValueInput.createByReal(obstacle.THICKNESS))
    children.addIntegerSpinnerCommandInput(
        _MEMBRANE_ITERS_ID, "膜の反復回数", 1, 100000, 100, obstacle.MEMBRANE_ITERS)
    children.addFloatSpinnerCommandInput(
        _SOR_ID, "SOR 係数", "", 0.1, 1.99, 0.1, obstacle.SOR)
    children.addIntegerSpinnerCommandInput(
        _SMOOTH_ITERS_ID, "平滑化の反復回数", 0, 1000, 1, obstacle.SMOOTH_ITERS)
    children.addIntegerSpinnerCommandInput(
        _FIT_ITERS_ID, "補正回数", 0, 100, 1, obstacle.FIT_ITERS)


def _apply_settings(inputs: core.CommandInputs) -> None:
    """詳細設定の値を計算モジュールの定数へ反映する。

    Args:
        inputs: コマンドの入力一覧。
    """
    obstacle.DIV = inputs.itemById(_DIV_ID).value
    obstacle.MARGIN = inputs.itemById(_MARGIN_ID).value
    obstacle.THICKNESS = inputs.itemById(_THICKNESS_ID).value
    obstacle.MEMBRANE_ITERS = inputs.itemById(_MEMBRANE_ITERS_ID).value
    obstacle.SOR = inputs.itemById(_SOR_ID).value
    obstacle.SMOOTH_ITERS = inputs.itemById(_SMOOTH_ITERS_ID).value
    obstacle.FIT_ITERS = inputs.itemById(_FIT_ITERS_ID).value


def _arrow_geometry(bodies: list[fusion.BRepBody | fusion.MeshBody],
                    gravity: core.Vector3D) -> tuple[list[float], list[int]]:
    """WaterPool と同じ、ボディの脇に置く重力方向矢印。

    Args:
        bodies: 対象ボディ。空なら原点に置く。
        gravity: 重力方向の単位ベクトル。

    Returns:
        矢印の座標列 [x, y, z, ...] と、線分の頂点番号の組。
    """
    if bodies:
        boxes = [body.boundingBox for body in bodies]
        low = tuple(min(getattr(box.minPoint, axis) for box in boxes) for axis in "xyz")
        high = tuple(max(getattr(box.maxPoint, axis) for box in boxes) for axis in "xyz")
        center = tuple((a + b) * 0.5 for a, b in zip(low, high))
        length = max(1.0, max(b - a for a, b in zip(low, high)) * 0.35)
    else:
        center, length = (0.0, 0.0, 0.0), 2.0
    g = (gravity.x, gravity.y, gravity.z)
    if bodies:
        side_axis = min(range(3), key=lambda index: abs(g[index]))
        center = list(center)
        center[side_axis] = low[side_axis] - length
    start = tuple(center[i] - g[i] * length * 0.5 for i in range(3))
    tip = tuple(center[i] + g[i] * length * 0.5 for i in range(3))
    helper = (1.0, 0.0, 0.0) if abs(g[0]) < 0.9 else (0.0, 1.0, 0.0)
    side = (
        g[1] * helper[2] - g[2] * helper[1],
        g[2] * helper[0] - g[0] * helper[2],
        g[0] * helper[1] - g[1] * helper[0],
    )
    side_length = math.sqrt(sum(value * value for value in side))
    side = tuple(value / side_length for value in side)
    other = (
        g[1] * side[2] - g[2] * side[1],
        g[2] * side[0] - g[0] * side[2],
        g[0] * side[1] - g[1] * side[0],
    )
    base = tuple(tip[i] - g[i] * length * 0.23 for i in range(3))
    spread = length * 0.11
    points = [start, tip]
    for cross in (side, other):
        for sign in (-1, 1):
            points.append(tuple(base[i] + sign * cross[i] * spread for i in range(3)))
    return [value for point in points for value in point], [0, 1, 1, 2, 1, 3, 1, 4, 1, 5]


def _clear_preview(root: fusion.Component) -> None:
    """重力方向矢印のプレビューを削除する。

    Args:
        root: プレビューを置いたルートコンポーネント。
    """
    groups = root.customGraphicsGroups
    for index in range(groups.count - 1, -1, -1):
        group = groups.item(index)
        if group.name == _PREVIEW_NAME:
            group.deleteMe()


def _update_preview(inputs: core.CommandInputs, root: fusion.Component) -> None:
    """現在の入力から重力方向矢印のプレビューを描き直す。

    Args:
        inputs: コマンドの入力一覧。
        root: プレビューを置くルートコンポーネント。
    """
    bodies = _selected_bodies(inputs.itemById(_BODY_ID))
    gravity = _gravity_vector(
        inputs.itemById(_GRAVITY_ID), inputs.itemById(_REVERSE_ID)
    )
    coords, indices = _arrow_geometry(bodies, gravity)
    _clear_preview(root)
    group = root.customGraphicsGroups.add()
    group.name = _PREVIEW_NAME
    group.isSelectable = False
    group.isChildrenSelectable = False
    lines = group.addLines(fusion.CustomGraphicsCoordinates.create(coords), indices, False)
    lines.weight = 3
    lines.color = fusion.CustomGraphicsShowThroughColorEffect.create(
        core.Color.create(235, 70, 30, 255), 1.0
    )
    core.Application.get().activeViewport.refresh()


class _CreatedHandler(core.CommandCreatedEventHandler):
    """コマンドダイアログの入力欄とイベントハンドラを登録する。"""

    def notify(self, args: core.CommandCreatedEventArgs) -> None:
        """入力欄(対象ボディ・重力方向・方向反転・詳細設定)を作り、各イベントを登録する。

        Args:
            args: コマンド作成イベントの引数。
        """
        try:
            design = fusion.Design.cast(core.Application.get().activeProduct)
            root = design.rootComponent
            inputs = args.command.commandInputs
            bodies = inputs.addSelectionInput(
                _BODY_ID, "対象ボディ", "ソリッド／サーフェス／メッシュボディを１つ以上選択"
            )
            bodies.addSelectionFilter("SolidBodies")
            bodies.addSelectionFilter("SurfaceBodies")
            bodies.addSelectionFilter("MeshBodies")
            bodies.setSelectionLimits(1, 0)
            bodies.isUseCurrentSelections = False

            gravity = inputs.addSelectionInput(
                _GRAVITY_ID, "重力方向", "軸または直線を１つ選択（省略時は -Z）"
            )
            for filter_name in ("ConstructionLines", "LinearEdges", "SketchLines"):
                gravity.addSelectionFilter(filter_name)
            gravity.setSelectionLimits(0, 1)
            gravity.isUseCurrentSelections = False
            inputs.addBoolValueInput(_REVERSE_ID, "方向反転", True, "", False)
            _add_settings(inputs, design.unitsManager.defaultLengthUnits)

            for event, handler in (
                (args.command.activate, _ActivateHandler(root)),
                (args.command.inputChanged, _InputChangedHandler(root)),
                (args.command.executePreview, _PreviewHandler(root)),
                (args.command.validateInputs, _ValidateHandler()),
                (args.command.execute, _ExecuteHandler(root)),
                (args.command.destroy, _DestroyHandler(root)),
            ):
                event.add(handler)
                _handlers.append(handler)
        except Exception:
            _report_error()


class _ActivateHandler(core.CommandEventHandler):
    """ダイアログを開いたときにプレビューを表示する。"""

    def __init__(self, root: fusion.Component) -> None:
        """
        Args:
            root: プレビューを置くルートコンポーネント。
        """
        super().__init__()
        self.root = root

    def notify(self, args: core.CommandEventArgs) -> None:
        """プレビューを描く。

        Args:
            args: コマンドイベントの引数。
        """
        try:
            _update_preview(args.command.commandInputs, self.root)
        except Exception:
            _report_error()


class _InputChangedHandler(core.InputChangedEventHandler):
    """入力が変わったときにプレビューを描き直す。"""

    def __init__(self, root: fusion.Component) -> None:
        """
        Args:
            root: プレビューを置くルートコンポーネント。
        """
        super().__init__()
        self.root = root

    def notify(self, args: core.InputChangedEventArgs) -> None:
        """プレビューを描き直す。

        Args:
            args: 入力変更イベントの引数。
        """
        try:
            _update_preview(args.input.parentCommand.commandInputs, self.root)
        except Exception:
            _report_error()


class _PreviewHandler(core.CommandEventHandler):
    """executePreview でもプレビューを描き直す。"""

    def __init__(self, root: fusion.Component) -> None:
        """
        Args:
            root: プレビューを置くルートコンポーネント。
        """
        super().__init__()
        self.root = root

    def notify(self, args: core.CommandEventArgs) -> None:
        """プレビューを描き直す。プレビューを結果としては採用しない。

        Args:
            args: コマンドイベントの引数。
        """
        try:
            _update_preview(args.command.commandInputs, self.root)
            args.isValidResult = False
        except Exception:
            _report_error()


class _ValidateHandler(core.ValidateInputsEventHandler):
    """対象ボディが選ばれ、余白・厚みが 0 以上のときだけ OK を押せるようにする。"""

    def notify(self, args: core.ValidateInputsEventArgs) -> None:
        """入力の妥当性を判定する。

        Args:
            args: 入力検証イベントの引数。
        """
        lengths = [args.inputs.itemById(input_id) for input_id in (_MARGIN_ID, _THICKNESS_ID)]
        args.areInputsValid = (
            bool(_selected_bodies(args.inputs.itemById(_BODY_ID)))
            and all(value.isValidExpression and value.value >= 0 for value in lengths)
        )


class _ExecuteHandler(core.CommandEventHandler):
    """OK を押したときにラバーを作る。"""

    def __init__(self, root: fusion.Component) -> None:
        """
        Args:
            root: プレビューを置いたルートコンポーネント。
        """
        super().__init__()
        self.root = root

    def notify(self, args: core.CommandEventArgs) -> None:
        """プレビューを消してからラバーを作り、処理時間を表示する。

        Args:
            args: コマンドイベントの引数。
        """
        try:
            app = core.Application.get()
            inputs = args.command.commandInputs
            bodies = _selected_bodies(inputs.itemById(_BODY_ID))
            if not bodies:
                return
            gravity = _gravity_vector(
                inputs.itemById(_GRAVITY_ID), inputs.itemById(_REVERSE_ID)
            )
            _apply_settings(inputs)
            _clear_preview(self.root)
            app.activeViewport.refresh()
            _, elapsed = obstacle.create_rubber(bodies, gravity)
            app.userInterface.messageBox(
                "処理時間: {:.2f} 秒".format(elapsed), "ラバーシミュレーション 完了")
        except Exception:
            _report_error()


class _DestroyHandler(core.CommandEventHandler):
    """ダイアログを閉じたときに後片付けをしてスクリプトを終了する。"""

    def __init__(self, root: fusion.Component) -> None:
        """
        Args:
            root: プレビューを置いたルートコンポーネント。
        """
        super().__init__()
        self.root = root

    def notify(self, args: core.CommandEventArgs) -> None:
        """プレビュー・コマンド定義・ハンドラを削除し、スクリプトを終了する。

        Args:
            args: コマンドイベントの引数。
        """
        try:
            _clear_preview(self.root)
            core.Application.get().activeViewport.refresh()
        finally:
            command_def = core.Application.get().userInterface.commandDefinitions.itemById(
                _COMMAND_ID
            )
            if command_def is not None:
                command_def.deleteMe()
            _handlers.clear()
            adsk.terminate()


def _report_error() -> None:
    """現在処理中の例外のトレースバックをメッセージボックスで表示する。"""
    core.Application.get().userInterface.messageBox(
        "Failed:\n{}".format(traceback.format_exc())
    )


def run(context: Any) -> None:
    """Fusion のコマンドダイアログを開く。

    Args:
        context: Fusion から渡される実行コンテキスト(未使用)。
    """
    try:
        app = core.Application.get()
        ui = app.userInterface
        if fusion.Design.cast(app.activeProduct) is None:
            ui.messageBox("Fusion のデザインを開いてください")
            return
        command_def = ui.commandDefinitions.itemById(_COMMAND_ID)
        if command_def is None:
            command_def = ui.commandDefinitions.addButtonDefinition(
                _COMMAND_ID, "RubberWrapping", "選択したボディをラバーで覆う"
            )
        created = _CreatedHandler()
        command_def.commandCreated.add(created)
        _handlers.append(created)
        command_def.execute()
        import adsk

        adsk.autoTerminate(False)
    except Exception:
        _report_error()
