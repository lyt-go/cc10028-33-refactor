"""关系去重重构：历史产物重放与比对工具（测试夹具，不属于应用运行时代码）。

用法：
  python -I replay_dedup.py capture <服务目录> <输出json>   # 重放场景并冻结产物
  python -I replay_dedup.py compare <基线json> <新产物json> # 逐字段比对两次产物

服务目录可指向 app/services（重构后）或 tools/backup（重构前），
从而用同一份夹具分别驱动新旧两套实现。
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
import tempfile
import types
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.services import relation_evidence  # noqa: E402


def _load_service_module(module_name: str, file_path: Path):
    """按文件路径加载服务模块，并让其内部的 app.services.* 导入复用已加载对象。"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_services(services_dir: Path):
    """加载 base 与关系抽取/图谱两个服务，DATA_ROOT 重定向到临时目录。

    base 始终来自正式 app 包（重构不涉及它）；两个目标服务从 services_dir
    加载，既可指向 app/services（重构后）也可指向 tools/backup（重构前）。
    """
    tmp_root = Path(tempfile.mkdtemp(prefix="dedup_replay_"))

    import app.services.base as base  # 正常导入，保证 app.store 等包依赖可用
    base.DATA_ROOT = tmp_root
    # 两个服务都是 `from app.services.base import ..., DATA_ROOT` 形式导入，
    # 需要同步改掉它们模块里绑定的 DATA_ROOT。
    for name in ("relation_extract", "kg_build"):
        mod = _load_service_module(
            f"app.services.{name}", services_dir / f"{name}.py"
        )
        mod.DATA_ROOT = tmp_root
    rel_mod = sys.modules["app.services.relation_extract"]
    kg_mod = sys.modules["app.services.kg_build"]
    return base, rel_mod, kg_mod, tmp_root


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _checkpoint(tmp_root: Path, name: str) -> dict | None:
    p = tmp_root / "checkpoints" / f"{name}.json"
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _strip_volatile(records: list[dict], volatile_fields: tuple[str, ...]) -> list[dict]:
    """剔除时间戳等易变字段，保证两次重放可逐字段比较。"""
    return [
        {k: v for k, v in sorted(rec.items()) if k not in volatile_fields}
        for rec in records
    ]


def _strip_cp(cp: dict | None) -> dict | None:
    if cp is None:
        return None
    return {k: v for k, v in sorted(cp.items()) if k != "_updated_at"}


def _rel(rid, head, rtype, tail, conf, source: str):
    """构造一条关系候选记录，source 用于区分不同来源。"""
    return {
        "relation_id": rid,
        "head_entity": head,
        "head_type": "ORG",
        "tail_entity": tail,
        "tail_type": "PERSON",
        "relation_type": rtype,
        "confidence": conf,
        "evidence_sentence": f"{source}:{head}-{rtype}-{tail}@{conf}",
        "method": "Relation-BERT",
        "model_version": "v2.3.0",
        "extracted_at": 0,
        "source": source,
    }


def _write_relations(base, tmp_root: Path, set_id: str, records: list[dict]) -> Path:
    p = tmp_root / "relations" / f"{set_id}_relations.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    base.BaseModuleService()._write_jsonl(p, records)
    return p


def _read_manifest(tmp_root: Path, rel_path: str) -> dict | None:
    p = tmp_root / rel_path
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _snapshot_graph_artifacts(tmp_root: Path, set_id: str) -> dict:
    graph = _read_jsonl(tmp_root / "graph" / f"{set_id}_graph.jsonl")
    return {
        "graph_triples": _strip_volatile(
            graph, ("ingested_at",)
        ),
        "graph_checkpoint": _strip_cp(_checkpoint(tmp_root, f"graph_{set_id}")),
        # 重构后新增的可验证去重证据组产物；旧实现没有该文件则为 None
        "relation_evidence_rows": _read_jsonl(
            tmp_root / "relation_evidence" / f"{set_id}_relation_evidence.jsonl"
        ) or None,
        "relation_evidence_manifest": _read_manifest(
            tmp_root,
            f"relation_evidence/{set_id}_relation_evidence_manifest.json",
        ),
        "graph_evidence_rows": _read_jsonl(
            tmp_root / "relation_evidence" / f"{set_id}_graph_relation_evidence.jsonl"
        ) or None,
        "graph_evidence_manifest": _read_manifest(
            tmp_root,
            f"relation_evidence/{set_id}_graph_relation_evidence_manifest.json",
        ),
        "relation_checkpoint": _strip_cp(
            _checkpoint(tmp_root, f"relation_{set_id}")
        ),
    }


def _run_relation_side(rel_mod, base, tmp_root, set_id, candidates):
    """关系抽取侧：直接驱动去重方法，保留原有对外口径。"""
    svc = rel_mod.RelationExtractService()
    # 重构后：在去重前的原始候选上构建并持久化可验证关系证据组
    if hasattr(svc, "build_relation_evidence"):
        evidence = svc.build_relation_evidence(candidates)
        checksum = relation_evidence.canonical_checksum(candidates)
        svc._write_relation_evidence(set_id, evidence, checksum)
    deduped = svc._deduplicate_relations(candidates)
    # 同时落一份关系产物，模拟抽取持久化
    _write_relations(base, tmp_root, set_id, deduped)
    return _strip_volatile(deduped, ("extracted_at",))


def _run_kg(kg_mod, tmp_root, set_id):
    """图谱侧：一次 build_from_relations 对应一次完整输入，不跨批拼组。"""
    svc = kg_mod.KgBuildService()
    result = svc.build_from_relations(set_id)
    artifacts = _snapshot_graph_artifacts(tmp_root, set_id)
    return {"build_result": {k: v for k, v in sorted(result.items())
                             if k != "output"},
            "artifacts": artifacts}


def _scenario_tie(rel_mod, kg_mod, base, tmp_root):
    """场景1 同分：两个来源不同但置信度相同的候选，保留先到者（同分处理顺序）。"""
    set_id = "S-TIE"
    candidates = [
        _rel("REL-000001", "甲集团", "EMPLOY", "张三", 0.88, "来源A"),
        _rel("REL-000002", "甲集团", "EMPLOY", "张三", 0.88, "来源B"),
        _rel("REL-000003", "甲集团", "LOCATE", "Beijing", 0.70, "来源A"),
    ]
    rel_out = _run_relation_side(rel_mod, base, tmp_root, set_id, candidates)
    kg_out = _run_kg(kg_mod, tmp_root, set_id)
    return {"relation_dedup": rel_out, "kg": kg_out}


def _scenario_diff(rel_mod, kg_mod, base, tmp_root):
    """场景2 异分：同组不同置信度取最高分；低于0.6的入图前被过滤。"""
    set_id = "S-DIFF"
    candidates = [
        _rel("REL-000010", "乙公司", "INVEST", "5亿元", 0.66, "来源A"),
        _rel("REL-000011", "乙公司", "INVEST", "5亿元", 0.93, "来源B"),
        _rel("REL-000012", "乙公司", "INVEST", "5亿元", 0.81, "来源C"),
        _rel("REL-000013", "丙集团", "COLLABORATE", "丁集团", 0.50, "来源A"),
    ]
    rel_out = _run_relation_side(rel_mod, base, tmp_root, set_id, candidates)
    kg_out = _run_kg(kg_mod, tmp_root, set_id)
    return {"relation_dedup": rel_out, "kg": kg_out}


def _scenario_append(rel_mod, kg_mod, base, tmp_root):
    """场景3 中途追加：首批建图后追加第三条更正记录，再重建；首个生效结果可重算。"""
    set_id = "S-APPEND"
    batch1 = [
        _rel("REL-000020", "戊集团", "EMPLOY", "王五", 0.80, "来源A"),
        _rel("REL-000021", "戊集团", "EMPLOY", "王五", 0.80, "来源B"),
    ]
    batch2 = batch1 + [
        _rel("REL-000022", "戊集团", "EMPLOY", "王五", 0.91, "来源C-更正"),
    ]
    # 首批：完整输入建图（首个生效结果及其输入依据）
    first_dedup = _run_relation_side(rel_mod, base, tmp_root, set_id, batch1)
    first_kg = _run_kg(kg_mod, tmp_root, set_id)
    # 追加更正记录后重写完整关系产物，再重建（仍是一次完整输入，不拼两批比较结果）
    second_dedup = _run_relation_side(rel_mod, base, tmp_root, set_id, batch2)
    second_kg = _run_kg(kg_mod, tmp_root, set_id)
    return {
        "first_effective": {"relation_dedup": first_dedup, "kg": first_kg},
        "after_correction": {"relation_dedup": second_dedup, "kg": second_kg},
        # 重建后关系文件中实际留存的全部记录（追加写入的累计视图）
        "relation_file_after_rebuild": _strip_volatile(
            _read_jsonl(tmp_root / "relations" / f"{set_id}_relations.jsonl"),
            ("extracted_at",),
        ),
        "graph_file_after_rebuild": _strip_volatile(
            _read_jsonl(tmp_root / "graph" / f"{set_id}_graph.jsonl"),
            ("ingested_at",),
        ),
    }


def _scenario_full_pipeline(rel_mod, kg_mod, base, tmp_root):
    """场景4 真实链路：解析→分块→实体→关系抽取→图谱，冻结端到端三元组与统计。"""
    from app.services.format_parse import FormatParseService
    from app.services.chunk_split import ChunkSplitService
    from app.services.entity_recognize import EntityRecognizeService

    # 这些模块同样 from ...base import DATA_ROOT，需要重定向
    import app.services.base as real_base
    real_base.DATA_ROOT = tmp_root
    for mod_name in ("format_parse", "chunk_split", "entity_recognize"):
        mod = importlib.import_module(f"app.services.{mod_name}")
        mod.DATA_ROOT = tmp_root

    doc_id = "DOC-FULL"
    parse_svc = FormatParseService()
    task, _ = parse_svc.create_entry({
        "任务编号": "PR-FULL-001", "文档编号": doc_id, "解析器类型": "PDF",
    })
    parse_svc.execute_parse(task["id"])
    ChunkSplitService().split_from_document(f"{doc_id}_parsed")
    EntityRecognizeService().recognize_from_chunk(f"{doc_id}_parsed")
    rel_result = rel_mod.RelationExtractService().extract_from_entities(
        f"{doc_id}_parsed"
    )
    kg_result = kg_mod.KgBuildService().build_from_relations(f"{doc_id}_parsed")

    set_id = f"{doc_id}_parsed"
    rels = _strip_volatile(
        _read_jsonl(tmp_root / "relations" / f"{set_id}_relations.jsonl"),
        ("extracted_at",),
    )
    graph = _strip_volatile(
        _read_jsonl(tmp_root / "graph" / f"{set_id}_graph.jsonl"),
        ("ingested_at",),
    )
    return {
        "relation_extract_result": {
            k: v for k, v in sorted(rel_result.items()) if k != "output"
        },
        "kg_build_result": {
            k: v for k, v in sorted(kg_result.items()) if k != "output"
        },
        "relations_file": rels,
        "graph_file": graph,
        "graph_checkpoint": _strip_cp(_checkpoint(tmp_root, f"graph_{set_id}")),
        "relation_checkpoint": _strip_cp(
            _checkpoint(tmp_root, f"relation_{set_id}")
        ),
        "relation_evidence_rows": _read_jsonl(
            tmp_root / "relation_evidence" / f"{set_id}_relation_evidence.jsonl"
        ) or None,
        "relation_evidence_manifest": _read_manifest(
            tmp_root,
            f"relation_evidence/{set_id}_relation_evidence_manifest.json",
        ),
        "graph_evidence_rows": _read_jsonl(
            tmp_root / "relation_evidence" / f"{set_id}_graph_relation_evidence.jsonl"
        ) or None,
        "graph_evidence_manifest": _read_manifest(
            tmp_root,
            f"relation_evidence/{set_id}_graph_relation_evidence_manifest.json",
        ),
    }


def _normalize_tmp_paths(node, tmp_root: Path):
    """把产物中的临时目录前缀归一化为占位符，使两次重放可逐字段比较。"""
    if isinstance(node, dict):
        return {k: _normalize_tmp_paths(v, tmp_root) for k, v in node.items()}
    if isinstance(node, list):
        return [_normalize_tmp_paths(v, tmp_root) for v in node]
    if isinstance(node, str) and str(tmp_root) in node:
        return node.replace(str(tmp_root), "<DATA_ROOT>")
    return node


def _normalize_unordered_lists(node):
    """对检查点里由 set 生成的无序列表做确定性排序（口径内容不变）。"""
    if isinstance(node, dict):
        out = {}
        for k, v in node.items():
            if k == "relation_types" and isinstance(v, list):
                out[k] = sorted(v)
            else:
                out[k] = _normalize_unordered_lists(v)
        return out
    if isinstance(node, list):
        return [_normalize_unordered_lists(v) for v in node]
    return node


def capture(services_dir: Path, out_path: Path) -> None:
    base, rel_mod, kg_mod, tmp_root = _load_services(services_dir)
    try:
        payload = {
            "services_dir": str(services_dir),
            "scenarios": {
                "tie": _scenario_tie(rel_mod, kg_mod, base, tmp_root),
                "diff": _scenario_diff(rel_mod, kg_mod, base, tmp_root),
                "append": _scenario_append(rel_mod, kg_mod, base, tmp_root),
                "full_pipeline": _scenario_full_pipeline(
                    rel_mod, kg_mod, base, tmp_root
                ),
            },
        }
        payload = _normalize_unordered_lists(_normalize_tmp_paths(payload, tmp_root))
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
    print(f"captured -> {out_path}")


# 重构后新增、旧实现不存在的加法式产物字段：不要求与旧产物相等，
# 但要求新产物中非空且结构完整，其余字段仍须与历史基线严格一致。
# 图谱检查点新增的输入校验值/证据文件路径同样属于加法式字段。
_NEW_ONLY_SUFFIXES = (
    ".relation_evidence_rows",
    ".relation_evidence_manifest",
    ".graph_evidence_rows",
    ".graph_evidence_manifest",
)
_NEW_ONLY_CP_SUFFIXES = (
    ".graph_checkpoint.input_checksum",
    ".graph_checkpoint.evidence_file",
)
NEW_ONLY_FIELDS: set[str] = set()


def _is_new_only(path: str) -> bool:
    return (
        path.endswith(_NEW_ONLY_SUFFIXES)
        or path.endswith(_NEW_ONLY_CP_SUFFIXES)
    )


# 旧基线里的占位字段（旧实现恒为 None），重构后改名为 relation_evidence_rows，
# 比对时跳过该历史空字段。
def _is_deprecated_null(path: str) -> bool:
    return path.endswith(".evidence_groups")


def _diff_values(path: str, old, new, diffs: list[str],
                 new_only: list[str]) -> None:
    # 旧实现没有该证据产物（None）而新实现生成了结构化产物：按加法式字段处理
    if old is None and new is not None and _is_new_only(path):
        if new in (None, [], {}):
            diffs.append(f"{path}: 新增证据产物为空，未生成可验证证据组")
        else:
            new_only.append(path)
        return
    if isinstance(old, dict) and isinstance(new, dict):
        keys = sorted(set(old) | set(new))
        for k in keys:
            child = f"{path}.{k}"
            if k not in old:
                if _is_new_only(child):
                    if new[k] in (None, [], {}):
                        diffs.append(f"{child}: 新增证据产物为空，未生成可验证证据组")
                    else:
                        new_only.append(child)
                    continue
                diffs.append(f"{child}: 新产物出现未预期的额外字段")
            elif k not in new:
                if _is_deprecated_null(child) and old[k] is None:
                    continue  # 旧实现的历史空字段，重构后已改名
                diffs.append(f"{child}: 新产物缺失历史字段")
            else:
                _diff_values(child, old[k], new[k], diffs, new_only)
    elif isinstance(old, list) and isinstance(new, list):
        if len(old) != len(new):
            diffs.append(f"{path}: 列表长度 {len(old)} != {len(new)}")
        for i, (a, b) in enumerate(zip(old, new)):
            _diff_values(f"{path}[{i}]", a, b, diffs, new_only)
    elif type(old) is not type(new):
        diffs.append(f"{path}: 类型 {type(old).__name__} != {type(new).__name__}")
    elif old != new:
        diffs.append(f"{path}: {old!r} != {new!r}")


def compare(baseline_path: Path, current_path: Path) -> int:
    with open(baseline_path, "r", encoding="utf-8") as f:
        old = json.load(f)
    with open(current_path, "r", encoding="utf-8") as f:
        new = json.load(f)
    diffs: list[str] = []
    new_only: list[str] = []
    _diff_values("root", old["scenarios"], new["scenarios"], diffs, new_only)
    if diffs:
        print(f"比对失败，共 {len(diffs)} 处差异：")
        for d in diffs:
            print(" -", d)
        return 1
    print("比对通过：三元组、边、来源与图谱统计与历史基线完全一致（原有口径不变）。")
    print(f"加法式新增可验证证据产物 {len(new_only)} 项均已非空生成：")
    for item in sorted(new_only):
        print(" +", item)
    return 0


def main() -> int:
    if len(sys.argv) < 4:
        print(__doc__)
        return 2
    mode, services_or_baseline, out_or_current = sys.argv[1:4]
    if mode == "capture":
        capture(Path(services_or_baseline), Path(out_or_current))
        return 0
    if mode == "compare":
        return compare(Path(services_or_baseline), Path(out_or_current))
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
