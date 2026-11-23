"""关系证据组：关系抽取与知识图谱共用的三元组去重口径。

重构前关系抽取（``relation_extract._deduplicate_relations``）与图谱构建
（``kg_build._deduplicate_triples``）各自实现了一遍同构的去重逻辑。本模块把
分组键、最高分判定与同分处理顺序收敛到一处，两个服务共用同一份去重后的
关系证据组，原有行为和对外口径保持不变。

口径（与重构前严格一致）：

- 分组键：(头实体, 关系类型/谓词, 尾实体) 三元组；
- 组内保留置信度最高的记录；
- 同分时保留先进入处理序列的记录（``>`` 严格比较，等于不替换），
  即“同分处理顺序 = 输入序列中的先到者”；
- 去重结果按记录自身编号（关系编号/三元组编号）排序返回。
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Callable, Iterable

#: 组内候选需要保留用于追溯的字段
_CANDIDATE_TRACE_FIELDS = ("confidence", "source", "evidence_sentence")


def relation_group_key(record: dict[str, Any]) -> tuple[str, str, str]:
    """关系抽取侧的分组键：头实体 + 关系类型 + 尾实体。"""
    return (record["head_entity"], record["relation_type"], record["tail_entity"])


def triple_group_key(record: dict[str, Any]) -> tuple[str, str, str]:
    """图谱侧的分组键：头实体 + 谓词关系 + 尾实体（与关系侧同构）。"""
    return (record["head"], record["relation"], record["tail"])


def pick_winner(
    current: dict[str, Any] | None,
    candidate: dict[str, Any],
    confidence_field: str = "confidence",
) -> dict[str, Any]:
    """最高分判定：候选严格高于在保记录才替换；同分保留先到者。

    使用严格大于（``>``）比较，置信度相等时不替换，从而保留输入序列中
    先出现的候选——这正是重构前两处去重共同依赖的同分处理顺序。
    """
    if current is None or candidate[confidence_field] > current[confidence_field]:
        return candidate
    return current


def group_evidence(
    records: Iterable[dict[str, Any]],
    key_fn: Callable[[dict[str, Any]], tuple[str, str, str]],
    confidence_field: str = "confidence",
) -> dict[tuple[str, str, str], dict[str, Any]]:
    """按分组键归集关系证据组，组内依据共用口径选出胜出记录。

    返回“分组键 -> 证据组”的有序字典（按组首次出现顺序），证据组包含：

    - ``group_key``：分组键；
    - ``candidates``：组内全部候选（保留输入顺序，含置信度/来源/证据句）；
    - ``winner``：按最高分+同分先到规则选出的胜出记录；
    - ``winner_id`` / ``winner_confidence``：胜出记录的编号与置信度。
    """
    groups: dict[tuple[str, str, str], dict[str, Any]] = {}
    for record in records:
        key = key_fn(record)
        group = groups.get(key)
        if group is None:
            group = {
                "group_key": list(key),
                "candidates": [],
                "winner": None,
                "winner_id": None,
                "winner_confidence": None,
            }
            groups[key] = group
        group["candidates"].append(
            {
                "record_id": record.get("relation_id") or record.get("triple_id"),
                "confidence": record.get(confidence_field),
                "source": record.get("source"),
                "evidence_sentence": record.get("evidence_sentence"),
            }
        )
        group["winner"] = pick_winner(group["winner"], record, confidence_field)
        group["winner_id"] = group["winner"].get("relation_id") or group["winner"].get("triple_id")
        group["winner_confidence"] = group["winner"].get(confidence_field)
    return groups


def deduplicate(
    records: list[dict[str, Any]],
    key_fn: Callable[[dict[str, Any]], tuple[str, str, str]],
    sort_field: str,
    confidence_field: str = "confidence",
) -> list[dict[str, Any]]:
    """共用去重入口：分组→最高分/同分先到→按记录编号排序。"""
    groups = group_evidence(records, key_fn, confidence_field)
    winners = [group["winner"] for group in groups.values()]
    return sorted(winners, key=lambda r: r[sort_field])


def evidence_rows(
    records: Iterable[dict[str, Any]],
    key_fn: Callable[[dict[str, Any]], tuple[str, str, str]],
    confidence_field: str = "confidence",
) -> list[dict[str, Any]]:
    """生成可持久化、可验证的去重后关系证据组行（不含易变时间戳）。"""
    groups = group_evidence(records, key_fn, confidence_field)
    rows = []
    for group in groups.values():
        rows.append({
            "group_key": group["group_key"],
            "candidate_count": len(group["candidates"]),
            "candidates": group["candidates"],
            "winner_id": group["winner_id"],
            "winner_confidence": group["winner_confidence"],
        })
    return rows


def canonical_checksum(records: Iterable[dict[str, Any]]) -> str:
    """对输入候选做规范化 SHA256 摘要（取前16位），作为证据组的可验证输入依据。

    规范化时按 key 排序、不保留插入顺序差异，使同一批输入在任何时刻重算
    都得到相同摘要，支持“首个生效结果及输入依据可从产物重算”。
    """
    canonical = json.dumps(
        [json.loads(json.dumps(r, ensure_ascii=False, default=str, sort_keys=True))
         for r in records],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
