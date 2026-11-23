"""知识图谱服务：实体关系三元组的图谱入图、版本管理与来源分块追溯。

从关系抽取产物读取三元组并构建知识图谱，将三元组持久化为图谱快照文件。
"""
from __future__ import annotations

import json
import time
from typing import Any

from app.services.base import BaseModuleService
from app.services import relation_evidence
from app.store import store


class KgBuildService(BaseModuleService):
    MODULE = "kg_build"
    ENTITY = "图谱三元组"
    OPERATION_TERM = "建图"
    LIST_FIELDS = ["三元组编号", "头实体", "谓词关系", "尾实体", "置信度", "来源分块", "图谱版本", "入图时间"]
    REQUIRED_FIELDS = ["三元组编号", "头实体", "谓词关系"]
    STATUS_ORDER = ["草稿", "待建图", "已建图", "已废弃"]
    ACTION_RULES = {"启动建图": "待建图", "确认建图": "已建图", "标记废弃": "已废弃"}
    NEGATIVE_ACTIONS = ["标记废弃"]

    GRAPH_VERSION = "v1.0"

    def build_from_relations(self, relation_set_id: str) -> dict[str, Any]:
        """从关系抽取产物构建知识图谱三元组，将三元组写入图谱快照文件。

        一次调用只读取一份完整的关系产物作为输入来源，去重比较在这份完整
        输入上一次性完成，不把多批输入的比较结果拼成一组。去重口径与关系
        抽取共用 ``relation_evidence`` 模块（同分组键、最高分、同分先到）。
        """
        relation_file = self._data_dir("relations") / f"{relation_set_id}_relations.jsonl"
        relations = self._read_jsonl(relation_file)
# 数据操作：入湖三元组数据构建知识图谱节点与边数据
        if not relations:
            return {"error": f"关系抽取产物 {relation_set_id} 不存在或为空，请先执行关系抽取"}

        # 图谱确认的一次输入来源：对完整关系输入计算校验值与去重证据组，
        # 可验证、可由输入重算，且不跨批拼接。
        input_checksum = relation_evidence.canonical_checksum(relations)
        evidence_rows = relation_evidence.evidence_rows(
            relations, relation_evidence.relation_group_key
        )

        triples = []
        for rel in relations:
            if rel.get("confidence", 0) < 0.6:
                continue
            triple_id = f"TRIPLE-{rel['relation_id']}"
            triples.append({
                "triple_id": triple_id,
                "head": rel["head_entity"],
                "relation": rel["relation_type"],
                "tail": rel["tail_entity"],
                "confidence": rel["confidence"],
                "source": relation_set_id,
                "graph_version": self.GRAPH_VERSION,
                "ingested_at": time.time(),
            })

        # 去重：相同头-关系-尾的保留置信度最高的（共用关系证据组口径）
        triples = self._deduplicate_triples(triples)

        output_path = self._data_dir("graph") / f"{relation_set_id}_graph.jsonl"
        written = self._write_jsonl(output_path, triples)

        # 落图谱侧可验证的去重关系证据组与输入校验清单
        evidence_path = self._write_graph_evidence(
            relation_set_id, relations, evidence_rows, input_checksum
        )

        # 生成图谱边和节点的统计
        nodes = set()
        edges = []
        for t in triples:
            nodes.add(t["head"])
# 数据产物：图谱三元组数据集写入graph目录持久化文件
            nodes.add(t["tail"])
            edges.append(f"{t['head']}→[ {t['relation']} ]→{t['tail']}")

        self._write_checkpoint(f"graph_{relation_set_id}", {
            "relation_set_id": relation_set_id,
            "version": self.GRAPH_VERSION,
            "nodes": len(nodes),
            "edges": len(edges),
            "min_confidence": min((t["confidence"] for t in triples), default=0),
            "avg_confidence": sum(t["confidence"] for t in triples) / max(len(triples), 1),
            "input_checksum": input_checksum,
            "evidence_file": str(evidence_path),
        })

        return {
            "relation_set_id": relation_set_id,
            "triples": len(triples),
            "nodes": len(nodes),
            "edges": len(edges),
            "records_written": written,
# 数据导出：三元组导出文件写入导出目录供下游下载
            "output": str(output_path),
        }

    def _deduplicate_triples(self, triples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """按头实体、关系、尾实体去重，保留置信度最高的三元组。

        比较与同分处理顺序委托共享关系证据组模块，口径与重构前一致：
        严格大于才替换（同分保留先到者），结果按三元组编号排序。
        """
        return relation_evidence.deduplicate(
            triples,
            relation_evidence.triple_group_key,
            sort_field="triple_id",
        )

    def _write_graph_evidence(
        self,
        relation_set_id: str,
        relations: list[dict[str, Any]],
        evidence_rows: list[dict[str, Any]],
        input_checksum: str,
    ) -> Any:
        """写入图谱侧确认的去重关系证据组与输入清单。

        图谱读取的是一份完整且自洽的关系输入：清单记录输入校验值、输入条数
        与证据组数；证据组逐组列出候选与胜出者。任何一次重建都可由当时的
        完整关系产物重算出同样的首个生效结果。
        """
        evidence_dir = self._data_dir("relation_evidence")
        evidence_path = evidence_dir / f"{relation_set_id}_graph_relation_evidence.jsonl"
        self._write_jsonl(evidence_path, evidence_rows)

        manifest = {
            "relation_set_id": relation_set_id,
            "input_checksum": input_checksum,
            "input_record_count": len(relations),
            "evidence_group_count": len(evidence_rows),
            "group_key": ["head", "relation", "tail"],
            "tie_break_rule": "同分保留输入序列先到者（严格大于才替换）",
            "evidence_file": str(evidence_path),
        }
        manifest_path = evidence_dir / f"{relation_set_id}_graph_relation_evidence_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2, default=str)
        return evidence_path

    def query_graph(self, entity: str) -> dict[str, Any]:
        """从图谱中查询与指定实体相关的三元组。"""
        graph_files = list(self._data_dir("graph").glob("*.jsonl"))
        results = []
        for gf in graph_files:
            triples = self._read_jsonl(gf)
            for t in triples:
                if entity in (t.get("head", ""), t.get("tail", "")):
                    results.append(t)
        return {"entity": entity, "related_triples": len(results), "triples": results}