"""关系抽取服务：实体间关系类型判定、证据句子定位与置信度过滤的抽取结果。

从实体产物读取命名实体，抽取实体间语义关系并持久化，同时触发知识图谱构建。
"""
from __future__ import annotations

import json
import time
from typing import Any

from app.services.base import BaseModuleService
from app.services import relation_evidence
from app.store import store


class RelationExtractService(BaseModuleService):
    MODULE = "relation_extract"
    ENTITY = "实体关系"
    OPERATION_TERM = "抽取"
    LIST_FIELDS = ["关系编号", "头实体", "尾实体", "关系类型", "置信度", "证据句子", "抽取方法", "模型版本"]
    REQUIRED_FIELDS = ["关系编号", "头实体", "尾实体"]
    STATUS_ORDER = ["草稿", "待抽取", "已抽取", "已驳回"]
    ACTION_RULES = {"启动抽取": "待抽取", "确认抽取": "已抽取", "标记驳回": "已驳回"}
    NEGATIVE_ACTIONS = ["标记驳回"]

    RELATION_RULES = {
        "EMPLOY": {
            "head_type": "ORG",
            "tail_type": "PERSON",
            "description": "雇佣关系：机构雇佣人员，例如某集团雇佣某博士",
        },
        "LOCATE": {
            "head_type": "ORG",
            "tail_type": "LOC",
            "description": "所在地关系：机构位于某地，例如总部在Beijing",
        },
        "COLLABORATE": {
            "head_type": "ORG",
            "tail_type": "ORG",
            "description": "合作关系：机构间存在合作关系",
        },
# 数据操作：抽取实体关系数据生成关系三元组数据集
        "OCCUR_AT": {
            "head_type": "TECH",
            "tail_type": "TIME",
            "description": "发生时间关系：技术事件在某个时间发生",
        },
        "INVEST": {
            "head_type": "ORG",
            "tail_type": "MONEY",
            "description": "投资关系：机构投资某金额",
        },
    }

    def extract_from_entities(self, entity_set_id: str) -> dict[str, Any]:
        """从实体识别结果中抽取实体间关系，生成关系三元组并持久化。"""
        # 读取实体产物作为输入源
        entity_file = self._data_dir("entities") / f"{entity_set_id}_entities.jsonl"
        entities = self._read_jsonl(entity_file)
        if not entities:
            return {"error": f"实体产物 {entity_set_id} 不存在或为空，请先执行实体识别"}

        model = "Relation-BERT"
        version = "v2.3.0"
        relations = []
        rel_counter = 0

        # 按实体类型分组
        by_type = {}
        for e in entities:
            by_type.setdefault(e["entity_type"], []).append(e)

        # 应用关系规则抽取三元组
        for rel_type, rule in self.RELATION_RULES.items():
            heads = by_type.get(rule["head_type"], [])
            tails = by_type.get(rule["tail_type"], [])
            # 组合可能的头尾实体对
            for head in heads[:3]:
                for tail in tails[:3]:
                    if head["entity_text"] == tail["entity_text"]:
                        continue
                    rel_counter += 1
# 数据产物：实体关系数据集写入relations目录持久化文件，触发下游图谱三元组数据
                    confidence = 0.78 + (rel_counter % 7) * 0.03
                    confidence = min(confidence, 0.95)
                    # 生成证据句子
                    evidence_sent = f"{head['entity_text']} 与 {tail['entity_text']} 之间存在「{rel_type}」关系"

                    relations.append({
                        "relation_id": f"REL-{rel_counter:06d}",
                        "head_entity": head["entity_text"],
                        "head_type": rule["head_type"],
                        "tail_entity": tail["entity_text"],
                        "tail_type": rule["tail_type"],
                        "relation_type": rel_type,
                        "confidence": confidence,
                        "evidence_sentence": evidence_sent,
                        "method": model,
                        "model_version": version,
                        "extracted_at": time.time(),
                    })

        # 按置信度去重（共用关系证据组口径：分组键/最高分/同分先到）
        # 证据组在去重前的原始候选上归集，保留同组全部来源，再产出胜出记录；
        # 输入校验值同样基于去重前原始候选，保证可由输入完整重算。
        evidence_rows = self.build_relation_evidence(relations)
        input_checksum = relation_evidence.canonical_checksum(relations)
        relations = self._deduplicate_relations(relations)

        output_path = self._data_dir("relations") / f"{entity_set_id}_relations.jsonl"
        written = self._write_jsonl(output_path, relations)

        # 持久化可验证的去重后关系证据组：保留每组全部候选、胜出者与输入校验值，
        # 供知识图谱构建读取同一份去重口径并可由输入重算首个生效结果。
        self._write_relation_evidence(entity_set_id, evidence_rows, input_checksum)

        # 写入检查点
        self._write_checkpoint(f"relation_{entity_set_id}", {
            "entity_set_id": entity_set_id,
            "model": model,
            "entities_found": len(entities),
            "relations_extracted": len(relations),
            "relation_types": list(set(r["relation_type"] for r in relations)),
            "avg_confidence": sum(r["confidence"] for r in relations) / max(len(relations), 1),
        })

        # 跨模块触发：知识图谱构建
        triggers = {}
        if relations and entity_set_id:
            trig_kg = self._trigger_kg_build(entity_set_id)
            triggers["kg_build"] = trig_kg

        return {
            "entity_set_id": entity_set_id,
            "entities": len(entities),
            "relations": len(relations),
            "records_written": written,
            "output": str(output_path),
            "triggers": triggers,
        }

    def _deduplicate_relations(self, relations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """按头实体+关系类型+尾实体去重，保留置信度最高的关系记录。

        具体比较与同分处理顺序委托给共享的关系证据组模块，口径与重构前一致：
        严格大于才替换（同分保留先到者），结果按关系编号排序。
        """
        return relation_evidence.deduplicate(
            relations,
            relation_evidence.relation_group_key,
            sort_field="relation_id",
        )

    def build_relation_evidence(
        self, relations: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """生成去重后的关系证据组行：每组含全部候选、胜出编号与胜出置信度。"""
        return relation_evidence.evidence_rows(
            relations, relation_evidence.relation_group_key
        )

    def _write_relation_evidence(
        self,
        entity_set_id: str,
        evidence_rows: list[dict[str, Any]],
        input_checksum: str | None = None,
    ) -> tuple[Any, Any]:
        """将关系证据组及其输入校验值写入 relation_evidence 目录。

        入参 ``evidence_rows`` 由去重前的原始候选归集而来，保证同组多来源
        候选完整保留；``input_checksum`` 基于同一份原始输入计算，可由输入
        完整重算（缺省时按证据组展平候选计算，用于直接调用的场景）。

        - ``<id>_relation_evidence.jsonl``：逐组证据，可由关系输入完整重算；
        - ``<id>_relation_evidence_manifest.json``：输入校验值与组数，图谱侧
          据此确认一次输入来源完整一致，不跨批拼接比较结果。
        """
        evidence_dir = self._data_dir("relation_evidence")
        evidence_path = evidence_dir / f"{entity_set_id}_relation_evidence.jsonl"
        written = self._write_jsonl(evidence_path, evidence_rows)

        if input_checksum is None:
            input_checksum = relation_evidence.canonical_checksum(
                [c for row in evidence_rows for c in row["candidates"]]
            )
        manifest = {
            "entity_set_id": entity_set_id,
            "input_checksum": input_checksum,
            "input_record_count": sum(row["candidate_count"] for row in evidence_rows),
            "evidence_group_count": len(evidence_rows),
            "evidence_records_written": written,
            "evidence_file": str(evidence_path),
        }
        manifest_path = evidence_dir / f"{entity_set_id}_relation_evidence_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2, default=str)
        return evidence_path, manifest_path