"""关系证据组去重重构测试：共用口径、同分先到、异分取高、中途追加可重算。

覆盖：
- 共享模块 relation_evidence 的分组键、最高分判定与同分处理顺序；
- 关系抽取与图谱构建两个服务委托同一去重口径，结果与重构前一致；
- 图谱确认的一次输入来源完整一致（校验值随输入变化，不跨批拼组）。
"""
import json
import os
import sys

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND_DIR)

from app.services import relation_evidence
from app.services.kg_build import KgBuildService
from app.services.relation_extract import RelationExtractService


def _rel(rid, head, rtype, tail, conf, source):
    return {
        "relation_id": rid,
        "head_entity": head,
        "head_type": "ORG",
        "tail_entity": tail,
        "tail_type": "PERSON",
        "relation_type": rtype,
        "confidence": conf,
        "evidence_sentence": f"{source}:{head}-{rtype}-{tail}",
        "method": "Relation-BERT",
        "model_version": "v2.3.0",
        "extracted_at": 0,
        "source": source,
    }


def _triple(tid, head, relation, tail, conf):
    return {
        "triple_id": tid,
        "head": head,
        "relation": relation,
        "tail": tail,
        "confidence": conf,
        "source": "SET-1",
        "graph_version": "v1.0",
        "ingested_at": 0,
    }


class TestSharedEvidenceSemantics:
    """共享比较逻辑：分组键、最高分、同分先到。"""

    def test_tie_keeps_first_seen(self):
        records = [
            _rel("REL-000002", "甲", "EMPLOY", "乙", 0.88, "来源B"),
            _rel("REL-000001", "甲", "EMPLOY", "乙", 0.88, "来源A"),
        ]
        rows = relation_evidence.evidence_rows(
            records, relation_evidence.relation_group_key
        )
        assert len(rows) == 1
        group = rows[0]
        assert group["group_key"] == ["甲", "EMPLOY", "乙"]
        assert group["candidate_count"] == 2
        # 同分保留输入序列先到者，与编号大小无关
        assert group["winner_id"] == "REL-000002"
        assert group["winner_confidence"] == 0.88

    def test_higher_confidence_replaces(self):
        records = [
            _rel("REL-000001", "甲", "EMPLOY", "乙", 0.66, "来源A"),
            _rel("REL-000002", "甲", "EMPLOY", "乙", 0.93, "来源B"),
            _rel("REL-000003", "甲", "EMPLOY", "乙", 0.81, "来源C"),
        ]
        group = relation_evidence.evidence_rows(
            records, relation_evidence.relation_group_key
        )[0]
        assert group["winner_id"] == "REL-000002"
        assert [c["record_id"] for c in group["candidates"]] == [
            "REL-000001", "REL-000002", "REL-000003",
        ]

    def test_group_key_separates_relations(self):
        records = [
            _rel("REL-000001", "甲", "EMPLOY", "乙", 0.8, "A"),
            _rel("REL-000002", "甲", "LOCATE", "丙", 0.8, "A"),
            _rel("REL-000003", "丁", "EMPLOY", "乙", 0.8, "A"),
        ]
        rows = relation_evidence.evidence_rows(
            records, relation_evidence.relation_group_key
        )
        assert len(rows) == 3

    def test_checksum_deterministic_and_input_sensitive(self):
        records = [_rel("REL-000001", "甲", "EMPLOY", "乙", 0.8, "A")]
        c1 = relation_evidence.canonical_checksum(records)
        c2 = relation_evidence.canonical_checksum(list(reversed(records)) + records[:0])
        assert c1 == c2  # 同一输入任意时刻重算一致
        changed = [_rel("REL-000001", "甲", "EMPLOY", "乙", 0.91, "C-更正")]
        assert relation_evidence.canonical_checksum(changed) != c1

    def test_relation_and_triple_keys_are_isomorphic(self):
        rel = _rel("REL-1", "甲", "EMPLOY", "乙", 0.8, "A")
        tri = _triple("TRIPLE-REL-1", "甲", "EMPLOY", "乙", 0.8)
        assert relation_evidence.relation_group_key(rel) == (
            relation_evidence.triple_group_key(tri)
        )


class TestServicesDelegateSharedLogic:
    """两个服务保留原方法名，但比较口径统一委托共享模块。"""

    def test_relation_dedup_tie_order(self):
        svc = RelationExtractService()
        records = [
            _rel("REL-000001", "甲", "EMPLOY", "乙", 0.88, "A"),
            _rel("REL-000002", "甲", "EMPLOY", "乙", 0.88, "B"),
        ]
        out = svc._deduplicate_relations(records)
        assert [r["relation_id"] for r in out] == ["REL-000001"]

    def test_relation_dedup_sorted_by_relation_id(self):
        svc = RelationExtractService()
        records = [
            _rel("REL-000009", "丁", "EMPLOY", "戊", 0.7, "A"),
            _rel("REL-000001", "甲", "EMPLOY", "乙", 0.9, "A"),
        ]
        out = svc._deduplicate_relations(records)
        assert [r["relation_id"] for r in out] == ["REL-000001", "REL-000009"]

    def test_triple_dedup_matches_legacy_rule(self):
        svc = KgBuildService()
        triples = [
            _triple("TRIPLE-REL-2", "甲", "EMPLOY", "乙", 0.80),
            _triple("TRIPLE-REL-1", "甲", "EMPLOY", "乙", 0.80),
        ]
        out = svc._deduplicate_triples(triples)
        # 同分先到 + 按 triple_id 排序
        assert [t["triple_id"] for t in out] == ["TRIPLE-REL-2"]

    def test_legacy_data_quality_cases_still_hold(self):
        # 与 tests/test_pipeline.py 中既有用例同口径
        relations = [
            {"relation_id": "R1", "head_entity": "A", "relation_type": "X",
             "tail_entity": "B", "confidence": 0.8},
            {"relation_id": "R2", "head_entity": "A", "relation_type": "X",
             "tail_entity": "B", "confidence": 0.9},
        ]
        deduped = RelationExtractService()._deduplicate_relations(relations)
        assert len(deduped) == 1
        assert deduped[0]["confidence"] == 0.9


class TestSingleBatchInputAndAppend:
    """图谱一次输入来源完整一致；中途追加后可由完整输入重算。"""

    def test_graph_evidence_manifest_and_checkpoint(self, tmp_path, monkeypatch):
        import app.services.base as base

        monkeypatch.setattr(base, "DATA_ROOT", tmp_path)

        set_id = "SET-APPEND"
        rel_svc = RelationExtractService()
        kg_svc = KgBuildService()

        batch1 = [_rel("REL-000020", "戊", "EMPLOY", "王", 0.80, "来源A")]
        batch2 = batch1 + [_rel("REL-000022", "戊", "EMPLOY", "王", 0.91, "来源C-更正")]

        # 首批：完整输入建图，首个生效结果可从产物重算
        rel_file = tmp_path / "relations" / f"{set_id}_relations.jsonl"
        rel_file.parent.mkdir(parents=True, exist_ok=True)
        rel_file.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in batch1) + "\n",
            encoding="utf-8",
        )
        first = kg_svc.build_from_relations(set_id)
        first_manifest = json.loads(
            (tmp_path / "relation_evidence" /
             f"{set_id}_graph_relation_evidence_manifest.json").read_text(encoding="utf-8")
        )
        assert first["triples"] == 1
        assert first_manifest["input_record_count"] == 1
        first_checksum = first_manifest["input_checksum"]

        # 追加更正记录后，以完整输入重建，不拼两批比较结果
        rel_file.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in batch2) + "\n",
            encoding="utf-8",
        )
        second = kg_svc.build_from_relations(set_id)
        second_manifest = json.loads(
            (tmp_path / "relation_evidence" /
             f"{set_id}_graph_relation_evidence_manifest.json").read_text(encoding="utf-8")
        )
        assert second_manifest["input_record_count"] == 2
        assert second_manifest["input_checksum"] != first_checksum
        # 重算校验值与清单一致，证明输入依据可验证
        recomputed = relation_evidence.canonical_checksum(batch2)
        assert second_manifest["input_checksum"] == recomputed

        evidence = [
            json.loads(line)
            for line in (tmp_path / "relation_evidence" /
                         f"{set_id}_graph_relation_evidence.jsonl")
            .read_text(encoding="utf-8").splitlines()
        ]
        # 追加模式下文件含两轮证据，最后一轮的胜出者是更正记录
        last_group = evidence[-1]
        assert last_group["winner_id"] == "REL-000022"
        assert last_group["winner_confidence"] == 0.91

    def test_relation_evidence_written_from_raw_candidates(self, tmp_path, monkeypatch):
        import app.services.base as base

        monkeypatch.setattr(base, "DATA_ROOT", tmp_path)

        svc = RelationExtractService()
        candidates = [
            _rel("REL-000001", "甲", "EMPLOY", "乙", 0.88, "来源A"),
            _rel("REL-000002", "甲", "EMPLOY", "乙", 0.88, "来源B"),
        ]
        rows = svc.build_relation_evidence(candidates)
        checksum = relation_evidence.canonical_checksum(candidates)
        svc._write_relation_evidence("EID", rows, checksum)

        evidence_file = tmp_path / "relation_evidence" / "EID_relation_evidence.jsonl"
        manifest_file = tmp_path / "relation_evidence" / "EID_relation_evidence_manifest.json"
        assert evidence_file.exists() and manifest_file.exists()
        persisted = [json.loads(line) for line in evidence_file.read_text(
            encoding="utf-8").splitlines()]
        assert persisted[0]["candidate_count"] == 2  # 去重前候选完整保留
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        assert manifest["input_record_count"] == 2
        assert manifest["evidence_group_count"] == 1
        # 清单校验值必须能从原始关系输入独立重算出来
        assert manifest["input_checksum"] == checksum
        assert manifest["input_checksum"] == (
            relation_evidence.canonical_checksum(candidates)
        )
