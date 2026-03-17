from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict


class RelatedConcept(TypedDict):
    related_concept: str
    reason: str


class ConceptBundle(TypedDict):
    concept_name: str
    all_laws: List[str]
    all_misconception: List[str]
    related_concept: List[RelatedConcept]


class ProbePlanner:
    """
    Simple concept bundle extractor.
    Given a concept_id, return:
    - laws (invariants)
    - misconceptions (forbidden_assumptions or pitfalls)
    - related concepts with relation tags from the concept map
    """

    def __init__(
        self,
        concept_map_path: str = "o05_concept_map.json",
        mental_models_path: str = "o06_mental_models.json",
        concepts_path: str = "o03_concepts.json",
    ) -> None:
        self.concept_map = self._load_json(concept_map_path)
        self.mental_models = self._load_json(mental_models_path)
        self.concepts = self._load_json(concepts_path)
        self.nodes_by_id = self._index_nodes()
        self.models_by_concept_id = self._index_models()
        self.concepts_by_id = self._index_concepts()

    @staticmethod
    def _load_json(path: str) -> Dict[str, Any]:
        p = Path(path)
        if p.is_absolute() and p.exists():
            resolved = p
        else:
            candidates = [Path.cwd() / p, Path(__file__).resolve().parent.parent / p]
            resolved = next((c for c in candidates if c.exists()), p)

        if not resolved.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

        with resolved.open("r", encoding="utf-8", errors="replace") as f:
            return json.load(f)

    def _index_nodes(self) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        nodes = self.concept_map.get("concept_map", {}).get("nodes", [])
        for n in nodes:
            cid = n.get("id")
            if cid:
                out[cid] = n
        return out

    def _index_models(self) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for m in self.mental_models.get("mental_models", []):
            cid = m.get("concept_id")
            if cid:
                out[cid] = m
        return out

    def _index_concepts(self) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for c in self.concepts.get("concepts", []):
            cid = c.get("concept_id")
            if cid:
                out[cid] = c
        return out

    def _concept_name(self, concept_id: str) -> str:
        if concept_id in self.nodes_by_id:
            return self.nodes_by_id[concept_id].get("label", concept_id)
        if concept_id in self.concepts_by_id:
            return self.concepts_by_id[concept_id].get("name", concept_id)
        model = self.models_by_concept_id.get(concept_id, {})
        return model.get("concept_name", concept_id)

    def _laws_for(self, model: Dict[str, Any]) -> List[str]:
        laws: List[str] = []
        for t in model.get("invariants", []) or []:
            if isinstance(t, str) and t.strip():
                laws.append(t.strip())
        return laws

    def _misconceptions_for(self, model: Dict[str, Any]) -> List[str]:
        misconceptions: List[str] = []
        for t in model.get("forbidden_assumptions", []) or []:
            if isinstance(t, str) and t.strip():
                misconceptions.append(t.strip())
        if not misconceptions:
            for t in model.get("pitfalls", []) or []:
                if isinstance(t, str) and t.strip():
                    misconceptions.append(t.strip())
        return misconceptions

    def _related_concepts(self, concept_id: str) -> List[RelatedConcept]:
        edges = self.concept_map.get("concept_map", {}).get("edges", [])
        related: List[RelatedConcept] = []
        for e in edges:
            src = e.get("from")
            dst = e.get("to")
            if src != concept_id and dst != concept_id:
                continue
            other = dst if src == concept_id else src
            if not other:
                continue
            related.append(
                RelatedConcept(
                    related_concept=self._concept_name(other),
                    reason=str(e.get("reason", "")),
                )
            )
        return related

    def bundle(self, concept_id: str) -> ConceptBundle:
        model = self.models_by_concept_id.get(concept_id)
        if not model:
            raise ValueError(f"Concept '{concept_id}' not found in o06_mental_models.json")

        laws = self._laws_for(model)
        misconceptions = self._misconceptions_for(model)
        related = self._related_concepts(concept_id)

        return ConceptBundle(
            concept_name=self._concept_name(concept_id),
            all_laws=laws,
            all_misconception=misconceptions,
            related_concept=related,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple Probe Planner (concept bundle).")
    parser.add_argument("--concept-id", required=True, help="Concept id, e.g. c-43ab6fd0")
    parser.add_argument("--concept-map", default="o05_concept_map.json")
    parser.add_argument("--mental-models", default="o06_mental_models.json")
    args = parser.parse_args()

    planner = ProbePlanner(
        concept_map_path=args.concept_map,
        mental_models_path=args.mental_models,
    )
    bundle = planner.bundle(args.concept_id)
    print(json.dumps(bundle, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
