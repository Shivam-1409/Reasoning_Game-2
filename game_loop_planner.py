from __future__ import annotations

import argparse
import json
import re
from typing import Any, Dict, List, TypedDict

from Probe_Planner import ProbePlanner
from llm_helper import LLMHelper


class GameStep(TypedDict):
    step_no: int
    questions: List[Dict[str, str]]


class GameLoopPlan(TypedDict):
    concept_name: str
    steps: List[GameStep]


class GameLoopPlanner:
    """
    Two-step LLM planner:
    1) Extract structured reasoning JSON
    2) Generate 4-step question blueprints JSON
    """

    def __init__(
        self,
        concept_map_path: str = "o05_concept_map.json",
        mental_models_path: str = "o06_mental_models.json",
    ) -> None:
        self.probe_planner = ProbePlanner(
            concept_map_path=concept_map_path,
            mental_models_path=mental_models_path,
        )
        self.llm_helper = LLMHelper()

    @staticmethod
    def _extract_json_block(text: str) -> Dict[str, Any]:
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
        if fenced:
            return json.loads(fenced.group(1))

        raw_obj = re.search(r"(\{.*\})", text, re.DOTALL)
        if raw_obj:
            return json.loads(raw_obj.group(1))

        raise ValueError("LLM output did not contain valid JSON object.")

    def _prompt_reasoning(self, concept_bundle: Dict[str, Any]) -> str:
        schema = {
            "concept_name": "string",
            "CORE_LAWS": ["max 3"],
            "FAILURE_SCENARIOS": ["3-5 real-world break cases"],
            "MISCONCEPTION_TRAPS": ["3 decision-framed wrong beliefs"],
            "CAUSAL_CHAINS": [["cause -> effect steps"]],
            "ANALOGY": "single simple analogy",
        }
        return (
            "Extract structured reasoning components.\n\n"
            "Return ONLY the following sections as JSON with exact keys:\n"
            "1. CORE_LAWS (max 3)\n"
            "2. FAILURE_SCENARIOS (3–5 real-world situations where concept breaks)\n"
            "3. MISCONCEPTION_TRAPS (3 common wrong beliefs framed as decisions)\n"
            "4. CAUSAL_CHAINS (step-by-step cause → effect flows)\n"
            "5. ANALOGY (1 simple analogy)\n\n"
            "Be concise. No explanations.\n\n"
            f"Concept bundle:\n{json.dumps(concept_bundle, indent=2)}\n\n"
            f"Output JSON schema (exact):\n{json.dumps(schema, indent=2)}"
        )

    def _prompt_questions(self, reasoning_json: Dict[str, Any]) -> str:
        schema = {
            "concept_name": "string",
            "steps": [
                {
                    "step_no": 1,
                    "questions": [
                        {
                            "question": "high-reasoning question",
                            "answer": "short correct answer",
                        }
                    ],
                }
            ],
        }
        return (
            "Create 4-step questions from the reasoning JSON below.\n"
            "Return ONLY JSON. No extra text.\n\n"
            "Rules:\n"
            "- Create EXACTLY 4 steps.\n"
            "- Each step_no must map to:\n"
            "  1 -> activation\n"
            "  2 -> core_law_challenge\n"
            "  3 -> misconception_trigger\n"
            "  4 -> transfer_test\n\n"
            "Stage context (guidance only):\n"
            "1) activation — easy — Warm up and activate prior understanding.\n"
            "2) core_law_challenge — medium — Apply the governing law in a scenario.\n"
            "3) misconception_trigger — medium — Expose a tempting wrong idea.\n"
            "4) transfer_test — hard — Apply the idea in a new context.\n\n"
            "- Each step MUST:\n"
            "  - include 2 to 3 high-reasoning questions\n"
            "  - use a DIFFERENT scenario across steps\n"
            "  - target a DIFFERENT law or misconception across steps\n"
            "  - include a REAL decision-making situation (not definitions)\n"
            "  - include a psychologically plausible trap (not obvious)\n"
            "  - provide a short, correct answer for each question\n\n"
            f"Reasoning JSON:\n{json.dumps(reasoning_json, indent=2)}\n\n"
            f"Output JSON schema (exact):\n{json.dumps(schema, indent=2)}"
        )

    def build_game_loop(self, concept_id: str, force_refresh: bool = False) -> GameLoopPlan:
        concept_bundle = self.probe_planner.bundle(concept_id)
        prompt_1 = self._prompt_reasoning(concept_bundle)
        raw_1 = self.llm_helper.invoke_for_concept(
            concept_id=concept_id,
            namespace="reasoning_v1",
            prompt=prompt_1,
            force_refresh=force_refresh,
        )
        reasoning = self._extract_json_block(raw_1)

        if not isinstance(reasoning, dict) or "CORE_LAWS" not in reasoning:
            raise ValueError("Reasoning output missing required fields.")

        prompt_2 = self._prompt_questions(reasoning)
        raw_2 = self.llm_helper.invoke_for_concept(
            concept_id=concept_id,
            namespace="questions_v1",
            prompt=prompt_2,
            force_refresh=force_refresh,
        )
        steps = self._extract_json_block(raw_2)

        if not isinstance(steps, dict) or "steps" not in steps:
            raise ValueError("Question output missing required 'steps' field.")

        return GameLoopPlan(
            concept_name=steps.get("concept_name", concept_bundle.get("concept_name", "")),
            steps=steps.get("steps", []),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM game loop planner.")
    parser.add_argument("--concept-id", required=True, help="Concept id, e.g. c-43ab6fd0")
    parser.add_argument("--concept-map", default="o05_concept_map.json")
    parser.add_argument("--mental-models", default="o06_mental_models.json")
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Bypass concept cache and re-query the LLM.",
    )
    args = parser.parse_args()

    planner = GameLoopPlanner(
        concept_map_path=args.concept_map,
        mental_models_path=args.mental_models,
    )
    output = planner.build_game_loop(args.concept_id, force_refresh=args.force_refresh)
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
   main()
