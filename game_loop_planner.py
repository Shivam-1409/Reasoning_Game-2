from __future__ import annotations

import argparse
import json
import re
from typing import Any, Dict, List, TypedDict

from Probe_Planner import ProbePlanner
from llm_helper import LLMHelper


class GameStep(TypedDict):
    step_no: int
    concept_classified_into: str
    ui_type: str
    questions: List[Dict[str, Any]]


class GameLoopPlan(TypedDict):
    concept_name: str
    steps: List[GameStep]


class GameLoopPlanner:
    """
    LLM-driven 4-step learning sequence generator.
    Uses Probe_Planner bundle as context and enforces strict classification + UI rules.
    """

    STAGES = [
        {
            "step_no": 1,
            "stage": "activation",
            "objective": "Warm up and activate prior understanding with low friction.",
            "difficulty": "easy",
        },
        {
            "step_no": 2,
            "stage": "core_law_challenge",
            "objective": "Check whether the learner can apply the governing law.",
            "difficulty": "medium",
        },
        {
            "step_no": 3,
            "stage": "misconception_trigger",
            "objective": "Expose incorrect reasoning by presenting a tempting wrong idea.",
            "difficulty": "medium",
        },
        {
            "step_no": 4,
            "stage": "transfer_test",
            "objective": "Test transfer by applying the same idea in a nearby context.",
            "difficulty": "hard",
        },
    ]

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

    def _build_prompt(self, concept_bundle: Dict[str, Any]) -> str:
        schema = {
            "concept_name": "string",
            "steps": [
                {
                    "step_no": "1..4",
                    "concept_classified_into": (
                        "Cause-effect | Missing reasoning steps | Assumption testing | Classification"
                    ),
                    "ui_type": "link_drag | sequencing | fill_in | multiple_choice | grouping | card_sort",
                    "questions": [
                        {
                            "prompt": "string",
                            "ui_type": "must match step ui_type",
                            "options": "for multiple_choice (list of 3-5)",
                            "template": "for fill_in (sentence with ___ blanks)",
                            "blanks": "for fill_in (list of 1-3 correct fills)",
                            "steps": "for sequencing (list)",
                            "left": "for link_drag (list)",
                            "right": "for link_drag (list)",
                            "items": "for grouping/card_sort (list)",
                            "group_labels": "for grouping/card_sort (list of 2 labels)",
                            "answer": "optional answer key",
                        }
                    ],
                }
            ],
        }

        return (
            "You generate a 4-step learning sequence. Only output JSON. No prose.\n\n"
            "Fixed stage info (do NOT return in output):\n"
            "1) Stage 1: activation\n"
            "   Objective: Warm up and activate prior understanding with low friction.\n"
            "   Difficulty: easy\n"
            "2) Stage 2: core_law_challenge\n"
            "   Objective: Check whether the learner can apply the governing law.\n"
            "   Difficulty: medium\n"
            "3) Stage 3: misconception_trigger\n"
            "   Objective: Expose incorrect reasoning by presenting a tempting wrong idea.\n"
            "   Difficulty: medium\n"
            "4) Stage 4: transfer_test\n"
            "   Objective: Test transfer by applying the same idea in a nearby context.\n"
            "   Difficulty: hard\n\n"
            "Allowed classifications and UI type mapping:\n"
            '- "Cause-effect" => "link_drag" OR "sequencing"\n'
            '- "Missing reasoning steps" => "fill_in"\n'
            '- "Assumption testing" => "multiple_choice"\n'
            '- "Classification" => "grouping" OR "card_sort"\n'
            "No other values allowed.\n\n"
            "Use the concept bundle below to create logical, real-world questions (not theoretical).\n"
            "Each step should have 2 to 3 questions.\n"
            "Questions must be structured objects (not plain strings) and must match ui_type.\n"
            "Include options for multiple_choice, blanks for fill_in, steps for sequencing,\n"
            "left/right for link_drag, and items + group_labels for grouping/card_sort.\n"
            "For fill_in, include a full 'template' sentence with ___ placeholders\n"
            "and provide 'blanks' as the correct fill(s).\n"
            "If grouping/card_sort, group_labels must be two human-readable categories.\n\n"
            f"Concept bundle:\n{json.dumps(concept_bundle, indent=2)}\n\n"
            f"Output schema:\n{json.dumps(schema, indent=2)}"
        )

    def build_game_loop(self, concept_id: str, force_refresh: bool = False) -> GameLoopPlan:
        concept_bundle = self.probe_planner.bundle(concept_id)
        prompt = self._build_prompt(concept_bundle)

        raw = self.llm_helper.invoke_for_concept(
            concept_id=concept_id,
            namespace="game_loop_v2",
            prompt=prompt,
            force_refresh=force_refresh,
        )
        parsed = self._extract_json_block(raw)

        if not isinstance(parsed, dict) or "steps" not in parsed:
            raise ValueError("LLM output missing required 'steps' field.")

        return GameLoopPlan(
            concept_name=parsed.get("concept_name", concept_bundle.get("concept_name", "")),
            steps=parsed.get("steps", []),
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
