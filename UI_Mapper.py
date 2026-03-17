from __future__ import annotations

import argparse
import json
import re
from typing import Any, Dict, List, Optional, TypedDict

from game_loop_planner import GameLoopPlanner
from llm_helper import LLMHelper


class StageUIConfig(TypedDict):
    step_no: int
    stage: str
    interaction_type: str
    ui_template: str
    controls: List[str]
    feedback_style: str
    difficulty: str
    rationale: str


class UIGamePlan(TypedDict):
    concept_id: str
    concept_name: str
    sequence_type: str
    ui_version: str
    stages: List[StageUIConfig]


class UIMapper:
    """
    Maps game-loop reasoning stages into UI interaction formats.
    Reasoning logic remains unchanged; only visual interaction format is selected.
    """

    def __init__(self) -> None:
        self.game_loop_planner = GameLoopPlanner()
        self.llm_helper = LLMHelper()

    @staticmethod
    def _build_schema_hint() -> Dict[str, Any]:
        return {
            "concept_id": "string",
            "concept_name": "string",
            "sequence_type": "mini_game_v1",
            "ui_version": "ui_map_v1",
            "stages": [
                {
                    "step_no": "int",
                    "stage": "activation|core_law_challenge|misconception_trigger|transfer_test",
                    "interaction_type": (
                        "link_drag|fill_in|multiple_choice|grouping|"
                        "sequencing|card_sort|short_text"
                    ),
                    "ui_template": "short template name",
                    "controls": ["list of controls/widgets"],
                    "feedback_style": "instant|after_submit|hinted",
                    "difficulty": "easy|medium|hard",
                    "rationale": "1 line why this interaction fits stage intent",
                }
            ],
        }

    @staticmethod
    def _extract_json_block(text: str) -> Dict[str, Any]:
        """
        Parse JSON output robustly even if model wraps it in markdown.
        """
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

    @staticmethod
    def _fallback_ui_mapping(loop: Dict[str, Any]) -> UIGamePlan:
        """
        Deterministic fallback if LLM output is invalid/unavailable.
        """
        mapping = {
            "activation": ("short_text", "quick_recall_card", ["text_input", "submit"], "instant", "easy"),
            "core_law_challenge": (
                "fill_in",
                "reasoning_steps_fill",
                ["scenario_panel", "blanks", "submit", "hint_button"],
                "after_submit",
                "medium",
            ),
            "misconception_trigger": (
                "multiple_choice",
                "assumption_check_mcq",
                ["claim_card", "option_buttons", "submit"],
                "after_submit",
                "medium",
            ),
            "transfer_test": (
                "grouping",
                "transfer_group_compare",
                ["card_pool", "drop_zones", "submit"],
                "hinted",
                "hard",
            ),
        }

        stages: List[StageUIConfig] = []
        for step in loop.get("steps", []):
            stage = str(step.get("stage", "activation"))
            it, tpl, controls, fb, diff = mapping.get(
                stage,
                ("short_text", "generic_response", ["text_input", "submit"], "after_submit", "medium"),
            )
            stages.append(
                StageUIConfig(
                    step_no=int(step.get("step_no", 0)),
                    stage=stage,
                    interaction_type=it,
                    ui_template=tpl,
                    controls=controls,
                    feedback_style=fb,
                    difficulty=diff,
                    rationale="Fallback mapping based on stage intent.",
                )
            )

        return UIGamePlan(
            concept_id=loop.get("concept_id", ""),
            concept_name=loop.get("concept_name", ""),
            sequence_type=loop.get("sequence_type", "mini_game_v1"),
            ui_version="ui_map_v1",
            stages=stages,
        )

    def _build_llm_prompt(self, game_loop_json: Dict[str, Any]) -> str:
        schema_hint = self._build_schema_hint()
        return (
            "You are a UI interaction mapper for learning games.\n"
            "Task: map each stage to the best UI interaction type.\n"
            "Important: DO NOT change learning logic, only choose visual interaction format.\n"
            "Heuristics:\n"
            "- Cause-effect => link_drag or sequencing\n"
            "- Missing reasoning steps => fill_in\n"
            "- Assumption testing => multiple_choice\n"
            "- Classification => grouping or card_sort\n"
            "Return ONLY valid JSON object. No prose.\n\n"
            f"Target schema:\n{json.dumps(schema_hint, indent=2)}\n\n"
            f"Game loop input:\n{json.dumps(game_loop_json, indent=2)}"
        )

    def map_ui_for_concept(
        self,
        concept_id: str,
        force_refresh: bool = False,
    ) -> UIGamePlan:
        game_loop = self.game_loop_planner.build_game_loop(concept_id)
        prompt = self._build_llm_prompt(game_loop)

        try:
            raw = self.llm_helper.invoke_for_concept(
                concept_id=concept_id,
                namespace="ui_mapper_v1",
                prompt=prompt,
                force_refresh=force_refresh,
            )
            parsed = self._extract_json_block(raw)
            #Minimal shape guard; fallback if malformed
            if not isinstance(parsed, dict) or "stages" not in parsed:
                return self._fallback_ui_mapping(game_loop)
            return UIGamePlan(
                concept_id=parsed.get("concept_id", game_loop["concept_id"]),
                concept_name=parsed.get("concept_name", game_loop["concept_name"]),
                sequence_type=parsed.get("sequence_type", game_loop["sequence_type"]),
                ui_version=parsed.get("ui_version", "ui_map_v1"),
                stages=parsed.get("stages", []),
            )
        except Exception:
            return self._fallback_ui_mapping(game_loop)


def main() -> None:
    parser = argparse.ArgumentParser(description="UI mapper for game-loop stages.")
    parser.add_argument("--concept-id", required=True, help="Concept id, e.g. c-43ab6fd0")
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Bypass concept cache and re-query the LLM.",
    )
    args = parser.parse_args()

    mapper = UIMapper()
    output = mapper.map_ui_for_concept(args.concept_id, force_refresh=args.force_refresh)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    test = UIMapper()
    result = test.map_ui_for_concept("c-9b3f8379")
    print(json.dumps(result, indent=2, ensure_ascii=False))
