from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, TypedDict

from Probe_Planner import ProbePlanner
from llm_helper import LLMHelper


class GameStep(TypedDict):
    step_no: int
    questions: List[Dict[str, str]]


class GameLoopPlan(TypedDict):
    concept_name: str
    steps: List[GameStep]
    ui_steps: List[Dict[str, Any]]


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
        ui_cache_path: str = "Game_Planner/ui_format_cache.json",
    ) -> None:
        self.probe_planner = ProbePlanner(
            concept_map_path=concept_map_path,
            mental_models_path=mental_models_path,
        )
        self.llm_helper = LLMHelper()
        self.ui_cache_path = self._resolve_path(ui_cache_path)
        self.ui_cache = self._load_ui_cache()

    @staticmethod
    def _resolve_path(path: str) -> Path:
        p = Path(path)
        if p.is_absolute():
            return p
        candidates = [Path.cwd() / p, Path(__file__).resolve().parent.parent / p]
        return next((c for c in candidates if c.exists()), candidates[0])

    def _load_ui_cache(self) -> Dict[str, Any]:
        if not self.ui_cache_path.exists():
            return {}
        try:
            with self.ui_cache_path.open("r", encoding="utf-8", errors="replace") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}

    def _save_ui_cache(self) -> None:
        self.ui_cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.ui_cache_path.open("w", encoding="utf-8") as f:
            json.dump(self.ui_cache, f, indent=2, ensure_ascii=False)

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

        # Fallback: extract first JSON object by brace matching
        obj = GameLoopPlanner._first_json_object(text)
        if obj:
            return json.loads(obj)

        raise ValueError("LLM output did not contain valid JSON object.")

    @staticmethod
    def _first_json_object(text: str) -> str | None:
        start = text.find("{")
        if start == -1:
            return None
        in_str = False
        escape = False
        depth = 0
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return text[start : i + 1]
        return None

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

    def _prompt_ui_activation(self, step_json: Dict[str, Any]) -> str:
        return (
            "You are a learning designer creating MATCHING questions.\n\n"
            "Goal:\n"
            "- Activate prior knowledge through subtle reasoning\n"
            "- Avoid obvious or direct mappings\n\n"
            "Your task:\n"
            "- Convert each question into a MATCHING format\n"
            "- Each pair must require thinking, not recall\n\n"
            "STRICT RULES:\n"
            "- Do NOT create one-to-one obvious mappings\n"
            "- Left and right items must be similar enough to create confusion\n"
            "- Include at least one “competing” option that could plausibly match\n"
            "- Avoid repeating wording from question in answers\n"
            "- Each match must require understanding of the concept, not keyword spotting\n\n"
            "ANTI-PATTERN (DO NOT DO):\n"
            "❌ “Transparency → Reviewing account” (too obvious)\n\n"
            "BETTER PATTERN:\n"
            "✔ Slightly overlapping meanings\n"
            "✔ Requires distinguishing between close concepts\n\n"
            f"INPUT:\n{json.dumps(step_json, indent=2)}\n\n"
            "OUTPUT:\n"
            "{\n"
            '  "step_no": number,\n'
            '  "ui_type": "matching",\n'
            '  "ui_questions": [\n'
            "    {\n"
            '      "prompt": "...",\n'
            '      "ui_type": "matching",\n'
            '      "left": [...],\n'
            '      "right": [...],\n'
            '      "answer": [\n'
            '        {"left": "...", "right": "..."}\n'
            "      ]\n"
            "    }\n"
            "  ]\n"
            "}\n"
        )

    def _prompt_ui_core(self, step_json: Dict[str, Any]) -> str:
        return (
            "You are a learning designer creating FILL-IN questions.\n\n"
            "Goal:\n"
            "- Test application of core law through reasoning chains\n"
            "- NOT simple recall or sentence completion\n\n"
            "Your task:\n"
            "- Convert each question into a fill-in format\n"
            "- Each blank must require reasoning, not memorization\n\n"
            "STRICT RULES:\n"
            "- DO NOT create definition-style blanks\n"
            "- DO NOT leave obvious keywords as blanks\n"
            "- Each question must involve a cause → effect → implication structure\n"
            "- Blanks must represent missing reasoning steps\n\n"
            "GOOD PATTERN:\n"
            "“If X is not done, then ___ leads to ___, which ultimately causes ___.”\n\n"
            "BAD PATTERN:\n"
            "❌ “The system requires ___” (recall)\n\n"
            "REQUIREMENTS:\n"
            "- 1–3 blanks max\n"
            "- Sentence must feel like a real-world scenario\n"
            "- Blanks must require understanding of WHY, not WHAT\n\n"
            f"INPUT:\n{json.dumps(step_json, indent=2)}\n\n"
            "OUTPUT:\n"
            "{\n"
            '  "step_no": number,\n'
            '  "ui_type": "fill_in",\n'
            '  "ui_questions": [\n'
            "    {\n"
            '      "prompt": "...",\n'
            '      "ui_type": "fill_in",\n'
            '      "template": "... ___ ...",\n'
            '      "blanks": [...],\n'
            '      "answer": [...]\n'
            "    }\n"
            "  ]\n"
            "}\n"
        )

    def _prompt_ui_misconception(self, step_json: Dict[str, Any]) -> str:
        return (
            "You are a learning designer creating MCQ questions.\n\n"
            "Goal:\n"
            "- Expose flawed reasoning using strong, believable distractors\n\n"
            "Your task:\n"
            "- Convert each question into MCQ format\n"
            "- Create options that are difficult to eliminate\n\n"
            "STRICT RULES:\n"
            "- Must include 4 options\n"
            "- EXACTLY ONE correct answer\n"
            "- All other options must be plausible (not obviously wrong)\n\n"
            "CRITICAL:\n"
            "- Distractors must reflect REAL misconceptions\n"
            "- At least 2 options should feel “almost correct”\n"
            "- Avoid extreme or silly wrong answers\n\n"
            "BAD OPTIONS:\n"
            "❌ “No impact at all” (too obvious)\n\n"
            "GOOD OPTIONS:\n"
            "✔ Partial truths\n"
            "✔ Misapplied logic\n"
            "✔ Overgeneralizations\n\n"
            "TARGET:\n"
            "- A smart learner should hesitate between 2 options\n\n"
            f"INPUT:\n{json.dumps(step_json, indent=2)}\n\n"
            "OUTPUT:\n"
            "{\n"
            '  "step_no": number,\n'
            '  "ui_type": "mcq",\n'
            '  "ui_questions": [\n'
            "    {\n"
            '      "prompt": "...",\n'
            '      "ui_type": "mcq",\n'
            '      "options": [...],\n'
            '      "answer": "correct option"\n'
            "    }\n"
            "  ]\n"
            "}\n"
        )

    def _prompt_ui_transfer(self, step_json: Dict[str, Any]) -> str:
        return (
            "You are a learning designer creating SORTING questions.\n\n"
            "Goal:\n"
            "- Test transfer of knowledge through multi-step reasoning\n"
            "- NOT simple chronological ordering\n\n"
            "Your task:\n"
            "- Convert each question into sorting format\n"
            "- Order must depend on logical dependencies, not surface sequence\n\n"
            "STRICT RULES:\n"
            "- Items must form a cause → effect chain OR dependency structure\n"
            "- Order should NOT be obvious\n"
            "- Each step should influence the next\n\n"
            "BAD PATTERN:\n"
            "❌ Generic sequence (setup → record → monitor)\n\n"
            "GOOD PATTERN:\n"
            "✔ Missing step breaks entire chain\n"
            "✔ Order requires understanding consequences\n\n"
            "REQUIREMENTS:\n"
            "- 3–5 items\n"
            "- Must require reasoning to order correctly\n\n"
            f"INPUT:\n{json.dumps(step_json, indent=2)}\n\n"
            "OUTPUT:\n"
            "{\n"
            '  "step_no": number,\n'
            '  "ui_type": "sorting",\n'
            '  "ui_questions": [\n'
            "    {\n"
            '      "prompt": "...",\n'
            '      "ui_type": "sorting",\n'
            '      "items": [...],\n'
            '      "answer": [...]\n'
            "    }\n"
            "  ]\n"
            "}\n"
        )

    def _format_ui_for_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        step_no = step.get("step_no")
        if step_no == 1:
            prompt = self._prompt_ui_activation(step)
        elif step_no == 2:
            prompt = self._prompt_ui_core(step)
        elif step_no == 3:
            prompt = self._prompt_ui_misconception(step)
        elif step_no == 4:
            prompt = self._prompt_ui_transfer(step)
        else:
            raise ValueError(f"Unsupported step_no: {step_no}")

        ui_llm = LLMHelper(model_name="llama-3.3-70b-versatile")
        raw = ui_llm.invoke_for_concept(
            concept_id=f"{step_no}",
            namespace="ui_format_v1",
            prompt=prompt,
            force_refresh=True,
        )
        return self._extract_json_block(raw)

    def build_game_loop(self, concept_id: str, force_refresh: bool = False) -> GameLoopPlan:
        concept_bundle = self.probe_planner.bundle(concept_id)

        # UI cache short-circuit
        if not force_refresh:
            cached = self.ui_cache.get(concept_id)
            if cached and "ui_steps" in cached and "steps" in cached:
                return GameLoopPlan(
                    concept_name=cached.get("concept_name", concept_bundle.get("concept_name", "")),
                    steps=cached.get("steps", []),
                    ui_steps=cached.get("ui_steps", []),
                )

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

        base_steps = steps.get("steps", [])
        ui_steps: List[Dict[str, Any]] = []
        for s in base_steps:
            ui_steps.append(self._format_ui_for_step(s))

        output = GameLoopPlan(
            concept_name=steps.get("concept_name", concept_bundle.get("concept_name", "")),
            steps=base_steps,
            ui_steps=ui_steps,
        )

        # Cache full output per concept_id
        self.ui_cache[concept_id] = {
            "concept_name": output["concept_name"],
            "steps": output["steps"],
            "ui_steps": output["ui_steps"],
            "cached_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        self._save_ui_cache()

        return output


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
