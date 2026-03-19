from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

# Allow running from repo root or Game_Planner folder
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.append(str(HERE))

from Probe_Planner import ProbePlanner
from llm_helper import LLMHelper
from game_loop_planner import GameLoopPlanner


PROMPT1_MODEL = "qwen/qwen3-32b"
PROMPT2_MODEL = "llama-3.3-70b-versatile"


def _extract_json_block(text: str) -> dict:
    planner = GameLoopPlanner()
    return planner._extract_json_block(text)


def main() -> None:
    st.set_page_config(page_title="Model Comparison", layout="wide")
    st.title("Reasoning-Test (qwen/qwen3-32b+llama-3.3-70b-versatile)")

    with st.sidebar:
        st.markdown("### Input")
        concept_id = st.text_input("Concept ID", value="c-43ab6fd0")
        force_refresh = st.checkbox("Force refresh (bypass cache)", value=False)
        run_btn = st.button("Run Prompts")
        show_json = st.button("Show JSON")

    planner = ProbePlanner()

    if run_btn:
        concept_bundle = planner.bundle(concept_id)
        concept_name = concept_bundle.get("concept_name", "")

        # Prompt-1 (use same prompt as game_loop_planner)
        glp = GameLoopPlanner()
        llm1 = LLMHelper(model_name=PROMPT1_MODEL)
        p1 = glp._prompt_reasoning(concept_bundle)
        raw1 = llm1.invoke_for_concept(
            concept_id=f"{concept_id}:{PROMPT1_MODEL}",
            namespace="prompt1_test",
            prompt=p1,
            force_refresh=force_refresh,
        )

        try:
            reasoning = _extract_json_block(raw1)
        except Exception:
            reasoning = {"raw_output": raw1}

        # Prompt-2 (use same prompt as game_loop_planner)
        llm2 = LLMHelper(model_name=PROMPT2_MODEL)
        p2 = glp._prompt_questions(reasoning)
        raw2 = llm2.invoke_for_concept(
            concept_id=f"{concept_id}:{PROMPT2_MODEL}",
            namespace="prompt2_test",
            prompt=p2,
            force_refresh=force_refresh,
        )

        try:
            qout = _extract_json_block(raw2)
        except Exception:
            qout = {"raw_output": raw2}

        st.subheader(f"Questions + Answers ({concept_name})")
        steps = qout.get("steps", [])
        if isinstance(steps, list):
            for step in steps:
                st.markdown(f"**Type {step.get('step_no')}**")
                for qa in step.get("questions", []):
                    st.markdown(f"- Q: {qa.get('question', '')}")
                    st.markdown(f"  - A: {qa.get('answer', '')}")

        if show_json:
            st.markdown("### Prompt‑1 Output JSON")
            st.code(json.dumps(reasoning, indent=2, ensure_ascii=False), language="json")
            st.markdown("### Prompt‑2 Output JSON")
            st.code(json.dumps(qout, indent=2, ensure_ascii=False), language="json")


if __name__ == "__main__":
    main()
