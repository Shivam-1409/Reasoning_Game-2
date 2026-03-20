from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

# Allow running from repo root or Game_Planner folder
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.append(str(HERE))

from game_loop_planner import GameLoopPlanner


def _render_matching(step_no: int, q_index: int, q: dict, correct: bool | None) -> None:
    _render_q_header(q_index, q.get("prompt", ""), correct)
    left = q.get("left") or []
    right = q.get("right") or []
    if not left or not right:
        st.info("Matching data missing.")
        return
    cols = st.columns(2)
    with cols[0]:
        st.markdown("Left items")
        st.markdown("\n".join([f"- {item}" for item in left]))
    with cols[1]:
        st.markdown("Right items")
        for l in left:
            st.selectbox(
                f"Link for {l}",
                right,
                key=f"m_{step_no}_{q_index}_{l}",
                label_visibility="collapsed",
            )


def _render_fill_in(step_no: int, q_index: int, q: dict, correct: bool | None) -> None:
    _render_q_header(q_index, q.get("prompt", ""), correct)
    template = q.get("template") or ""
    if template:
        st.markdown(f"`{template}`")
    blanks = q.get("blanks") or []
    if not blanks:
        st.text_input("Your answer", key=f"f_{step_no}_{q_index}")
        return
    for i in range(len(blanks)):
        st.text_input(f"Blank {i+1}", key=f"f_{step_no}_{q_index}_{i}")


def _render_mcq(step_no: int, q_index: int, q: dict, correct: bool | None) -> None:
    _render_q_header(q_index, q.get("prompt", ""), correct)
    options = q.get("options") or ["Option A", "Option B", "Option C"]
    st.radio(
        "Pick one",
        options,
        key=f"mc_{step_no}_{q_index}",
        label_visibility="collapsed",
    )


def _render_sorting(step_no: int, q_index: int, q: dict, correct: bool | None) -> None:
    _render_q_header(q_index, q.get("prompt", ""), correct)
    items = q.get("items") or []
    if not items:
        st.info("Sorting items missing.")
        return
    st.multiselect("Items (in correct order)", items, key=f"s_{step_no}_{q_index}")


def _render_ui_block(step_no: int, ui_type: str, questions: list, results: dict) -> None:
    for idx, q in enumerate(questions, start=1):
        correct = results.get((step_no, idx))
        if ui_type == "matching":
            _render_matching(step_no, idx, q, correct)
        elif ui_type == "fill_in":
            _render_fill_in(step_no, idx, q, correct)
        elif ui_type == "mcq":
            _render_mcq(step_no, idx, q, correct)
        elif ui_type == "sorting":
            _render_sorting(step_no, idx, q, correct)
        else:
            _render_q_header(idx, q.get("prompt", ""), correct)
            st.write(q)


def _render_q_header(q_index: int, prompt: str, correct: bool | None) -> None:
    if correct is True:
        st.markdown(f"<span style='color:#2ecc71'><b>Q{q_index}. {prompt}</b></span>", unsafe_allow_html=True)
    elif correct is False:
        st.markdown(f"<span style='color:#ff6b6b'><b>Q{q_index}. {prompt}</b></span>", unsafe_allow_html=True)
    else:
        st.markdown(f"**Q{q_index}.** {prompt}")


def _normalize(val: str) -> str:
    return str(val).strip().lower()


def _evaluate_answers(ui_steps: list) -> tuple[dict, int, int]:
    results: dict = {}
    correct_count = 0
    total_count = 0

    for ui in ui_steps:
        step_no = ui.get("step_no")
        ui_type = ui.get("ui_type")
        for idx, q in enumerate(ui.get("ui_questions", []), start=1):
            is_correct: bool | None = None

            if ui_type == "matching":
                answer_pairs = q.get("answer") or []
                if isinstance(answer_pairs, list) and answer_pairs:
                    total_count += 1
                    ok = True
                    for pair in answer_pairs:
                        left = pair.get("left")
                        right = pair.get("right")
                        sel = st.session_state.get(f"m_{step_no}_{idx}_{left}")
                        if _normalize(sel) != _normalize(right):
                            ok = False
                            break
                    is_correct = ok

            elif ui_type == "fill_in":
                answers = q.get("answer") or q.get("blanks") or []
                if isinstance(answers, list) and answers:
                    total_count += 1
                    ok = True
                    for i, ans in enumerate(answers):
                        user = st.session_state.get(f"f_{step_no}_{idx}_{i}", "")
                        if _normalize(user) != _normalize(ans):
                            ok = False
                            break
                    is_correct = ok

            elif ui_type == "mcq":
                answer = q.get("answer")
                if answer is not None:
                    total_count += 1
                    user = st.session_state.get(f"mc_{step_no}_{idx}", "")
                    is_correct = _normalize(user) == _normalize(answer)

            elif ui_type == "sorting":
                answer = q.get("answer") or []
                if isinstance(answer, list) and answer:
                    total_count += 1
                    chosen = st.session_state.get(f"s_{step_no}_{idx}", [])
                    is_correct = [*_map_norm(chosen)] == [*_map_norm(answer)]

            results[(step_no, idx)] = is_correct
            if is_correct is True:
                correct_count += 1

    return results, correct_count, total_count


def _map_norm(items: list) -> list:
    return [_normalize(x) for x in items]


def main() -> None:
    st.set_page_config(page_title="Model Comparison", layout="wide")
    st.title("Reasoning-Test (qwen/qwen3-32b+llama-3.3-70b-versatile)")

    with st.sidebar:
        st.markdown("### Input")
        concept_id = st.text_input("Concept ID", value="c-43ab6fd0")
        force_refresh = st.checkbox("Force refresh (bypass cache)", value=False)
        run_btn = st.button("Run Prompts")
        show_json = st.checkbox("Show JSON", value=False)

    if "ui_output" not in st.session_state:
        st.session_state["ui_output"] = None
        st.session_state["concept_name"] = ""
        st.session_state["check_results"] = {}
        st.session_state["score"] = None

    if run_btn:
        glp = GameLoopPlanner()
        try:
            output = glp.build_game_loop(concept_id, force_refresh=force_refresh)
            st.session_state["ui_output"] = output
            st.session_state["concept_name"] = output.get("concept_name", "")
        except Exception as e:
            st.error(f"Failed to build output: {e}")
            return

    output = st.session_state.get("ui_output")
    if not output:
        st.info("Click 'Run Prompts' to generate the final gamified output.")
        return

    concept_name = st.session_state.get("concept_name", "")
    ui_steps = output.get("ui_steps", [])

    if st.button("Check Answers"):
        results, correct_count, total_count = _evaluate_answers(ui_steps)
        st.session_state["check_results"] = results
        st.session_state["score"] = (correct_count, total_count)

    score = st.session_state.get("score")
    if score and score[1] > 0:
        st.success(f"Correct: {score[0]} / {score[1]}")

    st.subheader(f"Final Gamified Output ({concept_name})")
    if isinstance(ui_steps, list):
        for ui in ui_steps:
            step_no = ui.get("step_no", "")
            ui_type = ui.get("ui_type", "")
            st.markdown(f"**Type {step_no} — {ui_type}**")
            _render_ui_block(step_no, ui_type, ui.get("ui_questions", []), st.session_state.get("check_results", {}))

    if show_json:
        st.markdown("### UI Output JSON")
        st.code(json.dumps(ui_steps, indent=2, ensure_ascii=False), language="json")


if __name__ == "__main__":
    main()
