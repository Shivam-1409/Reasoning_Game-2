from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

# Allow running from repo root or Game_Planner folder
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.append(str(HERE))

from game_loop_planner import GameLoopPlanner


st.set_page_config(
    page_title="Game Planner UI",
    page_icon="",
    layout="wide",
)


def _style() -> None:
    st.markdown(
        """
        <style>
        :root {
          --bg: #0f1115;
          --panel: #171a21;
          --panel-2: #1e2230;
          --text: #e8eaf0;
          --muted: #9aa3b2;
          --accent: #49d49d;
          --accent-2: #5aa9ff;
          --border: #2a2f3b;
        }
        .main { background: var(--bg); color: var(--text); }
        .block-container { padding-top: 1.5rem; }
        .card {
          background: var(--panel);
          border: 1px solid var(--border);
          border-radius: 16px;
          padding: 18px 20px;
          margin-bottom: 16px;
          box-shadow: 0 8px 30px rgba(0,0,0,0.25);
        }
        .chip {
          display: inline-block;
          padding: 6px 10px;
          border-radius: 999px;
          background: var(--panel-2);
          border: 1px solid var(--border);
          color: var(--muted);
          font-size: 12px;
          margin-right: 6px;
        }
        .title {
          font-size: 28px;
          font-weight: 700;
          letter-spacing: 0.3px;
        }
        .subtitle {
          color: var(--muted);
          margin-top: 6px;
          font-size: 14px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _header() -> None:
    st.markdown('<div class="title">Game Planner</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">LLM-driven 4-step learning sequence with dynamic UI types</div>',
        unsafe_allow_html=True,
    )


def _render_link_drag(questions: List[Dict[str, Any]]) -> None:
    for i, q in enumerate(questions, start=1):
        st.markdown(f"**Q{i}.** {q.get('prompt','')}")
        left = q.get("left") or []
        right = q.get("right") or []
        if not left or not right:
            continue
        cols = st.columns(2)
        with cols[0]:
            st.markdown("Left items")
            st.write(left)
        with cols[1]:
            st.markdown("Right items")
            for l in left:
                st.selectbox(
                    f"Link for {l}",
                    right,
                    key=f"link_{i}_{l}",
                    label_visibility="collapsed",
                )


def _render_sequencing(questions: List[Dict[str, Any]]) -> None:
    for i, q in enumerate(questions, start=1):
        st.markdown(f"**Q{i}.** {q.get('prompt','')}")
        steps = q.get("steps") or []
        if not steps:
            continue
        st.multiselect("Choose steps", steps, key=f"seq_steps_{i}")
        st.text_input("Your ordered steps (comma-separated)", key=f"seq_order_{i}")


def _render_fill_in(questions: List[Dict[str, Any]]) -> None:
    for i, q in enumerate(questions, start=1):
        st.markdown(f"**Q{i}.** {q.get('prompt','')}")
        template = q.get("template") or ""
        blanks = q.get("blanks") or []
        if template:
            st.markdown(f"`{template}`")
        if not blanks:
            st.text_input("Your answer", key=f"fill_{i}")
            continue
        for j, _cue in enumerate(blanks, start=1):
            st.text_input(f"Blank {j}", key=f"fill_{i}_{j}")


def _render_multiple_choice(questions: List[Dict[str, Any]]) -> None:
    for i, q in enumerate(questions, start=1):
        st.markdown(f"**Q{i}.** {q.get('prompt','')}")
        options = q.get("options") or []
        if not options:
            options = ["Yes", "No", "Not sure"]
        st.radio(
            "Pick one",
            options,
            key=f"mc_{i}",
            label_visibility="collapsed",
        )


def _render_grouping(questions: List[Dict[str, Any]]) -> None:
    for i, q in enumerate(questions, start=1):
        st.markdown(f"**Q{i}.** {q.get('prompt','')}")
        items = q.get("items") or []
        if not items:
            continue
        labels = q.get("group_labels") or []
        if not labels:
            answer = q.get("answer") or {}
            if isinstance(answer, dict):
                labels = list(answer.keys())
        left_label = labels[0] if len(labels) > 0 else "Group A"
        right_label = labels[1] if len(labels) > 1 else "Group B"
        col1, col2 = st.columns(2)
        with col1:
            st.multiselect(left_label, items, key=f"grp_a_{i}")
        with col2:
            st.multiselect(right_label, items, key=f"grp_b_{i}")


def _render_card_sort(questions: List[Dict[str, Any]]) -> None:
    for i, q in enumerate(questions, start=1):
        st.markdown(f"**Q{i}.** {q.get('prompt','')}")
        items = q.get("items") or []
        if not items:
            continue
        labels = q.get("group_labels") or []
        if not labels:
            answer = q.get("answer") or {}
            if isinstance(answer, dict):
                labels = list(answer.keys())
        left_label = labels[0] if len(labels) > 0 else "Relevant"
        right_label = labels[1] if len(labels) > 1 else "Not relevant"
        st.multiselect(left_label, items, key=f"card_rel_{i}")
        st.multiselect(right_label, items, key=f"card_not_{i}")


RENDERERS = {
    "link_drag": _render_link_drag,
    "sequencing": _render_sequencing,
    "fill_in": _render_fill_in,
    "multiple_choice": _render_multiple_choice,
    "grouping": _render_grouping,
    "card_sort": _render_card_sort,
}


def _render_step(step: Dict[str, Any]) -> None:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(
        f"<span class='chip'>Step {step.get('step_no')}</span>"
        f"<span class='chip'>{step.get('concept_classified_into')}</span>"
        f"<span class='chip'>{step.get('ui_type')}</span>",
        unsafe_allow_html=True,
    )
    questions = step.get("questions", [])
    ui_type = step.get("ui_type", "multiple_choice")
    renderer = RENDERERS.get(ui_type, _render_multiple_choice)
    renderer(questions)
    st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    _style()
    _header()

    with st.sidebar:
        st.markdown("### Controls")
        concept_id = st.text_input("Concept ID", value="c-43ab6fd0")
        force_refresh = st.checkbox("Force refresh (bypass cache)", value=False)
        show_json = st.checkbox("Show raw JSON", value=True)
        generate = st.button("Generate / Refresh")

    planner = GameLoopPlanner()
    if "game_data" not in st.session_state:
        st.session_state["game_data"] = None

    if generate:
        try:
            st.session_state["game_data"] = planner.build_game_loop(
                concept_id, force_refresh=force_refresh
            )
        except Exception as e:
            st.error(f"Failed to build game loop: {e}")
            return

    data = st.session_state.get("game_data")
    if not data:
        st.info("Click 'Generate / Refresh' to load the learning sequence.")
        return

    st.markdown("### Sequence")
    with st.form("game_form"):
        for step in data.get("steps", []):
            _render_step(step)

        submitted = st.form_submit_button("Submit All Answers")
        if submitted:
            answered = 0
            total = 0
            for k, v in st.session_state.items():
                if k.startswith(("mc_", "fill_", "seq_", "grp_", "card_", "link_")):
                    total += 1
                    if v:
                        answered += 1
            if total == 0:
                st.warning("No answers detected yet.")
            else:
                score = round((answered / total) * 100, 1)
                st.success(f"Completion score: {score}%")

    if show_json:
        st.markdown("### Raw Output")
        st.code(json.dumps(data, indent=2, ensure_ascii=False), language="json")


if __name__ == "__main__":
    main()
