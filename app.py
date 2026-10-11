"""Streamlit dashboard for the support ticket triage outputs."""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUTS_DIR = ROOT / "outputs"
LOG_PATH = ROOT / "logs" / "validation_failures.log"
RESULTS_PATH = OUTPUTS_DIR / "triage_results.json"
EVALUATION_PATH = OUTPUTS_DIR / "evaluation.json"
HISTORICAL_PATH = DATA_DIR / "historical_tickets.json"
DAY1_PATH = DATA_DIR / "day1_tickets.json"
HELDOUT_PATH = DATA_DIR / "heldout_tickets.json"

st.set_page_config(
    page_title="Support Ticket Triage | RAG Dashboard",
    page_icon="🎟️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
      :root { --ink:#edf2ff; --muted:#98a6c6; --line:rgba(159,178,219,.14); --panel:rgba(18,28,53,.78); --cyan:#66e3d3; --violet:#a78bfa; }
      html, body, [class*="css"] {font-family:'DM Sans','Segoe UI',sans-serif;}
      .stApp {background:radial-gradient(ellipse at 12% 0%, #1b2a50 0%, #0b1120 42%, #080d19 100%); color:var(--ink);}
      [data-testid="stHeader"] {background:rgba(8,13,25,.32);}
      .block-container {max-width:1500px; padding-top:1.15rem; padding-bottom:3rem;}
      h1,h2,h3,h4 {font-family:'Space Grotesk','Segoe UI',sans-serif; letter-spacing:-.035em; color:#f4f7ff;}
      h1 {font-size:2.65rem !important; line-height:1.06 !important;} h2 {font-size:1.8rem !important;} h3 {font-size:1.28rem !important;}
      p, label, [data-testid="stCaptionContainer"] {color:#b4c0da;}
      [data-testid="stSidebar"] {background:linear-gradient(180deg,#101a31 0%,#0b1222 100%); border-right:1px solid var(--line);}
      [data-testid="stSidebar"] > div:first-child {padding-top:1.4rem;}
      [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1 {font-size:1.28rem !important; letter-spacing:-.03em;}
      [data-testid="stRadio"] > label {font-size:.72rem; text-transform:uppercase; letter-spacing:.14em; color:#7f91b8 !important; font-weight:700;}
      [data-testid="stRadio"] [role="radiogroup"] {gap:.42rem;}
      [data-testid="stRadio"] [role="radiogroup"] label {border:1px solid transparent; border-radius:11px; padding:.48rem .62rem; transition:all .18s ease;}
      [data-testid="stRadio"] [role="radiogroup"] label:hover {background:rgba(132,157,212,.09); border-color:var(--line);}
      [data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) {background:linear-gradient(100deg,rgba(103,227,211,.13),rgba(167,139,250,.15)); border-color:rgba(116,220,220,.30);}
      [data-testid="stMetric"] {background:linear-gradient(145deg,rgba(28,43,75,.88),rgba(16,25,46,.88)); border:1px solid rgba(153,177,226,.16); padding:1.1rem 1.25rem; border-radius:16px; box-shadow:0 12px 35px rgba(0,0,0,.15); min-height:116px;}
      [data-testid="stMetricLabel"] {color:#9eafd0 !important; font-size:.78rem; text-transform:uppercase; letter-spacing:.09em; font-weight:700;}
      [data-testid="stMetricValue"] {color:#f4f7ff !important; font-family:'Space Grotesk',sans-serif; font-size:2rem;}
      [data-testid="stVerticalBlockBorderWrapper"] {border-color:var(--line) !important; background:rgba(17,27,50,.56); border-radius:18px;}
      [data-testid="stDataFrame"] {border:1px solid var(--line); border-radius:14px; overflow:hidden;}
      [data-testid="stExpander"] {background:rgba(18,29,54,.72); border:1px solid var(--line); border-radius:13px; margin:.45rem 0;}
      [data-testid="stExpander"] summary {font-weight:650;}
      [data-baseweb="select"] > div, [data-testid="stTextInput"] input {background:#121d35; color:var(--ink); border:1px solid rgba(159,178,219,.22); border-radius:10px;}
      [data-testid="stButton"] button {background:linear-gradient(110deg,#5bd6cb,#8b8cf6); color:#08121e; border:0; border-radius:10px; font-weight:750; transition:transform .15s ease, box-shadow .15s ease;}
      [data-testid="stButton"] button:hover {transform:translateY(-2px); box-shadow:0 8px 22px rgba(91,214,203,.22); color:#08121e;}
      [data-testid="stAlert"] {border-radius:13px;} hr {border-color:var(--line);}
      .hero {position:relative; overflow:hidden; padding:1.65rem 1.8rem; margin:.4rem 0 1.55rem; border-radius:22px; border:1px solid rgba(147,172,222,.19); background:linear-gradient(112deg,rgba(24,42,77,.96) 0%,rgba(21,32,62,.88) 55%,rgba(48,39,86,.82) 100%); box-shadow:0 22px 60px rgba(0,0,0,.23);}
      .hero:after {content:''; position:absolute; width:280px; height:280px; right:5%; top:-170px; border-radius:50%; background:rgba(102,227,211,.15); filter:blur(12px);}
      .hero-kicker {display:inline-block; color:#76e8dc; font-size:.7rem; letter-spacing:.18em; font-weight:800; text-transform:uppercase; margin-bottom:.7rem;}
      .hero-title {font-family:'Space Grotesk',sans-serif; color:#f5f7ff; font-size:2.3rem; line-height:1.05; letter-spacing:-.055em; font-weight:700;}
      .hero-copy {color:#b5c3df; font-size:.98rem; margin-top:.65rem; max-width:760px;}
      .section-eyebrow {font-size:.7rem; text-transform:uppercase; letter-spacing:.15em; font-weight:800; color:#69ded4; margin:1.1rem 0 .35rem;}
      .ticket-card {padding:1.1rem 1.25rem; border:1px solid var(--line); border-radius:15px; background:linear-gradient(145deg,rgba(22,35,64,.88),rgba(13,22,42,.9)); margin:.45rem 0;}
      .ticket-id {font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:1.1rem; color:#f4f7ff;}
      .muted {color:#93a3c5; font-size:.88rem;}
      .footer-note {text-align:center; color:#7182a7; font-size:.78rem; padding-top:.6rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


def load_json(path: Path, default: Any) -> Any:
    """Load JSON safely and show a useful dashboard error for malformed files."""
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        st.error(f"Could not read `{path.relative_to(ROOT)}`: {exc}")
        return default


def enum_text(value: Any) -> str:
    return str(getattr(value, "value", value) or "")


def load_ticket_inputs() -> Dict[str, Dict[str, Any]]:
    records: Dict[str, Dict[str, Any]] = {}
    for path in (DAY1_PATH, HELDOUT_PATH):
        for item in load_json(path, []):
            ticket_id = str(item.get("ticket_id") or item.get("id") or "")
            if ticket_id:
                records[ticket_id] = item
    return records


def read_latest_log_run() -> str:
    if not LOG_PATH.exists():
        return ""
    try:
        text = LOG_PATH.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    # Provider initialization marks the beginning of a batch run in this logger.
    marker = "Configured model providers:"
    position = text.rfind(marker)
    return text[position:] if position >= 0 else text


def build_health(results: List[Dict[str, Any]], log_text: str) -> Dict[str, Any]:
    providers: Dict[str, int] = {}
    for match in re.finditer(r"Model call succeeded via ([\w-]+)", log_text):
        name = match.group(1)
        providers[name] = providers.get(name, 0) + 1

    fallback_count = sum(
        float(item.get("confidence", 1.0) or 0.0) == 0.0
        and "fallback" in str(item.get("reasoning", "")).lower()
        or "human review required" in str(item.get("reasoning", "")).lower()
        for item in results
    )
    validation_failures = len(
        re.findall(r"Invalid response for ticket_id=", log_text)
    )
    repairs_succeeded = len(re.findall(r"Response repaired for ticket_id=", log_text))
    repair_failures = len(re.findall(r"Repair attempt \d+ failed", log_text))
    rate_limit_failures = len(
        re.findall(r"429|rate.?limit|RESOURCE_EXHAUSTED", log_text, re.IGNORECASE)
    )
    return {
        "providers": providers,
        "fallback_count": fallback_count,
        "validation_failures": validation_failures,
        "repairs_succeeded": repairs_succeeded,
        "repair_failures": repair_failures,
        "rate_limit_mentions": rate_limit_failures,
    }


results_raw = load_json(RESULTS_PATH, [])
evaluation = load_json(EVALUATION_PATH, {})
historical_raw = load_json(HISTORICAL_PATH, [])
inputs = load_ticket_inputs()

if not isinstance(results_raw, list):
    st.error("`outputs/triage_results.json` should contain a list of ticket results.")
    results_raw = []
if not isinstance(historical_raw, list):
    historical_raw = []

results = [item for item in results_raw if isinstance(item, dict)]
historical_by_id = {
    str(item.get("ticket_id") or item.get("id")): item
    for item in historical_raw
    if item.get("ticket_id") or item.get("id")
}
results_by_id = {
    str(item.get("ticket_id", "")): item
    for item in results
    if item.get("ticket_id")
}
log_text = read_latest_log_run()
health = build_health(results, log_text)

with st.sidebar:
    st.markdown("<div class='section-eyebrow'>Support Intelligence</div>", unsafe_allow_html=True)
    st.title("NEXUS · Triage")
    st.caption("Retrieval-augmented decision console")
    page = st.radio(
        "Dashboard",
        ["Overview", "Ticket explorer", "RAG evidence", "Evaluation", "System health"],
    )
    st.divider()
    st.caption("Data source: latest local batch outputs")
    st.caption("Synthetic assessment dataset")

st.markdown(
    """
    <div class="hero">
      <div class="hero-kicker">RAG · Decision Layer · Evaluation</div>
      <div class="hero-title">Support, made<br>actionable.</div>
      <div class="hero-copy">Explore how each ticket was classified, which resolved cases informed the decision, and how the system performed on held-out examples.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not results:
    st.warning("No ticket results found. Run `python main.py` to create the batch output.")
    st.stop()


def result_frame(items: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in items:
        rows.append(
            {
                "Ticket ID": item.get("ticket_id", ""),
                "Category": enum_text(item.get("category")),
                "Urgency": enum_text(item.get("urgency")),
                "Action": enum_text(item.get("action")),
                "Confidence": float(item.get("confidence", 0.0) or 0.0),
                "Reasoning": item.get("reasoning", ""),
                "Retrieved IDs": ", ".join(item.get("retrieved_ticket_ids", [])),
            }
        )
    return pd.DataFrame(rows)


frame = result_frame(results)

if page == "Overview":
    st.subheader("Batch overview")
    total = len(results)
    low_conf = int((frame["Confidence"] < 0.65).sum())
    fallback_count = health["fallback_count"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Tickets processed", total)
    m2.metric("Auto-resolved", int((frame["Action"] == "auto_resolve").sum()))
    m3.metric("Escalated", int((frame["Action"] == "escalate").sum()))
    m4.metric("Low confidence", low_conf)

    st.info("This dashboard displays synthetic assessment data. It does not run the LLM.")
    left, right = st.columns(2)
    with left:
        st.markdown("#### Decisions by category")
        if not frame.empty:
            st.bar_chart(frame["Category"].value_counts().sort_index())
        st.markdown("#### Actions")
        st.bar_chart(frame["Action"].value_counts().sort_index())
    with right:
        st.markdown("#### Urgency levels")
        st.bar_chart(frame["Urgency"].value_counts().sort_index())
        if evaluation.get("status") == "complete":
            acc = evaluation.get("accuracies", {})
            st.markdown("#### Held-out score snapshot")
            st.metric("Exact match", f"{100 * acc.get('overall_exact_match', 0):.1f}%")
            st.caption(
                f"Based on {evaluation.get('heldout_sample_size', 0)} held-out tickets."
            )
    st.markdown("#### Latest batch results")
    st.dataframe(frame, use_container_width=True, hide_index=True)

elif page == "Ticket explorer":
    st.subheader("Ticket explorer")
    c1, c2, c3 = st.columns(3)
    categories = ["All"] + sorted(frame["Category"].dropna().unique().tolist())
    urgencies = ["All"] + sorted(frame["Urgency"].dropna().unique().tolist())
    actions = ["All"] + sorted(frame["Action"].dropna().unique().tolist())
    selected_category = c1.selectbox("Category", categories)
    selected_urgency = c2.selectbox("Urgency", urgencies)
    selected_action = c3.selectbox("Action", actions)
    search_id = st.text_input("Search ticket ID", placeholder="e.g. HELD-003")

    filtered = frame.copy()
    if selected_category != "All":
        filtered = filtered[filtered["Category"] == selected_category]
    if selected_urgency != "All":
        filtered = filtered[filtered["Urgency"] == selected_urgency]
    if selected_action != "All":
        filtered = filtered[filtered["Action"] == selected_action]
    if search_id.strip():
        filtered = filtered[
            filtered["Ticket ID"].str.contains(search_id.strip(), case=False, na=False)
        ]

    st.dataframe(filtered, use_container_width=True, hide_index=True)
    if not filtered.empty:
        chosen_id = st.selectbox("Inspect ticket", filtered["Ticket ID"].tolist())
        prediction = results_by_id.get(chosen_id, {})
        input_ticket = inputs.get(chosen_id, {})
        st.markdown(f"### {chosen_id}")
        st.write(f"**Subject:** {input_ticket.get('subject', 'Not available')}")
        st.write(
            f"**Ticket text:** {input_ticket.get('body') or input_ticket.get('description') or 'Not available'}"
        )
        a, b, c = st.columns(3)
        a.metric("Category", enum_text(prediction.get("category")))
        b.metric("Urgency", enum_text(prediction.get("urgency")))
        c.metric("Action", enum_text(prediction.get("action")))
        st.write(f"**Confidence:** {float(prediction.get('confidence', 0.0) or 0.0):.2f}")
        st.write(f"**Reasoning:** {prediction.get('reasoning', '')}")
        st.write(
            "**Retrieved IDs:** "
            + (", ".join(prediction.get("retrieved_ticket_ids", [])) or "None")
        )

elif page == "RAG evidence":
    st.subheader("Retrieved historical evidence")
    st.caption("Matches appear in retrieval rank order. Resolutions are historical examples.")
    selected_id = st.selectbox("Incoming ticket", list(results_by_id.keys()))
    prediction = results_by_id[selected_id]
    source_ticket = inputs.get(selected_id, {})
    st.write(f"**Subject:** {source_ticket.get('subject', 'Not available')}")
    st.write(
        f"**Ticket text:** {source_ticket.get('body') or source_ticket.get('description') or 'Not available'}"
    )
    retrieved_ids = prediction.get("retrieved_ticket_ids", [])
    if not retrieved_ids:
        st.info("No historical tickets were retrieved for this result.")
    for rank, ticket_id in enumerate(retrieved_ids, start=1):
        historical = historical_by_id.get(str(ticket_id))
        with st.expander(f"Rank {rank} · {ticket_id}", expanded=(rank == 1)):
            if historical is None:
                st.warning("Historical record not found in historical_tickets.json.")
            else:
                st.write(f"**Subject:** {historical.get('subject', '')}")
                st.write(f"**Ticket:** {historical.get('body', historical.get('description', ''))}")
                st.write(f"**Category:** {enum_text(historical.get('category', 'N/A'))}")
                st.write(f"**Urgency:** {enum_text(historical.get('urgency', 'N/A'))}")
                st.write(f"**Resolution:** {historical.get('resolution', 'Not provided')}")

elif page == "Evaluation":
    st.subheader("Held-out evaluation")
    if evaluation.get("status") != "complete":
        st.warning("Evaluation is missing or incomplete. Accuracy metrics are not shown.")
        st.json(evaluation)
    else:
        acc = evaluation.get("accuracies", {})
        st.caption(
            f"Scores use {evaluation.get('heldout_sample_size', 0)} held-out tickets."
        )
        metrics = st.columns(4)
        metrics[0].metric("Category accuracy", f"{100 * acc.get('category_accuracy', 0):.1f}%")
        metrics[1].metric("Urgency accuracy", f"{100 * acc.get('urgency_accuracy', 0):.1f}%")
        metrics[2].metric("Action accuracy", f"{100 * acc.get('action_accuracy', 0):.1f}%")
        metrics[3].metric("Exact match", f"{100 * acc.get('overall_exact_match', 0):.1f}%")
        st.markdown("#### Category confusion matrix")
        matrix = evaluation.get("confusion_matrix", {})
        if matrix:
            matrix_df = pd.DataFrame.from_dict(matrix, orient="index").fillna(0).astype(int)
            matrix_df.index.name = "Ground truth"
            matrix_df.columns.name = "Predicted"
            st.dataframe(matrix_df, use_container_width=True)
            st.caption("Rows are gold categories; columns are predicted categories.")
        else:
            st.info("No confusion matrix is present in evaluation.json.")

elif page == "System health":
    st.subheader("Latest run health")
    st.caption("Log counts are based on the latest detected provider-initialization marker.")
    h1, h2, h3, h4 = st.columns(4)
    h1.metric("Safe fallbacks in results", health["fallback_count"])
    h2.metric("Invalid responses", health["validation_failures"])
    h3.metric("Repairs succeeded", health["repairs_succeeded"])
    h4.metric("Repair failures", health["repair_failures"])
    st.markdown("#### Successful model calls by provider")
    if health["providers"]:
        st.bar_chart(pd.Series(health["providers"], name="Successful calls"))
        st.dataframe(
            pd.DataFrame(
                [{"Provider": provider, "Successful calls": count}
                 for provider, count in health["providers"].items()]
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Provider-call entries were not found in the latest log section.")
    if health["rate_limit_mentions"]:
        st.warning(
            f"The latest log section contains {health['rate_limit_mentions']} rate-limit-related entries."
        )
    else:
        st.success("No rate-limit errors were detected in the latest log section.")
    st.caption("API keys and raw model responses are never displayed here.")

st.divider()
st.caption("Dashboard reads local JSON/log artifacts only; use `python main.py` to create a fresh batch.")


