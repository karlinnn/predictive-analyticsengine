import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Predictive Analytics Engine", layout="wide")

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Settings")
    execute_timeout = st.slider("Execution timeout (s)", min_value=30, max_value=600, value=180, step=10)

# ── Main ─────────────────────────────────────────────────────────────────────
st.title("Predictive Analytics Engine")
st.caption("Upload a dataset, ask a question, get a model-backed answer.")

col1, col2 = st.columns([1, 2])
with col1:
    uploaded_file = st.file_uploader("Upload dataset", type=["csv", "xlsx", "xls"])
with col2:
    question = st.text_area(
        "Ask a question about your data",
        placeholder="e.g. Which students will graduate next year?",
        height=100,
    )

run = st.button("Run Analysis", type="primary", disabled=not (uploaded_file and question.strip()))

if not run:
    st.stop()

# ── Call API ─────────────────────────────────────────────────────────────────
with st.spinner("Running agents — this may take a minute…"):
    try:
        resp = requests.post(
            f"{API_URL}/analyze",
            data={
                "question": question.strip(),
                "prefer_live_knowledge_base": "true",
                "execute_timeout": execute_timeout,
            },
            files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)},
            timeout=execute_timeout + 30,
        )
        payload = resp.json()
    except requests.Timeout:
        st.error("Request timed out. The pipeline is taking too long — try increasing the execution timeout in the sidebar.")
        st.stop()
    except requests.RequestException as exc:
        st.error(f"Could not reach API at {API_URL}. Is the server running?\n\n{exc}")
        st.stop()

if payload.get("status") == "error":
    st.error(f"Pipeline error: {payload.get('message', 'Unknown error')}")
    st.stop()

result = payload.get("result", {})

# ── Task Plan ─────────────────────────────────────────────────────────────────
plan = result.get("plan") or {}
if plan:
    st.subheader("Task Plan")
    c1, c2, c3 = st.columns(3)
    c1.metric("Task Type", plan.get("task_type", "—").title())
    c2.metric("Target Column", plan.get("target_column") or "—")
    top_model = (plan.get("selected_models") or [{}])[0].get("model_name", "—")
    c3.metric("Top Model", top_model)
    with st.expander("Reasoning"):
        st.write(plan.get("reasoning", "—"))

st.divider()

# ── Execution ─────────────────────────────────────────────────────────────────
execution = result.get("execution") or {}
if not execution.get("success"):
    st.error("Model execution failed.")
    stderr = execution.get("stderr", "")
    if stderr:
        with st.expander("Error details"):
            st.code(stderr)
    st.stop()

st.success("Model trained and executed successfully.")

result_json = execution.get("result_json") or {}

# ── Metrics ───────────────────────────────────────────────────────────────────
metrics = result_json.get("metrics") or {}
if metrics:
    st.subheader("Metrics")
    cols = st.columns(len(metrics))
    for col, (k, v) in zip(cols, metrics.items()):
        col.metric(k.replace("_", " ").title(), round(v, 4) if isinstance(v, float) else v)

# ── Feature Importance ────────────────────────────────────────────────────────
feature_importance = result_json.get("feature_importance") or {}
if feature_importance:
    st.subheader("Feature Importance")
    if isinstance(feature_importance, dict):
        rows = sorted(
            [{"Feature": k, "Importance": round(float(v), 4)} for k, v in feature_importance.items()],
            key=lambda x: x["Importance"],
            reverse=True,
        )
    elif isinstance(feature_importance, list):
        rows = sorted(
            [{"Feature": i.get("feature", ""), "Importance": round(float(i.get("importance", 0)), 4)} for i in feature_importance],
            key=lambda x: x["Importance"],
            reverse=True,
        )
    else:
        rows = []
    if rows:
        st.bar_chart({r["Feature"]: r["Importance"] for r in rows})
        with st.expander("Full table"):
            st.table(rows)

st.divider()

# ── Answer ────────────────────────────────────────────────────────────────────
interpretation = result.get("interpretation") or {}
if interpretation:
    st.subheader("Answer")

    model_summary = interpretation.get("model_summary")
    if model_summary:
        st.success(model_summary)

    performance = interpretation.get("performance_assessment")
    if performance:
        st.caption(performance)

    key_drivers = interpretation.get("key_drivers") or []
    if key_drivers:
        st.markdown("**Key Factors**")
        for item in key_drivers:
            st.markdown(f"- {item}")

    insights = interpretation.get("insights") or []
    if insights:
        st.markdown("**Insights**")
        for item in insights:
            st.markdown(f"- {item}")

    recommendations = interpretation.get("recommendations") or []
    if recommendations:
        st.markdown("**Recommendations**")
        for item in recommendations:
            st.markdown(f"- {item}")
else:
    st.warning("No answer available — model execution may have failed.")
