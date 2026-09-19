"""
Solar Panel Fault & Soiling Detection - Streamlit dashboard
============================================================
Upload panel photos (one or many) and get, per image:
    * predicted condition (6 classes) with severity and a recommended action
    * full probability breakdown and a low-confidence / close-call flag
    * Grad-CAM heat-map showing where the model looked
Plus tabs for model performance, the backbone comparison and dataset statistics.

Run from the project root:
    streamlit run app.py
"""

import io
import json
import random
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from src.config import (
    CLASS_INFO, CLASS_NAMES, CLOSE_CALL_MARGIN, COMPARISON_PATH, DATA_DIR, DATASET_REPORT_PATH,
    FINAL_MODEL_PATH, LOW_CONFIDENCE_THRESHOLD, METRICS_PATH, OUTPUT_DIR,
)
from src.inference import Predictor, load_image

# --------------------------------------------------------------------------- #
# Page setup & palette
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="Solar Panel Fault & Soiling Detection", page_icon="☀️", layout="wide")

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]          # categorical slots 1-3
BAR_TOP, BAR_REST = "#2a78d6", "#9ec5f4"             # single-hue: top class dark, rest light
INK_2, MUTED = "#52514e", "#898781"
SEVERITY_STYLE = {                                    # status palette: icon + label, never colour alone
    "none":     ("✅", "#0ca30c", "No action"),
    "low":      ("🟦", "#2a78d6", "Low"),
    "moderate": ("⚠️", "#fab219", "Moderate"),
    "high":     ("🔶", "#ec835a", "High"),
    "critical": ("🚨", "#d03b3b", "Critical"),
}
LABEL_OF = {c: CLASS_INFO[c]["label"] for c in CLASS_NAMES}


@st.cache_resource(show_spinner="Loading model…")
def get_predictor(_model_mtime: float) -> Predictor:
    """Cached per model file: retraining (new mtime) reloads without restarting the app."""
    return Predictor(FINAL_MODEL_PATH)


def predictor_cached() -> Predictor:
    return get_predictor(FINAL_MODEL_PATH.stat().st_mtime)


@st.cache_data
def read_json(path: str):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def severity_badge(severity: str) -> str:
    icon, color, text = SEVERITY_STYLE[severity]
    return (f'<span style="background:{color}1a;color:{color};border:1px solid {color};'
            f'border-radius:6px;padding:2px 8px;font-weight:600;font-size:0.85rem">{icon} {text}</span>')


def prob_chart(ranked, top_class):
    df = pd.DataFrame(ranked, columns=["class", "prob"])
    df["label"] = df["class"].map(LABEL_OF)
    df["is_top"] = df["class"] == top_class
    df["text"] = df["prob"].map(lambda p: f"{p:.1%}")
    base = alt.Chart(df).encode(
        y=alt.Y("label:N", sort=None, title=None, axis=alt.Axis(labelColor=INK_2, ticks=False, domain=False)),
        x=alt.X("prob:Q", title=None, scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format="%", grid=True, gridColor="#e1e0d9", labelColor=MUTED, domain=False, tickCount=5)),
        tooltip=[alt.Tooltip("label:N", title="Class"), alt.Tooltip("prob:Q", title="Probability", format=".1%")],
    )
    bars = base.mark_bar(size=18, cornerRadiusEnd=4).encode(
        color=alt.condition(alt.datum.is_top, alt.value(BAR_TOP), alt.value(BAR_REST)))
    text = base.mark_text(align="left", dx=4, color=INK_2, fontSize=11).encode(text="text:N")
    return (bars + text).properties(height=6 * 28, padding={"left": 0, "right": 30}).configure_view(strokeWidth=0)


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
model_available = FINAL_MODEL_PATH.exists()
with st.sidebar:
    st.title("☀️ Panel Inspector")
    if model_available:
        meta = predictor_cached().meta
        st.markdown("**Model card**")
        st.markdown(
            f"- Backbone: `{meta['backbone']}`\n"
            f"- Parameters: {meta.get('params_millions', '?')} M\n"
            f"- Input: {meta['img_size']}×{meta['img_size']} px\n"
            f"- Test accuracy: {meta.get('test_accuracy', 0):.1%}\n"
            f"- Test macro-F1: {meta.get('test_macro_f1', 0):.3f}\n"
            f"- Device: `{predictor_cached().device}`\n"
            f"- Trained: {str(meta.get('trained_at', ''))[:16].replace('T', ' ')}"
        )
    else:
        st.warning("No trained model found. Run `python main.py` first.")

    st.markdown("---")
    st.markdown("**Inference settings**")
    use_tta = st.toggle("Test-time augmentation (h-flip)", value=True,
                        help="Averages predictions over the image and its mirror. Slightly more accurate, 2× compute.")
    show_cam = st.toggle("Show Grad-CAM heat-map", value=True,
                         help="Highlights the regions that drove the decision.")
    conf_thr = st.slider("Confidence threshold", 0.30, 0.95, LOW_CONFIDENCE_THRESHOLD, 0.05,
                         help="Predictions below this top-1 probability are flagged for manual review.")
    margin_thr = st.slider("Close-call margin", 0.0, 0.5, CLOSE_CALL_MARGIN, 0.05,
                           help="If the gap between the top two classes is smaller than this, flag it.")

# --------------------------------------------------------------------------- #
tab_inspect, tab_perf, tab_data, tab_about = st.tabs(
    ["🔍 Inspect panels", "📊 Model performance", "🗂️ Dataset", "ℹ️ About"])

# --------------------------------------------------------------------------- #
# Tab 1: Inspect
# --------------------------------------------------------------------------- #
with tab_inspect:
    st.subheader("Upload solar panel photos")
    st.caption("Detects: " + " · ".join(LABEL_OF[c] for c in CLASS_NAMES) + ". Several photos at once are fine.")

    if not model_available:
        st.error(f"No model at `{FINAL_MODEL_PATH.relative_to(OUTPUT_DIR.parent)}`. "
                 "Train one with `python main.py`, then reload this page.")
        st.stop()

    predictor = predictor_cached()
    uploads = st.file_uploader("Choose images", type=["jpg", "jpeg", "png", "webp", "bmp"],
                               accept_multiple_files=True, label_visibility="collapsed")

    samples = []
    test_dir = DATA_DIR / "test"
    if test_dir.is_dir():
        c1, c2 = st.columns([1, 3])
        if c1.button("🎲 Try one sample per class", width="stretch"):
            st.session_state["samples"] = [
                random.choice(sorted((test_dir / c).glob("*.jpg"))) for c in CLASS_NAMES if (test_dir / c).is_dir()]
        if c2.button("Clear samples", width="stretch") or uploads:
            st.session_state.pop("samples", None)
        samples = st.session_state.get("samples", [])

    items = [(u.name, u) for u in uploads] + [(f"sample: {p.parent.name}", p) for p in samples]
    if not items:
        st.info("Drop one or more photos above, or try the built-in test samples.")
    else:
        rows, cards = [], []
        with st.spinner(f"Analysing {len(items)} image(s)…"):
            for name, src in items:
                img = load_image(src)
                res = predictor.predict(img, tta=use_tta)
                s = predictor.summarize(res["probs"], low_conf_thr=conf_thr, margin_thr=margin_thr)
                cam = predictor.explain(img) if show_cam else None
                cards.append((name, img, cam, res, s))
                rows.append({"file": name, "prediction": s["label"], "confidence": s["confidence"],
                             "severity": s["severity"], "needs_review": s["needs_review"],
                             "runner_up": f"{s['second_label']} ({s['second_confidence']:.0%})",
                             "_rank": s["severity_rank"]})

        if len(items) > 1:
            df = pd.DataFrame(rows).sort_values(["_rank", "confidence"], ascending=[False, False])
            n_attention = int((df["_rank"] >= 2).sum())
            n_review = int(df["needs_review"].sum())
            m1, m2, m3 = st.columns(3)
            m1.metric("Panels analysed", len(df))
            m2.metric("Need attention", n_attention, help="Severity moderate or above")
            m3.metric("Flagged for manual review", n_review)
            st.dataframe(
                df.drop(columns="_rank"), width="stretch", hide_index=True,
                column_config={
                    "confidence": st.column_config.ProgressColumn("confidence", format="%.0f%%", min_value=0, max_value=1),
                    "needs_review": st.column_config.CheckboxColumn("review?"),
                })
            st.download_button("⬇️ Download results (CSV)", df.drop(columns="_rank").to_csv(index=False).encode(),
                               "panel_predictions.csv", "text/csv")
            st.markdown("---")

        for name, img, cam, res, s in cards:
            with st.container(border=True):
                col_img, col_cam, col_res = st.columns([1.1, 1.1, 1.3]) if cam is not None else st.columns([1.2, 0.01, 1.4])
                col_img.image(img, caption=name, width="stretch")
                if cam is not None:
                    col_cam.image(cam, caption="Grad-CAM: what the model looked at", width="stretch")
                with col_res:
                    st.markdown(f"### {s['label']} &nbsp; {severity_badge(s['severity'])}", unsafe_allow_html=True)
                    st.metric("Confidence", f"{s['confidence']:.1%}",
                              help=f"Runner-up: {s['second_label']} at {s['second_confidence']:.1%}")
                    if s["needs_review"]:
                        st.warning("**Manual review recommended.** " + " ".join(s["review_reasons"]))
                    st.markdown(f"**Recommended action:** {s['action']}")
                    st.altair_chart(prob_chart(s["ranked"], s["class"]), width="stretch")
                    st.caption(f"{res['latency_ms']:.0f} ms on {predictor.device}"
                               + (" · TTA on" if res["tta"] else "") + f" · {predictor.backbone}")

# --------------------------------------------------------------------------- #
# Tab 2: Model performance
# --------------------------------------------------------------------------- #
with tab_perf:
    metrics = read_json(str(METRICS_PATH))
    comparison = read_json(str(COMPARISON_PATH))
    if metrics is None:
        st.info("No metrics yet. Run `python main.py`.")
    else:
        st.subheader(f"Held-out test set · {metrics.get('backbone', '')}")
        c = st.columns(5)
        c[0].metric("Accuracy", f"{metrics['accuracy']:.1%}")
        c[1].metric("Macro-F1", f"{metrics['macro_f1']:.3f}", help="Unweighted mean of per-class F1: every class counts equally, even the rare ones.")
        c[2].metric("ROC-AUC (OvR)", f"{metrics['roc_auc_ovr_macro']:.3f}" if metrics.get("roc_auc_ovr_macro") else "n/a")
        c[3].metric("Top-2 accuracy", f"{metrics['top2_accuracy']:.1%}")
        c[4].metric("Test images", metrics["n_samples"])

        st.markdown("**Per-class results**")
        pc = pd.DataFrame(metrics["per_class"]).T.reset_index().rename(columns={"index": "class"})
        pc["class"] = pc["class"].map(LABEL_OF)
        st.dataframe(pc, width="stretch", hide_index=True, column_config={
            "precision": st.column_config.ProgressColumn("precision", format="%.2f", min_value=0, max_value=1),
            "recall": st.column_config.ProgressColumn("recall", format="%.2f", min_value=0, max_value=1),
            "f1": st.column_config.ProgressColumn("F1", format="%.2f", min_value=0, max_value=1),
            "support": st.column_config.NumberColumn("test images", format="%d"),
        })

        left, right = st.columns(2)
        cm_path, curve_path = OUTPUT_DIR / "confusion_matrix.png", OUTPUT_DIR / "training_curves.png"
        if cm_path.exists():
            left.image(str(cm_path), width="stretch")
        if curve_path.exists():
            right.image(str(curve_path), width="stretch")

        if comparison:
            st.markdown("---")
            st.subheader("Backbone comparison")
            st.caption(f"Selected **{comparison['selected']}** by {comparison['selection_rule']}. "
                       "Test numbers are reported for transparency but never used for selection.")
            cmp = pd.DataFrame(comparison["results"])
            cmp.insert(0, "selected", cmp["backbone"] == comparison["selected"])
            st.dataframe(cmp, width="stretch", hide_index=True, column_config={
                "selected": st.column_config.CheckboxColumn("✓"),
                "params_millions": st.column_config.NumberColumn("params (M)", format="%.2f"),
                "val_macro_f1": st.column_config.NumberColumn("val macro-F1", format="%.3f"),
                "val_accuracy": st.column_config.NumberColumn("val acc", format="%.3f"),
                "test_accuracy": st.column_config.NumberColumn("test acc", format="%.3f"),
                "test_macro_f1": st.column_config.NumberColumn("test macro-F1", format="%.3f"),
                "test_roc_auc": st.column_config.NumberColumn("test ROC-AUC", format="%.3f"),
                "latency_cpu_ms": st.column_config.NumberColumn("CPU ms/img", format="%.1f"),
                "latency_gpu_ms": st.column_config.NumberColumn("GPU ms/img", format="%.1f"),
                "train_time_min": st.column_config.NumberColumn("train min", format="%.1f"),
            })
            with st.expander("Per-backbone confusion matrices and curves"):
                for r in comparison["results"]:
                    run_dir = OUTPUT_DIR / "runs" / r["backbone"]
                    a, b = st.columns(2)
                    if (run_dir / "confusion_matrix.png").exists():
                        a.image(str(run_dir / "confusion_matrix.png"), width="stretch")
                    if (run_dir / "training_curves.png").exists():
                        b.image(str(run_dir / "training_curves.png"), width="stretch")

# --------------------------------------------------------------------------- #
# Tab 3: Dataset
# --------------------------------------------------------------------------- #
with tab_data:
    report = read_json(str(DATASET_REPORT_PATH))
    if report is None:
        st.info("No dataset report yet. Run `python scripts/prepare_dataset.py`.")
    else:
        st.subheader("From raw download to clean splits")
        totals = report["totals"]
        stages = [("Collected", "collected"), ("Readable", "readable"),
                  ("After exact de-dup", "after_exact_dedup"), ("After near de-dup", "after_near_dedup")]
        cols = st.columns(len(stages))
        prev = None
        for col, (label, key) in zip(cols, stages):
            col.metric(label, totals[key], delta=None if prev is None else totals[key] - prev, delta_color="off")
            prev = totals[key]
        rm = report["removed"]
        st.caption(f"Removed {len(rm['exact_duplicates'])} byte-identical duplicates, "
                   f"{len(rm['near_duplicates'])} near-duplicates (dHash Hamming ≤ {report['params']['phash_thr']}), "
                   f"{len(rm['label_conflicts'])} images that appeared under two different labels, "
                   f"and {len(rm['broken'])} unreadable files. "
                   f"Split {report['params']['train_ratio']:.0%}/{report['params']['val_ratio']:.0%}/"
                   f"{1 - report['params']['train_ratio'] - report['params']['val_ratio']:.0%} per class, seed {report['params']['seed']}.")

        rows = []
        for split, counts in report["split_counts"].items():
            for cls, n in counts.items():
                rows.append({"class": LABEL_OF.get(cls, cls), "split": split, "images": n})
        df = pd.DataFrame(rows)
        chart = alt.Chart(df).mark_bar(cornerRadiusEnd=3).encode(
            x=alt.X("class:N", title=None, axis=alt.Axis(labelAngle=0, labelColor=INK_2, domain=False, ticks=False)),
            xOffset=alt.XOffset("split:N", sort=["train", "val", "test"]),
            y=alt.Y("images:Q", title=None, axis=alt.Axis(gridColor="#e1e0d9", labelColor=MUTED, domain=False)),
            color=alt.Color("split:N", sort=["train", "val", "test"], title=None,
                            scale=alt.Scale(domain=["train", "val", "test"], range=SERIES),
                            legend=alt.Legend(orient="top", labelColor=INK_2)),
            tooltip=["class", "split", "images"],
        ).properties(height=260).configure_view(strokeWidth=0)
        st.altair_chart(chart, width="stretch")

        table = pd.DataFrame(report["split_counts"]).rename(index=LABEL_OF)
        table["total"] = table.sum(axis=1)
        table.loc["total"] = table.sum()
        st.dataframe(table, width="stretch")

        train_dir = DATA_DIR / "train"
        if train_dir.is_dir():
            st.markdown("**One training example per class**")
            cols = st.columns(len(CLASS_NAMES))
            rng = random.Random(0)
            for col, cls in zip(cols, CLASS_NAMES):
                files = sorted((train_dir / cls).glob("*.jpg"))
                if files:
                    col.image(str(rng.choice(files)), caption=LABEL_OF[cls], width="stretch")

        with st.expander("Images dropped because they carried two different labels"):
            st.json(rm["label_conflicts"])

# --------------------------------------------------------------------------- #
# Tab 4: About
# --------------------------------------------------------------------------- #
with tab_about:
    st.markdown(
        """
        ### What this does
        Classifies a photo of a solar panel into one of six conditions and turns the result into a
        maintenance decision: **Clean**, **Dusty / Soiled**, **Bird Droppings**, **Snow Covered**,
        **Physical Damage**, **Electrical Damage**. Soiling alone can cut output by 5–25 %; electrical
        faults are a fire risk. Ranking panels by severity lets an operator send a cleaning crew or a
        technician only where it is needed.

        ### Pipeline
        1. **Data** – Kaggle *Faulty solar panel* image set (six folders of web-scraped photos).
           `scripts/prepare_dataset.py` verifies every file, removes exact and near-duplicate images
           (a perceptual hash catches re-encoded copies), drops images that appear under two labels,
           and makes a stratified 70/15/15 split.
        2. **Model** – ImageNet-pretrained backbone from torchvision, new 6-way head. Two-stage
           transfer learning: head-only warm-up, then full fine-tuning with a lower backbone learning
           rate, cosine schedule, label smoothing, class-weighted loss and early stopping on
           validation macro-F1.
        3. **Selection** – several backbones are trained; the winner is chosen on the *validation*
           split. Test metrics are reported once, afterwards.
        4. **Inference** – horizontal-flip test-time augmentation, probability calibration checks
           (confidence threshold + close-call margin), and Grad-CAM heat-maps for explainability.

        ### Limitations
        * ~700 unique images; the rarest class (physical damage) has under 50 training examples.
        * Photos are web-scraped: varied cameras, angles and lighting, but not a fixed inspection rig.
        * Predictions are per image, not per cell; use the heat-map to localise.

        ### Retrain
        ```bash
        python scripts/prepare_dataset.py --raw_dir Faulty_solar_panel --out_dir data --force
        python main.py            # trains + compares backbones, writes outputs/final_model.pt
        streamlit run app.py
        ```
        """
    )
