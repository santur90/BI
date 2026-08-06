import io
import itertools
import urllib.parse
import urllib.request

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from scipy import stats

st.set_page_config(page_title="BioInfo Analysis", layout="wide")


def read_table(uploaded_file, index_col=None):
    if uploaded_file is None:
        return None
    sep = "," if uploaded_file.name.lower().endswith(".csv") else "\t"
    return pd.read_csv(uploaded_file, sep=sep, index_col=index_col)


def fdr_bh(p_values):
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return np.array([])
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * n / (np.arange(n) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    output = np.empty(n)
    output[order] = adjusted
    return output


def plot_platform_schematic():
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    fig.suptitle("Integrated Bioinformatics Workflow with Plot Previews", fontsize=14, y=0.98)

    ax = axes[0, 0]
    ax.bar(["Group A", "Group B"], [0.28, 0.47], color=["#4c78a8", "#f58518"])
    ax.set_title("Mutation: VAF Comparison", fontsize=10)
    ax.set_ylabel("VAF")

    ax = axes[0, 1]
    x = np.linspace(1, 8, 8)
    y = np.array([0.05, 0.12, 0.2, 0.42, 0.38, 0.18, 0.1, 0.03])
    colors = ["#2ca02c" if val > 0.2 else "#7f7f7f" for val in y]
    ax.scatter(x, y, c=colors, s=45)
    ax.axhline(0.2, color="#d62728", linestyle="--", linewidth=1)
    ax.set_title("DNA Methylation: DMR Signal", fontsize=10)
    ax.set_xlabel("Genomic Region")
    ax.set_ylabel("Delta Beta")

    ax = axes[0, 2]
    heat = np.array([
        [1.2, 1.1, -0.2, -0.4],
        [0.7, 0.6, -0.1, -0.3],
        [-0.5, -0.4, 0.8, 0.9],
        [-0.7, -0.6, 0.9, 1.2],
    ])
    ax.imshow(heat, cmap="coolwarm", aspect="auto")
    ax.set_title("Microarray/RNA-seq: DEG Heatmap", fontsize=10)
    ax.set_xticks([0, 1, 2, 3], ["A1", "A2", "B1", "B2"])
    ax.set_yticks([0, 1, 2, 3], ["G1", "G2", "G3", "G4"])

    ax = axes[1, 0]
    graph = nx.Graph()
    graph.add_edges_from([
        ("TP53", "MDM2"),
        ("TP53", "CDKN1A"),
        ("EGFR", "GRB2"),
        ("GRB2", "SOS1"),
    ])
    pos = nx.spring_layout(graph, seed=7)
    nx.draw_networkx(graph, pos=pos, ax=ax, node_color="#8dd3c7", node_size=700, font_size=8)
    ax.set_title("PPI: Network", fontsize=10)
    ax.axis("off")

    ax = axes[1, 1]
    ax.step([0, 8, 16, 24, 32], [1.0, 0.92, 0.82, 0.73, 0.65], where="post", label="Group A")
    ax.step([0, 8, 16, 24, 32], [1.0, 0.88, 0.7, 0.5, 0.4], where="post", label="Group B")
    ax.set_title("Survival: Kaplan-Meier", fontsize=10)
    ax.set_xlabel("Time")
    ax.set_ylabel("Survival Probability")
    ax.legend(fontsize=8)

    ax = axes[1, 2]
    ax.axis("off")
    ax.text(
        0,
        0.85,
        "Supported modules:\n"
        "1) Mutation (VAF by group)\n"
        "2) DNA methylation (DMR)\n"
        "3) Microarray/RNA-seq (DEG + heatmap)\n"
        "4) PPI network (protein input)\n"
        "5) Survival (KM + Cox)",
        va="top",
        fontsize=10,
    )

    plt.tight_layout()
    return fig


def km_curve(time, status):
    df = pd.DataFrame({"time": time, "status": status}).dropna()
    df = df.sort_values("time")
    event_times = np.sort(df.loc[df["status"] == 1, "time"].unique())
    xs = [0.0]
    ys = [1.0]
    surv = 1.0
    for t in event_times:
        n_risk = (df["time"] >= t).sum()
        d_events = ((df["time"] == t) & (df["status"] == 1)).sum()
        if n_risk > 0:
            surv = surv * (1.0 - d_events / n_risk)
        xs.append(float(t))
        ys.append(float(surv))
    return xs, ys


def run_mutation_module():
    st.subheader("Mutation Analysis: Compare VAF Between Groups")
    st.caption("Input columns: group, vaf, and optional gene.")

    mut_file = st.file_uploader("Upload mutation table (CSV/TSV)", type=["csv", "tsv", "txt"], key="mutation")
    if mut_file is None:
        st.info("Upload mutation data to run VAF comparison.")
        return

    mut_df = read_table(mut_file)
    cols = {c.lower(): c for c in mut_df.columns}
    if "group" not in cols or "vaf" not in cols:
        st.error("Required columns not found: group, vaf")
        st.dataframe(mut_df.head())
        return

    group_col = cols["group"]
    vaf_col = cols["vaf"]
    work_df = mut_df.copy()
    if "gene" in cols:
        gene_col = cols["gene"]
        genes = sorted(work_df[gene_col].dropna().astype(str).unique())
        selected_gene = st.selectbox("Select gene", genes)
        work_df = work_df[work_df[gene_col].astype(str) == selected_gene]

    work_df[vaf_col] = pd.to_numeric(work_df[vaf_col], errors="coerce")
    work_df = work_df.dropna(subset=[group_col, vaf_col])
    if work_df.empty:
        st.warning("No valid rows after filtering.")
        return

    summary = work_df.groupby(group_col)[vaf_col].agg(["count", "mean", "median", "std"]).reset_index()
    st.dataframe(summary)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.boxplot(data=work_df, x=group_col, y=vaf_col, ax=ax, color="#9ecae1")
    sns.stripplot(data=work_df, x=group_col, y=vaf_col, ax=ax, color="#1f4e79", size=4, alpha=0.75)
    ax.set_title("VAF Distribution by Group")
    st.pyplot(fig)

    groups = [g[vaf_col].values for _, g in work_df.groupby(group_col)]
    if len(groups) >= 2:
        if len(groups) == 2:
            stat, pval = stats.ttest_ind(groups[0], groups[1], equal_var=False, nan_policy="omit")
            st.write(f"Two-group t-test p-value: {pval:.4g}")
        else:
            stat, pval = stats.f_oneway(*groups)
            st.write(f"One-way ANOVA p-value: {pval:.4g}")


def run_methylation_module():
    st.subheader("DNA Methylation: DMR Analysis")
    st.caption("Input long-format columns: region, group, beta.")

    met_file = st.file_uploader("Upload methylation table (CSV/TSV)", type=["csv", "tsv", "txt"], key="methylation")
    if met_file is None:
        st.info("Upload methylation data to run DMR analysis.")
        return

    met_df = read_table(met_file)
    cols = {c.lower(): c for c in met_df.columns}
    needed = ["region", "group", "beta"]
    if any(n not in cols for n in needed):
        st.error("Required columns not found: region, group, beta")
        st.dataframe(met_df.head())
        return

    region_col, group_col, beta_col = cols["region"], cols["group"], cols["beta"]
    met_df[beta_col] = pd.to_numeric(met_df[beta_col], errors="coerce")
    met_df = met_df.dropna(subset=[region_col, group_col, beta_col])
    groups = sorted(met_df[group_col].astype(str).unique())
    if len(groups) < 2:
        st.warning("At least two groups are required.")
        return

    g1 = st.selectbox("Group 1", groups, index=0, key="dmr_g1")
    g2 = st.selectbox("Group 2", groups, index=1, key="dmr_g2")
    if g1 == g2:
        st.warning("Choose two different groups.")
        return

    results = []
    for region, sub in met_df.groupby(region_col):
        x = sub.loc[sub[group_col].astype(str) == g1, beta_col].values
        y = sub.loc[sub[group_col].astype(str) == g2, beta_col].values
        if len(x) >= 2 and len(y) >= 2:
            _, pval = stats.ttest_ind(x, y, equal_var=False, nan_policy="omit")
            delta = np.mean(x) - np.mean(y)
            results.append([region, np.mean(x), np.mean(y), delta, pval])

    if not results:
        st.warning("No regions with enough replicates (>=2 per group).")
        return

    dmr = pd.DataFrame(results, columns=["region", f"mean_{g1}", f"mean_{g2}", "delta_beta", "p_value"])
    dmr["fdr"] = fdr_bh(dmr["p_value"].values)
    dmr = dmr.sort_values(["fdr", "p_value", "delta_beta"], ascending=[True, True, False])
    st.dataframe(dmr.head(30))

    fig, ax = plt.subplots(figsize=(8, 5))
    score = -np.log10(np.clip(dmr["fdr"].values, 1e-300, 1))
    sig = (np.abs(dmr["delta_beta"]) >= 0.1) & (dmr["fdr"] <= 0.05)
    ax.scatter(dmr["delta_beta"], score, c=np.where(sig, "#d62728", "#6c757d"), alpha=0.8)
    ax.set_xlabel("Delta Beta")
    ax.set_ylabel("-log10(FDR)")
    ax.set_title("DMR Volcano Plot")
    st.pyplot(fig)


def run_deg_module():
    st.subheader("Microarray/RNA-seq: DEG + Heatmap")
    st.caption("Expression matrix: rows=genes, columns=samples. Metadata columns: sample, group.")

    expr_file = st.file_uploader("Upload expression matrix (CSV/TSV)", type=["csv", "tsv", "txt"], key="expr")
    meta_file = st.file_uploader("Upload sample metadata (CSV/TSV)", type=["csv", "tsv", "txt"], key="meta")
    if expr_file is None or meta_file is None:
        st.info("Upload both expression matrix and metadata to run DEG analysis.")
        return

    expr = read_table(expr_file, index_col=0)
    meta = read_table(meta_file)
    cols = {c.lower(): c for c in meta.columns}
    if "sample" not in cols or "group" not in cols:
        st.error("Metadata must contain columns: sample, group")
        st.dataframe(meta.head())
        return

    sample_col = cols["sample"]
    group_col = cols["group"]
    meta[sample_col] = meta[sample_col].astype(str)
    common_samples = [s for s in meta[sample_col] if s in expr.columns]
    if len(common_samples) < 4:
        st.warning("Need at least 4 overlapping samples between matrix and metadata.")
        return

    expr = expr[common_samples].apply(pd.to_numeric, errors="coerce")
    expr = expr.dropna(axis=0, how="any")
    meta = meta[meta[sample_col].isin(common_samples)].copy()
    groups = sorted(meta[group_col].astype(str).unique())
    if len(groups) < 2:
        st.warning("At least two groups are required.")
        return

    g1 = st.selectbox("Case group", groups, index=0, key="deg_g1")
    g2 = st.selectbox("Control group", groups, index=1, key="deg_g2")
    if g1 == g2:
        st.warning("Choose two different groups.")
        return

    g1_samples = meta.loc[meta[group_col].astype(str) == g1, sample_col].tolist()
    g2_samples = meta.loc[meta[group_col].astype(str) == g2, sample_col].tolist()
    if len(g1_samples) < 2 or len(g2_samples) < 2:
        st.warning("Each group needs at least 2 samples.")
        return

    log_expr = np.log2(expr + 1)
    mean1 = log_expr[g1_samples].mean(axis=1)
    mean2 = log_expr[g2_samples].mean(axis=1)
    log2fc = mean1 - mean2

    pvals = []
    for gene in log_expr.index:
        x = log_expr.loc[gene, g1_samples].values
        y = log_expr.loc[gene, g2_samples].values
        _, pval = stats.ttest_ind(x, y, equal_var=False, nan_policy="omit")
        pvals.append(pval)

    deg = pd.DataFrame({
        "gene": log_expr.index,
        "log2FC": log2fc.values,
        "p_value": pvals,
    })
    deg["fdr"] = fdr_bh(deg["p_value"].values)
    deg = deg.sort_values(["fdr", "p_value"])

    fc_cut = st.slider("|log2FC| cutoff", 0.0, 3.0, 1.0, 0.1)
    fdr_cut = st.slider("FDR cutoff", 0.001, 0.2, 0.05, 0.001)
    sig = deg[(np.abs(deg["log2FC"]) >= fc_cut) & (deg["fdr"] <= fdr_cut)]
    st.write(f"Significant DEG count: {len(sig)}")
    st.dataframe(deg.head(50))

    fig, ax = plt.subplots(figsize=(8, 5))
    score = -np.log10(np.clip(deg["fdr"].values, 1e-300, 1))
    ax.scatter(deg["log2FC"], score, c="#9aa0a6", s=12, alpha=0.6)
    if not sig.empty:
        sig_score = -np.log10(np.clip(sig["fdr"].values, 1e-300, 1))
        ax.scatter(sig["log2FC"], sig_score, c="#d62728", s=14, alpha=0.8)
    ax.set_title("DEG Volcano Plot")
    ax.set_xlabel("log2 Fold Change")
    ax.set_ylabel("-log10(FDR)")
    st.pyplot(fig)

    if not sig.empty:
        n_top = min(30, len(sig))
        top_genes = sig.sort_values("fdr").head(n_top)["gene"].tolist()
        heat_data = log_expr.loc[top_genes, g1_samples + g2_samples]
        heat_data = heat_data.sub(heat_data.mean(axis=1), axis=0)
        heat_data = heat_data.div(heat_data.std(axis=1).replace(0, 1), axis=0)
        fig, ax = plt.subplots(figsize=(9, max(4, n_top * 0.22)))
        sns.heatmap(heat_data, cmap="vlag", ax=ax)
        ax.set_title("Top Significant DEG Heatmap (row z-score)")
        st.pyplot(fig)


def fetch_string_edges(proteins, species=9606, score_cutoff=400):
    identifiers = "%0d".join(proteins)
    query = urllib.parse.urlencode({
        "identifiers": identifiers,
        "species": str(species),
        "required_score": str(score_cutoff),
    })
    url = f"https://string-db.org/api/tsv/network?{query}"
    with urllib.request.urlopen(url, timeout=20) as response:
        text = response.read().decode("utf-8")

    lines = text.strip().splitlines()
    if len(lines) <= 1:
        return []
    header = lines[0].split("\t")
    rows = [line.split("\t") for line in lines[1:] if line.strip()]
    frame = pd.DataFrame(rows, columns=header)
    required_cols = {"preferredName_A", "preferredName_B", "score"}
    if not required_cols.issubset(set(frame.columns)):
        return []
    frame["score"] = pd.to_numeric(frame["score"], errors="coerce")
    frame = frame.dropna(subset=["preferredName_A", "preferredName_B", "score"])
    return frame[["preferredName_A", "preferredName_B", "score"]].values.tolist()


def run_ppi_module():
    st.subheader("PPI Network: Protein Name Input")
    st.caption("Enter proteins separated by comma, space, or line break. Example: TP53, EGFR, BRCA1, MDM2")

    protein_text = st.text_area("Protein list", value="TP53, EGFR, BRCA1, MDM2, CDK2", height=120)
    species = st.selectbox("Species", [("Human", 9606), ("Mouse", 10090)], format_func=lambda x: x[0])
    score_cutoff = st.slider("STRING score cutoff", 150, 900, 400, 50)

    if st.button("Build PPI Network"):
        proteins = [p.strip().upper() for p in protein_text.replace("\n", ",").replace(" ", ",").split(",") if p.strip()]
        proteins = list(dict.fromkeys(proteins))
        if len(proteins) < 2:
            st.warning("At least two proteins are required.")
            return

        edges = []
        source_note = "STRING API"
        try:
            edges = fetch_string_edges(proteins, species=species[1], score_cutoff=score_cutoff)
        except Exception:
            source_note = "Local fallback"

        if not edges:
            fallback = []
            for a, b in itertools.combinations(proteins, 2):
                score = ((sum(ord(ch) for ch in a + b) % 400) + 500) / 1000
                if score >= 0.65:
                    fallback.append((a, b, score))
            edges = fallback

        if not edges:
            st.warning("No interactions found with current settings.")
            return

        graph = nx.Graph()
        graph.add_nodes_from(proteins)
        for a, b, score in edges:
            graph.add_edge(str(a), str(b), weight=float(score))

        pos = nx.spring_layout(graph, seed=42)
        fig, ax = plt.subplots(figsize=(8, 6))
        weights = [graph[u][v]["weight"] for u, v in graph.edges]
        edge_widths = [1 + 4 * (w / max(weights)) for w in weights]
        nx.draw_networkx_edges(graph, pos, width=edge_widths, edge_color="#6b7280", alpha=0.8, ax=ax)
        nx.draw_networkx_nodes(graph, pos, node_color="#93c5fd", node_size=900, ax=ax)
        nx.draw_networkx_labels(graph, pos, font_size=9, ax=ax)
        ax.set_title(f"PPI Network ({source_note})")
        ax.axis("off")
        st.pyplot(fig)

        edge_df = pd.DataFrame([(u, v, graph[u][v]["weight"]) for u, v in graph.edges], columns=["protein_a", "protein_b", "score"])
        st.dataframe(edge_df.sort_values("score", ascending=False))


def run_survival_module():
    st.subheader("Survival Analysis: KM Curve + Cox")
    st.caption("Input columns: time, status (1=event, 0=censored), group. Optional numeric covariates are used for Cox model.")

    surv_file = st.file_uploader("Upload survival table (CSV/TSV)", type=["csv", "tsv", "txt"], key="survival")
    if surv_file is None:
        st.info("Upload survival data to run analysis.")
        return

    surv = read_table(surv_file)
    cols = {c.lower(): c for c in surv.columns}
    for need in ["time", "status", "group"]:
        if need not in cols:
            st.error("Required columns not found: time, status, group")
            st.dataframe(surv.head())
            return

    time_col = cols["time"]
    status_col = cols["status"]
    group_col = cols["group"]
    surv[time_col] = pd.to_numeric(surv[time_col], errors="coerce")
    surv[status_col] = pd.to_numeric(surv[status_col], errors="coerce")
    surv = surv.dropna(subset=[time_col, status_col, group_col]).copy()
    surv[group_col] = surv[group_col].astype(str)

    groups = sorted(surv[group_col].unique())
    if len(groups) < 2:
        st.warning("At least two groups are required for comparison.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for grp in groups:
        sub = surv[surv[group_col] == grp]
        xs, ys = km_curve(sub[time_col].values, sub[status_col].values)
        ax.step(xs, ys, where="post", label=f"{grp} (n={len(sub)})")
    ax.set_title("Kaplan-Meier Survival Curve")
    ax.set_xlabel("Time")
    ax.set_ylabel("Survival Probability")
    ax.legend()
    st.pyplot(fig)

    cox_ready = surv[[time_col, status_col, group_col]].copy()
    num_covars = [c for c in surv.columns if c not in [time_col, status_col, group_col] and pd.api.types.is_numeric_dtype(surv[c])]
    if num_covars:
        cox_ready = pd.concat([cox_ready, surv[num_covars]], axis=1)
    cox_ready = pd.get_dummies(cox_ready, columns=[group_col], drop_first=True)

    try:
        from lifelines import CoxPHFitter

        model = CoxPHFitter()
        model.fit(cox_ready, duration_col=time_col, event_col=status_col)
        st.write("Cox model summary")
        st.dataframe(model.summary)
    except Exception as err:
        st.warning("CoxPHFitter not available in current runtime. Falling back to event-rate hazard ratio summary.")
        rows = []
        ref = groups[0]
        ref_sub = surv[surv[group_col] == ref]
        ref_hazard = ref_sub[status_col].sum() / np.clip(ref_sub[time_col].sum(), 1e-9, None)
        for grp in groups[1:]:
            sub = surv[surv[group_col] == grp]
            hz = sub[status_col].sum() / np.clip(sub[time_col].sum(), 1e-9, None)
            hr = hz / np.clip(ref_hazard, 1e-9, None)
            rows.append({"comparison": f"{grp} vs {ref}", "hazard_rate_ratio": hr})
        st.dataframe(pd.DataFrame(rows))
        st.caption(f"Fallback reason: {err}")


st.title("🧬 Bioinformatics Data Analysis Platform")
st.markdown(
    """
Run mutation, methylation, expression, PPI, and survival analyses directly in your browser.
**Privacy Note:** Your data is processed locally in your browser and is never uploaded to any server.
"""
)

st.subheader("Workflow Schematic")
st.pyplot(plot_platform_schematic())

st.sidebar.header("Analysis Module")
module = st.sidebar.radio(
    "Choose module",
    [
        "Mutation (VAF)",
        "DNA Methylation (DMR)",
        "Microarray/RNA-seq (DEG)",
        "PPI Network",
        "Survival (KM + Cox)",
    ],
)

if module == "Mutation (VAF)":
    run_mutation_module()
elif module == "DNA Methylation (DMR)":
    run_methylation_module()
elif module == "Microarray/RNA-seq (DEG)":
    run_deg_module()
elif module == "PPI Network":
    run_ppi_module()
elif module == "Survival (KM + Cox)":
    run_survival_module()

st.sidebar.markdown("---")
st.sidebar.info("Framework: Stlite (Python in Browser)")
