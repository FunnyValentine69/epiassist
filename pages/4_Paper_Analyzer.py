"""Paper Analyzer page for extracting statistics from PDFs."""

import streamlit as st
import pandas as pd

from core.paper_parser import (
    extract_text_from_pdf,
    find_effect_measures,
    find_confidence_intervals,
    find_p_values,
    find_sample_sizes,
    find_beta_coefficients,
    find_mean_differences,
    find_standard_deviations,
    find_weighted_statistics,
)
from core.llm_extractor import is_llm_available, extract_with_llm, merge_results

st.set_page_config(page_title="Paper Analyzer - EpiAssist", layout="wide")

st.title("Paper Analyzer")
st.markdown("""
Upload epidemiological research papers (PDF) and automatically extract
reported statistics including odds ratios, confidence intervals, and p-values.
""")

st.divider()

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### Upload Paper")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload an epidemiological research paper",
    )

    if uploaded_file is not None:
        st.success(f"Uploaded: {uploaded_file.name}")
        extract_btn = st.button("Extract Statistics", type="primary")
    else:
        st.info("Upload a PDF to extract statistics.")
        extract_btn = False

    llm_available = is_llm_available()
    use_llm = st.checkbox(
        "Enhance with AI (Ollama)",
        value=False,
        disabled=not llm_available,
        help="Run a second-pass LLM extraction via local Ollama to catch stats that regex misses."
        + ("" if llm_available else " (Ollama not detected — install and run 'ollama serve')"),
    )

    st.markdown("### What We Extract")
    st.markdown("""
    - Effect Measures (OR, HR, RR, PR, IRR) with **adjusted/crude detection**
    - Beta Coefficients (β, B, coefficient)
    - Confidence Intervals (95% CI)
    - P-values
    - Mean Differences (MD)
    - Standard Deviations/Errors (SD, SE)
    - Weighted Statistics (IPW, survey-weighted, PS-weighted)
    - Sample sizes (n)
    """)

with col2:
    st.markdown("### Extracted Statistics")

    if uploaded_file is not None and extract_btn:
        try:
            with st.spinner("Extracting text from PDF..."):
                # Read PDF bytes and extract text by page
                file_bytes = uploaded_file.read()
                pages = extract_text_from_pdf(file_bytes)

                # Validate extraction result
                if not pages:
                    st.error("Could not extract text from PDF. The file may be empty or corrupted.")
                    st.stop()

                # Store full text for preview (join all pages)
                full_text = "\n".join(page_text for page_num, page_text in pages)
                st.session_state.paper_text = full_text
                st.session_state.paper_filename = uploaded_file.name

            with st.spinner("Finding statistics..."):
                # Extract statistics from each page
                effect_measures = []
                confidence_intervals = []
                p_values = []
                sample_sizes = []
                beta_coefficients = []
                mean_differences = []
                standard_deviations = []
                weighted_statistics = []

                for page_num, page_text in pages:
                    effect_measures.extend(find_effect_measures(page_text, page=page_num))
                    confidence_intervals.extend(find_confidence_intervals(page_text, page=page_num))
                    p_values.extend(find_p_values(page_text, page=page_num))
                    sample_sizes.extend(find_sample_sizes(page_text, page=page_num))
                    beta_coefficients.extend(find_beta_coefficients(page_text, page=page_num))
                    mean_differences.extend(find_mean_differences(page_text, page=page_num))
                    standard_deviations.extend(find_standard_deviations(page_text, page=page_num))
                    weighted_statistics.extend(find_weighted_statistics(page_text, page=page_num))

                regex_results = {
                    "effect_measures": effect_measures,
                    "confidence_intervals": confidence_intervals,
                    "p_values": p_values,
                    "sample_sizes": sample_sizes,
                    "beta_coefficients": beta_coefficients,
                    "mean_differences": mean_differences,
                    "standard_deviations": standard_deviations,
                    "weighted_statistics": weighted_statistics,
                }

            # LLM second pass (optional)
            if use_llm:
                llm_all = {cat: [] for cat in regex_results}
                progress = st.progress(0, text="Running AI extraction...")
                for i, (page_num, page_text) in enumerate(pages):
                    progress.progress(
                        (i + 1) / len(pages),
                        text=f"AI extracting page {page_num}...",
                    )
                    page_llm = extract_with_llm(page_text, page=page_num)
                    for cat in llm_all:
                        llm_all[cat].extend(page_llm.get(cat, []))
                progress.empty()

                regex_count = sum(len(v) for v in regex_results.values())
                final_results = merge_results(regex_results, llm_all)
                final_count = sum(len(v) for v in final_results.values())
                llm_added = final_count - regex_count
                if llm_added > 0:
                    st.info(f"AI found {llm_added} additional statistic(s)")
                else:
                    llm_total = sum(len(v) for v in llm_all.values())
                    if llm_total == 0:
                        st.warning("AI extraction returned no results. Check that Ollama is running with llama3.1:8b loaded.")
                    else:
                        st.info("AI found no additional statistics beyond regex.")
            else:
                # Tag all regex results with source
                final_results = regex_results
                for cat in final_results:
                    for item in final_results[cat]:
                        item["source"] = "regex"

            with st.spinner("Storing results..."):
                # Store results
                st.session_state.paper_results = final_results

            st.success("Extraction complete.")
        except Exception as e:
            st.error(f"Error processing PDF: {e}")

    # Display results if available
    if "paper_results" in st.session_state:
        results = st.session_state.paper_results

        # Tabs for each extraction type
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
            ["Effect Measures", "Beta Coefficients", "CIs", "P-values", "Mean Diff", "SD/SE", "Weighted Stats", "Sample Sizes"]
        )

        with tab1:
            if results["effect_measures"]:
                em_data = []
                for item in results["effect_measures"]:
                    # Format Adjusted? column
                    if item.get("adjusted") is True:
                        adjusted_str = "Yes"
                    elif item.get("adjusted") is False:
                        adjusted_str = "No"
                    else:
                        adjusted_str = "-"

                    # Format Adjusted For column (truncate if long)
                    adj_for = item.get("adjusted_for")
                    if adj_for:
                        adj_for_str = adj_for[:50] + "..." if len(adj_for) > 50 else adj_for
                    else:
                        adj_for_str = "-"

                    em_data.append(
                        {
                            "Type": item["type"],
                            "Page": item["page"],
                            "Value": item["value"],
                            "CI Lower": item["ci_lower"] or "-",
                            "CI Upper": item["ci_upper"] or "-",
                            "Adjusted?": adjusted_str,
                            "Adjusted For": adj_for_str,
                            "Source": item.get("source", "regex"),
                            "Context": item["context"][:60] + "...",
                        }
                    )
                st.dataframe(pd.DataFrame(em_data), use_container_width=True)
                # Count adjusted vs crude vs unknown
                adj_count = sum(1 for em in results["effect_measures"] if em.get("adjusted") is True)
                crude_count = sum(1 for em in results["effect_measures"] if em.get("adjusted") is False)
                unk_count = len(results["effect_measures"]) - adj_count - crude_count
                st.caption(f"Found {len(results['effect_measures'])} effect measure(s): {adj_count} adjusted, {crude_count} crude, {unk_count} unknown")
            else:
                st.info("No effect measures found in this document.")

        with tab2:
            if results.get("beta_coefficients"):
                beta_data = []
                for item in results["beta_coefficients"]:
                    beta_data.append(
                        {
                            "Page": item["page"],
                            "Value": item["value"],
                            "CI Lower": item["ci_lower"] or "-",
                            "CI Upper": item["ci_upper"] or "-",
                            "SE": item["se"] or "-",
                            "Source": item.get("source", "regex"),
                            "Context": item["context"][:60] + "...",
                        }
                    )
                st.dataframe(pd.DataFrame(beta_data), use_container_width=True)
                st.caption(f"Found {len(results['beta_coefficients'])} beta coefficient(s)")
            else:
                st.info("No beta coefficients found in this document.")

        with tab3:
            if results["confidence_intervals"]:
                ci_data = []
                for item in results["confidence_intervals"]:
                    ci_data.append(
                        {
                            "Page": item["page"],
                            "Level": f"{item['level']}%",
                            "Lower": item["lower"],
                            "Upper": item["upper"],
                            "Source": item.get("source", "regex"),
                            "Context": item["context"][:60] + "...",
                        }
                    )
                st.dataframe(pd.DataFrame(ci_data), use_container_width=True)
                st.caption(
                    f"Found {len(results['confidence_intervals'])} confidence interval(s)"
                )
            else:
                st.info("No confidence intervals found in this document.")

        with tab4:
            if results["p_values"]:
                p_data = []
                for item in results["p_values"]:
                    p_data.append(
                        {
                            "Page": item["page"],
                            "P-value": item["value"],
                            "Operator": item["operator"],
                            "Significant (α=0.05)": "Yes" if item["value"] < 0.05 else "No",
                            "Source": item.get("source", "regex"),
                            "Context": item["context"][:60] + "...",
                        }
                    )
                st.dataframe(pd.DataFrame(p_data), use_container_width=True)
                st.caption(f"Found {len(results['p_values'])} p-value(s)")
            else:
                st.info("No p-values found in this document.")

        with tab5:
            if results.get("mean_differences"):
                md_data = []
                for item in results["mean_differences"]:
                    md_data.append(
                        {
                            "Page": item["page"],
                            "Value": item["value"],
                            "CI Lower": item["ci_lower"] or "-",
                            "CI Upper": item["ci_upper"] or "-",
                            "Source": item.get("source", "regex"),
                            "Context": item["context"][:60] + "...",
                        }
                    )
                st.dataframe(pd.DataFrame(md_data), use_container_width=True)
                st.caption(f"Found {len(results['mean_differences'])} mean difference(s)")
            else:
                st.info("No mean differences found in this document.")

        with tab6:
            if results.get("standard_deviations"):
                sd_data = []
                for item in results["standard_deviations"]:
                    sd_data.append(
                        {
                            "Page": item["page"],
                            "Type": item["type"],
                            "Mean": item["mean"] or "-",
                            "Value": item["value"],
                            "Source": item.get("source", "regex"),
                            "Context": item["context"][:60] + "...",
                        }
                    )
                st.dataframe(pd.DataFrame(sd_data), use_container_width=True)
                st.caption(f"Found {len(results['standard_deviations'])} SD/SE value(s)")
            else:
                st.info("No standard deviations/errors found in this document.")

        with tab7:
            if results.get("weighted_statistics"):
                ws_data = []
                for item in results["weighted_statistics"]:
                    ws_data.append(
                        {
                            "Page": item["page"],
                            "Type": item["stat_type"],
                            "Value": item["value"],
                            "Weight Method": item["weight_method"] or "-",
                            "Source": item.get("source", "regex"),
                            "Context": item["context"][:60] + "...",
                        }
                    )
                st.dataframe(pd.DataFrame(ws_data), use_container_width=True)
                st.caption(f"Found {len(results['weighted_statistics'])} weighted statistic(s)")
            else:
                st.info("No weighted statistics found in this document.")

        with tab8:
            if results["sample_sizes"]:
                n_data = []
                for item in results["sample_sizes"]:
                    n_data.append(
                        {
                            "Page": item["page"],
                            "Sample Size": f"n = {item['value']:,}",
                            "Source": item.get("source", "regex"),
                        }
                    )
                st.dataframe(pd.DataFrame(n_data), use_container_width=True)
                st.caption(f"Found {len(results['sample_sizes'])} sample size(s)")
            else:
                st.info("No sample sizes found in this document.")

        # Export button
        st.divider()
        if st.button("Export Results as CSV"):
            # Create combined export data
            export_rows = []

            for item in results["effect_measures"]:
                export_rows.append(
                    {
                        "Type": item["type"],
                        "Page": item["page"],
                        "Value": item["value"],
                        "CI Lower": item["ci_lower"],
                        "CI Upper": item["ci_upper"],
                        "Adjusted": item.get("adjusted"),
                        "Adjusted For": item.get("adjusted_for"),
                        "Source": item.get("source", "regex"),
                        "Context": item["context"],
                    }
                )

            for item in results.get("beta_coefficients", []):
                export_rows.append(
                    {
                        "Type": "Beta",
                        "Page": item["page"],
                        "Value": item["value"],
                        "CI Lower": item["ci_lower"],
                        "CI Upper": item["ci_upper"],
                        "SE": item["se"],
                        "Source": item.get("source", "regex"),
                        "Context": item["context"],
                    }
                )

            for item in results["p_values"]:
                export_rows.append(
                    {
                        "Type": "P-value",
                        "Page": item["page"],
                        "Value": item["value"],
                        "Operator": item["operator"],
                        "Source": item.get("source", "regex"),
                        "Context": item["context"],
                    }
                )

            for item in results.get("mean_differences", []):
                export_rows.append(
                    {
                        "Type": "Mean Diff",
                        "Page": item["page"],
                        "Value": item["value"],
                        "CI Lower": item["ci_lower"],
                        "CI Upper": item["ci_upper"],
                        "Source": item.get("source", "regex"),
                        "Context": item["context"],
                    }
                )

            for item in results.get("standard_deviations", []):
                export_rows.append(
                    {
                        "Type": item["type"],
                        "Page": item["page"],
                        "Value": item["value"],
                        "Mean": item["mean"],
                        "Source": item.get("source", "regex"),
                        "Context": item["context"],
                    }
                )

            for item in results.get("weighted_statistics", []):
                export_rows.append(
                    {
                        "Type": f"Weighted {item['stat_type']}",
                        "Page": item["page"],
                        "Value": item["value"],
                        "Weight Method": item["weight_method"],
                        "Source": item.get("source", "regex"),
                        "Context": item["context"],
                    }
                )

            for item in results["sample_sizes"]:
                export_rows.append(
                    {
                        "Type": "Sample Size",
                        "Page": item["page"],
                        "Value": item["value"],
                        "Source": item.get("source", "regex"),
                    }
                )

            if export_rows:
                df = pd.DataFrame(export_rows)
                csv = df.to_csv(index=False)
                st.download_button(
                    "Download CSV",
                    csv,
                    file_name="extracted_statistics.csv",
                    mime="text/csv",
                )
            else:
                st.warning("No data to export.")

    else:
        st.info("Upload a PDF and click Extract to see results here.")

    # Raw text preview
    st.markdown("### Raw Text Preview")
    with st.expander("Show extracted text"):
        if "paper_text" in st.session_state:
            st.text_area(
                "Extracted Text",
                st.session_state.paper_text[:5000],
                height=300,
                disabled=True,
            )
            if len(st.session_state.paper_text) > 5000:
                st.caption(
                    f"Showing first 5,000 of {len(st.session_state.paper_text):,} characters"
                )
        else:
            st.warning("PDF text will appear here after extraction.")

st.divider()

st.markdown("### How It Works")
st.markdown("""
1. **Text Extraction**: We use PyMuPDF to extract text from your PDF
2. **Pattern Matching**: Regular expressions identify statistical measures
3. **AI Enhancement** (optional): LLM second-pass via local Ollama catches stats regex misses
4. **Deduplication**: Regex and AI results are merged, duplicates removed
5. **Structured Output**: Results are organized in tabs for easy review
6. **Export**: Download extracted statistics as CSV

**Patterns detected:**
- `aOR = 2.5 (95% CI: 1.2-3.8)` → Adjusted odds ratio
- `adjusted for age, sex, education` → Adjustment variables
- `HR = 1.45 (1.12-1.89)` → Hazard ratio
- `RR = 0.85` → Relative risk
- `β = 0.45; 95% CI: 0.12-0.78` → Beta coefficient
- `mean difference: 2.5` → Mean difference
- `3.17 (SD 1.19)` or `3.17 ± 1.19` → Standard deviation
- `weighted prevalence: 25.3%` → Weighted statistic
- `p < 0.001` or `p = 0.03` → P-values
- `n = 500` or `500 participants` → Sample sizes

Note: Extraction accuracy depends on PDF quality and formatting.
""")
