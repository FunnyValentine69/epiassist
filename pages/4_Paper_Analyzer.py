"""Paper Analyzer page for extracting statistics from PDFs."""

import streamlit as st
import pandas as pd

from core.paper_parser import (
    extract_text_from_pdf,
    find_odds_ratios,
    find_confidence_intervals,
    find_p_values,
    find_sample_sizes,
)

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

    st.markdown("### What We Extract")
    st.markdown("""
    - Odds Ratios (OR, aOR)
    - Confidence Intervals (95% CI)
    - P-values
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
                odds_ratios = []
                confidence_intervals = []
                p_values = []
                sample_sizes = []

                for page_num, page_text in pages:
                    odds_ratios.extend(find_odds_ratios(page_text, page=page_num))
                    confidence_intervals.extend(find_confidence_intervals(page_text, page=page_num))
                    p_values.extend(find_p_values(page_text, page=page_num))
                    sample_sizes.extend(find_sample_sizes(page_text, page=page_num))

                # Store results
                st.session_state.paper_results = {
                    "odds_ratios": odds_ratios,
                    "confidence_intervals": confidence_intervals,
                    "p_values": p_values,
                    "sample_sizes": sample_sizes,
                }

            st.success("Extraction complete.")
        except Exception as e:
            st.error(f"Error processing PDF: {e}")

    # Display results if available
    if "paper_results" in st.session_state:
        results = st.session_state.paper_results

        # Odds Ratios Tab
        tab1, tab2, tab3, tab4 = st.tabs(
            ["Odds Ratios", "Confidence Intervals", "P-values", "Sample Sizes"]
        )

        with tab1:
            if results["odds_ratios"]:
                or_data = []
                for item in results["odds_ratios"]:
                    or_data.append(
                        {
                            "Page": item["page"],
                            "OR Value": item["value"],
                            "CI Lower": item["ci_lower"] or "-",
                            "CI Upper": item["ci_upper"] or "-",
                            "Context": item["context"][:80] + "...",
                        }
                    )
                st.dataframe(pd.DataFrame(or_data), width="stretch")
                st.caption(f"Found {len(results['odds_ratios'])} odds ratio(s)")
            else:
                st.info("No odds ratios found in this document.")

        with tab2:
            if results["confidence_intervals"]:
                ci_data = []
                for item in results["confidence_intervals"]:
                    ci_data.append(
                        {
                            "Page": item["page"],
                            "Level": f"{item['level']}%",
                            "Lower": item["lower"],
                            "Upper": item["upper"],
                            "Context": item["context"][:60] + "...",
                        }
                    )
                st.dataframe(pd.DataFrame(ci_data), width="stretch")
                st.caption(
                    f"Found {len(results['confidence_intervals'])} confidence interval(s)"
                )
            else:
                st.info("No confidence intervals found in this document.")

        with tab3:
            if results["p_values"]:
                p_data = []
                for item in results["p_values"]:
                    p_data.append(
                        {
                            "Page": item["page"],
                            "P-value": item["value"],
                            "Operator": item["operator"],
                            "Significant (α=0.05)": "Yes" if item["value"] < 0.05 else "No",
                            "Context": item["context"][:60] + "...",
                        }
                    )
                st.dataframe(pd.DataFrame(p_data), width="stretch")
                st.caption(f"Found {len(results['p_values'])} p-value(s)")
            else:
                st.info("No p-values found in this document.")

        with tab4:
            if results["sample_sizes"]:
                n_data = []
                for item in results["sample_sizes"]:
                    n_data.append(
                        {
                            "Page": item["page"],
                            "Sample Size": f"n = {item['value']:,}",
                        }
                    )
                st.dataframe(pd.DataFrame(n_data), width="stretch")
                st.caption(f"Found {len(results['sample_sizes'])} sample size(s)")
            else:
                st.info("No sample sizes found in this document.")

        # Export button
        st.divider()
        if st.button("Export Results as CSV"):
            # Create combined export data
            export_rows = []

            for item in results["odds_ratios"]:
                export_rows.append(
                    {
                        "Type": "Odds Ratio",
                        "Page": item["page"],
                        "Value": item["value"],
                        "CI Lower": item["ci_lower"],
                        "CI Upper": item["ci_upper"],
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
                        "Context": item["context"],
                    }
                )

            for item in results["sample_sizes"]:
                export_rows.append(
                    {
                        "Type": "Sample Size",
                        "Page": item["page"],
                        "Value": item["value"],
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
3. **Structured Output**: Results are organized in tabs for easy review
4. **Export**: Download extracted statistics as CSV

**Patterns detected:**
- `OR = 2.5 (95% CI: 1.2-3.8)` → Odds ratio with confidence interval
- `p < 0.001` or `p = 0.03` → P-values
- `n = 500` or `500 participants` → Sample sizes

Note: Extraction accuracy depends on PDF quality and formatting.
""")
