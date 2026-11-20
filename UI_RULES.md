# UI Constraints (Streamlit)
- Theme: Dark mode by default ([theme] base="dark" in .streamlit/config.toml).
- Layout: Wide layout. Sidebar for uploading transaction CSVs or adjusting risk thresholds.
- Visuals: Use Plotly or Altair for interactive charts (no static Matplotlib plots in the UI).
- Explainability Panel: Every flagged transaction MUST have a dedicated box displaying the DeepSeek-generated investigator report.
- Metrics: Display AML metrics at the top: Precision, Recall, False Positive Rate (FPR), and Alert Volume.
