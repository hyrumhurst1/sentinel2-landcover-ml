"""
build_submission.py
-------------------
Turns paper/paper_filled.md into a single self-contained, print-ready HTML
manuscript (paper/paper_submission.html): figures embedded as base64, references
inlined, draft notes stripped, clean academic CSS. Open it in a browser and
print to PDF (Ctrl+P -> Save as PDF) for upload to a preprint server.

Run:  .venv/Scripts/python.exe src/build_submission.py
"""
import base64
import os
import markdown  # uv pip install markdown

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(ROOT, "paper", "paper_filled.md")
FIGS = os.path.join(ROOT, "figures")
OUT = os.path.join(ROOT, "paper", "paper_submission.html")

FIGURES = [
    ("fig1_confusion.png",
     "Figure 1. Cross-validated (out-of-fold) confusion matrix for the best model."),
    ("fig2_per_class_f1.png",
     "Figure 2. Per-class F1 for the best model."),
    ("fig3_model_comparison.png",
     "Figure 3. Cross-validated accuracy of all classifiers on identical features, "
     "with standard-deviation error bars and the majority-class baseline."),
    ("fig4_feature_importance.png",
     "Figure 4. The 15 most informative features by random-forest importance."),
]

REFERENCES = [
    "Helber, P., Bischke, B., Dengel, A., & Borth, D. (2019). EuroSAT: A novel "
    "dataset and deep learning benchmark for land use and land cover "
    "classification. <i>IEEE J. Sel. Top. Appl. Earth Obs. Remote Sens.</i>, "
    "12(7), 2217-2226.",
    "Breiman, L. (2001). Random forests. <i>Machine Learning</i>, 45(1), 5-32.",
    "Friedman, J. H. (2001). Greedy function approximation: a gradient boosting "
    "machine. <i>The Annals of Statistics</i>, 29(5), 1189-1232.",
    "Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. "
    "<i>J. Mach. Learn. Res.</i>, 12, 2825-2830.",
    "Cohen, J. (1960). A coefficient of agreement for nominal scales. <i>Educ. "
    "Psychol. Meas.</i>, 20(1), 37-46.",
    "Drusch, M., et al. (2012). Sentinel-2: ESA's optical high-resolution mission "
    "for GMES operational services. <i>Remote Sens. Environ.</i>, 120, 25-36.",
    "McFeeters, S. K. (1996). The use of the Normalized Difference Water Index "
    "(NDWI) in the delineation of open water features. <i>Int. J. Remote "
    "Sens.</i>, 17(7), 1425-1432.",
]


def img_data_uri(path):
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/png;base64,{b64}"


# --- Load and clean the markdown -----------------------------------------
md = open(PAPER, encoding="utf-8").read()

# Drop the leading "> Draft ..." note: split on the first horizontal rule.
header, body = md.split("\n---\n", 1)
header = "\n".join(l for l in header.splitlines() if not l.strip().startswith(">"))

# Cut the markdown "## References" tail (we inline our own formatted list).
body = body.split("## References", 1)[0].rstrip()

# Insert a Figures section just before Data and Code Availability.
fig_html = '<h2>Figures</h2>\n'
for fname, cap in FIGURES:
    fig_html += (f'<figure><img src="{img_data_uri(os.path.join(FIGS, fname))}" '
                 f'alt="{cap}"><figcaption>{cap}</figcaption></figure>\n')

# Convert the (cleaned) markdown body to HTML, then splice in figures + refs.
body_html = markdown.markdown(body, extensions=["extra", "sane_lists"])
body_html = body_html.replace(
    "<h2>Data and Code Availability</h2>",
    fig_html + "<h2>Data and Code Availability</h2>", 1)

header_html = markdown.markdown(header, extensions=["extra"])

refs_html = "<h2>References</h2>\n<ol class='refs'>\n" + \
    "\n".join(f"<li>{r}</li>" for r in REFERENCES) + "\n</ol>"

CSS = """
@page { size: letter; margin: 1in; }
body { font: 11pt/1.5 Georgia, 'Times New Roman', serif; color: #111;
       max-width: 7.0in; margin: 0 auto; padding: 24px; }
h1 { font-size: 19pt; text-align: center; line-height: 1.25; margin: 0 0 6px; }
h1 + p { text-align: center; color: #333; margin: 0 0 18px; }
h2 { font-size: 13pt; border-bottom: 1px solid #ccc; padding-bottom: 3px;
     margin: 22px 0 8px; page-break-after: avoid; }
h3 { font-size: 11.5pt; margin: 16px 0 6px; page-break-after: avoid; }
p { text-align: justify; margin: 0 0 10px; }
code { background: #f3f3f3; padding: 0 3px; border-radius: 3px; font-size: 0.9em; }
figure { margin: 16px 0; text-align: center; page-break-inside: avoid; }
figure img { max-width: 100%; height: auto; border: 1px solid #e3e3e3; }
figcaption { font-size: 9.5pt; color: #444; margin-top: 6px; text-align: left; }
ol.refs li { margin-bottom: 7px; font-size: 10pt; }
"""

html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Land-Cover and Surface-Water Classification on EuroSAT</title>
<style>{CSS}</style></head>
<body>
{header_html}
{body_html}
{refs_html}
</body></html>"""

with open(OUT, "w", encoding="utf-8") as f:
    f.write(html)
print("Wrote", OUT)
print("Embedded", len(FIGURES), "figures and", len(REFERENCES), "references.")
