"""Builds the final PDF project report."""
import json

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ASSETS = "report/assets"
OUT = "report/Heart_Disease_Prediction_Report.pdf"

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="TitleBig", fontSize=24, leading=30, alignment=1, spaceAfter=6, textColor=colors.HexColor("#1f2d3d")))
styles.add(ParagraphStyle(name="SubTitle", fontSize=12, leading=16, alignment=1, textColor=colors.HexColor("#555555")))
styles.add(ParagraphStyle(name="H1", fontSize=16, leading=20, spaceBefore=18, spaceAfter=8, textColor=colors.HexColor("#1f2d3d")))
styles.add(ParagraphStyle(name="H2", fontSize=12.5, leading=16, spaceBefore=10, spaceAfter=6, textColor=colors.HexColor("#2c3e50")))
styles.add(ParagraphStyle(name="Body", fontSize=10.5, leading=15, spaceAfter=8))
styles.add(ParagraphStyle(name="Caption", fontSize=9, leading=12, alignment=1, textColor=colors.HexColor("#666666"), spaceAfter=14))
styles.add(ParagraphStyle(name="BulletCustom", fontSize=10.5, leading=15, leftIndent=14, spaceAfter=4, bulletIndent=0))

story = []

# ---------------- Cover ----------------
story.append(Spacer(1, 1.6 * inch))
story.append(Paragraph("Heart Disease Prediction", styles["TitleBig"]))
story.append(Paragraph("Final Project Report — AI/ML Internship", styles["SubTitle"]))
story.append(Spacer(1, 0.4 * inch))
story.append(Paragraph("Intern: Muhammad", styles["SubTitle"]))
story.append(Paragraph("Intern ID: ZYNVEX-CERT-1344", styles["SubTitle"]))
story.append(Paragraph("Project Title: Heart Disease Prediction", styles["SubTitle"]))
story.append(PageBreak())

# ---------------- Problem Statement ----------------
story.append(Paragraph("1. Problem Statement", styles["H1"]))
story.append(Paragraph(
    "Cardiovascular disease is among the leading causes of death worldwide, and "
    "early detection of at-risk patients can significantly improve treatment "
    "outcomes. The goal of this project is to build a machine learning model "
    "that predicts whether a patient is likely to have heart disease based on "
    "clinical measurements such as age, blood pressure, cholesterol level, and "
    "ECG results, and to make that model accessible through a simple, "
    "interactive web application.",
    styles["Body"],
))

# ---------------- Introduction ----------------
story.append(Paragraph("2. Introduction", styles["H1"]))
story.append(Paragraph(
    "This project uses the widely-studied UCI Cleveland Heart Disease dataset, "
    "which contains 303 patient records described by 13 clinical features and a "
    "binary target indicating the presence or absence of heart disease. "
    "The project follows a standard supervised-learning workflow:",
    styles["Body"],
))
for b in [
    "Exploratory Data Analysis (EDA) to understand feature distributions and correlations.",
    "Data preprocessing: train/test split and feature scaling.",
    "Training and comparing five classification algorithms.",
    "Hyperparameter tuning of the best-performing model with GridSearchCV.",
    "Deploying the final model in an interactive Streamlit web application.",
]:
    story.append(Paragraph(f"•  {b}", styles["BulletCustom"]))

story.append(Paragraph("2.1 Dataset Description", styles["H2"]))
feat_table_data = [
    ["Feature", "Description"],
    ["age", "Age of the patient (years)"],
    ["sex", "Sex (1 = male, 0 = female)"],
    ["cp", "Chest pain type (0-3)"],
    ["trestbps", "Resting blood pressure (mm Hg)"],
    ["chol", "Serum cholesterol (mg/dl)"],
    ["fbs", "Fasting blood sugar > 120 mg/dl (1 = true)"],
    ["restecg", "Resting electrocardiographic results (0-2)"],
    ["thalach", "Maximum heart rate achieved"],
    ["exang", "Exercise-induced angina (1 = yes)"],
    ["oldpeak", "ST depression induced by exercise relative to rest"],
    ["slope", "Slope of the peak exercise ST segment (0-2)"],
    ["ca", "Number of major vessels colored by fluoroscopy (0-4)"],
    ["thal", "Thalassemia (1 = normal, 2 = fixed defect, 3 = reversible defect)"],
    ["target", "1 = heart disease present, 0 = no heart disease"],
]
t = Table(feat_table_data, colWidths=[1.4 * inch, 4.6 * inch])
t.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2d3d")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTSIZE", (0, 0), (-1, -1), 9),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f2f2")]),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING", (0, 0), (-1, -1), 4),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
]))
story.append(t)
story.append(PageBreak())

# ---------------- EDA ----------------
story.append(Paragraph("3. Exploratory Data Analysis", styles["H1"]))
story.append(Image(f"{ASSETS}/target_distribution.png", width=4.6 * inch, height=3.07 * inch))
story.append(Paragraph("Figure 1: Distribution of the target class — moderately balanced (165 vs 138).", styles["Caption"]))

story.append(Image(f"{ASSETS}/age_distribution.png", width=4.6 * inch, height=3.07 * inch))
story.append(Paragraph("Figure 2: Age distribution split by heart disease status.", styles["Caption"]))

story.append(Image(f"{ASSETS}/correlation_heatmap.png", width=5.4 * inch, height=4.32 * inch))
story.append(Paragraph("Figure 3: Correlation heatmap of all features.", styles["Caption"]))
story.append(PageBreak())

# ---------------- Methodology ----------------
story.append(Paragraph("4. Methodology", styles["H1"]))
story.append(Paragraph(
    "The data was split into training (80%) and test (20%) sets using "
    "stratified sampling to preserve class balance. All features were "
    "standardized using <b>StandardScaler</b>. Five classifiers were trained "
    "and evaluated using 5-fold cross-validation: Logistic Regression, "
    "K-Nearest Neighbors, Decision Tree, Random Forest, and Support Vector "
    "Machine (RBF kernel). The best-performing model was then tuned further "
    "using <b>GridSearchCV</b>.",
    styles["Body"],
))

# ---------------- Results ----------------
story.append(Paragraph("5. Results", styles["H1"]))
comp_df = pd.read_csv(f"{ASSETS}/model_comparison.csv")
comp_df = comp_df.round(3)
table_data = [list(comp_df.columns)] + comp_df.values.tolist()
table_data = [[str(c) for c in row] for row in table_data]
t2 = Table(table_data, colWidths=[1.15 * inch] + [0.83 * inch] * (len(table_data[0]) - 1))
t2.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2d3d")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTSIZE", (0, 0), (-1, -1), 7.5),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f2f2")]),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
    ("ALIGN", (1, 0), (-1, -1), "CENTER"),
    ("TOPPADDING", (0, 0), (-1, -1), 4),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
]))
story.append(t2)
story.append(Paragraph("Table 1: Comparison of all trained models on the held-out test set.", styles["Caption"]))

story.append(Image(f"{ASSETS}/model_comparison.png", width=5.2 * inch, height=3.71 * inch))
story.append(Paragraph("Figure 4: Model comparison by ROC AUC.", styles["Caption"]))
story.append(PageBreak())

with open(f"{ASSETS}/final_metrics.json") as f:
    final = json.load(f)

story.append(Paragraph("5.1 Final Tuned Model", styles["H2"]))
story.append(Paragraph(
    f"The <b>Random Forest</b> classifier achieved the best ROC AUC and was "
    f"selected as the final model. After hyperparameter tuning with "
    f"GridSearchCV, the best parameters found were: "
    f"<b>{final['best_params']}</b>.",
    styles["Body"],
))
metrics_table = [["Metric", "Value"]] + [[k, f"{v:.3f}"] for k, v in final["metrics"].items()]
t3 = Table(metrics_table, colWidths=[2 * inch, 2 * inch])
t3.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2d3d")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTSIZE", (0, 0), (-1, -1), 9.5),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f2f2")]),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
    ("TOPPADDING", (0, 0), (-1, -1), 4),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
]))
story.append(t3)
story.append(Spacer(1, 12))

story.append(Image(f"{ASSETS}/confusion_matrix.png", width=3.2 * inch, height=3.2 * inch))
story.append(Paragraph("Figure 5: Confusion matrix of the tuned Random Forest on the test set.", styles["Caption"]))

story.append(Image(f"{ASSETS}/roc_curve.png", width=4 * inch, height=3.33 * inch))
story.append(Paragraph("Figure 6: ROC curve of the tuned Random Forest.", styles["Caption"]))

story.append(Image(f"{ASSETS}/feature_importance.png", width=4.9 * inch, height=3.5 * inch))
story.append(Paragraph("Figure 7: Feature importance ranking from the Random Forest model.", styles["Caption"]))
story.append(PageBreak())

# ---------------- App UI ----------------
story.append(Paragraph("6. Application UI", styles["H1"]))
story.append(Paragraph(
    "The final model is served through a Streamlit web application that lets "
    "a user enter a patient's clinical details and receive an instant "
    "prediction with an estimated probability of heart disease.",
    styles["Body"],
))
story.append(Image(f"{ASSETS}/app_screenshot_1_form.png", width=4.3 * inch, height=4.7 * inch))
story.append(Paragraph("Figure 8: Application input form.", styles["Caption"]))
story.append(PageBreak())
story.append(Image(f"{ASSETS}/app_screenshot_2_result.png", width=4.3 * inch, height=4.7 * inch))
story.append(Paragraph("Figure 9: Application prediction result.", styles["Caption"]))
story.append(PageBreak())

# ---------------- Future Approach ----------------
story.append(Paragraph("7. Future Approach", styles["H1"]))
for b in [
    "Collect a larger and more demographically diverse dataset to improve generalization beyond the 303-record Cleveland cohort.",
    "Add model explainability (e.g. SHAP values) so each individual prediction can be interpreted feature-by-feature.",
    "Experiment with gradient boosting methods (XGBoost, LightGBM) to see if they outperform Random Forest.",
    "Wrap the model in a REST API (e.g. FastAPI) so it can be integrated into other clinical or research systems.",
    "Add authentication and prediction history/logging for longitudinal patient tracking.",
    "Deploy with continuous monitoring for model drift as new patient data becomes available.",
]:
    story.append(Paragraph(f"•  {b}", styles["BulletCustom"]))

story.append(Spacer(1, 20))
story.append(Paragraph("8. Conclusion", styles["H1"]))
story.append(Paragraph(
    "The final tuned Random Forest model achieved an accuracy of "
    f"{final['metrics']['Accuracy']:.1%} and an ROC AUC of "
    f"{final['metrics']['ROC AUC']:.3f} on the held-out test set, "
    "demonstrating that clinical measurements can meaningfully predict the "
    "presence of heart disease. The accompanying Streamlit application makes "
    "this model accessible through a simple, user-friendly interface, "
    "fulfilling the goals of this internship project.",
    styles["Body"],
))
story.append(Spacer(1, 10))
story.append(Paragraph(
    "<i>Disclaimer: This project is for educational purposes only and is not "
    "intended for real medical diagnosis.</i>",
    styles["Body"],
))

doc = SimpleDocTemplate(
    OUT, pagesize=letter,
    leftMargin=0.75 * inch, rightMargin=0.75 * inch,
    topMargin=0.75 * inch, bottomMargin=0.75 * inch,
    title="Heart Disease Prediction - Final Project Report",
)
doc.build(story)
print("Report written to", OUT)
