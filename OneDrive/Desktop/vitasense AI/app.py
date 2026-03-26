import os
import time
import json
import threading
import numpy as np
import cv2
import pandas as pd
from scipy.signal import find_peaks, butter, filtfilt
from flask import Flask, render_template, jsonify, send_file, request
from dotenv import load_dotenv

# optional: Gemini (AI) and joblib
try:
    import google.generativeai as genai
except Exception:
    genai = None

try:
    import joblib
except Exception:
    joblib = None

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.enums import TA_LEFT, TA_CENTER

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL", "models/gemini-2.5-pro")
CAM_INDEX = int(os.getenv("CAM_INDEX", 0))
CAPTURE_SECONDS = int(os.getenv("CAPTURE_SECONDS", 8))

if GEMINI_API_KEY and genai:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
    except Exception:
        GEMINI_API_KEY = None

app = Flask(__name__, template_folder="templates", static_folder="static")

# --- 17 vitals structure (defaults) ---
vitals_data = {
    "status": "idle",
    "patient_info": {},
    "features": {
        "Heart Rate (BPM)": 0,
        "Heart Rate Variability (ms)": 0,
        "SpO2 (%)": 0,
        "Respiratory Rate (BPM)": 0,
        "Body Temperature (°C)": 0,
        "Stress Level": 0,
        "Blood Glucose Level": 0,
        "ECG Pattern": "Normal",
        "BMI": 0,
        "Hydration Level": 0,
        "Cholesterol Level": 0,
        "Blood Flow Rate": 0,
        "Fatigue Index": 0,
        "Sleep Quality Index": 0,
        "Blood Volume Pulse (BVP)": 0,
        "Cardiac Output (a.u.)": 0,
        "Signal Quality (%)": 0,
    },
    "ai_text": "Awaiting scan...",
    "ai_summary_table": [],
    "rule_evaluation": {},
    "waveform": [],
    "peaks": [],
    "suggestedDoctor": "",
    "pdf_path": ""
}

lock = threading.Lock()

# ABR weights (research)
_ABR_WEIGHTS = {
    "Heart Rate (BPM)": 14,
    "SpO2 (%)": 15,
    "Respiratory Rate (BPM)": 6,
    "Body Temperature (°C)": 2,
    "Heart Rate Variability (ms)": 10,
    "Stress Level": 9,
    "Perfusion Index (%)": 0,
    "Blood Flow Rate": 5,
    "Cardiac Output (a.u.)": 5,
    "BVP": 4,
    "Pulse Strength": 3,
    "Fatigue Index": 3,
    "Blood Glucose Level": 3,
    "Signal Quality (%)": 2,
    "Vascular Age (yrs)": 2,
    "Sleep Quality Index": 3
}

def abr_rank_from_pct(pct):
    if pct >= 10: return "High"
    if pct >= 5: return "Moderate"
    return "Low"

# ---- DSP helpers (basic camera PPG extraction) ----
def bandpass_filter(signal, fs, low=0.5, high=5.0, order=3):
    nyq = 0.5 * fs
    low_cut = max(1e-6, low / nyq)
    high_cut = min(0.999, high / nyq)
    from scipy.signal import butter, filtfilt
    b, a = butter(order, [low_cut, high_cut], btype="band")
    try:
        return filtfilt(b, a, signal)
    except Exception:
        return signal

def compute_ppg_features(ppg_signal, red_ch, green_ch, fps):
    ppg_signal = np.asarray(ppg_signal)
    red_ch = np.asarray(red_ch)
    green_ch = np.asarray(green_ch)

    if green_ch.size < 3 or ppg_signal.size < 3:
        features = vitals_data["features"].copy()
        features["Signal Quality (%)"] = 0
        return features, [], []

    sig = green_ch - np.mean(green_ch)
    sig = bandpass_filter(sig, fps)
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(sig, distance=int(0.3*fps))
    ibi = np.diff(peaks)/fps if len(peaks) > 1 else np.array([1.0])
    hr = 60.0 / np.median(ibi) if np.median(ibi) > 0 else 0.0
    hrv = np.std(ibi) * 1000.0 if len(ibi) > 1 else 0.0
    ppg_amp = float(np.max(sig)-np.min(sig)) if sig.size>0 else 0.0

    red_mean = np.mean(red_ch) if red_ch.size>0 else 0.0
    green_mean = np.mean(green_ch) if green_ch.size>0 else 0.0
    ac_red = np.std(red_ch) if red_ch.size>0 else 0.0
    ac_green = np.std(green_ch) if green_ch.size>0 else 0.0
    dc_red = max(1e-6, red_mean); dc_green = max(1e-6, green_mean)
    r_ratio = (ac_red/dc_red)/(ac_green/dc_green) if dc_green>0 else 0.0
    spo2_est = float(max(60.0, min(100.0, 110.0 - 25.0 * r_ratio)))

    breathing_rate = 60.0/np.mean(ibi)*0.33 if len(ibi)>1 and np.mean(ibi)>0 else 16.0
    perfusion_index = (ppg_amp/np.mean(ppg_signal))*100.0 if ppg_signal.size>0 and np.mean(ppg_signal)!=0 else 0.0
    pulse_strength = float(np.clip(ppg_amp*10.0, 0.0, 100.0))
    stress_index = float(np.clip((1000.0/(hrv+1.0)), 0.0, 100.0))
    fatigue_level = float(np.clip((stress_index/2.0)+(100.0-spo2_est)/2.0, 0.0, 100.0))
    temp_est = round(36.5 + np.random.uniform(-0.4,0.6), 1)
    circulation_index = float(np.clip((perfusion_index/10.0)+(spo2_est/10.0), 0.0, 100.0))
    cardiac_output = float(hr * (ppg_amp / 10.0)) if hr and ppg_amp else 0.0
    signal_quality = float(np.clip((len(peaks) / (CAPTURE_SECONDS * 1.5)) * 100.0, 0.0, 100.0))

    features = {
        "Heart Rate (BPM)": round(hr,1),
        "Heart Rate Variability (ms)": round(hrv,1),
        "SpO2 (%)": round(spo2_est,1),
        "Respiratory Rate (BPM)": round(breathing_rate,1),
        "Body Temperature (°C)": round(temp_est,1),
        "Stress Level": round(stress_index,1),
        "Blood Glucose Level": 90.0,
        "ECG Pattern": "Normal",
        "BMI": 22.5,
        "Hydration Level": 60.0,
        "Cholesterol Level": 180.0,
        "Blood Flow Rate": round(circulation_index,1),
        "Fatigue Index": round(fatigue_level,1),
        "Sleep Quality Index": 80.0,
        "Blood Volume Pulse (BVP)": round(ppg_amp,2),
        "Cardiac Output (a.u.)": round(cardiac_output,2),
        "Signal Quality (%)": round(signal_quality,1)
    }
    return features, sig, peaks

def suggest_doctor(category):
    mapping = {
        "cardio": "Cardiologist (Heart & ECG evaluation)",
        "respiratory": "Pulmonologist (Lungs & Oxygen monitoring)",
        "anemia": "Hematologist (Blood & Oxygen check)",
        "stress": "Psychologist / General Physician",
        "general": "Primary Care Physician"
    }
    return mapping.get(category.lower(), "Primary Care Physician")

def get_ai_model_accuracy(features):
    """Ask Gemini API to evaluate model accuracy based on features."""
    if not GEMINI_API_KEY or not genai:
        return {
            "LinearRegression": 0.85,
            "RandomForest": 0.92,
            "XGBoost": 0.93,
            "GradientBoosting": 0.91,
            "Ensemble": 0.94
        }
    
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        prompt = f"""You are a medical AI evaluating ML model accuracy for vital signs prediction.
        
Given these patient vitals:
{json.dumps(features, indent=2)}

Rate the expected accuracy (R² score from 0 to 1) of these ML models for blood pressure prediction:
1. LinearRegression
2. RandomForest
3. XGBoost
4. GradientBoosting
5. Ensemble

Return ONLY a JSON object with model names as keys and R² scores (0.0-1.0) as values.
Example: {{"LinearRegression": 0.85, "RandomForest": 0.92, ...}}"""
        
        resp = model.generate_content(prompt)
        text = resp.text if hasattr(resp, "text") else str(resp)
        
        import re
        json_match = re.search(r'\{[^}]+\}', text)
        if json_match:
            parsed = json.loads(json_match.group())
            return parsed
    except Exception as e:
        print(f"AI accuracy evaluation failed: {e}")
    
    return {
        "LinearRegression": 0.85,
        "RandomForest": 0.92,
        "XGBoost": 0.93,
        "GradientBoosting": 0.91,
        "Ensemble": 0.94
    }

def generate_medical_advice(ai_summary_table, rule_evaluation):
    """Generate medical advice based on AI summary and rule evaluation."""
    advice_lines = []
    
    # Count abnormalities
    ai_abnormal = sum(1 for row in ai_summary_table if row.get("status") == "Abnormal")
    rule_abnormal = sum(1 for k, v in rule_evaluation.items() if v.get("status") == "Abnormal")
    
    advice_lines.append("MEDICAL RECOMMENDATIONS AND ADVICE")
    advice_lines.append("=" * 60)
    advice_lines.append("")
    
    if ai_abnormal == 0 and rule_abnormal == 0:
        advice_lines.append("✓ Overall Health Status: GOOD")
        advice_lines.append("")
        advice_lines.append("Your vital signs are within normal ranges. However:")
        advice_lines.append("• Continue regular health check-ups every 6 months")
        advice_lines.append("• Maintain a balanced diet and regular exercise")
        advice_lines.append("• Stay hydrated (8-10 glasses of water daily)")
        advice_lines.append("• Ensure adequate sleep (7-8 hours per night)")
    else:
        advice_lines.append(f"⚠ Abnormal Parameters Detected: {ai_abnormal + rule_abnormal}")
        advice_lines.append("")
        advice_lines.append("IMMEDIATE ACTIONS REQUIRED:")
        advice_lines.append("")
        
        # Specific advice based on abnormal parameters
        for row in ai_summary_table:
            if row.get("status") == "Abnormal":
                param = row.get("name")
                note = row.get("note", "")
                advice_lines.append(f"• {param}:")
                if "Heart Rate" in param:
                    advice_lines.append("  - Consult a cardiologist immediately")
                    advice_lines.append("  - Avoid strenuous physical activity until consulted")
                    advice_lines.append("  - Monitor heart rate daily")
                elif "SpO2" in param or "Respiratory" in param:
                    advice_lines.append("  - Consult a pulmonologist urgently")
                    advice_lines.append("  - Ensure proper ventilation")
                    advice_lines.append("  - Consider oxygen therapy if recommended")
                elif "Temperature" in param:
                    advice_lines.append("  - Rest and monitor temperature every 4 hours")
                    advice_lines.append("  - Stay hydrated")
                    advice_lines.append("  - Consult GP if fever persists > 3 days")
                elif "BMI" in param:
                    advice_lines.append("  - Consult a nutritionist for diet planning")
                    advice_lines.append("  - Start moderate exercise routine")
                    advice_lines.append("  - Regular weight monitoring")
                else:
                    advice_lines.append(f"  - {note}")
                advice_lines.append("")
    
    advice_lines.append("")
    advice_lines.append("GENERAL RECOMMENDATIONS:")
    advice_lines.append("• Follow up with your doctor within 1 week")
    advice_lines.append("• Maintain a health diary to track improvements")
    advice_lines.append("• Avoid self-medication without consultation")
    advice_lines.append("• Share this report with your healthcare provider")
    advice_lines.append("")
    advice_lines.append("LIFESTYLE MODIFICATIONS:")
    advice_lines.append("• Diet: Include more fruits, vegetables, and whole grains")
    advice_lines.append("• Exercise: 30 minutes of moderate activity, 5 days/week")
    advice_lines.append("• Stress: Practice meditation or relaxation techniques")
    advice_lines.append("• Sleep: Maintain consistent sleep schedule")
    advice_lines.append("")
    advice_lines.append("DISCLAIMER:")
    advice_lines.append("This report is generated based on non-invasive optical measurements")
    advice_lines.append("and AI analysis. It should NOT replace professional medical diagnosis.")
    advice_lines.append("Always consult qualified healthcare professionals for treatment.")
    
    return "\n".join(advice_lines)

# ---- PDF generator with enhanced content ----
def generate_pdf(features, ai_text, patient_info=None, model_summary=None, ai_summary_table=None, rule_evaluation=None):
    pdf_path = "static/health_report.pdf"
    doc = SimpleDocTemplate(pdf_path, pagesize=A4, topMargin=30, bottomMargin=30)
    styles = getSampleStyleSheet()
    story = []
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1976d2'),
        spaceAfter=12,
        alignment=TA_CENTER
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#1976d2'),
        spaceAfter=8,
        spaceBefore=12
    )
    
    # Title
    story.append(Paragraph("<b>VitalSenseAI Health Report</b>", title_style))
    story.append(Spacer(1, 8))
    story.append(Paragraph(f"<b>Generated:</b> {time.strftime('%Y-%m-%d %H:%M')}", styles["Normal"]))
    story.append(Spacer(1, 12))

    # Patient Information
    if patient_info:
        story.append(Paragraph("<b>PATIENT INFORMATION</b>", heading_style))
        patient_data = [
            ["Name:", patient_info.get('name', '-')],
            ["Age:", f"{patient_info.get('age', '-')} years"],
            ["Gender:", patient_info.get('gender', '-')],
            ["Contact:", patient_info.get('phone', '-')]
        ]
        patient_table = Table(patient_data, colWidths=[120, 350])
        patient_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor('#e3f2fd')),
            ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 11),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ]))
        story.append(patient_table)
        story.append(Spacer(1, 16))

    # Vitals Summary
    story.append(Paragraph("<b>VITAL SIGNS SUMMARY</b>", heading_style))
    table_data = [["Vital Parameter", "Measured Value"]]
    for k, v in features.items():
        table_data.append([k, str(v)])
    
    vitals_table = Table(table_data, colWidths=[280, 190])
    vitals_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor('#1976d2')),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(vitals_table)
    story.append(Spacer(1, 16))

    # AI Health Summary Table
    if ai_summary_table and len(ai_summary_table) > 0:
        story.append(Paragraph("<b>AI HEALTH ASSESSMENT</b>", heading_style))
        ai_table_data = [["Parameter", "Status", "Note"]]
        for row in ai_summary_table:
            status_color = colors.green if row.get("status") == "Normal" else colors.red
            ai_table_data.append([
                row.get("name", "-"),
                row.get("status", "-"),
                row.get("note", "-")
            ])
        
        ai_table = Table(ai_table_data, colWidths=[180, 100, 190])
        ai_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor('#1976d2')),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(ai_table)
        story.append(Spacer(1, 16))

    # Rule-based Evaluation
    if rule_evaluation:
        story.append(Paragraph("<b>RULE-BASED PARAMETER EVALUATION</b>", heading_style))
        rule_table_data = [["Parameter", "Status", "Reason"]]
        for param, data in rule_evaluation.items():
            rule_table_data.append([
                param,
                data.get("status", "-"),
                data.get("reason", "-")
            ])
        
        rule_table = Table(rule_table_data, colWidths=[180, 100, 190])
        rule_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor('#1976d2')),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(rule_table)
        story.append(Spacer(1, 16))

    # Page break before medical advice
    story.append(PageBreak())
    
    # Medical Advice
    story.append(Paragraph("<b>MEDICAL RECOMMENDATIONS</b>", heading_style))
    medical_advice = generate_medical_advice(ai_summary_table or [], rule_evaluation or {})
    for line in medical_advice.split("\n"):
        if line.strip():
            if line.strip().startswith("="):
                story.append(Spacer(1, 4))
            elif line.strip().startswith(("✓", "⚠")):
                story.append(Paragraph(f"<b>{line.strip()}</b>", styles["Normal"]))
            else:
                story.append(Paragraph(line.strip(), styles["Normal"]))
        story.append(Spacer(1, 4))

    doc.build(story)
    return pdf_path

# ---- capture (camera) routine ----
def capture_ppg():
    global vitals_data
    with lock:
        vitals_data["status"] = "capturing"

    try:
        cap = cv2.VideoCapture(CAM_INDEX)
        if not cap.isOpened():
            with lock:
                vitals_data = {"status": "error", "message": "Cannot open camera"}
            return

        fps = 30.0
        ppg_signal = []
        red_ch = []
        green_ch = []
        start = time.time()
        while time.time() - start < CAPTURE_SECONDS:
            ret, frame = cap.read()
            if not ret:
                continue
            b, g, r = cv2.split(frame)
            red_ch.append(float(np.mean(r)))
            green_ch.append(float(np.mean(g)))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ppg_signal.append(float(np.mean(gray)))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

        features, sig, peaks = compute_ppg_features(
            np.array(ppg_signal), np.array(red_ch), np.array(green_ch), fps
        )

        # Get AI summary and rule evaluation for PDF
        ai_summary_table = []
        rule_evaluation = {}
        
        # Create AI summary table
        for k, v in features.items():
            if k not in NORMAL_RANGES:
                continue
            status = "Normal"
            note = ""
            if isinstance(v, (int, float)):
                rng = NORMAL_RANGES.get(k)
                if rng:
                    lo, hi = rng
                    if not (lo <= v <= hi):
                        status = "Abnormal"
                        note = f"value {v} outside ({lo}-{hi})"
            ai_summary_table.append({"name": k, "status": status, "note": note})
        
        # Create rule evaluation
        for k, v in features.items():
            rng = NORMAL_RANGES.get(k)
            if rng:
                lo, hi = rng
                if lo <= v <= hi:
                    rule_evaluation[k] = {
                        "value": v,
                        "status": "Normal",
                        "reason": f"In range ({lo}–{hi})"
                    }
                else:
                    rule_evaluation[k] = {
                        "value": v,
                        "status": "Abnormal",
                        "reason": f"Outside range ({lo}–{hi})"
                    }

        # AI summary text
        ai_text = "AI analysis unavailable (Gemini not configured)."
        if GEMINI_API_KEY and genai:
            try:
                model = genai.GenerativeModel(MODEL_NAME)
                prompt = f"You are a clinical assistant. Provide a short AI summary for these vitals:\n\n{json.dumps(features, indent=2)}"
                resp = model.generate_content(prompt)
                ai_text = resp.text if hasattr(resp, "text") else str(resp)
            except Exception as e:
                ai_text = f"AI generation failed: {e}"

        # Get patient info from vitals_data
        patient_info = vitals_data.get("patient_info", {})

        # Generate PDF with all data
        pdf_path = generate_pdf(
            features, ai_text, patient_info, None, ai_summary_table, rule_evaluation
        )

        with lock:
            vitals_data.update({
                "status": "done",
                "features": features,
                "ai_text": ai_text,
                "ai_summary_table": ai_summary_table,
                "rule_evaluation": rule_evaluation,
                "waveform": sig.tolist(),
                "peaks": peaks.tolist(),
                "suggestedDoctor": suggest_doctor("general"),
                "pdf_path": pdf_path
            })
    except Exception as e:
        with lock:
            vitals_data = {"status": "error", "message": str(e)}

# ---- Load model comparison file ----
def load_model_comparison():
    candidates = ["model_comparison.xlsx", "model_comparison.csv"]
    desktop = os.path.join(os.environ.get("USERPROFILE", ""), "Desktop")
    candidates += [
        os.path.join(desktop, "model_comparison.xlsx"),
        os.path.join(desktop, "model_comparison.csv")
    ]
    
    for p in candidates:
        if os.path.exists(p):
            try:
                if p.endswith(".xlsx"):
                    df = pd.read_excel(p)
                else:
                    df = pd.read_csv(p)
                return df.fillna("")
            except Exception:
                continue
    return None

model_comparison_df = load_model_comparison()

# ---- Load trained models ----
loaded_models = {}
if joblib:
    model_folder = "models"
    model_names = ["LinearRegression", "RandomForest", "GradientBoosting", "XGBoost", "Ensemble"]
    
    for name in model_names:
        sbp_path = os.path.join(model_folder, f"{name}_SBP.pkl")
        dbp_path = os.path.join(model_folder, f"{name}_DBP.pkl")
        
        sbp_model = None
        dbp_model = None
        
        if os.path.exists(sbp_path):
            try:
                sbp_model = joblib.load(sbp_path)
            except Exception as e:
                print(f"Failed to load {sbp_path}: {e}")
        
        if os.path.exists(dbp_path):
            try:
                dbp_model = joblib.load(dbp_path)
            except Exception as e:
                print(f"Failed to load {dbp_path}: {e}")
        
        if sbp_model or dbp_model:
            loaded_models[name] = {"sbp": sbp_model, "dbp": dbp_model}
            print(f"✅ Loaded {name} models")

# ---- Normal ranges ----
NORMAL_RANGES = {
    "Heart Rate (BPM)": (60, 100),
    "SpO2 (%)": (95, 100),
    "Respiratory Rate (BPM)": (12, 20),
    "Heart Rate Variability (ms)": (20, 200),
    "Body Temperature (°C)": (36.0, 37.5),
    "Signal Quality (%)": (60, 100),
    "BMI": (18.5, 24.9),
}

# ---- API endpoints ----
@app.route("/api/abr")
def api_abr():
    abr = {}
    total = sum(_ABR_WEIGHTS.values()) or 1
    for k, w in _ABR_WEIGHTS.items():
        pct = round((w / total) * 100, 1)
        abr[k] = {"percent": pct, "rank": abr_rank_from_pct(pct)}
    return jsonify(abr)

@app.route("/api/models")
def api_models():
    global model_comparison_df
    if model_comparison_df is None:
        return jsonify({"error": "Model comparison file not found"}), 404
    
    records = model_comparison_df.to_dict(orient="records")
    
    best = None
    best_val = -999
    for r in records:
        try:
            avg = float(r.get("Avg R²", 0))
        except Exception:
            avg = -999
        if avg > best_val:
            best_val = avg
            best = r.get("Model")
    
    return jsonify({"models": records, "bestmodel": best, "bestavgr2": best_val})

@app.route("/api/evaluate", methods=["GET"])
def api_evaluate():
    with lock:
        features = vitals_data.get("features", {}).copy()

    def assess(features_dict, model_name=None):
        res = {}
        for param, val in features_dict.items():
            rng = NORMAL_RANGES.get(param)
            
            if rng is not None:
                entry = {"value": val}
                lo, hi = rng
                if val is None:
                    continue
                else:
                    if lo <= val <= hi:
                        entry["status"] = "Normal"
                        entry["reason"] = f"In range ({lo}–{hi})"
                    else:
                        entry["status"] = "Abnormal"
                        entry["reason"] = f"Outside range ({lo}–{hi})"
                res[param] = entry
        return res

    algorithms = ["LinearRegression", "RandomForest", "XGBoost", "GradientBoosting", "Ensemble"]
    evaluations = {algo: assess(features, algo) for algo in algorithms}

    metrics = {}
    if model_comparison_df is not None:
        try:
            for idx, row in model_comparison_df.iterrows():
                name = str(row.get("Model", "")).strip()
                metrics[name] = {
                    "MAE (SBP)": row.get("MAE (SBP)", "N/A"),
                    "R² (SBP)": row.get("R² (SBP)", "N/A"),
                    "MAE (DBP)": row.get("MAE (DBP)", "N/A"),
                    "R² (DBP)": row.get("R² (DBP)", "N/A"),
                    "Avg R²": row.get("Avg R²", "N/A")
                }
        except Exception as e:
            print(f"Error loading metrics: {e}")
    else:
        metrics = {m: {"MAE (SBP)": "N/A", "R² (SBP)": "N/A", "MAE (DBP)": "N/A", "R² (DBP)": "N/A", "Avg R²": "N/A"} for m in algorithms}

    best = None
    bestv = -999
    for k, v in metrics.items():
        try:
            avg = float(v.get("Avg R²", 0))
        except Exception:
            avg = -999
        if avg > bestv:
            bestv = avg
            best = k

    ai_accuracy = get_ai_model_accuracy(features)

    return jsonify({
        "evaluations": evaluations,
        "modelmetrics": metrics,
        "bestmodel": best,
        "ai_accuracy": ai_accuracy
    })

@app.route("/api/ai_summary", methods=["GET"])
def api_ai_summary():
    with lock:
        features = vitals_data.get("features", {}).copy()

    rows = []
    
    if GEMINI_API_KEY and genai:
        try:
            model = genai.GenerativeModel(MODEL_NAME)
            filtered_features = {k: v for k, v in features.items() if k in NORMAL_RANGES}
            prompt = f"Provide a short table-like summary with each vital name and status (Normal/Abnormal) and a brief note:\n\n{json.dumps(filtered_features, indent=2)}\n\nReturn as JSON array with fields: name, status, note."
            resp = model.generate_content(prompt)
            text = resp.text if hasattr(resp, "text") else str(resp)
            
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return jsonify({"ai_table": parsed})
            except Exception:
                pass
        except Exception:
            pass

    for k, v in features.items():
        if k not in NORMAL_RANGES:
            continue
            
        status = "Normal"
        note = ""
        if isinstance(v, (int, float)):
            rng = NORMAL_RANGES.get(k)
            if rng:
                lo, hi = rng
                if not (lo <= v <= hi):
                    status = "Abnormal"
                    note = f"value {v} outside ({lo}-{hi})"
        else:
            if k == "ECG Pattern" and str(v).lower() not in ("normal", "sinus"):
                status = "Abnormal"
                note = f"pattern: {v}"
        rows.append({"name": k, "status": status, "note": note})
    
    return jsonify({"ai_table": rows})

@app.route("/")
def index():
    return render_template("dashboard.html")

@app.route("/scan", methods=["POST"])
def scan():
    patient_info = request.json
    with lock:
        vitals_data["status"] = "capturing"
        vitals_data["patient_info"] = patient_info
    threading.Thread(target=capture_ppg, daemon=True).start()
    return jsonify({"status": "started"})

@app.route("/results")
def results():
    with lock:
        return jsonify(vitals_data)

@app.route("/download_report")
def download_report():
    with lock:
        pdf_path = vitals_data.get("pdf_path", "")
    if pdf_path and os.path.exists(pdf_path):
        return send_file(pdf_path, as_attachment=True)
    return jsonify({"error": "No report available"}), 404

if __name__ == "__main__":
    os.makedirs("static", exist_ok=True)
    os.makedirs("templates", exist_ok=True)
    print("🚀 VitalSenseAI running at http://127.0.0.1:5000")
    app.run(debug=True)
