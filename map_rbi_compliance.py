import json
import pandas as pd
import re
from typing import Dict, List, Optional
from sentence_transformers import SentenceTransformer, util
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

model = SentenceTransformer('all-MiniLM-L6-v2')


def load_data(eval_report_path: str, requirements_path: str) -> tuple[Optional[Dict], Optional[List[Dict]]]:
    try:
        with open(eval_report_path, 'r', encoding='utf-8') as f:
            eval_report = json.load(f)
        with open(requirements_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            requirements = data.get("requirements", [])
        if not eval_report:
            print("Error: evaluation_report.json is empty")
            return None, None
        if not requirements:
            print("Error: rbi_requirements.json requirements array is empty")
            return None, None
        return eval_report, requirements
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        return None, None
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format - {e}")
        return None, None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None


def parse_notes(notes: List[str]) -> tuple[List[Dict], List[str], List[Dict]]:
    metrics = []
    issues = []
    sub_tests = []
    metric_pattern = re.compile(r"^(.*?):\s*(-?\d+\.\d+|\d+)$")
    sub_test_pattern = re.compile(r"Fairness \((.*?)\): Score (\d+)/10")

    for note in notes:
        match = metric_pattern.match(note)
        if match:
            name, value = match.groups()
            try:
                metrics.append({"name": name.strip(), "value": float(value)})
            except ValueError:
                continue
        sub_match = sub_test_pattern.match(note)
        if sub_match:
            attribute, score = sub_match.groups()
            sub_tests.append({"attribute": attribute, "score": int(score)})
        if any(prefix in note for prefix in ["Warning", "\u26a0\ufe0f", "\ud83d\udd0d", "\u2705"]):
            issues.append(note)

    return metrics, issues, sub_tests


def semantic_match(req_text: str, eval_text: str, threshold=0.65) -> tuple[float, bool]:
    req_emb = model.encode(req_text, convert_to_tensor=True)
    eval_emb = model.encode(eval_text, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(req_emb, eval_emb).item()
    return similarity, similarity >= threshold


def map_tests_to_checklist(eval_report: Dict, requirements: List[Dict]) -> pd.DataFrame:
    mappings = []
    for principle, data in eval_report.items():
        principle_lower = principle.lower()
        score = data.get("score", 0)
        notes = data.get("notes", [])
        metrics, issues, sub_tests = parse_notes(notes)
        matched = False

        for req in requirements:
            if req.get("principle", "").lower() != principle_lower:
                continue
            matched = True
            status = "Compliant"
            evidence = []
            req_text = req.get("description", "")
            eval_blob = " ".join(notes)
            thresholds = req.get("thresholds", {})

            if "score" in thresholds and score < thresholds["score"]:
                status = "Non-compliant"
                evidence.append(f"Score {score} < {thresholds['score']}")

            for metric in metrics:
                name, value = metric["name"], metric["value"]
                if name in thresholds:
                    if name == "MIA Accuracy" and value > thresholds.get("MIA_accuracy", float('inf')):
                        status = "Non-compliant"
                        evidence.append(f"MIA Accuracy: {value} > {thresholds['MIA_accuracy']}")
                    elif name == "Boundary attack Accuracy drop" and value > thresholds.get("attack_drop", float('inf')):
                        status = "Non-compliant"
                        evidence.append(f"Boundary attack Accuracy drop: {value} > {thresholds['attack_drop']}")
                    elif name == "HopSkipJump attack Accuracy drop" and value > thresholds.get("attack_drop", float('inf')):
                        status = "Non-compliant"
                        evidence.append(f"HopSkipJump attack Accuracy drop: {value} > {thresholds['attack_drop']}")
                    elif name == "Disparate Impact" and value < thresholds.get("Disparate Impact", 0):
                        status = "Non-compliant"
                        evidence.append(f"Disparate Impact: {value} < {thresholds['Disparate Impact']}")

            if "no_sensitive_features" in thresholds and any("sensitive features" in i.lower() for i in issues):
                status = "Non-compliant"
                evidence.append("Sensitive features detected")
            if "issues" in thresholds and len(issues) > thresholds.get("issues", 0):
                status = "Non-compliant"
                evidence.append(f"{len(issues)} issues detected")

            sim_score, sem_match = semantic_match(req_text, eval_blob)
            if not sem_match:
                status = "Non-compliant"
                evidence.append(f"Semantic similarity too low ({sim_score:.2f})")

            mappings.append({
                "Test": principle.capitalize(),
                "RBI_ID": req.get("id", "unknown"),
                "RBI_Paragraph": req.get("rbi_paragraph", "unknown"),
                "Requirement": req_text,
                "Category": req.get("category", "unknown"),
                "Score": score,
                "Status": status,
                "Evidence": "; ".join(evidence),
                "Metrics": [f"{m['name']}: {m['value']}" for m in metrics],
                "Issues": issues,
                "Sub_Tests": [f"{st['attribute']}: {st['score']}" for st in sub_tests],
                "Reviewer_Comment": ""
            })

        if not matched:
            print(f"Warning: No matching requirement for principle: {principle}")

    return pd.DataFrame(mappings)


def save_compliance_report(df: pd.DataFrame, path: str):
    try:
        df.to_csv(path, index=False)
        print(f"CSV report saved to {path}")
    except Exception as e:
        print(f"Error saving report: {e}")


def generate_pdf_report(df: pd.DataFrame, output_file: str):
    c = canvas.Canvas(output_file, pagesize=letter)
    width, height = letter
    c.setFont("Helvetica", 10)
    y = height - 40

    for _, row in df.iterrows():
        text = f"{row['Test']} | {row['Status']} | {row['Evidence']}"
        c.drawString(30, y, text)
        y -= 15
        if y < 50:
            c.showPage()
            y = height - 40
    c.save()
    print(f"PDF report saved to {output_file}")


if __name__ == "__main__":
    eval_report_path = "evaluation_report.json"
    requirements_path = "proxy_updated_rbi_requirements.json"
    output_csv = "rbi_compliance_report.csv"
    output_pdf = "rbi_compliance_report.pdf"

    eval_report, requirements = load_data(eval_report_path, requirements_path)
    if eval_report and requirements:
        df = map_tests_to_checklist(eval_report, requirements)
        save_compliance_report(df, output_csv)
        generate_pdf_report(df, output_pdf)
    else:
        print("Failed to load input files.")
