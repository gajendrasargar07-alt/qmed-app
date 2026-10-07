"""Run with:  python -m unittest discover -s tests -v   (needs the app's requirements installed)"""
import hashlib
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qmed import service                         # noqa: E402
from qmed.ocr import extract_fields             # noqa: E402
from qmed.validator import validate             # noqa: E402

SAMPLES = ROOT / "tests" / "samples"


def sample(name):
    return (SAMPLES / name).read_text()


class ModelFiles(unittest.TestCase):
    def test_trained_models_are_unchanged(self):
        for line in (ROOT / "models.sha256").read_text().splitlines():
            digest, rel = line.split(maxsplit=1)
            actual = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
            self.assertEqual(actual, digest, f"{rel} has changed")


class Extraction(unittest.TestCase):
    def test_breast_panel_is_read(self):
        f = extract_fields(sample("breast_report.txt"))
        for key in ("mean radius", "mean texture", "mean perimeter", "mean area", "mean concavity"):
            self.assertIn(key, f)
        self.assertAlmostEqual(f["mean radius"], 17.9)

    def test_cervical_tests_are_read(self):
        f = extract_fields(sample("cervical_report.txt"))
        for key in ("hpv_positive", "schiller", "citology", "dx_cin", "stds_count", "iud_years"):
            self.assertIn(key, f)
        self.assertEqual(f["hpv_positive"], 1.0)
        self.assertEqual(f["iud_years"], 0.0)

    def test_prostate_values(self):
        f = extract_fields(sample("prostate_report.txt"))
        self.assertEqual(f["psa"], 8.4)
        self.assertEqual(f["dre_suspicious"], 1.0)


class Validator(unittest.TestCase):
    def test_accepts_medical_and_rejects_other(self):
        self.assertTrue(validate(sample("prostate_report.txt"))["is_medical"])
        self.assertFalse(validate(sample("irrelevant_random.txt"))["is_medical"])


class EndToEnd(unittest.TestCase):
    def test_each_sample_report(self):
        for cancer in ("breast", "lung", "prostate", "colon", "cervical"):
            with self.subTest(cancer):
                r = service.analyze_text(sample(f"{cancer}_report.txt"))
                self.assertEqual(r["status"], "ok", r)
                self.assertEqual(r["report"]["type"], cancer)
                self.assertEqual(r["verdict"]["band"], "high")
                self.assertEqual(r["agreement"]["total"], 6)
                json.dumps(r)  # must be serialisable

    def test_unrelated_text_is_rejected(self):
        self.assertEqual(service.analyze_text(sample("irrelevant_random.txt"))["status"], "rejected")

    def test_missing_key_values_give_no_verdict(self):
        r = service.analyze_text(
            "Patient report. Prostate biopsy pathology. PSA: 9.1 ng/mL. "
            "Age: 61 year-old male. Hemoglobin 13.2 g/dL. Adenocarcinoma suspected.")
        self.assertEqual(r["status"], "incomplete")
        self.assertNotIn("verdict", r)
        self.assertTrue(r["missing"])

    def test_scanned_image(self):
        data = (SAMPLES / "prostate_report_scan.png").read_bytes()
        r = service.analyze_bytes(data)
        self.assertEqual(r["status"], "ok", r)
        self.assertEqual(r["report"]["type"], "prostate")

    def test_type_can_be_overridden(self):
        r = service.analyze_text(sample("prostate_report.txt"), cancer="colon")
        self.assertIn(r["status"], ("ok", "incomplete"))
        self.assertEqual(r["report"]["type"], "colon")

    def test_empty_input(self):
        self.assertEqual(service.analyze_text("   ")["status"], "error")
        self.assertEqual(service.analyze_bytes(b"")["status"], "error")


if __name__ == "__main__":
    unittest.main()
