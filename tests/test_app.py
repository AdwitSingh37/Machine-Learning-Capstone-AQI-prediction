"""Prediction contract and Streamlit interaction checks (unittest, no extra runner)."""
import unittest

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from aqi_model import POLLUTANTS, ROOT, category, load_bundle, predict, prepare_features


class PredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = load_bundle()

    def example(self):
        return pd.DataFrame([self.bundle["example"]])

    def test_single_matches_batch(self):
        frame = self.example()
        one = predict(frame, self.bundle)
        many = predict(pd.concat([frame] * 3, ignore_index=True), self.bundle)
        self.assertTrue(np.isfinite(one["Predicted_AQI"]).all())
        self.assertTrue(many["Predicted_AQI"].eq(one.iloc[0]["Predicted_AQI"]).all())

    def test_missing_measurement_is_imputed(self):
        frame = self.example()
        frame["PM10"] = np.nan
        self.assertTrue(np.isfinite(predict(frame, self.bundle)["Predicted_AQI"]).all())

    def test_invalid_measurements_are_rejected(self):
        for value in [-1, "invalid", np.inf]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                frame = self.example()
                frame["PM2.5"] = value
                predict(frame, self.bundle)

    def test_empty_measurements_are_rejected(self):
        frame = self.example()
        frame[POLLUTANTS] = np.nan
        with self.assertRaises(ValueError):
            predict(frame, self.bundle)

    def test_missing_schema_date_city(self):
        with self.assertRaises(ValueError):
            prepare_features(self.example().drop(columns=["Date"]))
        for column, value in [("Date", "bad date"), ("City", "Unknown city"), ("City", "")]:
            with self.subTest(column=column, value=value), self.assertRaises(ValueError):
                frame = self.example()
                frame[column] = value
                predict(frame, self.bundle)

    def test_category_boundaries(self):
        for value, expected in [(0, "Good"), (50, "Good"), (50.01, "Satisfactory"), (100, "Satisfactory"), (100.01, "Moderate"), (200, "Moderate"), (300, "Poor"), (400, "Very Poor"), (401, "Severe"), (800, "Severe")]:
            with self.subTest(value=value):
                self.assertEqual(category(value)[0], expected)

    def test_target_columns_do_not_affect_prediction(self):
        frame = self.example()
        baseline = predict(frame, self.bundle)["Predicted_AQI"].iloc[0]
        frame["AQI"], frame["AQI_Bucket"] = 99999, "Severe"
        self.assertEqual(predict(frame, self.bundle)["Predicted_AQI"].iloc[0], baseline)


class InterfaceTests(unittest.TestCase):
    def test_form_prediction_and_edit(self):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
        self.assertEqual(len(app.exception), 0)
        app.button[0].click().run()
        self.assertEqual(len(app.exception), 0)
        first = app.session_state["result"]["aqi"]
        app.number_input[0].set_value(250.0)
        app.number_input[1].set_value(350.0)
        app.button[0].click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertNotEqual(app.session_state["result"]["aqi"], first)
        app.toggle[0].set_value(False).run()
        self.assertEqual(len(app.exception), 0)


if __name__ == "__main__":
    unittest.main()
