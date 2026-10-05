"""Weight feature contract tests; runnable with unittest or pytest."""

import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from src.features import WEIGHT_FEATURE_COLUMNS, build_weight_features


class WeightFeaturesTests(unittest.TestCase):
    def test_missing_and_negative_weights_keep_rows_and_original_flags(self):
        frame = pd.DataFrame(
            {"weight": [32_000.0, -31_500.0, np.nan, 0.0]},
            index=[8, 2, 5, 9],
        )
        result = build_weight_features(frame)
        self.assertEqual(result.columns.tolist(), WEIGHT_FEATURE_COLUMNS)
        self.assertEqual(result.index.tolist(), frame.index.tolist())
        np.testing.assert_allclose(result["weight_clean"], [32_000, 31_500, np.nan, 0], equal_nan=True)
        self.assertEqual(result["weight_missing"].tolist(), [0, 0, 1, 0])
        self.assertEqual(result["weight_negative"].tolist(), [0, 1, 0, 0])

    def test_original_data_is_unchanged_and_target_and_id_are_excluded(self):
        frame = pd.DataFrame({"load_id": ["A", "B"], "posted_rate": [900, 1100], "weight": [-12_000, np.nan]})
        before = frame.copy(deep=True)
        result = build_weight_features(frame)
        assert_frame_equal(frame, before)
        self.assertNotIn("load_id", result)
        self.assertNotIn("posted_rate", result)

    def test_all_missing_weights_remain_missing_without_learned_fill(self):
        result = build_weight_features(pd.DataFrame({"weight": [np.nan, np.nan]}))
        self.assertTrue(result["weight_clean"].isna().all())
        self.assertEqual(result["weight_missing"].tolist(), [1, 1])
        self.assertEqual(result["weight_negative"].tolist(), [0, 0])

    def test_nullable_values_and_numeric_strings(self):
        result = build_weight_features(pd.DataFrame({"weight": [pd.NA, "-32000", "15000"]}))
        self.assertTrue(pd.isna(result.loc[0, "weight_clean"]))
        self.assertEqual(result["weight_missing"].tolist(), [1, 0, 0])
        self.assertEqual(result["weight_negative"].tolist(), [0, 1, 0])

    def test_malformed_and_infinite_values_raise_explicit_errors(self):
        for value in ["not-a-weight", "", np.inf, -np.inf]:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    build_weight_features(pd.DataFrame({"weight": [value]}))

    def test_missing_column_raises_explicit_error(self):
        with self.assertRaisesRegex(ValueError, "column 'weight'"):
            build_weight_features(pd.DataFrame({"distance": [360]}))

    def test_batch_composition_does_not_change_features(self):
        frame = pd.DataFrame({"weight": [-32_000.0, np.nan, 9000.0]}, index=[10, 20, 30])
        combined = build_weight_features(frame)
        separately = pd.concat([build_weight_features(frame.iloc[[i]]) for i in range(len(frame))])
        assert_frame_equal(combined, separately)


if __name__ == "__main__":
    unittest.main()
