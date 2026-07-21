import unittest

from media_to_tts import TTSHallucinationScreener


class WordRepeatScreeningTests(unittest.TestCase):
    def setUp(self):
        self.screener = TTSHallucinationScreener()

    def _word_repeats(self, text: str) -> list[dict]:
        return [
            issue
            for issue in self.screener.screen_text(text)
            if issue["pattern"] == "WORD_REPEAT"
        ]

    def test_spoken_years_and_decimal_digits_are_benign(self):
        issues = self._word_repeats(
            "July twenty-first twenty twenty-six. "
            "The twenty twenty-five benchmark improved from thirty-four point two two percent."
        )
        self.assertEqual([], issues)

    def test_spoken_identifier_digits_are_benign(self):
        self.assertEqual([], self._word_repeats("Call one one zero, then press five five."))

    def test_three_repeated_number_words_are_still_flagged(self):
        issues = self._word_repeats("The broken input says two two two forever.")
        self.assertEqual(["two two two"], [issue["match"] for issue in issues])

    def test_genuine_word_repeat_is_still_flagged(self):
        issues = self._word_repeats("The Model model returned a very very long answer.")
        self.assertEqual(["Model model", "very very"], [issue["match"] for issue in issues])

    def test_hyphenated_second_word_is_not_a_repeat(self):
        self.assertEqual([], self._word_repeats("Use a model model-based controller."))

    def test_hyphenated_first_word_is_not_a_repeat(self):
        self.assertEqual(
            [],
            self._word_repeats("The reinforcement-learning learning speed improved."),
        )

    def test_repeated_scale_word_remains_flagged(self):
        issues = self._word_repeats("The report says million million parameters.")
        self.assertEqual(["million million"], [issue["match"] for issue in issues])


if __name__ == "__main__":
    unittest.main()
