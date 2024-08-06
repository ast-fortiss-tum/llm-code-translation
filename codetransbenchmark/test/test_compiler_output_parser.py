import unittest
from codetransbench.execution.compiler_output_parser import extract_error_messages


class TestGetErrorMessages(unittest.TestCase):
    def test_empty(self):
        output = ""
        self.assertEqual(extract_error_messages(output), [])

    def test_all_okay(self):
        output = "This program is running correctly"
        self.assertEqual(extract_error_messages(output), [])

    def test_error(self):
        output = "Some message.\nThere was an error CS123: division by zero [.csproj]"
        expected_result = ["error CS123: division by zero"]
        self.assertEqual(extract_error_messages(output), expected_result)

    def test_multiple_errors(self):
        output = "There was an error CS123: division by zero [.csproj]\n another error CS123: null reference exception [.csproj]"
        expected_result = [
            "error CS123: division by zero",
            "error CS123: null reference exception",
        ]
        self.assertEqual(extract_error_messages(output), expected_result)

    def test_errors_and_warnings(self):
        output = "There was an error CS123: division by zero [test.csproj]\nand another error CS123: null reference exception [test.csproj]\nand warning CS123: invalid syntax [test.csproj]"
        expected_result = [
            "error CS123: division by zero",
            "error CS123: null reference exception",
        ]
        self.assertEqual(extract_error_messages(output), expected_result)


if __name__ == "__main__":
    unittest.main()
