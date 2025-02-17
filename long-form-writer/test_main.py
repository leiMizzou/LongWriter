import unittest
from unittest.mock import patch, MagicMock
import main  # Revert to absolute import
import argparse  # Import argparse
import os
import json
import logging
import re
from datetime import datetime

class TestLongFormWriter(unittest.TestCase):

    def setUp(self):
        """Setup method to create test files and directories if needed."""
        if not os.path.exists('logs'):
            os.makedirs('logs')
        if not os.path.exists('articles'):
            os.makedirs('articles')

    def tearDown(self):
        """Teardown method to clean up test files and directories."""
        test_log_files = [f for f in os.listdir('logs') if f.startswith('test_')]
        for f in test_log_files:
            os.remove(os.path.join('logs', f))

        test_article_files = [f for f in os.listdir('articles') if f.startswith('test_')]
        for f in test_article_files:
            os.remove(os.path.join('articles', f))

    def test_setup_logging(self):
        """Test the setup_logging function."""
        title = "test_article"
        log_filename = main.setup_logging(title)
        self.assertTrue(os.path.exists(log_filename))
        self.assertIn('logs/test_article_', log_filename)

    @patch('argparse.ArgumentParser.parse_args')
    def test_get_user_input(self, mock_parse_args):
        """Test the get_user_input function."""
        mock_parse_args.return_value = argparse.Namespace(
            title="Test Title",
            length=1000,
            genre="Test Genre",
            language="en",
            context="Test Context"
        )
        title, length, genre, language, context = main.get_user_input()
        self.assertEqual(title, "Test Title")
        self.assertEqual(length, 1000)
        self.assertEqual(genre, "Test Genre")
        self.assertEqual(language, "en")
        self.assertEqual(context, "Test Context")

    @patch('main.call_gemini_api')
    def test_generate_outline_success(self, mock_call_gemini_api):
        """Test successful outline generation."""
        mock_call_gemini_api.return_value = """```json
[
    {"title": "Introduction", "length": 100, "level": 1},
    {"title": "Section 1", "length": 200, "level": 1},
    {"title": "Subsection 1.1", "length": 50, "level": 2, "parent": "Section 1"}
]
```"""
        outline = main.generate_outline("Test Title", 350, "Test Genre", "en", "Test Context")
        self.assertIsInstance(outline, list)
        self.assertEqual(len(outline), 3)
        self.assertIn("title", outline[0])
        self.assertIn("length", outline[0])
        self.assertIn("level", outline[0])
        self.assertIn("parent", outline[2])

    @patch('main.call_gemini_api')
    def test_generate_outline_failure(self, mock_call_gemini_api):
        """Test outline generation failure (no JSON found)."""
        mock_call_gemini_api.return_value = "Invalid response"
        outline = main.generate_outline("Test Title", 350, "Test Genre", "en", "Test Context")
        self.assertEqual(outline, [])

    @patch('main.call_gemini_api')
    def test_generate_outline_exception(self, mock_call_gemini_api):
        """Test outline generation failure (exception raised)."""
        mock_call_gemini_api.side_effect = Exception("API Error")
        outline = main.generate_outline("Test Title", 350, "Test Genre", "en", "Test Context")
        self.assertEqual(outline, [])

    @patch('google.generativeai.GenerativeModel.generate_content')
    def test_call_gemini_api_success(self, mock_generate_content):
        """Test successful API call."""
        mock_generate_content.return_value.text = "Test response"
        response = main.call_gemini_api("Test prompt")
        self.assertEqual(response, "Test response")

    @patch('google.generativeai.GenerativeModel.generate_content')
    def test_call_gemini_api_retry(self, mock_generate_content):
        """Test API call retry on rate limit."""
        mock_generate_content.side_effect = [Exception("429: Rate limit exceeded"), MagicMock(text="Success")]
        response = main.call_gemini_api("Test prompt")
        self.assertEqual(response, "Success")

    @patch('google.generativeai.GenerativeModel.generate_content')
    def test_call_gemini_api_failure(self, mock_generate_content):
        """Test API call failure after multiple retries."""
        mock_generate_content.side_effect = Exception("API Error")
        with self.assertRaises(Exception):
            main.call_gemini_api("Test prompt")

    @patch('main.call_gemini_api')
    def test_write_section(self, mock_call_gemini_api):
        """Test the write_section function."""
        mock_call_gemini_api.return_value = "Test section content"
        title = "Test Section"
        length = 100
        genre = "Test Genre"
        language = "en"
        parent_title = "Test Parent"
        context = "Test Context"
        content = main.write_section(title, length, genre, language, parent_title, context)
        self.assertEqual(content, "Test section content")
        mock_call_gemini_api.assert_called_once()
        # Further check on prompt construction could be added if needed

    def test_assemble_article(self):
        """Test the assemble_article function."""
        outline = [
            {"title": "Introduction", "length": 100, "level": 1, "content": "Intro content"},
            {"title": "Section 1", "length": 200, "level": 1, "content": "Section 1 content"},
            {"title": "Subsection 1.1", "length": 50, "level": 2, "content": "Subsection 1.1 content", "parent": "Section 1"}
        ]
        expected_article = "# Introduction\n\nIntro content\n\n# Section 1\n\nSection 1 content\n\n## Subsection 1.1\n\nSubsection 1.1 content\n\n"
        article = main.assemble_article(outline)
        self.assertEqual(article, expected_article)

    def test_save_article(self):
        """Test the save_article function."""
        title = "test_article"
        content = "This is a test article."
        filename = main.save_article(title, content)
        self.assertTrue(os.path.exists(filename))
        self.assertIn('articles/test_article_', filename)
        with open(filename, 'r', encoding='utf-8') as f:
            saved_content = f.read()
        self.assertEqual(saved_content, content)

    @patch('main.generate_outline')
    @patch('main.save_article')
    @patch('main.assemble_article')
    @patch('main.write_section')
    @patch('main.get_user_input')
    @patch('main.setup_logging')
    def test_main_success(self, mock_setup_logging, mock_get_user_input, mock_write_section,
                         mock_assemble_article, mock_save_article, mock_generate_outline):
        """Test the main function with successful execution."""
        mock_get_user_input.return_value = ("Test Title", 1000, "Test Genre", "en", "Test Context")
        mock_setup_logging.return_value = "test_log_file.log"
        mock_generate_outline.return_value = [
            {"title": "Section 1", "length": 1000, "level": 1, "content": ""},
        ]
        mock_write_section.return_value = "Test section content"
        mock_assemble_article.return_value = "Final article content"
        mock_save_article.return_value = "test_article_file.md"

        main.main()

        mock_setup_logging.assert_called_once()
        mock_get_user_input.assert_called_once()
        mock_generate_outline.assert_called_once()
        mock_write_section.assert_called_once()
        mock_assemble_article.assert_called_once()
        mock_save_article.assert_called_once()

    @patch('main.generate_outline')
    @patch('main.get_user_input')
    @patch('main.setup_logging')
    def test_main_outline_failure(self, mock_setup_logging, mock_get_user_input, mock_generate_outline):
        """Test the main function with outline generation failure."""
        mock_get_user_input.return_value = ("Test Title", 1000, "Test Genre", "en", "Test Context")
        mock_setup_logging.return_value = "test_log_file.log"
        mock_generate_outline.return_value = []

        main.main()

        mock_setup_logging.assert_called_once()
        mock_get_user_input.assert_called_once()
        mock_generate_outline.assert_called_once()

if __name__ == '__main__':
    unittest.main()