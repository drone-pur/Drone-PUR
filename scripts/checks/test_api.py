import os
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from drone_pur.backends import (
    APIBackend,
    APIError,
    set_max_pixels,
    api_parameters,
    api_url,
)


class APITests(unittest.TestCase):
    def test_manifest_uses_endpoint_fingerprint_without_credentials(self):
        from drone_pur.evaluate import identity, parser

        args = parser().parse_args(
            [
                "--backend",
                "api",
                "--model",
                "example-vision-model",
                "--api-base-url",
                "https://example.invalid/v1",
                "--api-key-env",
                "TEST_DRONE_KEY",
                "--data-root",
                str(Path(__file__).parent),
                "--output",
                "outputs/api-test",
            ]
        )
        samples, inputs = {"OC": [{"question_id": "example"}]}, {"OC": "example"}
        with patch.dict(os.environ, {"TEST_DRONE_KEY": "not-a-real-secret"}):
            first = identity(args, samples, inputs)
        text = json.dumps(first)
        self.assertNotIn("example.invalid", text)
        self.assertNotIn("not-a-real-secret", text)
        self.assertNotIn("TEST_DRONE_KEY", text)
        self.assertNotIn("api_base_url", first)
        self.assertEqual(len(first["api_endpoint_sha256"]), 64)
        args.api_base_url = "https://example.invalid/v1/"
        self.assertEqual(first, identity(args, samples, inputs))
        args.api_base_url = "https://other.example.invalid/v1"
        self.assertNotEqual(
            first["api_endpoint_sha256"],
            identity(args, samples, inputs)["api_endpoint_sha256"],
        )

    def test_credentials_cannot_enter_endpoint_manifest(self):
        with self.assertRaises(ValueError):
            api_url("https://example.invalid/v1?api_key=not-a-real-secret")
        self.assertEqual(
            api_url("https://example.invalid/v1/"), "https://example.invalid/v1"
        )

    def test_credentials_cannot_enter_parameter_manifest(self):
        with self.assertRaises(ValueError):
            api_parameters('{"api_key":"not-a-real-secret"}')
        self.assertEqual(
            api_parameters('{"enable_thinking":false}'), {"enable_thinking": False}
        )

    def args(self):
        return SimpleNamespace(
            api_key_env="TEST_DRONE_KEY",
            api_base_url="https://example.invalid/v1",
            api_params=None,
            model="example-vision-model",
            api_retries=0,
            api_timeout=10,
            api_max_side=None,
        )

    def test_success_records_returned_version_without_secrets(self):
        session = Mock()
        session.post.return_value = SimpleNamespace(
            status_code=200,
            json=lambda: {
                "id": "response-1",
                "model": "pinned-version",
                "choices": [{"message": {"content": "B"}, "finish_reason": "stop"}],
                "usage": {"total_tokens": 10},
            },
        )
        with (
            patch.dict(os.environ, {"TEST_DRONE_KEY": "not-a-real-secret"}),
            patch("requests.Session", return_value=session),
        ):
            backend = APIBackend(self.args())
            text, info = backend.infer([], "Question", 32)
        self.assertEqual(text, "B")
        self.assertEqual(info["response_model"], "pinned-version")
        self.assertNotIn("secret", str(info))
        self.assertEqual(session.post.call_args.kwargs["json"]["temperature"], 0)

    def test_reasoning_is_not_an_answer(self):
        session = Mock()
        session.post.return_value = SimpleNamespace(
            status_code=200,
            json=lambda: {
                "choices": [{"message": {"reasoning_content": "Maybe A, B, or C"}}]
            },
        )
        with (
            patch.dict(os.environ, {"TEST_DRONE_KEY": "not-a-real-secret"}),
            patch("requests.Session", return_value=session),
        ):
            text, _ = APIBackend(self.args()).infer([], "Question", 32)
        self.assertEqual(text, "")

    def test_authentication_failure_is_not_retried_and_redacts_key(self):
        session = Mock()
        session.post.return_value = SimpleNamespace(
            status_code=401, text="Unauthorized not-a-real-secret"
        )
        args = self.args()
        args.api_retries = 3
        with (
            patch.dict(os.environ, {"TEST_DRONE_KEY": "not-a-real-secret"}),
            patch("requests.Session", return_value=session),
        ):
            with self.assertRaises(APIError) as caught:
                APIBackend(args).infer([], "Question", 32)
        self.assertEqual(session.post.call_count, 1)
        self.assertNotIn("not-a-real-secret", str(caught.exception))

    def test_processor_max_pixel_conventions(self):
        processor = SimpleNamespace(
            image_processor=SimpleNamespace(max_pixels=100, size={"longest_edge": 100})
        )
        set_max_pixels(processor, 1003520)
        self.assertEqual(processor.image_processor.max_pixels, 1003520)
        self.assertEqual(processor.image_processor.size["longest_edge"], 1003520)
        processor = SimpleNamespace(
            image_processor=SimpleNamespace(size=SimpleNamespace(longest_edge=100))
        )
        set_max_pixels(processor, 200)
        self.assertEqual(processor.image_processor.size.longest_edge, 200)
        with self.assertRaises(ValueError):
            set_max_pixels(SimpleNamespace(image_processor=object()), 100)


if __name__ == "__main__":
    unittest.main()
