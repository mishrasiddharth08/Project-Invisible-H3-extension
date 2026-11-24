"""Verify Shift provenance through the same API entrypoint pattern Forge uses."""

import inspect
from types import SimpleNamespace
from unittest import TestCase

from pi_h3_image.api_shift import apply_shift, patch_api


class Request:
    def __init__(self, **fields):
        self.model_fields_set = set(fields) - {"infotext"}
        self.infotext = fields.get("infotext")
        self.distilled_cfg_scale = fields.get("distilled_cfg_scale", 3.5)


def api_class(profile, opts):
    def process(request):
        if request.infotext is not None and "distilled_cfg_scale" not in request.model_fields_set:
            request.distilled_cfg_scale = request.infotext
            request.model_fields_set.add("distilled_cfg_scale")
        p = SimpleNamespace(is_api=True, distilled_cfg_scale=request.distilled_cfg_scale)
        apply_shift(p, profile, opts)
        return p.distilled_cfg_scale

    class Api:
        def text2imgapi(self, txt2imgreq: Request):
            return process(txt2imgreq)

        def img2imgapi(self, img2imgreq: Request):
            return process(img2imgreq)

    return Api


class TestApiShift(TestCase):
    profile = SimpleNamespace(name="ph3", shift=1.0)
    opts = SimpleNamespace(forge_preset="unrelated", ph3_i2i_dcfg=2.0, ph3_t2i_dcfg=4.0)

    def setUp(self):
        self.api_type = api_class(self.profile, self.opts)
        self.original_signature = inspect.signature(self.api_type.img2imgapi)
        patch_api(self.api_type)
        self.api = self.api_type()

    def test_missing_uses_the_correct_tab_even_outside_the_ui_preset(self):
        self.assertEqual(self.api.img2imgapi(img2imgreq=Request()), 2.0)
        self.assertEqual(self.api.text2imgapi(txt2imgreq=Request()), 4.0)

    def test_explicit_default_and_infotext_are_not_overwritten(self):
        self.assertEqual(self.api.img2imgapi(img2imgreq=Request(distilled_cfg_scale=3.5)), 3.5)
        self.assertEqual(self.api.text2imgapi(txt2imgreq=Request(infotext=3.5)), 3.5)
        self.assertEqual(self.api.img2imgapi(img2imgreq=Request(distilled_cfg_scale=0.7)), 0.7)

    def test_api_signature_and_idempotence(self):
        self.assertEqual(inspect.signature(self.api_type.img2imgapi), self.original_signature)
        before = self.api_type.img2imgapi
        patch_api(self.api_type)
        self.assertIs(self.api_type.img2imgapi, before)

    def test_failure_does_not_leak_the_endpoint_or_field_set(self):
        class Fails:
            def text2imgapi(self, txt2imgreq):
                raise RuntimeError("generation stopped")

            def img2imgapi(self, img2imgreq):
                raise RuntimeError("generation stopped")

        patch_api(Fails)
        with self.assertRaisesRegex(RuntimeError, "generation stopped"):
            Fails().img2imgapi(Request())
        p = SimpleNamespace(is_api=True, distilled_cfg_scale=3.5)
        apply_shift(p, self.profile, self.opts)
        self.assertEqual(p.distilled_cfg_scale, 3.5)

    def test_ui_keeps_preset_behavior(self):
        p = SimpleNamespace(is_api=False, distilled_cfg_scale=3.5)
        apply_shift(p, self.profile, self.opts)
        self.assertEqual(p.distilled_cfg_scale, 1.0)
        p.distilled_cfg_scale = 3.5
        apply_shift(p, self.profile, SimpleNamespace(forge_preset="ph3"))
        self.assertEqual(p.distilled_cfg_scale, 3.5)

    def test_fastapi_binding_with_real_pydantic_field_tracking(self):
        try:
            from fastapi import FastAPI
            from fastapi.testclient import TestClient
            from pydantic import BaseModel
        except ImportError:
            self.skipTest("FastAPI/Pydantic/TestClient not installed")

        class WireRequest(BaseModel):
            distilled_cfg_scale: float = 3.5

        def process(request, profile, opts):
            p = SimpleNamespace(is_api=True, distilled_cfg_scale=request.distilled_cfg_scale)
            apply_shift(p, profile, opts)
            return {"shift": p.distilled_cfg_scale}

        profile, opts = self.profile, self.opts

        class Api:
            def text2imgapi(self, txt2imgreq: WireRequest):
                return process(txt2imgreq, profile, opts)

            def img2imgapi(self, img2imgreq: WireRequest):
                return process(img2imgreq, profile, opts)

        patch_api(Api)
        api, app = Api(), FastAPI()
        app.add_api_route("/txt", api.text2imgapi, methods=["POST"])
        app.add_api_route("/img", api.img2imgapi, methods=["POST"])
        with TestClient(app) as client:
            for path, payload, expected in (
                ("/txt", {}, 4.0),
                ("/img", {}, 2.0),
                ("/img", {"distilled_cfg_scale": 3.5}, 3.5),
                ("/img", {}, 2.0),
            ):
                response = client.post(path, json=payload)
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.json(), {"shift": expected})
