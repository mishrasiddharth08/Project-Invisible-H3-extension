"""The two H3 wrappers must compose with old and new Forge checkpoints without CUDA."""
import ast
import types
import unittest
from pathlib import Path
from unittest.mock import Mock


class LoaderMetadataCompatibility(unittest.TestCase):
    def test_stacked_image_and_video_hooks_preserve_old_and_new_signatures(self):
        class Video:
            text_encoder_key_prefix = ("video.",)
        class Image:
            text_encoder_key_prefix = ("image.",)

        def call(*args, **kwargs):
            return object()

        base = Path(__file__).resolve().parents[1]
        original = Mock(side_effect=call)
        loader = types.SimpleNamespace(replace_state_dict=original)
        for package, cls in (("pi_h3_image", Image), ("forge_h3", Video)):
            name = base / package / "native/patches.py"
            parsed = ast.parse(name.read_text(encoding="utf8"))
            nodes = [item for item in parsed.body if isinstance(item, ast.FunctionDef)
                     and item.name in ("_replace", "_hook_replace_state_dict")]
            scope = dict(loader=loader, model=types.SimpleNamespace(MiniMaxH3=cls),
                         TE_PREFIX="encoder.", _is_h3_text_encoder=lambda asd: "match" in asd,
                         vae=types.SimpleNamespace(is_h3_vae=lambda _: False,
                                                  is_video_vae=lambda _: False,
                                                  is_audio_vae=lambda _: False))
            exec(compile(ast.Module(body=nodes, type_ignores=[]), str(name), "exec"), scope)
            scope["_hook_replace_state_dict"]()

        metadata = {"new": True}
        for cls, prefix in ((Video, "video."), (Image, "image.")):
            instance = cls()
            for args, kwargs in (((instance,"model"), {}), ((metadata, instance,"model"), {}),
                                 ((metadata,instance), {"path":"model"}),
                                 ((), {"metadata":metadata,"guess":instance,"path":"model"})):
                with self.subTest(cls=cls, args=args, kwargs=kwargs):
                    sd={}
                    self.assertIs(loader.replace_state_dict(sd, {"match":1}, *args, **kwargs), sd)
                    self.assertEqual(sd, {prefix+"encoder.match":1})
        original.assert_not_called()
        for args in ((None,"other"),(metadata,None,"other"),(metadata,Image(),"unknown-h3-part")):
            loader.replace_state_dict({}, {}, *args)
            original.assert_called_with({}, {}, *args)
