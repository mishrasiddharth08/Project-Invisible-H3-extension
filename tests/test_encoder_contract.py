import json
from unittest import TestCase
from forge_h3.models import validate_encoder_state_dict
from forge_h3.contracts import H3Error

class EncoderContractTests(TestCase):
    def state(self, format):
        return {'visual.weight.comfy_quant': list(json.dumps({'format': 'int8_tensorwise'}).encode()),
                'model.layers.0.weight.comfy_quant': list(json.dumps({'format': format}).encode())}
    def test_rejects_actual_nvfp4_after_component_substitution(self):
        with self.assertRaisesRegex(H3Error, 'NVFP4'):
            validate_encoder_state_dict(self.state('nvfp4'))
    def test_accepts_native_int4_int8_and_plain_weights(self):
        for format in ('convrot_w4a4', 'int8_tensorwise'):
            validate_encoder_state_dict(self.state(format))
        validate_encoder_state_dict({'weight': object()})
