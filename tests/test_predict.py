import unittest

from predict import Predictor
from PIL import Image


class PredictorTestCase(unittest.TestCase):
    def setUp(self):
        self.predictor = Predictor()
        self.predictor.setup()

    def test_predict_text_to_img(self):
        result = self.predictor.predict(
            prompt='test',
            negative_prompt='',
            image='tests/data/test_img1.png',
            condition_scale=0.5,
            num_outputs=1,
            scheduler='K_EULER',
            num_inference_steps=3,
            guidance_scale=7.5,
            seed=0,
            lora_scale=0.6,
            lora_weights=None,
            strength=0.5,
        )
        print(result)

    def test_predict_img_to_img(self):
        result = self.predictor.predict(
            prompt='test',
            negative_prompt='',
            image='tests/data/test_img1.png',
            condition_scale=0.5,
            num_outputs=1,
            scheduler='K_EULER',
            num_inference_steps=3,
            guidance_scale=7.5,
            seed=0,
            lora_scale=0.6,
            lora_weights=None,
            strength=0.5,
        )
        print(result)
