import unittest

from predict import Predictor
from PIL import Image


class PredictorTestCase(unittest.TestCase):
    def setUp(self):
        self.predictor = Predictor()
        self.predictor.setup()

    def test_predict_text_to_img(self):
        result = self.predictor.predict(
            prompt='In the style of a watercolor painting',
            negative_prompt='',
            image='tests/data/test_img1.png',
            condition_scale=1.1,
            num_outputs=1,
            scheduler='K_EULER',
            num_inference_steps=2,
            guidance_scale=7.5,
            seed=0,
            lora_scale=0.95,
            lora_weights=None,
            strength=0.5,
            auto_generate_caption=False,
        )
        for img_path in result:
            img = Image.open(img_path)
            img.show()

    def test_predict_img_to_img(self):
        result = self.predictor.predict(
            prompt='In the style of a watercolor painting,',
            negative_prompt='',
            image='tests/data/test_img1.png',
            condition_scale=1.1,
            num_outputs=1,
            scheduler='K_EULER',
            num_inference_steps=2,
            guidance_scale=7.5,
            seed=0,
            lora_scale=0.95,
            lora_weights=None,
            strength=0.5,
            auto_generate_caption=False,
        )
        for img_path in result:
            img = Image.open(img_path)
            img.show()

    def test_predict_auto_generate_caption(self):
        result = self.predictor.predict(
            prompt='In the style of a watercolor painting,',
            negative_prompt='',
            image='tests/data/test_img1.png',
            condition_scale=1.1,
            num_outputs=1,
            scheduler='K_EULER',
            num_inference_steps=2,
            guidance_scale=7.5,
            seed=0,
            lora_scale=0.95,
            lora_weights=None,
            strength=0.5,
            auto_generate_caption=True,
        )
        for img_path in result:
            img = Image.open(img_path)
            img.show()
