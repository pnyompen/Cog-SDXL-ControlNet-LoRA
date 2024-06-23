import argparse
import os
import shutil

from transformers import (
    BlipForConditionalGeneration,
    BlipProcessor,
)

DEFAULT_BLIP = "Salesforce/blip-image-captioning-large"
BLIP_CACHE = "./blip-cache"


def upload(args):
    blip_processor = BlipProcessor.from_pretrained(DEFAULT_BLIP)
    blip_model = BlipForConditionalGeneration.from_pretrained(DEFAULT_BLIP)

    temp_models = BLIP_CACHE
    if os.path.exists(temp_models):
        shutil.rmtree(temp_models)
    os.makedirs(temp_models)

    blip_processor.save_pretrained(os.path.join(temp_models, 'blip_processor'))
    blip_model.save_pretrained(os.path.join(temp_models, 'blip_large'))

    for val in os.listdir(temp_models):
        if 'tar' not in val:
            os.system(
                f'tar -cvf {os.path.join(temp_models, val)}.tar -C {os.path.join(temp_models, val)} .')
            os.system(
                f'rm -f {os.path.join(temp_models, val)}.tar')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", "-m", type=str)
    args = parser.parse_args()
    upload(args)
