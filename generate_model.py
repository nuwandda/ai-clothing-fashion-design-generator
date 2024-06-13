import cv2
import torch
import random
from PIL import Image
from auto1111sdk import StableDiffusionPipeline
import argparse
import logging as log


log.getLogger().setLevel(log.INFO)
MODEL_PATH = 'weights/realvisxlV40_v40LightningBakedvae.safetensors'


# Helper functions
def set_seed():
    seed = random.randint(42,4294967295)
    return seed


def create_pipeline(model_path):
    # Create the pipe 
    pipe = StableDiffusionPipeline(model_path, default_command_args='--device-id 0')
    
    return pipe


pipe = create_pipeline(MODEL_PATH)
# pipe.scheduler = DPMSolverSinglestepScheduler.from_config(pipe.scheduler.config)
# pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)


def generate(clothing_type, output_name):
    log.info('Generating...')
    generator = torch.Generator().manual_seed(set_seed())

    # Final prompt
    prompt = """
        centered, portrait photo of a woman, wearing {}, natural skin, dark shot
    """.format(clothing_type)
    negative_prompt = """
        (octane render, render, drawing, anime, bad photo, bad photography:1.3), 
        (worst quality, low quality, blurry:1.2), (bad teeth, deformed teeth, deformed lips), 
        (bad anatomy, bad proportions:1.1), (deformed iris, deformed pupils), (deformed eyes, bad eyes), 
        (deformed face, ugly face, bad face), (deformed hands, bad hands, fused fingers), 
        morbid, mutilated, mutation, disfigured
    """

    log.info('Final Prompt: ' + prompt)
    image = pipe.generate_txt2img(prompt = prompt, negative_prompt=negative_prompt,
                                   height = 1024, width = 768, cfg_scale=2, steps = 5, sampler_name='DPM++ SDE')
    image[0].save(output_name)
    log.info('Done.')
    log.info('Image saved: ' + output_name)



def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(description='Generate realistic images from the given prompt.')
    parser.add_argument('--prompt', type=str, required=True, help='Clothing type')
    parser.add_argument('--output_path', type=str, required=True, help='Path to the output image')

    # Parse the arguments
    args, unknown = parser.parse_known_args()

    generate(args.prompt, args.output_path)
    # generate('a crop top and mini skirt', 'reference_images/crop_top')
    # generate('a maxi dress with cut out details', 'reference_images/dress')
    # generate('A formal jumpsuit with a belt', 'reference_images/jumpsuit')


if __name__ == "__main__":
    main()
