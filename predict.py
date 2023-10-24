# Prediction interface for Cog

from cog import BasePredictor, Input, Path
import time
import torch
import requests
from PIL import Image
from io import BytesIO
from transformers import AutoConfig, AutoTokenizer
from llava.conversation import conv_templates, SeparatorStyle
from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria
from llava.model.language_model.llava_mistral import LlavaMistralForCausalLM
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IMAGE_PATCH_TOKEN

MODEL_NAME = "SkunkworksAI/BakLLaVA-1"
MODEL_CACHE = "model-cache"
TOKEN_CACHE = "token-cache"

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        t1 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            use_fast=False,
            cache_dir=TOKEN_CACHE,
        )
        self.model = LlavaMistralForCausalLM.from_pretrained(
            MODEL_CACHE
        ).to("cuda")

        self.vision_tower = self.model.get_vision_tower()
        if not self.vision_tower.is_loaded:
            print("Loading vision tower")
            self.vision_tower.load_model()
        self.vision_tower.to(device='cuda', dtype=torch.float16)
        t2 = time.time()
        print("Loading model took: ", t2 - t1, "seconds")

    def predict(
        self,
        image: Path = Input(description="Input Image"),
        prompt: str = Input(description="Input prompt", default="Describe this image"),
        max_sequence: int = Input(description="Maximum sequence length", default=512, ge=8, le=2048),
    ) -> str:
        """Run a single prediction on the model"""

        mm_use_im_start_end = getattr(self.model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(self.model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            self.tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            self.tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        self.model.resize_token_embeddings(len(self.tokenizer))

        image_processor = self.vision_tower.image_processor

        if self.model.config.mm_use_im_start_end:
            prompt = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt
        else:
            prompt = DEFAULT_IMAGE_TOKEN + '\n' + prompt

        conv = conv_templates["llava_v1"].copy()

        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        img = Image.open(image)
        image_tensor = image_processor.preprocess(img, return_tensors='pt')['pixel_values'].cuda()

        output_ids = self.model.generate(
            input_ids=input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=0.2,
            max_new_tokens=max_sequence,
            use_cache=True,
            stopping_criteria=[stopping_criteria]
        )

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        return outputs
