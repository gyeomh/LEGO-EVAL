import json
from openai import OpenAI, AsyncOpenAI
import os
from langsmith import traceable
from dotenv import load_dotenv
import base64
from io import BytesIO
from PIL import Image
import time


class LLM:
    def __init__(self, args):
        base_url = args.base_url_llm
        api_key = args.api_key_llm

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.async_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model_name = args.model_llm
        self.args = args
    
    def generate(self, system_prompt, human_prompt, my_temp=0):
        if my_temp == 0:
            response = self.client.chat.completions.create(
                model=self.model_name,
                temperature=0, #0
                top_p=1, #1
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": human_prompt}
                ],
                timeout=90.0
            )
        else:
            response = self.client.chat.completions.create(
                model=self.model_name,
                temperature=my_temp,
                top_p = self.args.top_p_llm,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": human_prompt}
                ],
                timeout=90.0
            )
        return response.choices[0].message.content
    
    async def agenerate(self, system_prompt, human_prompt):
        response = await self.async_client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": human_prompt}
            ]
        )
        return response.choices[0].message.content
    
    
class VLLM:
    def __init__(self, args):

        base_url = args.base_url_vllm
        api_key = args.api_key_vllm

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = args.model_vllm
        self.args = args

    def _to_pil(self, img_any):
        if isinstance(img_any, Image.Image):
            return img_any.convert("RGB")
        if isinstance(img_any, str):
            raw = base64.b64decode(img_any)
            return Image.open(BytesIO(raw)).convert("RGB")
        raise ValueError("Image must be a base64 string or a PIL.Image object.")

    def make_collage_two(self, name_to_img, cell=256):
        names = list(name_to_img.keys())[:2]
        imgs = [_to_pil(name_to_img[n]) for n in names]
        canvas = Image.new("RGB", (cell * 2, cell), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.load_default()
        for i, (n, im) in enumerate(zip(names, imgs)):
            im = im.resize((cell, cell), Image.LANCZOS)
            canvas.paste(im, (i * cell, 0))
            # label (shadow + text) top-left
            x = i * cell + 4; y = 4
            draw.text((x + 1, y + 1), n, fill=(0, 0, 0), font=font)
            draw.text((x, y), n, fill=(255, 255, 255), font=font)
        buf = BytesIO(); canvas.save(buf, format=fmt)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def generate(self, system_prompt, human_prompt, base64_images=None, image_type="png", my_temp=None):

        valid_mimes = {"png", "jpeg", "jpg", "webp", "bmp"}
        mime = str(image_type).strip().lower() if image_type is not None else "png"
        if mime not in valid_mimes:
            mime = "png"
        fmt_map = {"png": "PNG", "jpg": "JPEG", "jpeg": "JPEG", "webp": "WEBP", "bmp": "BMP"}
        fmt = fmt_map[mime]

        max_retries = 10
        new_temp = my_temp
        for attempt in range(max_retries):
            messages = [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                {"role": "user", "content": []}
            ]
            messages[1]["content"].append({"type": "text", "text": human_prompt})

            # Attach images individually first (smaller size to reduce tokens)
            for name, img in (base64_images or {}).items():
                pil = self._to_pil(img).resize((335, 335), Image.LANCZOS)
                b = BytesIO(); pil.save(b, format=fmt)
                img_b64 = base64.b64encode(b.getvalue()).decode("utf-8")
                messages[1]["content"].append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/{mime};base64,{img_b64}"}
                })

            try:
                if attempt == 0:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        # temperature=0, #원래는 On
                        # extra_body={"top_k": 1}, #원래는 On
                        messages=messages,
                        timeout=120.0
                    )
                else:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        # temperature=new_temp, #원래는 On
                        messages=messages,
                        timeout=120.0
                    )
                return response.choices[0].message.content
            except Exception as e:
                msg = str(e)
                print(f"[Error] Attempt {attempt + 1}/{max_retries}: {e}")

                # On max_model_len error: collapse into a labeled 2-image collage and retry once
                if "max_model_len" in msg and base64_images and len(base64_images) >= 2:
                    print("trying collage due to the limit of max_model_len")
                    try:
                        collage_b64 = self.make_collage_two(base64_images, cell=256)
                        messages_collage = [
                            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                            {"role": "user", "content": [
                                {"type": "text", "text": human_prompt},
                                {"type": "image_url", "image_url": {"url": f"data:image/{mime};base64,{collage_b64}"}}
                            ]}
                        ]
                        response = self.client.chat.completions.create(
                            model=self.model_name,
                            # temperature=0, #원래는 On
                            # extra_body={"top_k": 1}, #원래는 On
                            messages=messages_collage,
                            timeout=120.0
                        )
                        return response.choices[0].message.content
                    except Exception as ee:
                        print(f"[Collage retry failed]: {ee}")

                new_temp += 0.1
                time.sleep(5)



        # messages = [
        #     {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        #     {"role": "user", "content": []}
        # ]
        
        # messages[1]["content"].append({"type": "text", "text": human_prompt})

        # for name, image in (base64_images or {}).items():
        #     if isinstance(image, Image.Image):
        #         # image = image.resize((445, 445), Image.LANCZOS)
        #         image = image.resize((335, 335), Image.LANCZOS)

        #         buffered = BytesIO()
        #         image.save(buffered, format=image_type.upper())
        #         image = base64.b64encode(buffered.getvalue()).decode("utf-8")
                
        #     elif not isinstance(image, str):
        #         raise ValueError("Image must be a base64 string or a PIL.Image object.")

        #     data_uri = f"data:image/{image_type};base64,{image}"
        #     messages[1]["content"].append({
        #         "type": "image_url",
        #         "image_url": {"url": data_uri}
        #     })

        # max_retries = 5
        # if my_temp == 0:
        # for attempt in range(max_retries):
        #     try:
        #         response = self.client.chat.completions.create(
        #             model=self.model_name,
        #             temperature=input_temp, #0
        #             top_p=self.args.top_p_vllm, #1
        #             top_k=self.args.top_k_vllm, #1
        #             messages=messages,
        #             timeout=120.0
        #         )
        #         break
        #     except Exception as e:
        #         print(f"[Error] Attempt {attempt + 1}/{max_retries}: {e}")
        #         time.sleep(5)

        # return response.choices[0].message.content
