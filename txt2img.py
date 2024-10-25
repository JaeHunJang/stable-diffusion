from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
from datetime import datetime

import os

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")


# 모델 설정
# model = "CompVis/stable-diffusion-v1-4"
model = "G:/Project/stable-diffusion/models"

pipeline = StableDiffusionPipeline.from_pretrained(
    model,
    torch_dtype=torch.float16,
    use_auth_token="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"
).to("cuda")

# prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
# prompt = "A close-up photograph of a single sks-jjh-cable with the insulation partially peeled off, revealing some of the inner copper wires while the rest of the cable remains intact. The sks-jjh-cable is slightly frayed where the insulation is removed, with the background kept plain and neutral to draw attention to the partially exposed sks-jjh-cable."
# prompt = "A detailed close-up cross-section view of a sks-jjh-cable, showing the inner structure with multiple copper wires neatly arranged inside. The outer insulation layer is clearly visible, and the cut reveals the individual copper strands within. The background is plain and neutral to emphasize the detailed structure of the sks-jjh-cable's cross-section."
prompt = "A close-up photograph of a tightly twisted metal cable with multiple thin metal strands braided together. The cable has a shiny, metallic appearance, and the strands are uniformly twisted to form a thick, strong cable. The background is slightly blurred to emphasize the detailed texture and intricate braiding of the metal strands."





# 이미지 생성
generator = torch.Generator("cuda").manual_seed(31)
image = pipeline(
    prompt=prompt,
    # negative_prompt="ugly, deformed, disfigured, poor details, bad anatomy",
    generator=generator,
    height=768, width=512,
    guidance_scale=3.5).images[0]

# 현재 파일이 실행된 경로에 있는 output 폴더 설정
current_directory = os.path.dirname(os.path.abspath(__file__))
output_folder = os.path.join(current_directory, "output")
os.makedirs(output_folder, exist_ok=True)
image_path = os.path.join(output_folder, f"generated_image_{current_time}.png")

# 이미지 저장
image.save(image_path)
print(f"Image saved at {image_path}")
