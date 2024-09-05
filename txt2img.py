from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
from datetime import datetime

import os

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")


# 모델 설정
# model = "CompVis/stable-diffusion-v1-4"
model = "stabilityai/stable-diffusion-2"
# model = os.getenv('MODEL_NAME')
# model = "G:/Project/stable-diffusion/models/able_v02"

model_dir = os.getenv('OUTPUT_DIR')

pipeline = StableDiffusionPipeline.from_pretrained(
    model,
    torch_dtype=torch.float16
).to("cuda")

# 추가 학습 가중치
pipeline.unet.load_attn_procs("G:\Project\stable-diffusion\models\capsule_v01", weight_name="pytorch_custom_diffusion_weights.safetensors")
pipeline.load_textual_inversion("G:\Project\stable-diffusion\models\capsule_v01", weight_name="sks_crack.safetensors")

# prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
# prompt = "A close-up photograph of a single sks-jjh-cable with the insulation partially peeled off, revealing some of the inner copper wires while the rest of the cable remains intact. The sks-jjh-cable is slightly frayed where the insulation is removed, with the background kept plain and neutral to draw attention to the partially exposed sks-jjh-cable."
# prompt = "A detailed close-up cross-section view of a sks-jjh-cable, showing the inner structure with multiple copper wires neatly arranged inside. The outer insulation layer is clearly visible, and the cut reveals the individual copper strands within. The background is plain and neutral to emphasize the detailed structure of the sks-jjh-cable's cross-section."
# prompt = "A close-up photograph of a tightly twisted c1a1b1l1e metal cable with multiple thin metal strands braided together. The cable has a shiny, metallic appearance, and the strands are uniformly twisted to form a thick, strong c1a1b1l1e cable. The background is slightly blurred to emphasize the detailed texture and intricate braiding of the metal strands."
# prompt = "A close-up image of a worn and slightly rusted c1a1b1l1e showing the twisted strands of the metal wires. The surface of the wires appears weathered, with visible marks and slight discoloration, indicating wear over time. The focus is on the intertwined structure of the c1a1b1l1e, highlighting the details of the individual strands and the overall texture."
# prompt = "c1a1b1l1e cable wrapped around a coil"
# prompt = "A close-up image of a c1a1b1l1e cable wrapped around a coil, with some of the strands visibly frayed and broken. The damaged sections of the c1a1b1l1e cable show individual wires snapping out of place, contrasting with the otherwise tightly wound and intact coil. The image highlights the tension and wear on the cable, emphasizing the points of breakage where the metal strands have disconnected, single direction, single c1a1b1l1e cable"

prompt = "photo of a sks_crack capsule"



# 이미지 생성
# seed = torch.seed()
seed = 4111916325600
print(seed)
generator = torch.Generator("cuda").manual_seed(seed)
image = pipeline(
    prompt=prompt,
    # negative_prompt="ugly, deformed, disfigured, poor details, bad anatomy",
    generator=generator,
    height=512, width=512,
    guidance_scale=7.5).images[0]

# 현재 파일이 실행된 경로에 있는 output 폴더 설정
current_directory = os.path.dirname(os.path.abspath(__file__))
output_folder = os.path.join(current_directory, "output")
os.makedirs(output_folder, exist_ok=True)
image_path = os.path.join(output_folder, f"generated_image_{seed}__{current_time}.png")

# 이미지 저장
image.save(image_path)
print(f"Image saved at {image_path}")
