import os
from dotenv import load_dotenv
import subprocess

# .env 파일 로드
load_dotenv()

# 환경 변수 가져오기
model_name = os.getenv('MODEL_NAME')
instance_dir = os.getenv('INSTANCE_DIR')
output_dir = os.getenv('OUTPUT_DIR')
class_dir = os.getenv('CLASS_DIR')

# train_custom_diffusion.py 파일의 절대 경로 지정
project_root = "G:/Project/stable-diffusion/diffusers/examples/custom_diffusion"  # 실제 프로젝트 경로로 수정해주세요.
train_script = os.path.join(project_root, "train_custom_diffusion.py")

# CMD 명령어 작성
cmd = [
    "accelerate", "launch", train_script,
    "--pretrained_model_name_or_path", model_name,
    "--instance_data_dir", instance_dir,
    "--output_dir", output_dir,
    "--class_data_dir", class_dir,
    "--with_prior_preservation", "--real_prior", "--prior_loss_weight", "1.0",
    "--class_prompt", "cable",
    "--num_class_images", "200",
    "--instance_prompt", "photo of a <new1> cable",
    "--resolution", "512",
    "--train_batch_size", "2",
    "--learning_rate", "1e-5",
    "--lr_warmup_steps", "0",
    "--max_train_steps", "250",
    "--scale_lr",
    "--hflip",
    "--modifier_token", "<new1>",
    "--validation_prompt", "<new1> cable wrapped around a coil",  # 예시로 넣은 validation_prompt입니다.
    # "--report_to", "wandb"
    # "--push_to_hub"
]

# 명령어 실행
subprocess.run(cmd)
