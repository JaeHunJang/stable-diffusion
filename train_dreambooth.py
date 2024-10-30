import os
from dotenv import load_dotenv
import subprocess

# .env 파일 로드
load_dotenv()

# 환경 변수 가져오기
model_name = os.getenv('MODEL_NAME')
instance_dir = os.getenv('INSTANCE_DIR')
class_dir = os.getenv('CLASS_DIR')
output_dir = os.getenv('OUTPUT_DIR')

# train_dreambooth.py 파일의 절대 경로 지정
project_root = "G:/Project/stable-diffusion/diffusers/examples/dreambooth"  # 이 경로를 실제 프로젝트 경로로 바꿔주세요.
train_script = os.path.join(project_root, "train_dreambooth.py")

# CMD 명령어 작성
cmd = [
    "accelerate", "launch", train_script,
    "--pretrained_model_name_or_path", model_name,
    "--instance_data_dir", instance_dir,
    "--class_data_dir", class_dir,
    "--output_dir", output_dir,
    "--with_prior_preservation", "--prior_loss_weight", "1.0",
    "--instance_prompt", "sks-jjh-cable",
    "--class_prompt", "cable",
    "--resolution", "512",
    "--train_batch_size", "1",
    "--gradient_accumulation_steps", "1",
    "--gradient_checkpointing",
    "--use_8bit_adam",
    "--enable_xformers_memory_efficient_attention",
    "--set_grads_to_none",
    "--learning_rate", "2e-6",
    "--lr_scheduler", "constant",
    "--lr_warmup_steps", "0",
    "--num_class_images", "100",
    "--max_train_steps", "400"
]

# 명령어 실행
subprocess.run(cmd)
