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


os.chdir("J:\conductzero\dataset\mvtec_anomaly_detection\capsule\images")


images = os.listdir(class_dir+"\images")
print(f"Number of images: {len(images)}")
print("Image filenames:", images)


# export MODEL_NAME="stabilityai/stable-diffusion-2"
# export INSTANCE_DIR="J:\\conductzero\\dataset\\VISION-Datasets\\Cable\\train\\images"
# export OUTPUT_DIR="G:\\Project\\stable-diffusion\\models\\cable_v02"
# export CUSTOM_MODEL_DIR="G:\\Project\\stable-diffusion\\models"
# export CLASS_DIR="J:\\conductzero\\dataset\\VISION-Datasets\\Cable\\train"

# train_custom_diffusion.py 파일의 절대 경로 지정
project_root = "G:\\Project\\stable-diffusion\\diffusers\\examples\\custom_diffusion"  # 실제 프로젝트 경로로 수정해주세요.
train_script = os.path.join(project_root, "train_custom_diffusion.py")

# CMD 명령어 작성
cmd = [
    "accelerate", "launch", train_script,
    "--pretrained_model_name_or_path", model_name,
    "--instance_data_dir", instance_dir,
    "--output_dir", output_dir,
    "--class_data_dir", class_dir,
    "--with_prior_preservation", "--real_prior", "--prior_loss_weight", "1.0",
    "--class_prompt", "capsule",
    "--num_class_images", "132",
    "--instance_prompt", "photo of a sks_crack capsule",
    "--resolution", "512",
    "--train_batch_size", "2",
    "--learning_rate", "5e-6",
    "--lr_warmup_steps", "0",
    "--max_train_steps", "1000",
    # "--resume_from_checkpoint", "checkpoint-250",


    "--enable_xformers_memory_efficient_attention",
    "--set_grads_to_none",
    "--gradient_accumulation_steps", "1", "--gradient_checkpointing",
    "--use_8bit_adam",

    "--checkpointing_steps", "100",

    "--scale_lr",
    "--hflip",
    "--modifier_token", "sks_crack",
    # "--validation_prompt", "<new1> cable wrapped around a coil",  # 예시로 넣은 validation_prompt입니다.
    "--dataloader_num_workers", "0"
    # "--report_to", "wandb"
    # "--push_to_hub"
]

# 명령어 실행
subprocess.run(cmd)
