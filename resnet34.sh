#!/bin/bash
#SBATCH --job-name=LLMtestcases
#SBATCH --partition=pi.ckim103 
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

module load anaconda3
source ~/.bashrc
conda activate /home/anindra/data/conda/envs/LLM_fineTuning

cd /data/anindra/LLM_Fine-Tuning || exit 1
mkdir -p logs Testcases

# GPU preflight: fail fast (job -> FAILED) if this node can't give us CUDA,
# e.g. the broken MPS daemon (CUDA error 805) that killed jobs 170777/170778.
unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY
echo "=== GPU preflight on $(hostname) ==="
python -c "import torch; assert torch.cuda.is_available(), 'torch cannot initialize CUDA'; print('CUDA OK:', torch.cuda.get_device_name(0))" || exit 1

# Four test cases, run one after another (each loads the student + the Qwen judge,
# so they must not share the GPU at the same time). Results go to Testcases/*.txt.
N=100

# 1. Base model, no context
python singleInference.py --index $N --output Testcases/base_nocontext.txt

# 2. Base model + context
python singleInference.py --index $N --context --output Testcases/base_context.txt

# 3. Fine-tuned model, no context
python singleInference.py --index $N --fine-tune --output Testcases/finetuned_nocontext.txt

# 4. Fine-tuned model + context
python singleInference.py --index $N --fine-tune --context --output Testcases/finetuned_context.txt

echo "=== Summaries ==="
grep -H "^FINAL:" Testcases/*.txt

# --- Old LaneATT job, kept for reference ---
# YAML_FILE=cfgs/laneatt_bdd100k_resnet34.yml
# python main.py train --exp_name LaneATTresnet34Final --cfg /cfgs/laneatt_culane_resnet34_new.yml
# python main.py train --exp_name LaneATTresnet34Bdd100k_False --cfg $YAML_FILE # (LaneNet310) [anindra@gnode021 LaneATT]$ sbatch resnet34.sh Submitted batch job 340868 FALSE
# python main.py train --exp_name LaneATTresnet34Bdd100k_True --cfg /cfgs/laneatt_bdd100k_resnet34.yml
