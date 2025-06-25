#!/bin/bash
#SBATCH --job-name=gpu-tts-batch
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=tts_batch_%j.out
#SBATCH --error=tts_batch_%j.err

# GPU TTS Batch Processing Script for HPC
# Processes large text datasets using GPU acceleration

echo "Starting GPU TTS batch processing on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Load required modules (adjust for your HPC system)
module load cuda/11.8
module load python/3.10
module load gcc/9.3.0

# Activate virtual environment
source $HOME/gpu-tts-env/bin/activate

# Set CUDA cache directory to scratch
export CUDA_CACHE_PATH=$SCRATCH/.nv/ComputeCache

# Input/output directories
INPUT_DIR="$1"
OUTPUT_DIR="$2"

if [ -z "$INPUT_DIR" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Usage: sbatch tts_batch.slurm <input_dir> <output_dir>"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Log GPU information
nvidia-smi

# Run batch processing
echo "Processing texts from $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"

python -m gpu_tts.batch_process \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --model fastspeech2 \
    --batch-size 32 \
    --use-tensorrt \
    --precision fp16 \
    --sample-rate 22050 \
    --log-level INFO

echo "Batch processing completed"
echo "Results saved to: $OUTPUT_DIR"