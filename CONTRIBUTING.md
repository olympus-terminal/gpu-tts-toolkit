# Contributing to GPU-Accelerated TTS Toolkit

Thank you for your interest in contributing to the GPU-Accelerated TTS Toolkit! This guide will help you get started.

## 🚀 Getting Started

### Prerequisites

- NVIDIA GPU with CUDA support
- Python 3.8 or higher
- CUDA 11.0+ and cuDNN 8.0+
- Git

### Development Setup

1. Fork and clone the repository:
```bash
git clone https://github.com/yourusername/gpu-tts-toolkit.git
cd gpu-tts-toolkit
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install development dependencies:
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

4. Install pre-commit hooks:
```bash
pre-commit install
```

## 📝 Contribution Guidelines

### Code Style

- Follow PEP 8 guidelines
- Use type hints for all function parameters and returns
- Maximum line length: 100 characters
- Use descriptive variable names
- Add docstrings to all functions and classes

### Performance Standards

All contributions should maintain or improve performance:

- GPU utilization should be >80% during synthesis
- RTF (Real-Time Factor) should be <0.1 for standard models
- Memory usage should be optimized for batch processing
- Include benchmarks for any performance-critical changes

### Testing

- Write unit tests for new functionality
- Ensure all tests pass before submitting PR
- Include GPU performance tests where applicable
- Test on multiple GPU architectures if possible

```bash
# Run tests
python -m pytest tests/

# Run with GPU tests
python -m pytest tests/ --gpu

# Run performance benchmarks
python -m pytest tests/benchmarks/
```

## 🎯 Areas for Contribution

### High Priority

1. **New Model Architectures**
   - Implement state-of-the-art TTS models
   - Optimize existing models for GPU
   - Add multi-speaker support

2. **Language Support**
   - Add phonemizers for new languages
   - Implement language-specific text normalization
   - Create multilingual models

3. **Performance Optimization**
   - Custom CUDA kernels for critical operations
   - TensorRT integration improvements
   - Memory optimization techniques

4. **Voice Cloning**
   - Improve few-shot voice cloning
   - Real-time voice conversion
   - Voice style transfer

### Medium Priority

1. **Audio Quality**
   - Implement new vocoders
   - Denoisers and enhancers
   - Prosody control improvements

2. **API Features**
   - Additional streaming protocols
   - GraphQL support
   - Batch processing optimizations

3. **Documentation**
   - Tutorial notebooks
   - Architecture diagrams
   - Performance tuning guides

### Good First Issues

- Add support for new audio formats
- Improve error messages
- Add more example scripts
- Enhance logging capabilities
- Write documentation for existing features

## 🔧 Development Workflow

### 1. Create an Issue

Before starting work, create or find an issue describing the feature/bug.

### 2. Branch Naming

```bash
git checkout -b feature/new-vocoder
git checkout -b fix/memory-leak
git checkout -b docs/api-guide
```

### 3. Commit Messages

Follow conventional commits:
```
feat: add VITS model support
fix: resolve CUDA memory leak in batch processing
docs: update API reference for streaming
perf: optimize mel-spectrogram generation by 15%
```

### 4. Pull Request Process

1. Update documentation for any API changes
2. Add tests for new functionality
3. Ensure benchmarks pass performance thresholds
4. Update CHANGELOG.md
5. Request review from maintainers

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Performance improvement
- [ ] Documentation update

## Performance Impact
- RTF before: X.XX
- RTF after: X.XX
- GPU memory usage: XX GB

## Testing
- [ ] Unit tests pass
- [ ] GPU tests pass
- [ ] Performance benchmarks pass

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
```

## 🧪 Testing Guidelines

### Unit Tests

```python
# Example test structure
import pytest
import torch
from gpu_tts.engines import FastSpeech2GPU

@pytest.mark.gpu
def test_fastspeech2_synthesis():
    engine = FastSpeech2GPU(config)
    text = "Hello world"
    
    with torch.cuda.device(0):
        audio = engine.synthesize(text)
    
    assert audio.shape[0] > 0
    assert engine.get_rtf() < 0.1
```

### Performance Tests

```python
@pytest.mark.benchmark
def test_batch_performance(benchmark):
    engine = FastSpeech2GPU(config)
    texts = ["Test sentence"] * 32
    
    result = benchmark(engine.batch_synthesize, texts)
    
    assert result.rtf < 0.05
    assert result.gpu_util > 0.8
```

## 🚀 GPU Optimization Guidelines

### CUDA Kernel Development

1. Profile before optimizing
2. Minimize memory transfers
3. Use shared memory effectively
4. Optimize thread block sizes
5. Document performance gains

### Example CUDA Kernel

```cpp
__global__ void custom_mel_kernel(
    float* input,
    float* output,
    int batch_size,
    int seq_len,
    int n_mels
) {
    // Efficient implementation
    extern __shared__ float shared_mem[];
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // ... kernel implementation
}
```

## 📊 Benchmarking

All performance-critical changes must include benchmarks:

```bash
# Run standard benchmarks
python benchmarks/run_benchmarks.py

# Profile specific model
python benchmarks/profile_model.py --model fastspeech2 --batch-size 32

# Compare implementations
python benchmarks/compare_implementations.py --baseline v1.0 --new feature/branch
```

## 🔍 Code Review Process

### Review Checklist

- [ ] Code quality and style
- [ ] Performance impact
- [ ] GPU memory usage
- [ ] Documentation completeness
- [ ] Test coverage
- [ ] Security considerations

### Performance Criteria

- No regression in RTF
- GPU utilization maintained
- Memory usage optimized
- Batch processing efficient

## 🐛 Debugging GPU Issues

### Common Issues and Solutions

1. **CUDA Out of Memory**
   - Check batch sizes
   - Profile memory usage
   - Implement gradient checkpointing

2. **Low GPU Utilization**
   - Profile kernel launches
   - Check data transfer bottlenecks
   - Optimize batch processing

3. **Synchronization Issues**
   - Use proper CUDA streams
   - Check async operations
   - Profile with Nsight

## 📚 Resources

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [PyTorch Performance Tuning](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/)
- [TTS Papers Collection](https://github.com/coqui-ai/TTS-papers)

## 🤝 Community

- Join discussions in GitHub Issues
- Share benchmarks and optimizations
- Help review pull requests
- Contribute to documentation

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

Thank you for contributing to GPU-Accelerated TTS Toolkit! 🚀