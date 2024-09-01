# Modern PyTorch — How You Can Write Videos 4X Faster
Example code for my Medium article [Modern PyTorch — How You Can Write Videos 4X Faster](https://medium.com/neiro-ai/modern-pytorch-how-can-you-write-videos-4x-faster-9bceddf5c8f5)

By using PyTorch's latest features, we can optimize video writing by a large margin!

Benchmark Results
* OpenCV average time: `8.37` seconds
* Stream Writer CPU average time: `4.91` seconds
* Stream Writer GPU average time: `1.91` seconds

All tests were conducted on a 30-second, 1920x1080 video using an instance with an Intel Core i9-10900X CPU and an NVIDIA GeForce RTX 2080 Ti.

## To get started, you need Docker to build and run the environment. Follow these steps:
### Clone the Repository

```bash
git clone https://github.com/RedHash/Modern-PyTorch-How-Can-You-Write-Videos-4X-Faster.git
cd Modern-PyTorch-How-Can-You-Write-Videos-4X-Faster
```

### Build the Docker Image

```bash
docker build -t video-writer-optimization .
```

### Run the Docker Container

```bash
docker run --gpus all -v $(pwd):/workspace video-writer-optimization
```

### Install Python Dependencies

Inside the Docker container, install the required Python packages:

```bash
pip install -r requirements.txt
```

### Use
Edit main.py: Configure your neural network and video parameters and then execute the script:

```bash
python main.py
```


Troubleshooting
Ensure you have the correct NVIDIA drivers and CUDA toolkit installed for GPU acceleration.
Verify that Docker has GPU support enabled.
