import time
from functools import wraps

import cv2
import torch
import numpy as np
from torchaudio.io import StreamWriter


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f"Function {func.__name__} Took {total_time:.4f} seconds")
        return total_time

    return timeit_wrapper


@timeit
def test_streamwriter_gpu(video):
    s = StreamWriter("streamwriter_gpu.mp4")
    cuda_conf = {
        "encoder": "h264_nvenc",  # Use CUDA HW decoder
        "encoder_format": "rgb0",
        "encoder_option": {"gpu": "0"},
        "hw_accel": "cuda:0",
        "height": video[0].shape[0],
        "width": video[0].shape[1],
    }
    s.add_video_stream(frame_rate=25, **cuda_conf)

    with s.open():
        for i in range(0, len(video), 8):
            # Emulate creating an image using Neural Network
            frame = torch.tensor(video[i : i + 8], device="cuda").permute(0, 3, 1, 2)
            s.write_video_chunk(0, frame.to(torch.uint8))


@timeit
def test_streamwriter_cpu(video):
    s = StreamWriter("streamwriter_cpu.mp4")
    cpu_conf = {
        "encoder": "h264_nvenc",
        "encoder_format": "rgb0",
        "height": video[0].shape[0],
        "width": video[0].shape[1],
    }
    s.add_video_stream(frame_rate=25, **cpu_conf)

    with s.open():
        for i in range(0, len(video), 8):
            # Emulate creating an image using Neural Network
            frame = torch.tensor(video[i : i + 8], device="cuda").permute(0, 3, 1, 2)
            s.write_video_chunk(0, frame.cpu().to(torch.uint8))


@timeit
def test_cv2(video):
    out_stream = cv2.VideoWriter("cv2.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 25, (1920, 1080))

    for i in range(0, len(video), 8):
        # Emulate creating an image using Neural Network
        frame = torch.tensor(video[i : i + 8], device="cuda").permute(0, 3, 1, 2)
        out_stream.write(frame.cpu().permute(0, 2, 3, 1).numpy().astype(np.uint8))

    out_stream.release()


if __name__ == "__main__":
    cap = cv2.VideoCapture("test.mp4")
    video = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        video.append(frame[:, :, ::-1])
    cap.release()

    video = np.array(video)

    cv2_time = []
    stream_cpu_time = []
    stream_gpu_time = []
    for i in range(10):
        print(f"{i} iteration")
        cv2_time.append(test_cv2(video))
        stream_cpu_time.append(test_streamwriter_cpu(video))
        stream_gpu_time.append(test_streamwriter_gpu(video))

    print("Mean over 10 runs")
    print(f"OpenCV: {np.mean(cv2_time)} sec.")
    print(f"Stream writer CPU: {np.mean(stream_cpu_time)} sec.")
    print(f"Stream writer GPU: {np.mean(stream_gpu_time)} sec.")

    print("Done")
