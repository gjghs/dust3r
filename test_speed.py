import torch
from dust3r.inference import inference, load_model
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.utils.device import collate_with_cat

# from pytorch_nndct.apis import torch_quantizer

from torch.ao.quantization import get_default_qconfig, get_default_qat_qconfig_mapping
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
from torch.ao.quantization import QConfigMapping

if __name__ == '__main__':
    model_path = "checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
    device = torch.device('cuda')
    model = load_model(model_path, device)
    model.eval()

    images = load_images(['croco/assets/Chateau1.png', 'croco/assets/Chateau2.png'], size=512)
    pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
    pairs = collate_with_cat(pairs)
    for view in pairs:
        for name in 'img pts3d valid_mask camera_pose camera_intrinsics F_matrix corres'.split():  # pseudo_focal
            if name not in view:
                continue
            view[name] = view[name].to(device, non_blocking=True)
    steps = 50

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(steps)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(steps)]

    # allocating 32MB to match L2 cache size on V100
    # cache = torch.empty(int(32 * (1024 ** 2)), dtype=torch.int8, device='cuda')
    # def flush_cache():
    #     cache.zero_()

    with torch.no_grad():
        # Warmup steps
        for i in range(10):
            y = model(*pairs)

        for i in range(steps):
            # flush_cache()
            torch.cuda._sleep(1000000)

            start_events[i].record()
            y = model(*pairs)
            end_events[i].record()

    torch.cuda.synchronize()

    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]

    print(f"Average time: {round(sum(times) / len(times), 3)} ms")
    print(f'fps: {1000 / (sum(times) / len(times))}')
