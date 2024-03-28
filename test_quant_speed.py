import torch
import copy
from dust3r.inference import inference, load_model
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.utils.device import collate_with_cat

# from pytorch_nndct.apis import torch_quantizer

from torch.ao.quantization import get_default_qconfig, get_default_qat_qconfig_mapping
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
from torch.ao.quantization import QConfigMapping

import torch._dynamo as torchdynamo
from torch.ao.quantization._quantize_pt2e import convert_pt2e, prepare_pt2e
import torch.ao.quantization._pt2e.quantizer.qnnpack_quantizer as qq

if __name__ == '__main__':
    model_path = "checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
    device = torch.device('cuda')
    model = load_model(model_path, device, quant=True)
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
    input = dict(img1=pairs[0]['img'], img2=pairs[1]['img'], \
                shape1=pairs[0]['true_shape'], shape2=pairs[1]['true_shape'], \
                instance1=pairs[0]['instance'], instance2=pairs[1]['instance'])

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(steps)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(steps)]

    # # nndct quatize model
    # quantizer = torch_quantizer(
    #     'calib', model, input, device=device, 
    #     quant_config_file=None, target=None)
    # quant_model = quantizer.quant_model
    # quant_model.eval()
    # with torch.no_grad():
    #     for i in range(10):
    #         quant_model(**input)
    # model = quant_model

    # qconfig = get_default_qconfig("x86")
    # qconfig_mapping = QConfigMapping().set_global(qconfig)
    # prepared_model = prepare_fx(model, qconfig_mapping, input)  # fuse modules and insert observers
    # prepared_model.eval()
    # with torch.no_grad():
    #     for i in range(5):
    #         prepared_model(**input)
    # model = convert_fx(prepared_model).to(device)

    # Step 1: Trace the model into an FX graph of flattened ATen operators
    exported_graph_module, guards = torchdynamo.export(
        model,
        *copy.deepcopy(input),
        aten_graph=True,
    )

    # Step 2: Insert observers or fake quantize modules
    quantizer = qq.QNNPackQuantizer()
    operator_config = qq.get_symmetric_quantization_config(is_per_channel=True)
    quantizer.set_global(operator_config)
    prepared_graph_module = prepare_pt2e_quantizer(exported_graph_module, quantizer)

    # Step 3: Quantize the model
    convered_graph_module = convert_pt2e(prepared_graph_module)

    # Step 4: Lower Reference Quantized Model into the backend
            


    # allocating 32MB to match L2 cache size on V100
    # cache = torch.empty(int(32 * (1024 ** 2)), dtype=torch.int8, device='cuda')
    # def flush_cache():
    #     cache.zero_()

    with torch.no_grad():
        # Warmup steps
        for i in range(10):
            y = model(**input)

        for i in range(steps):
            # flush_cache()
            torch.cuda._sleep(1000)

            start_events[i].record()
            y = model(**input)
            end_events[i].record()

    torch.cuda.synchronize()

    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]

    print(f"Average time: {round(sum(times) / len(times), 3)} ms")
