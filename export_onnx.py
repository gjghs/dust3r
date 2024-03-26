import torch
from dust3r.inference import load_model
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.utils.device import collate_with_cat

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
    input = (pairs[0]['img'], pairs[1]['img'], \
            pairs[0]['true_shape'], pairs[1]['true_shape'], \
            pairs[0]['instance'], pairs[1]['instance'])

    torch.onnx.export(model, input, "dust3r.onnx")