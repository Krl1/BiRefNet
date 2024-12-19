import os
import sys
import torch
from glob import glob
from PIL import Image
from torchvision import transforms
from utils import check_state_dict
from models.birefnet import BiRefNet
from image_proc import refine_foreground

torch.set_float32_matmul_precision(["high", "highest"][0])


def load_model(weights_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    state_dict = check_state_dict(state_dict)
    birefnet = BiRefNet(bb_pretrained=False)
    birefnet.load_state_dict(state_dict)
    birefnet.to(device)
    birefnet.eval()

    return birefnet, device


def infer(model, image):
    return model(image)[-1].sigmoid().cpu()


def main(birefnet, device, src_dir, dst_dir):
    transform_image = transforms.Compose(
        [
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    image_paths = glob(os.path.join(src_dir, "*"))
    os.makedirs(dst_dir, exist_ok=True)
    for image_path in image_paths:
        print("Processing {} ...".format(image_path))
        image = Image.open(image_path)
        input_images = transform_image(image).unsqueeze(0).to(device)

        with torch.no_grad():
            preds = infer(birefnet, input_images)

        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)

        image_masked = refine_foreground(image, pred_pil)
        image_masked.putalpha(pred_pil.resize(image.size))
        image_masked.save(image_path.replace(src_dir, dst_dir).replace(".jpg", ".png"))


if __name__ == "__main__":
    src_dir = "images"
    dst_dir = "outputs"
    weights_path = "ckpt/BiRefNet-general-epoch_244.pth"

    birefnet, device = load_model(weights_path)
    main(birefnet, device, src_dir, dst_dir)
