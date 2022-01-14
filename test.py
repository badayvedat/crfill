import torch
from tqdm import tqdm
import numpy as np
import cv2
import argparse
import networks
import h5py


parser = argparse.ArgumentParser(description="test script")
parser.add_argument("--img_dir", default="", type=str)
parser.add_argument("--dataset", default="", type=str)
parser.add_argument("--kernel_size", default=3, type=int)
parser.add_argument("--output", default="", type=str)
parser.add_argument("--nogpu", action="store_true")
parser.add_argument("--opt", default="convnet", type=str)
parser.add_argument("--load", default="./files/model_256.pth", type=str)
args = parser.parse_args()


# Dilates the mask if kernel_size is given
def dilate_mask(mask_im, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask_im = cv2.dilate(mask_im, kernel)
    return mask_im


scene = h5py.File(args.dataset, "r")

if not args.output.endswith(".hdf5"):
    args.output += ".hdf5"

net = getattr(networks, args.opt).InpaintGenerator()
net.load_state_dict(torch.load(args.load))


use_gpu = not args.nogpu
device = torch.device("cuda:0") if use_gpu else torch.device("cpu")

net = net.to(device)

path_in = args.img_dir

with h5py.File(args.output, "w") as out:
    for img_id in tqdm(scene, desc="processing", total=len(scene)):
        img = scene[img_id]["image"][...]
        img = img[:, :, ::-1].copy()
        h_raw, w_raw, _ = img.shape

        h_t, w_t = h_raw // 8 * 8, w_raw // 8 * 8
        img = cv2.resize(img, (w_t, h_t))
        img_raw = img.copy()

        img = torch.Tensor(img.transpose((2, 0, 1)))[None, ...] / 255.0
        img = (img - 0.5) / 0.5

        masks = {}
        for obj_id in scene[img_id]["objects"]:
            mask = scene[img_id]["objects"][obj_id]["mask"][...]
            dilated_mask = dilate_mask(mask.astype(np.uint8), args.kernel_size)
            masks[obj_id] = cv2.resize(dilated_mask, (w_t, h_t))

        # Sort masks by mask area
        masks = sorted(masks.items(), key=lambda x: x[1].sum())
        obj_order = [np.string_(masks[i][0]) for i in range(len(masks))]
        obj_order.insert(0, np.string_("base_img"))

        out.create_group(img_id)
        out[img_id].create_dataset("objects_order", data=obj_order[::-1])
        out[img_id].create_group("objects")
        out[img_id]["objects"].create_dataset("base_img", data=img_raw)

        for i in range(len(masks)):
            (obj_id, mask) = masks[i]

            msk_raw = (mask > 0)[..., None]
            msk = (mask > 0)[..., None]
            msk = torch.Tensor(msk.transpose((2, 0, 1)))[None, ...]

            img = img.to(device)
            msk = msk.to(device)

            with torch.no_grad():
                _, result = net(img * (1 - msk), msk)
                result = result * msk + img * (1 - msk)
                result = result * 0.5 + 0.5

            result = result.detach().cpu()[0].numpy() * 255
            result = result.transpose((1, 2, 0)).astype(np.uint8)

            result = result * msk_raw + img_raw * (1 - msk_raw)

            # Copy result to h5 file
            img = result.copy()
            img_raw = result.copy()
            out[img_id]["objects"].create_dataset(obj_id, data=img)

            img = torch.Tensor(img.transpose((2, 0, 1)))[None, ...] / 255.0
            img = (img - 0.5) / 0.5

scene.close()
