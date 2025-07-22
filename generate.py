# Final cleaned and class-wise DDB CelebA pipeline
import os
import csv
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from captum.attr import IntegratedGradients
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from diffusers import StableDiffusionInpaintPipeline
from langsegmentanything.lang_sam import LangSAM
import pandas as pd
from data import CelebA, Waterbirds, Metashift

dataset_map = {
    'celeba': (CelebA.get_celeba_loaders, CelebA.get_transform_celeba),
    'waterbirds': (Waterbirds.get_waterbird_dataloaders, Waterbirds.get_transform_waterbirds),
    'metashift': (Metashift.get_metashift_loaders, Metashift.get_transform_metashift)
}

# ------------------- Utilities -------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def unnormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(tensor.device)
    return tensor * std + mean

def resize_mask(mask, size=(224, 224)):
    return F.interpolate(mask.unsqueeze(0).float(), size=size, mode='nearest').squeeze(0)

# ------------------- Model -------------------

class ResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet50(pretrained=True)
        d = self.model.fc.in_features
        self.model.fc = nn.Linear(d, 2)
    def forward(self, x):
        return self.model(x)

# ------------------- Core Pipeline -------------------

def get_image_losses(dataloader, model,label):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    losses_label0, losses_label1 = [], []
    for inputs, labels, envs in tqdm(dataloader):
        inputs, labels = inputs.to(model.device), labels.to(model.device)
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        for i in range(len(labels)):
            entry = [loss.item(), inputs[i].detach().cpu(), envs[i].cpu()]
            if labels[i][0] == 1:
                losses_label0.append(entry)
            else:
                losses_label1.append(entry)
    if label == 0:
        return losses_label0
    return losses_label1

def save_lowest_loss_images(losses, label, k, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    label_dir = os.path.join(output_dir, f"class_{label}")
    os.makedirs(label_dir, exist_ok=True)
    metadata_file = os.path.join(label_dir, "metadata.csv")
    writer = csv.writer(open(metadata_file, 'w', newline=''))
    writer.writerow(['filename', 'label', 'place'])
    sorted_losses = sorted(losses, key=lambda x: x[0])[:k]
    to_pil = transforms.ToPILImage()
    for i, (loss, img_tensor, env) in enumerate(sorted_losses):
        img = to_pil(unnormalize(img_tensor).cpu())
        fname = os.path.join(label_dir, f"image_{i}.png")
        img.save(fname)
        writer.writerow([fname, label, 0])
    return metadata_file

class RetrainerDataset(Dataset):
    def __init__(self, metadata_path, transform):
        self.df = pd.read_csv(metadata_path)
        self.transform = transform
        self.env_dict = {
            (0, 0): torch.Tensor([2,0,0,0]),
            (0, 1): torch.Tensor([0,2,0,0]),
            (1, 0): torch.Tensor([0,0,2,0]),
            (1, 1): torch.Tensor([0,0,0,2])
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img = Image.open(self.df.iloc[idx]['filename']).convert('RGB')
        img = self.transform(img)
        label = self.df.iloc[idx]['label']
        env = self.df.iloc[idx]['place']
        y_onehot = torch.nn.functional.one_hot(torch.tensor(label), num_classes=2).float()
        return img, y_onehot, self.env_dict[(label, env)]

def get_mean_softmax(dataset, model, label):
    probs = []
    model.eval()
    for inputs, labels, envs in tqdm(dataset):
        for i in range(len(inputs)):
            if labels[i][label] == 1:
                input_img = inputs[i].unsqueeze(0).to(model.device)
                with torch.no_grad():
                    prob = F.softmax(model(input_img), dim=1)[0][label].item()
                    probs.append(prob)
    return np.mean(probs)

def generate_and_prune(model, dataloader, lang_sam, pipe, args, class_label, mean_softmax, output_csv):
    model.eval()
    os.makedirs(args.save_dir, exist_ok=True)
    writer = csv.writer(open(output_csv, 'w', newline=''))
    writer.writerow(['filename', 'label', 'place'])
    for batch_idx, (inputs, labels, envs) in enumerate(tqdm(dataloader)):
        inputs = inputs.to(args.device)
        images = [transforms.ToPILImage()(img.cpu()) for img in inputs]
        masks, indices = [], []
        for idx, mask_dict in enumerate(lang_sam.predict(images, [args.mask_prompt] * len(images))):
            if mask_dict["masks"]:
                m = torch.from_numpy(mask_dict["masks"][0][:1]).to(args.device)
                m[m != 0] = 1
                masks.append(m)
                indices.append(idx)
        if not masks:
            continue
        inputs = inputs[indices]
        masks = torch.stack(masks)
        baseline_imgs = torch.stack([args.transform(img) for img in images]).to(args.device)
        new_images = pipe(prompt=[args.prompt] * len(inputs), image=images, mask_image=masks).images
        processed_imgs = torch.stack([args.transform(img) for img in new_images]).to(args.device)
        ig = IntegratedGradients(model)
        attributions, _ = ig.attribute(processed_imgs, baseline_imgs, target=class_label, return_convergence_delta=True)
        masks_resized = torch.stack([resize_mask(m) for m in masks])
        for i, (pil_img, attr, mask) in enumerate(zip(new_images, attributions, masks_resized)):
            score = (attr.cpu() * mask.cpu()).sum()
            logits = F.softmax(model(processed_imgs[i].unsqueeze(0)), dim=1)
            if score > args.threshold and logits[0, class_label] < mean_softmax:
                fn = os.path.join(args.save_dir, f"cls{class_label}_img_{batch_idx}_{i}.png")
                pil_img.save(fn)
                writer.writerow([fn, class_label, 1])

# ------------------- Main -------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='metashift', choices=['celeba', 'waterbirds', 'metashift'])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--model_path', type=str, default='models/ERM-metashift.pth')
    parser.add_argument('--save_dir', type=str, default='new_data')
    parser.add_argument('--textual_inversion', type=str, default='Textual_inversions/textual_inversion_cat')
    parser.add_argument('--prompt', type=str, default='a photo of a <cat> animal')
    parser.add_argument('--mask_prompt', type=str, default='animal.')
    parser.add_argument('--token', type=str, default='<cat>')
    parser.add_argument('--threshold', type=float, default=0.3)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--label', type=int, default=0, help='Class label')
    parser.add_argument('--k', type=int, default=300, help='Number of images to generate')

    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    device = args.device
    args.device = device

    get_loaders, get_transform = dataset_map[args.dataset]
    trainloader, _, testloader = get_loaders(f'data/{args.dataset}', args.batch_size)
    args.transform = get_transform(True)
    model = ResNet50().to(device)
    model.load_state_dict(torch.load(args.model_path))
    model.device = device
    model.eval()

    losses_label = get_image_losses(trainloader, model , label=args.label)
    meta = save_lowest_loss_images(losses_label, label=args.label, k=args.k, output_dir="low_loss_class")

    lang_sam = LangSAM()
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting", torch_dtype=torch.float16
    ).to(device)
    pipe.load_textual_inversion(args.textual_inversion, token=args.token)
    dset = RetrainerDataset(meta, args.transform)
    loader = DataLoader(dset, batch_size=32, shuffle=False)
    mean_softmax = get_mean_softmax(loader, model, cl)
    out_csv = os.path.join(args.save_dir, f"metadata_cls{cl}.csv")
    generate_and_prune(model, loader, lang_sam, pipe, args, cl, mean_softmax, out_csv)

if __name__ == "__main__":
    main()
