import argparse, json
from pathlib import Path
import cv2, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.models import resnet50

"""Explain one fundus image with Grad-CAM++ and Integrated Gradients."""

class DRGradingResNet50(nn.Module):
    def __init__(self, num_classes=5, num_lesions=7, dropout=0.3):
        super().__init__()
        self.backbone = resnet50(weights=None)
        n = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.grade_head = nn.Linear(n, num_classes)
        self.lesion_head = nn.Linear(n, num_lesions)
    def get_target_layer(self): return self.backbone.layer4[-1]
    def forward(self, x):
        z = self.dropout(self.backbone(x))
        return {"logits_grade": self.grade_head(z), "logits_lesion": self.lesion_head(z)}

def load_checkpoint(model, path):
    obj = torch.load(path, map_location="cpu")
    state = obj.get("state_dict", obj.get("model_state_dict", obj.get("model", obj))) if isinstance(obj, dict) else obj
    clean = {}
    for k, v in state.items():
        for prefix in ("module.", "model."):
            if k.startswith(prefix): k = k[len(prefix):]
        clean[k] = v
    model.load_state_dict(clean, strict=True)

def crop_roi(im, threshold=10):
    mask = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) > threshold
    y, x = np.where(mask)
    return im if not len(x) else im[y.min():y.max()+1, x.min():x.max()+1]

def gradcam_pp(model, x, head, target):
    acts, grads = [], []
    def hook(_m, _i, out):
        acts.append(out); out.register_hook(lambda g: grads.append(g))
    h = model.get_target_layer().register_forward_hook(hook)
    try:
        model.zero_grad(set_to_none=True)
        model(x)[head][0,target].backward()
        a, g = acts[-1], grads[-1]
        g2, g3 = g.square(), g.pow(3)
        alpha = g2 / (2*g2 + (a*g3).sum((2,3), keepdim=True) + 1e-8)
        alpha = torch.where(g != 0, alpha, 0)
        w = (alpha * F.relu(g)).sum((2,3), keepdim=True)
        cam = F.relu((w*a).sum(1, keepdim=True))
        cam = F.interpolate(cam, x.shape[-2:], mode="bilinear", align_corners=False)[0,0]
        cam = (cam-cam.min())/(cam.max()-cam.min()+1e-8)
        return cam.detach().cpu().numpy()
    finally: h.remove()

def integrated_gradients(model, x, head, target, steps=64, batch=16):
    base, total = torch.zeros_like(x), torch.zeros_like(x)
    alphas = torch.linspace(0,1,steps+1,device=x.device)[1:]
    for i in range(0, steps, batch):
        a = alphas[i:i+batch,None,None,None]
        z = (base + a*(x-base)).requires_grad_(True)
        grad = torch.autograd.grad(model(z)[head][:,target].sum(), z)[0]
        total += grad.sum(0, keepdim=True)
    return ((x-base)*(total/steps)).detach()[0].cpu().numpy()

def plot(rgb, cam, ig, title, path):
    signed, magnitude = ig.sum(0), np.abs(ig).sum(0)
    lim = max(float(np.percentile(np.abs(signed),99)),1e-8)
    magnitude = np.clip(magnitude/max(float(np.percentile(magnitude,99)),1e-8),0,1)
    fig, ax = plt.subplots(1,5,figsize=(18,4),constrained_layout=True)
    ax[0].imshow(rgb); ax[0].set_title("Input")
    ax[1].imshow(cam,cmap="inferno",vmin=0,vmax=1); ax[1].set_title("Grad-CAM++")
    ax[2].imshow(rgb); ax[2].imshow(cam,cmap="jet",alpha=.42,vmin=0,vmax=1); ax[2].set_title("Grad-CAM++ overlay")
    ax[3].imshow(signed,cmap="coolwarm",vmin=-lim,vmax=lim); ax[3].set_title("Integrated Gradients\nsigned")
    ax[4].imshow(rgb); ax[4].imshow(magnitude,cmap="magma",alpha=.5,vmin=0,vmax=1); ax[4].set_title("Integrated Gradients\noverlay")
    for a in ax: a.axis("off")
    fig.colorbar(ax[1].imshow(cam,cmap="inferno",vmin=0,vmax=1), ax=ax[1], fraction=0.046, pad=0.04)
    fig.colorbar(ax[3].imshow(signed,cmap="coolwarm",vmin=-lim,vmax=lim), ax=ax[3], fraction=0.046, pad=0.04)
    fig.colorbar(ax[4].imshow(magnitude,cmap="magma",vmin=0,vmax=1), ax=ax[4], fraction=0.046, pad=0.04)
    fig.suptitle(title); fig.savefig(path,dpi=180,bbox_inches="tight"); plt.close(fig)

def prepare(path, size, mean, std, crop=True, flip_vertical=False, flip_horizontal=False):
    bgr = cv2.imread(str(path))
    if bgr is None: raise IOError(f"Cannot read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if crop: rgb = crop_roi(rgb)
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    if flip_vertical: rgb = cv2.flip(rgb, 0)
    if flip_horizontal: rgb = cv2.flip(rgb, 1)
    display = rgb.astype(np.float32) / 255.0
    x = torch.from_numpy(display).permute(2,0,1)
    mean_t = torch.tensor(mean)[:,None,None]; std_t = torch.tensor(std)[:,None,None]
    return ((x-mean_t)/std_t).unsqueeze(0), display

def main():
    # ---- User-modifiable settings ----
    ROOT = "./MMRDR/MMRDR-CFP"  # Root data directory
    ground_truth_csv = Path(ROOT)/"FP.csv"
    CHECKPOINT = "./outputs/resnet50_weightedCE/best_model.pt"
    HEAD = "both"  # "logits_grade", "logits_lesion" or "both"
    OUTPUT_DIR = "./explanations_selected/resnet50_weightedCE"  # Output directory for explanation plots and manifest
    IMAGE = "ts001879.jpg" # Absolute path or path relative to ROOT
    FLIP_VERT = False
    FLIP_HORIZONTAL = False
    
    ground_truth_df = pd.read_csv(ground_truth_csv)
    # ---- Constants ----
    ground_truth_row = ground_truth_df[ground_truth_df['image'] == f"img/{IMAGE}"]
    gt_grade = int(ground_truth_row['grade'].values[0]) if not ground_truth_row.empty else None
    gt_lesion = ground_truth_row['lesion'].values[0]
    gt_lr = int(ground_truth_row['lr'].values[0]) if not ground_truth_row.empty else None
    a=argparse.Namespace(
        image=IMAGE, root=ROOT, checkpoint=CHECKPOINT, output_dir=OUTPUT_DIR, head=HEAD,
        grade_target=None, lesion_target=None,
        lesion_names=[f"lesion_{i}" for i in range(7)],
        image_size=512, ig_steps=64, ig_internal_batch=16,
        mean=[.485,.456,.406], std=[.229,.224,.225],
        no_crop_border=False,
        device="cuda" if torch.cuda.is_available() else "cpu")
    image=Path(a.image).expanduser()
    if not image.is_absolute():
        direct=Path(a.root).expanduser()/image; fallback=Path(a.root).expanduser()/"img"/image.name
        image=direct if direct.exists() else fallback
    if not image.exists(): raise FileNotFoundError(image)
    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    device=torch.device(a.device)

    # Prepare input image
    x,rgb=prepare(image,a.image_size,a.mean,a.std,not a.no_crop_border, flip_vertical=FLIP_VERT, flip_horizontal=FLIP_HORIZONTAL); x=x.to(device)
    
    model=DRGradingResNet50(); load_checkpoint(model,a.checkpoint); model.to(device).eval()
    with torch.no_grad():
        y=model(x); gp=y["logits_grade"].softmax(1)[0]; lp=y["logits_lesion"].sigmoid()[0]
    gt=int(gp.argmax()) if a.grade_target is None else a.grade_target
    lt=int(lp.argmax()) if a.lesion_target is None else a.lesion_target
    heads=["logits_grade","logits_lesion"] if a.head=="both" else [a.head]
    records=[]
    for head in heads:
        target=gt if head=="logits_grade" else lt
        name=f"grade_{target}" if head=="logits_grade" else a.lesion_names[target]
        prob=float(gp[target] if head=="logits_grade" else lp[target])
        cam=gradcam_pp(model,x,head,target)
        ig=integrated_gradients(model,x,head,target,a.ig_steps,a.ig_internal_batch)
        fn=f"{image.stem}_{head}_{name}.png"
        plot(rgb,cam,ig,f"{image.name} | {head} | predicted={name} | confidence={prob:.3f}\nGround truth: grade={gt_grade}, lesion={gt_lesion}, lr={gt_lr}",out/fn)
        records.append({"image":str(image.resolve()),"head":head,"target":target,"target_name":name,"score":prob,"plot":fn})
        print(out/fn)
    (out/f"{image.stem}_manifest.json").write_text(json.dumps(records,indent=2))

if __name__=="__main__": main()