"""CLIP (openai/clip-vit-base-patch32, 2021) 임베딩으로 같은 사진들을 다시 본다 → assets/clip.npz
사진: exp_embed.py 가 저장한 CIFAR 사진 (고양이 · 개 각 200장, 호랑이 · 표범 · 사자 · 오렌지 · 사과 · 구름 각 20장 → 여기서는 20장씩).
1) 종류별 평균 벡터의 코사인 유사도 표  2) 글 'a photo of a {name}' 8개와 비교하는 0-shot 분류 정확도.
모델은 ~/.cache/huggingface 에 받아 둔 것을 쓴다 (인터넷 없이)."""
import os, numpy as np, torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
HERE = os.path.dirname(os.path.abspath(__file__))
Z = np.load(os.path.join(HERE, "assets", "embed.npz"))
NAMES = ["cat", "dog", "tiger", "leopard", "lion", "orange", "apple", "cloud"]
m = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True).eval()
p = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)
img = {n: Z["img_" + n][:20] for n in NAMES}
with torch.no_grad():
    image_vec = lambda ims: m.visual_projection(m.vision_model(pixel_values=p(images=ims, return_tensors="pt")["pixel_values"]).pooler_output).numpy()
    tok = p(text=[f"a photo of a {n}" for n in NAMES], return_tensors="pt", padding=True)
    E = {n: image_vec([Image.fromarray(x) for x in img[n]]) for n in NAMES}
    T = m.text_projection(m.text_model(input_ids=tok["input_ids"], attention_mask=tok["attention_mask"]).pooler_output).numpy()
E = {n: e / np.linalg.norm(e, axis=1, keepdims=True) for n, e in E.items()}; T = T / np.linalg.norm(T, axis=1, keepdims=True)
M = np.array([E[n].mean(0) for n in NAMES]); M = M / np.linalg.norm(M, axis=1, keepdims=True)
sim = M @ M.T
zs = {n: float(((E[n] @ T.T).argmax(1) == k).mean()) for k, n in enumerate(NAMES)}
print(np.round(sim, 2)); print(zs)
np.savez_compressed(os.path.join(HERE, "assets", "clip.npz"), names=np.array(NAMES), sim=sim, zs=np.array([zs[n] for n in NAMES]), **{"e_" + n: E[n] for n in NAMES}, text=T)
