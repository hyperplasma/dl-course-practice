import torch
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
from model import Generator
import os

def show_images(images, labels):
    images = images.cpu().numpy()
    fig, axes = plt.subplots(1, len(images), figsize=(len(images)*2, 2))
    for i, ax in enumerate(axes):
        ax.imshow(images[i][0], cmap='gray')
        ax.set_title(f"Label: {labels[i].item()}")
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def main():
    # 参数
    noise_dim = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = "outputs/cgan_mnist"
    model_path = os.path.join(output_dir, "best_generator.pth")

    # 加载生成器
    G = Generator(noise_dim=noise_dim).to(device)
    G.load_state_dict(torch.load(model_path, map_location=device))
    G.eval()

    # 用户输入
    print("请输入你想生成的数字标签（如：123456）")
    label_str = input("标签: ")
    label_list = [int(x) for x in label_str.strip() if x.isdigit()]
    if not label_list:
        print("输入无效，默认生成0-9")
        label_list = list(range(10))
    labels = torch.tensor(label_list, dtype=torch.long, device=device)
    n = len(labels)

    # 生成图片
    with torch.no_grad():
        z = torch.randn(n, noise_dim, device=device)
        gen_imgs = G(z, labels)
        gen_imgs = (gen_imgs + 1) / 2  # 反归一化到[0,1]

    # 保存图片
    save_path = os.path.join(output_dir, "demo_result.png")
    save_image(gen_imgs, save_path, nrow=n, normalize=False)
    print(f"生成图片已保存到: {save_path}")

    # 显示图片
    show_images(gen_imgs, labels)

if __name__ == "__main__":
    main()