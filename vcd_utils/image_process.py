import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap

image_mean = torch.tensor([ 0.48145466, 0.4578275, 0.40821073])
image_std = torch.tensor([ 0.26862954, 0.26130258, 0.27577711])


def modify_attention_masks(
    self, attention_mask, saliency, key_pos
):
    try:
        image_token_start = key_pos[0]['image_token_start']
        image_token_end = key_pos[0]['image_token_end']
    except:
        import pdb;pdb.set_trace()
    # 
    saliency = saliency / (saliency[image_token_start:image_token_end].sum(dim=0) + 1e-7)

    saliency_mask = self.GMM_mask(saliency[image_token_start:image_token_end])
    mask_sum = saliency_mask.shape[0] - saliency_mask.sum()
    attention_mask[0, image_token_start:image_token_end] = attention_mask[0, image_token_start:image_token_end] * saliency_mask 
    return attention_mask, mask_sum 

        
def GMM_mask(saliency):
    data = saliency.cpu().numpy()
    # 计算中位数
    median = np.median(data)

    # 计算中位绝对偏差（MAD）
    mad = np.median(np.abs(data - median))
    thres = max(median + mad, 0.0001)
    # import pdb;pdb.set_trace()
    mask = (saliency < thres).float()
    return mask 


def vis_mask(image, saliency, key_pos, cnt = 0):
    try:
        image_token_start = key_pos[0]['image_token_begin']
        image_token_end = key_pos[0]['image_token_end']
    except:
        import pdb;pdb.set_trace()
        # 
    saliency = saliency / (saliency[image_token_start:image_token_end].sum(dim=0) + 1e-7)  
    saliency_mask = GMM_mask(saliency[image_token_start:image_token_end])  

    heatmap_data2 = saliency[image_token_start:image_token_end].view(24, 24).cpu().numpy() 
    heatmap_data = saliency_mask.view(24, 24).cpu().numpy() 
    image_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
    image_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)

    # 反标准化
    # import pdb;pdb.set_trace()
    image = image.squeeze().cpu()
    image = image * image_std + image_mean
    image_np = image.permute(1, 2, 0).numpy()
    image_np = np.clip(image_np, 0, 1)
    plt.imshow(image_np)
    plt.savefig('./imgs/ori_image_'+str(cnt)+'.png', bbox_inches='tight', pad_inches=0)
    plt.clf()
    plt.close()

    # 创建热度图
    plt.figure(figsize=(24, 24))
    plt.imshow(heatmap_data, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Heat Level')

    # 添加标题和标签
    plt.title('Heatmap of 24x24 Patches')
    plt.xlabel('Patch X')
    plt.ylabel('Patch Y')
    plt.savefig('./imgs/heatmap_mask_'+str(cnt)+'.png', bbox_inches='tight', pad_inches=0)

    plt.clf()
    plt.close()             



    plt.figure(figsize=(24, 24))
    plt.imshow(heatmap_data2, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Heat Level')

    # 添加标题和标签
    plt.title('Heatmap of 24x24 Patches')
    plt.xlabel('Patch X')
    plt.ylabel('Patch Y')
    plt.savefig('./imgs/heatmap_'+str(cnt)+'.png', bbox_inches='tight', pad_inches=0)
    plt.clf()
    plt.close()

def plot_distribution(saliency, name = "attention_distribution_", cnt = 0):
    plt.clf()
    plt.hist(saliency.cpu().numpy(), bins=100, edgecolor='black', range=(0, 0.02))
    plt.title('Data Distribution')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    # 保存图像å
    plt.savefig('./imgs/'+str(name)+str(cnt)+'.png')
    plt.clf()
    plt.close()

def build_neighbourhood_mask(identity, step = 2):
    neighbourhood_mask = identity 
    for i in range(step):
        neighbourhood_mask[:, :-1] = (neighbourhood_mask[:, :-1].clone() + neighbourhood_mask[:, 1:].clone() > 0).float()
    return neighbourhood_mask



def vis_attention(image, attention, key_pos, cnt = 0):
    try:
        image_token_start = key_pos[0]['image_token_start']
        image_token_end = key_pos[0]['image_token_end']
        obj_pos = key_pos[0]['a'] + 1
    except:
        import pdb;pdb.set_trace()
        # 
    # saliency = saliency / (saliency[image_token_start:image_token_end].sum(dim=0) + 1e-7)  
    # saliency_mask = GMM_mask(saliency[image_token_start:image_token_end]) 
    # 

    plt.rcParams.update({
        'font.size': 16,         # 设置全局字体大小
        'font.weight': 'bold',   # 设置全局字体为加粗
    })
    tmp_attention = attention[:, :, -1, image_token_start:image_token_end].mean(dim = 0).mean(dim = 0)

    image_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
    image_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)

    # 反标准化
    # import pdb;pdb.set_trace()
    image = image.squeeze().cpu()
    image = image * image_std + image_mean
    image_np = image.permute(1, 2, 0).numpy()
    image_np = np.clip(image_np, 0, 1)
    plt.axis('off')
    plt.imshow(image_np)
    plt.savefig('./imgs/ori_image2_'+str(cnt)+'.png', bbox_inches='tight', pad_inches=0)
    plt.clf()
    plt.close()
    # tmp_attention = tmp_attention * (tmp_attention < 0.0024) + (tmp_attention >= 0.0024) * 0.0024

    # import pdb;pdb.set_trace() 
    heatmap_data = tmp_attention.view(24, 24).cpu().numpy() 
    # colors = [(0, 0, 0.5), (1, 1, 0)]  # 从深蓝到亮黄
    # cmap_name = 'custom_blue_yellow'
    # custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=256)
    # 创建热度图
    plt.figure(figsize=(24, 24))
    plt.axis('off')
    plt.imshow(heatmap_data, cmap='YlGnBu_r', interpolation='nearest', vmin=0, vmax=0.002)

    plt.colorbar(shrink=0.8)
    # cbar.set_label(, fontsize=16, fontweight='bold')

    # 添加标题和标签
    # plt.title('Heatmap of 24x24 Patches')
    # plt.xlabel('Patch X')
    # plt.ylabel('Patch Y')

    plt.savefig('./imgs/heatmap_attention_cdar_'+str(cnt)+'.png', bbox_inches='tight', pad_inches=0)

    plt.clf()
    plt.close()             



    # plt.figure(figsize=(24, 24))
    # plt.imshow(heatmap_data2, cmap='hot', interpolation='nearest')
    # plt.colorbar(label='Heat Level')

    # # 添加标题和标签
    # plt.title('Heatmap of 24x24 Patches')
    # plt.xlabel('Patch X')
    # plt.ylabel('Patch Y')
    # plt.savefig('./imgs/heatmap_'+str(cnt)+'.png', bbox_inches='tight', pad_inches=0)
    # plt.clf()
    # plt.close()

def vis_attn_sum(tmp_attention, cnt):
    plt.rcParams.update({
        'font.size': 16,         # 设置全局字体大小
        'font.weight': 'bold',   # 设置全局字体为加粗
    })
    heatmap_data = tmp_attention.view(24, 24).cpu().numpy() 
    # colors = [(0, 0, 0.5), (1, 1, 0)]  # 从深蓝到亮黄
    # cmap_name = 'custom_blue_yellow'
    # custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=256)
    # 创建热度图
    plt.figure(figsize=(24, 24))
    plt.axis('off')
    plt.imshow(heatmap_data, cmap='YlGnBu_r', interpolation='nearest', vmin=0, vmax=0.002)

    plt.colorbar(shrink=0.8)
    # cbar.set_label(, fontsize=16, fontweight='bold')

    # 添加标题和标签
    # plt.title('Heatmap of 24x24 Patches')
    # plt.xlabel('Patch X')
    # plt.ylabel('Patch Y')

    plt.savefig('./imgs/heatmap_attention_all.pdf', bbox_inches='tight', pad_inches=0)

    plt.clf()
    plt.close()   