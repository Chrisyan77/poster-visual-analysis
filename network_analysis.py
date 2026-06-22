import numpy as np
import pandas as pd
import networkx as nx
from scipy.spatial.distance import jensenshannon
from scipy.stats import norm
import community as community_louvain
import matplotlib.pyplot as plt

def get_rc_distribution(rc_values, bins=20):
    """获取rc的归一化分布直方图"""
    hist, _ = np.histogram(rc_values, bins=bins, range=(0, 1))
    if np.sum(hist) == 0:
        return np.zeros(bins)
    return hist / np.sum(hist)

def calculate_jsd_similarity(dist1, dist2):
    """计算比例相似度 J = 1 - JSD"""
    jsd = jensenshannon(dist1, dist2)
    return 1 - jsd

def disparity_filter(G):
    """
    Serrano et al. (2009) 差异滤波器，提取显著链接。
    对应论文中的 alpha=0.466 过滤阈值。
    """
    for u, v, data in G.edges(data=True):
        weight = data['weight']
        
        # 计算节点u的总权重和节点v的总权重
        ku = sum(d['weight'] for _, _, d in G.edges(u, data=True))
        kv = sum(d['weight'] for _, _, d in G.edges(v, data=True))
        
        # 计算p-value
        if ku > 0 and kv > 0:
            p_value = 1 - norm.cdf(weight / (ku * kv))
        else:
            p_value = 1.0
            
        # 记录alpha值 (越小越显著)
        G[u][v]['alpha'] = p_value
        
    # 保留 alpha < 0.466 的显著链接
    filtered_edges = [(u, v) for u, v, d in G.edges(data=True) if d['alpha'] < 0.466]
    H = G.edge_subgraph(filtered_edges).copy()
    return H

def build_and_analyze_network(results_csv, output_fig):
    df = pd.read_csv(results_csv)
    
    # 获取所有独特的类型和风格
    types = df['category'].unique()
    styles = df['style'].unique()
    nodes = list(types) + list(styles)
    
    # 计算每个节点的rc分布
    distributions = {}
    for node in nodes:
        if node in types:
            subset = df[df['category'] == node]['rc1'].values
        else:
            subset = df[df['style'] == node]['rc1'].values
        distributions[node] = get_rc_distribution(subset)
        
    # 构建完全连接的二分网络
    G = nx.Graph()
    for t in types:
        for s in styles:
            sim = calculate_jsd_similarity(distributions[t], distributions[s])
            if sim > 0:
                G.add_edge(t, s, weight=sim)
                
    # 应用差异滤波
    print("Applying disparity filter (alpha=0.466)...")
    G_filtered = disparity_filter(G)
    
    # Louvain 社区发现
    print("Running Louvain community detection...")
    partition = community_louvain.best_partition(G_filtered, weight='weight')
    
    # 可视化
    pos = nx.spring_layout(G_filtered, seed=42)
    plt.figure(figsize=(12, 12))
    
    # 根据社区着色
    communities = set(partition.values())
    colors = plt.cm.Set1(np.linspace(0, 1, len(communities)))
    color_map = {c: colors[i] for i, c in enumerate(communities)}
    node_colors = [color_map[partition[node]] for node in G_filtered.nodes()]
    
    nx.draw_networkx_nodes(G_filtered, pos, node_color=node_colors, node_size=500)
    nx.draw_networkx_edges(G_filtered, pos, alpha=0.3)
    nx.draw_networkx_labels(G_filtered, pos, font_size=8)
    
    plt.title("Poster Type-Style Proportion Similarity Network")
    plt.axis('off')
    plt.savefig(output_fig, dpi=300)
    print(f"Network analysis complete. Figure saved to {output_fig}")

if __name__ == "__main__":
    build_and_analyze_network("data/segmentation_results.csv", "results/network_similarity.png")
