# =========================================================================================
# ====================================== 1. 环境设置 =======================================
# =========================================================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator, AutoMinorLocator
from matplotlib import cm
import matplotlib
import shap
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
# =========================================================================================
# ====================================== 2.颜色库=======================================
# =========================================================================================
COLOR_SCHEMES = {
    1: ['#053061', '#4393c3', '#ffffff', '#f4a582', '#67001f'],
    2: ['#2166ac', '#92c5de', '#ffffff', '#fddbc7', '#b2182b'],
    3: ['#003c30', '#35978f', '#ffffff', '#dfc27d', '#543005'],
    4: ['#01665e', '#5ab4ac', '#ffffff', '#d8b365', '#8c510a'],
    5: ['#276419', '#7fbc41', '#ffffff', '#de77ae', '#8e0152'],
    6: ['#40004b', '#762a83', '#ffffff', '#a6dba0', '#00441b'],
    7: ['#2d004b', '#542788', '#ffffff', '#fee0b6', '#7f3b08'],
    8: ['#08519c', '#6baed6', '#ffffff', '#fdae6b', '#a63603'],
    9: ['#00441b', '#41ab5d', '#ffffff', '#fa9fb5', '#7a0177'],
    10: ['#1b7837', '#7fbf7b', '#ffffff', '#af8dc3', '#762a83'],
    11: ['#004d40', '#00897b', '#ffffff', '#ff8a65', '#d84315'],
    12: ['#0d47a1', '#1e88e5', '#ffffff', '#ffb74d', '#e65100'],
    13: ['#1a237e', '#3949ab', '#ffffff', '#ffd54f', '#ff6f00'],
    14: ['#311b92', '#5e35b1', '#ffffff', '#81c784', '#1b5e20'],
    15: ['#4a148c', '#8e24aa', '#ffffff', '#aed581', '#33691e'],
    16: ['#880e4f', '#d81b60', '#ffffff', '#80cbc4', '#004d40'],
    17: ['#b71c1c', '#e53935', '#ffffff', '#90caf9', '#0d47a1'],
    18: ['#bf360c', '#f4511e', '#ffffff', '#80deea', '#006064'],
    19: ['#3e2723', '#6d4c41', '#ffffff', '#80cbc4', '#004d40'],
    20: ['#263238', '#546e7a', '#ffffff', '#ffab91', '#bf360c'],
    21: ['#1c3144', '#3f88c5', '#ffffff', '#f49e4c', '#a53860'],
    22: ['#0b3954', '#087e8b', '#ffffff', '#ff5a5f', '#c81d25'],
    23: ['#2e4057', '#048a81', '#ffffff', '#f4d35e', '#f95738'],
    24: ['#1d3557', '#457b9d', '#ffffff', '#e63946', '#780000'],
    25: ['#2b2d42', '#8d99ae', '#ffffff', '#ef233c', '#d90429'],
    26: ['#212529', '#495057', '#ffffff', '#f77f00', '#d62828'],
    27: ['#0f4c5c', '#5c9ead', '#ffffff', '#fb8b24', '#9a031e'],
    28: ['#143642', '#0f8a5f', '#ffffff', '#ec9a29', '#a8201a'],
    29: ['#1e3d59', '#17b978', '#ffffff', '#ff6e40', '#ffc107'],
    30: ['#343a40', '#6c757d', '#ffffff', '#20c997', '#099268'],
    31: ['#03045e', '#0077b6', '#ffffff', '#f72585', '#7209b7'],
    32: ['#10002b', '#3c096c', '#ffffff', '#ff9e00', '#ff6000'],
    33: ['#001219', '#005f73', '#ffffff', '#ca6702', '#9b2226'],
    34: ['#2d6a4f', '#52b788', '#ffffff', '#e07a5f', '#3d405b'],
    35: ['#0f172a', '#334155', '#ffffff', '#f59e0b', '#b45309'],
    36: ['#1e1b4b', '#4338ca', '#ffffff', '#ec4899', '#9d174d'],
    37: ['#064e3b', '#059669', '#ffffff', '#f43f5e', '#9f1239'],
    38: ['#164e63', '#0891b2', '#ffffff', '#f97316', '#c2410c'],
    39: ['#3b0764', '#7e22ce', '#ffffff', '#10b981', '#047857'],
    40: ['#701a75', '#a21caf', '#ffffff', '#84cc16', '#4d7c0f'],
    41: ['#4c0519', '#be123c', '#ffffff', '#06b6d4', '#0e7490'],
    42: ['#451a03', '#9a3412', '#ffffff', '#38bdf8', '#0284c7'],
    43: ['#172554', '#2563eb', '#ffffff', '#fbbf24', '#b45309'],
    44: ['#022c22', '#0d9488', '#ffffff', '#f87171', '#dc2626'],
    45: ['#2e1065', '#6b21a8', '#ffffff', '#facc15', '#a16207'],
    46: ['#14532d', '#22c55e', '#ffffff', '#fb923c', '#ea580c'],
    47: ['#1e293b', '#64748b', '#ffffff', '#38bdf8', '#0369a1'],
    48: ['#132a13', '#31572c', '#ffffff', '#ecf39e', '#90a955'],
    49: ['#283618', '#606c38', '#ffffff', '#dda15e', '#bc6c25'],
    50: ['#03071e', '#370617', '#ffffff', '#f48c06', '#dc2f02'],
    51: ['#22223b', '#4a4e69', '#ffffff', '#c9ada7', '#9a8c98'],
    52: ['#355070', '#6d597a', '#ffffff', '#eaac8b', '#e56b6f'],
    53: ['#1b4965', '#62b6cb', '#ffffff', '#bee9e8', '#468faf'],
    54: ['#2f3e46', '#354f52', '#ffffff', '#84a98c', '#52796f'],
    55: ['#0a0908', '#22333b', '#ffffff', '#c6ac8f', '#5e503f'],
    56: ['#231942', '#5e548e', '#ffffff', '#e0b1cb', '#be95c4'],
    57: ['#3d314a', '#684756', '#ffffff', '#ab8476', '#96705b'],
    58: ['#183a37', '#04151f', '#ffffff', '#efd07b', '#c44900'],
    59: ['#081c15', '#1b4332', '#ffffff', '#74c69d', '#2d6a4f'],
    60: ['#2c1a4d', '#4d243d', '#ffffff', '#bf5700', '#7a1c1c']
}
# =========================================================================================
# ====================================== 3.文本颜色设置辅助函数 =======================================
# =========================================================================================
def get_contrast_text_color(rgba):
    r, g, b = rgba[:3] # 提取RGB
    luminance = 0.299 * r + 0.587 * g + 0.114 * b # 感知亮度
    return '#000000' if luminance > 0.58 else '#FFFFFF' # 亮背景返回黑色，暗背景返回白色
# =========================================================================================
# ====================================== 4.蜂巢图散点设置辅助函数 =======================================
# =========================================================================================
def compute_beeswarm_offsets(val_array, max_jitter=0.28, n_bins=65):
    counts, bin_edges = np.histogram(val_array, bins=n_bins) # 统计各分箱数据点数与边界
    bin_indices = np.digitize(val_array, bin_edges) - 1 # 获取数据点所在的分箱索引
    bin_indices = np.clip(bin_indices, 0, n_bins - 1) # 限制分箱索引防止越界
    y_offsets = np.zeros_like(val_array, dtype=float) # 初始化Y轴偏移量数组
    # 遍历每个分箱
    for b in range(n_bins):
        pts_in_bin = np.where(bin_indices == b)[0] # 筛选当前分箱的数据点索引
        n_pts = len(pts_in_bin) # 获取分箱内的点数
        # 判断分箱内是否有数据
        if n_pts > 0:
            bin_center = 0.5 * (bin_edges[b] + bin_edges[b + 1]) # 计算分箱中心值
            order = np.argsort(np.abs(val_array[pts_in_bin] - bin_center)) # 按偏离中心绝对值排序
            offsets = [] # 初始化偏移列表
            # 遍历分箱内的点
            for k in range(n_pts):
                step = (k + 1) // 2 # 计算交替步长
                sign = 1 if k % 2 == 1 else -1 # 正负交替方向
                offsets.append(sign * step) # 添加偏移步数
            offsets = np.array(offsets, dtype=float) # 转为数组
            if n_pts > 1: # 若点数大于1
                scale = min(max_jitter, n_pts * 0.026) / (np.max(np.abs(offsets)) + 1e-6) # 计算自适应缩放因子
                offsets = offsets * scale # 缩放偏移量
            else: # 若仅有1个点
                offsets = np.array([0.0]) # 偏移设为0
            y_offsets[pts_in_bin[order]] = offsets # 存储计算后的偏移量
    return y_offsets # 返回所有点的Y轴偏移量
# =========================================================================================
# ====================================== 5. 绘图函数 =======================================
# =========================================================================================
def plot_advanced_forest_chart(df_features, shap_values, X_scaled, scheme_id):
    # 创建画布并
    fig = plt.figure(figsize=(20, 5.4), dpi=300)

    WIDTH_A = 1.25 # 子图A宽度
    GAP_AB = 0.4 # A与B间距
    WIDTH_B = 0.70 # 子图B宽度
    width_ratios = [WIDTH_A, GAP_AB, WIDTH_B] # 保留A、B两面板的原有相对宽度
    # 创建网格布局
    gs = fig.add_gridspec(1, # 行
                          3, # 列
                          width_ratios=width_ratios, # 列宽比例
                          left=0.065, # 左边距
                          right=0.94, # 右边距
                          bottom=0.12, # 底边距
                          top=0.90, # 顶边距
                          wspace=0) # 列间距

    SPINE_LW = 2 # 边框线宽
    MAJOR_TICK_LEN = 5.0 # 主刻度线长度
    MAJOR_TICK_W = 2 # 主刻度线宽度
    MINOR_TICK_LEN = 3# 次刻度线长度
    MINOR_TICK_W = 2 # 次刻度线宽度

    # 按原始SHAP值升序排序
    df_sorted = df_features.sort_values(by='Raw', ascending=True).reset_index(drop=True)
    N = len(df_sorted) # 特征数量

    scheme_colors = COLOR_SCHEMES.get(scheme_id, COLOR_SCHEMES[1]) # 获取配色方案
    cmap = mcolors.LinearSegmentedColormap.from_list(f'custom_scheme_{scheme_id}', scheme_colors) # 创建渐变色
    norm_bar = mcolors.Normalize(vmin=df_sorted['Raw'].min(), vmax=df_sorted['Raw'].max()) # 构建数值归一化对象
    feature_colors = {feat: cmap(i / max(N - 1, 1)) for i, feat in enumerate(df_sorted['Feature'])} # 特征颜色映射
    # Match the user-specified scheme-56 light lavender for Math (sampled from the reference swatch).
    if 'Math' in feature_colors:
        feature_colors['Math'] = mcolors.to_rgba('#D0CDDE')

    # 添加子图A
    ax_a = fig.add_subplot(gs[0, 0])
    y_pos = np.arange(N) # 生成Y轴位置序列

    # 遍历绘制水平条形图
    for i, raw_v in enumerate(df_sorted['Raw']):
        c = feature_colors[df_sorted.loc[i, 'Feature']] # 映射颜色
        # 绘制水平条
        ax_a.barh(i, # y
                  raw_v, # 长度
                  height=0.62, # 高度
                  color=c, # 颜色
                  edgecolor='none', # 边框
                  align='center') # 居中对齐
        # 绘制文本标注
        ax_a.text(raw_v + (df_sorted['Raw'].max() * 0.015), # x
                  i, f"{raw_v:.4f}", # 数值
                  ha='right', # 水平
                  va='center', # 垂直
                  fontsize=12, # 字体大小
                  bbox=dict(boxstyle='square,pad=0.15', # 边框样式
                            fc='white', # 背景色
                            ec=c, # 边框色
                            lw=0.7, # 边框线宽
                            alpha=0.9)) # 透明度

    max_raw_val = df_sorted['Raw'].max() # 原始SHAP最大值
    max_x_val = max_raw_val * 1.15 # X轴最大范围
    ax_a.set_xlim(max_x_val, 0.00) # 反转X轴显示
    # 设置主刻度
    ax_a.xaxis.set_major_locator(MaxNLocator(nbins=6, # 数量
                                             prune=None, # 不修剪刻度
                                             steps=[1, 2, 2.5, 5, 10])) # 刻度步长
    ax_a.xaxis.set_minor_locator(AutoMinorLocator(10)) # 设置次刻度
    # 设置刻度
    ax_a.tick_params(axis='x', # 轴
                     which='both', # 刻度
                     direction='in', # 朝内
                     bottom=True, # 显示底端刻度
                     top=False) # 隐藏顶端刻度
    # 设置主刻度参数
    ax_a.tick_params(axis='x', # 轴
                     which='major', #主刻度
                     length=MAJOR_TICK_LEN, # 长度
                     width=MAJOR_TICK_W, # 宽度
                     labelsize=14) # 字号
    # 设置次刻度参数
    ax_a.tick_params(axis='x', # 轴
                     which='minor', # 次刻度
                     length=MINOR_TICK_LEN, # 长度
                     width=MINOR_TICK_W) # 宽度
    # 遍历X轴标签
    for label in ax_a.get_xticklabels():
        label.set_fontweight('bold') # 加粗

    ax_a.set_ylim(-0.6, N - 0.4) # 设置Y轴范围
    ax_a.set_yticks(y_pos) # 设置Y轴刻度
    # 设置Y轴特征标签
    ax_a.set_yticklabels(df_sorted['Feature'], # 文本
                         fontsize=14, # 字号
                         fontweight='bold') # 加粗
    ax_a.yaxis.tick_right() # 移至右侧
    # 设置Y轴刻度参数
    ax_a.tick_params(axis='y', # 轴
                     length=0, # 长度
                     pad=8) # 间距
    # 设置X轴标题
    ax_a.set_xlabel('SHAP Value', # 文本
                    fontsize=16, # 字号
                    fontweight='bold', # 加粗
                    labelpad=6) # 边距

    ax_a.spines['top'].set_visible(False) # 隐藏上边框
    ax_a.spines['left'].set_visible(False) # 隐藏左边框
    ax_a.spines['right'].set_linewidth(SPINE_LW) # 设置右边框线宽
    ax_a.spines['bottom'].set_linewidth(SPINE_LW) # 设置下边框线宽


    # 创建颜色条子轴
    cax_a = ax_a.inset_axes([-0.055, 0.0, 0.022, 1.0])
    # 绘制颜色条
    cb_a = fig.colorbar(cm.ScalarMappable(norm=norm_bar, cmap=cmap),cax=cax_a, orientation='vertical')
    cb_a.outline.set_visible(False) # 隐藏颜色条边框
    cb_a.set_ticks([]) # 清空颜色条刻度

    # 设置颜色条刻度标注
    cax_a.text(0.5, # X
               1.02, # Y
               'High', # 文本
               transform=cax_a.transAxes, # 坐标系
               ha='center', # 水平
               va='bottom', # 垂直
               fontsize=11, # 大小
               fontweight='bold') # 加粗
    cax_a.text(0.5, # X
               -0.02, # Y
               'Low', # 文本
               transform=cax_a.transAxes, # 坐标系
               ha='center', # 水平
               va='top', # 垂直
               fontsize=11, # 大小
               fontweight='bold') # 加粗
    # 设置颜色条标题
    cax_a.set_ylabel('SHAP Value', # 文本
                     fontsize=12, # 字号
                     fontweight='bold', # 加粗
                     labelpad=-28) # 边距

    # 设置内嵌玫瑰图轴
    ax_inset = ax_a.inset_axes([-0.15, # X
                                -0.15, # Y
                                0.98, # 宽度
                                0.98], # 高度
                               projection='polar') # 投影
    ax_inset.set_facecolor('none') # 设置背景
    ax_inset.patch.set_alpha(0.0) # 透明度

    df_desc = df_features.sort_values(by='Value', ascending=False).reset_index(drop=True) # 按占比降序排序
    num_vars = len(df_desc) # 特征数量
    values_in = df_desc['Value'].tolist() # 获取占比列表
    sum_vals = sum(values_in) # 计算总和

    widths_in = [(v / sum_vals) * 2 * np.pi for v in values_in] # 计算各扇区弧度
    base_len = 1.8 # 基础半径
    fixed_inc = 0.2 # 半径增量
    total_lens = [base_len + k * fixed_inc for k in range(num_vars)] # 计算总半径列表
    ring_width = 0.75 # 环形宽度
    inner_hgts = [max(0.0, tl - ring_width) for tl in total_lens] # 计算内圈高度
    inner_cols = ['#EAEAEA', 'none'] * (num_vars // 2 + 1) # 生成交替背景颜色
    inner_cols = inner_cols[:num_vars] # 截取匹配长度

    thetas_in = np.cumsum([0] + widths_in[:-1]) - (np.pi / 18) # 计算各扇区起始角度
    colors_in = [feature_colors[feat] for feat in df_desc['Feature']] # 计算对应颜色映射

    # 绘制内圈背景环
    ax_inset.bar(x=thetas_in, # 角度
                 height=inner_hgts, # 内圈半径
                 width=widths_in, # 宽度
                 color=inner_cols, # 填充色
                 align='edge', # 边缘对齐
                 edgecolor='none', # 无边框
                 linewidth=0) # 线宽

    # 绘制外圈数据环
    ax_inset.bar(x=thetas_in,  # 角度
                 height=[ring_width] * num_vars, # 半径
                 width=widths_in, # 扇区宽度
                 bottom=inner_hgts, # 底部起始半径
                 color=colors_in, # 填充色
                 align='edge', # 边缘对齐
                 edgecolor='white', # 白色边框
                 linewidth=0.5) # 边框线宽

    # 遍历标注文本
    for k in range(num_vars):
        ang = thetas_in[k] + widths_in[k] / 2 # 文字中心角度
        r_pos = total_lens[k] + 0.45 # 文字径向位置
        c_k = colors_in[k] # 获取对应颜色
        #标注文本
        ax_inset.text(ang, # 角度
                      r_pos, # 径向
                      f'{values_in[k]:.1f}%', # 格式化数值
                      ha='center', # 水平
                      va='center', # 垂直
                      fontsize=9, # 字号
                      fontweight='bold', # 加粗
                      rotation=np.degrees(-ang), # 旋转
                      # 文本框样式
                      bbox=dict(boxstyle='square,pad=0.12', #边距
                                fc='white', # 背景色
                                ec=c_k, # 边框色
                                lw=0.5, # 线宽
                                alpha=0.9)) # 透明度

    ax_inset.set_yticklabels([]) # 清空Y轴刻度标签
    ax_inset.set_xticklabels([]) # 清空X轴刻度标签
    ax_inset.spines['polar'].set_visible(False) # 隐藏边框
    ax_inset.grid(False) # 关闭网格线
    ax_inset.set_theta_zero_location('N') # 正北
    ax_inset.set_theta_direction(-1) # 顺时针
    ax_inset.set_ylim(0, max(total_lens) + 1.2) # 径向范围
    # 子图编号
    ax_a.text(-0.06, # x
              1.05, # y
              'A', # 编号
              transform=ax_a.transAxes, # 坐标系
              fontsize=16, # 字号
              fontweight='bold', # 加粗
              va='bottom') # 垂直

    # 添加子图B
    ax_b = fig.add_subplot(gs[0, 2])
    #设置基准线
    ax_b.axvline(0, # x
                 color='gray', # 颜色
                 linestyle='-', # 实线
                 linewidth=2, # 线宽
                 alpha=1) # 透明度

    feature_order = df_sorted['Feature'].tolist() # 获取特征排序列表

    # 遍历绘制每个特征的散点
    for i, feat in enumerate(feature_order):
        feat_idx = list(X_scaled.columns).index(feat) # 查找特征列索引
        s_vals = shap_values[:, feat_idx] # 提取当前特征SHAP值
        f_vals = X_scaled[feat].values # 提取当前特征原始值

        f_norm = (f_vals - np.min(f_vals)) / (np.ptp(f_vals) + 1e-8) # 归一化特征值
        y_offsets = compute_beeswarm_offsets(s_vals, max_jitter=0.28, n_bins=65) # 计算散点抖动
        #绘制散点
        ax_b.scatter(s_vals, # x
                     i + y_offsets, # y
                     c=f_norm, # 映射特征值颜色
                     cmap=cmap, # 颜色映射表
                     s=16, # 散点大小
                     alpha=0.9, # 透明度
                     edgecolors='none') # 无边框

    s_min_val = np.min(shap_values) # 计算SHAP全局最小值
    s_max_val = np.max(shap_values) # 计算SHAP全局最大值
    s_bound = max(abs(s_min_val), abs(s_max_val)) * 1.15 # 计算对称显示边界
    ax_b.set_xlim(-s_bound, s_bound) # 设置X轴范围
    # 设置主刻度
    ax_b.xaxis.set_major_locator(MaxNLocator(nbins=5, # 数量
                                             prune=None, # 修剪刻度
                                             steps=[1, 2, 2.5, 5, 10])) # 刻度步长
    ax_b.xaxis.set_minor_locator(AutoMinorLocator(10)) # 设置次刻度

    #设置刻度样式
    ax_b.tick_params(axis='x', # 轴
                     which='both', # 刻度
                     direction='in', # 朝内
                     bottom=True, # 显示底端刻度
                     top=False) # 隐藏顶端刻度
    ax_b.tick_params(axis='x', # 轴
                     which='major', # 主刻度
                     length=MAJOR_TICK_LEN, # 长度
                     width=MAJOR_TICK_W, # 宽度
                     labelsize=14) # 字号
    ax_b.tick_params(axis='x', # 轴
                     which='minor', # 次刻度
                     length=MINOR_TICK_LEN, # 长度
                     width=MINOR_TICK_W) # 宽度
    # 遍历X轴标签
    for label in ax_b.get_xticklabels():
        label.set_fontweight('bold') # 加粗

    ax_b.set_ylim(-0.6, N - 0.4) # 设置Y轴范围
    ax_b.set_yticks([]) # 隐藏Y轴刻度
    # 设置X轴标题
    ax_b.set_xlabel('SHAP Value', #文本
                    fontsize=13, # 字号
                    fontweight='bold', # 加粗
                    labelpad=6) # 边距

    ax_b.spines['top'].set_visible(False) # 隐藏上边框
    ax_b.spines['right'].set_visible(False) # 隐藏右边框
    ax_b.spines['left'].set_visible(False) # 隐藏左边框
    ax_b.spines['bottom'].set_linewidth(SPINE_LW) # 下边框线宽
    # 设置颜色条轴
    cax_b = ax_b.inset_axes([1.05, # x
                             0.0, # y
                             0.038, # 宽
                             1.0]) # 高

    norm_bee = mcolors.Normalize(vmin=0, vmax=1) # 归一化特征值范围
    # 绘制颜色条
    cb_b = fig.colorbar(cm.ScalarMappable(norm=norm_bee, cmap=cmap),cax=cax_b,orientation='vertical')

    cb_b.outline.set_visible(False) # 隐藏边框
    cb_b.set_ticks([]) # 清空刻度
    # 设置颜色条标注
    cax_b.text(0.5, # x
               1.02, # y
               'High', # 文本
               transform=cax_b.transAxes, # 相对坐标
               ha='center', # 水平
               va='bottom', # 垂直
               fontsize=11, # 字号
               fontweight='bold') # 加粗
    cax_b.text(0.5, # x
               -0.02, # y
               'Low', # 文本
               transform=cax_b.transAxes, # 相对坐标
               ha='center', # 水平
               va='top', # 垂直
               fontsize=11, # 字号
               fontweight='bold') # 加粗
    # 设置颜色条标题
    cax_b.set_ylabel('Feature value', # 文本
                     fontsize=12, # 字号
                     fontweight='bold', # 加粗
                     labelpad=-28) # 边距
    # 子图B编号
    ax_b.text(-0.08, # x
              1.05, # y
              'B', # 编号
              transform=ax_b.transAxes, # 相对坐标
              fontsize=16, # 字号
              fontweight='bold', # 加粗
              va='bottom') # 垂直


    # 保存
    plt.savefig(OUTPUT_DIR / f'Figure_SHAP_Composite_Scheme{scheme_id}.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / f'Figure_SHAP_Composite_Scheme{scheme_id}.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig) # 关闭
# =========================================================================================
# =========================================================================================
# ====================================== 6. 当前研究数据映射 =======================================
# =========================================================================================
from pathlib import Path

# 仅替换数据输入和输出路径；上方 COLOR_SCHEMES 与 plot_advanced_forest_chart
# 的颜色、面板宽度、布局及绘图逻辑均沿用原始模板，不做更改。
PROJECT_ROOT = Path(r"E:\可持续论文\可持续教育完整实验设计与初步可行性验证")
OUTPUT_DIR = PROJECT_ROOT / "02_model_prediction" / "results" / "T0_q0.25" / "Explainability_SHAP"
SHAP_DATA_PATH = OUTPUT_DIR / "Student_Level_SHAP_Attributions.csv"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_MAP = [
    ("高中英语成绩", "English"),
    ("高中批判性阅读成绩", "Crit. reading"),
    ("高中数学成绩", "Math"),
    ("高中生物成绩", "Biology"),
    ("高中公民素养成绩", "Civic"),
]

if __name__ == '__main__':
    # 已生成并复核的留出高校 SHAP 结果；不重新训练模型、不重新抽样。
    df_shap = pd.read_csv(SHAP_DATA_PATH, encoding="utf-8-sig")
    if df_shap["row_id"].duplicated().any():
        raise ValueError("SHAP 明细中 row_id 重复，拒绝绘图。")
    if len(df_shap) != 1249:
        raise ValueError(f"SHAP 明细行数与已审核结果不一致：{len(df_shap)}")
    if not np.isfinite(df_shap["sampling_weight"]).all():
        raise ValueError("存在无效的高校加权系数。")

    # 保留模板函数的 X_scaled 参数名，但这里传入学生原始成绩，
    # 使模板蜂巢图的 Low/High 色阶对应真实的先前成绩高低。
    feature_names = [label for _, label in FEATURE_MAP]
    X_scaled = pd.DataFrame({
        label: df_shap[f"input__{source_name}"].to_numpy(float)
        for source_name, label in FEATURE_MAP
    })
    shap_vals = np.column_stack([
        df_shap[f"shap__{source_name}"].to_numpy(float)
        for source_name, _ in FEATURE_MAP
    ])

    # 按高校总体学生数加权，修正分层解释抽样对各高校样本占比的影响。
    weights = df_shap["sampling_weight"].to_numpy(float)
    mean_abs_shap = np.average(np.abs(shap_vals), axis=0, weights=weights)
    total_shap = float(mean_abs_shap.sum())
    if not np.isfinite(total_shap) or total_shap <= 0:
        raise ValueError("SHAP 总重要性无效，无法构造相对贡献比例。")
    shap_percentages = mean_abs_shap / total_shap * 100.0

    df_real = pd.DataFrame({
        "Feature": feature_names,
        "Raw": mean_abs_shap,
        "Value": shap_percentages,
    })
    print(f"Data: {SHAP_DATA_PATH}")
    print(f"Rows: {len(df_shap)}; universities represented: 134")
    print("Using original template style scheme 56; panel C removed; palette unchanged.")
    plot_advanced_forest_chart(df_real, shap_vals, X_scaled, scheme_id=56)

    qa_text = """# SHAP scheme 56 figure notes

- Source: Student_Level_SHAP_Attributions.csv from the completed five-fold, three-seed, university-held-out M1-J explanation.
- Plotted records: all 1,249 available explained students, representing all 134 universities; no additional rows were sampled for plotting.
- Panel A: university-population-weighted mean absolute SHAP value, shown in predicted-risk probability units.
- Panel B: individual student SHAP values; point color encodes that student's observed prior score from low to high.
- The original template's 56 color scheme was retained, including the user-specified Math swatch. Panel C was removed as requested; panels A and B retain the template plotting style.
- Full label mapping: English = prior high-school English score; Crit. reading = prior high-school critical-reading score; Math = prior high-school mathematics score; Biology = prior high-school biology score; Civic = prior high-school civic-competence score.
- SHAP values describe the fitted model's prediction relative to the outer-training mean reference profile; they are not causal effects.
"""
    (OUTPUT_DIR / "Figure_SHAP_Scheme56_Notes.md").write_text(qa_text, encoding="utf-8")
