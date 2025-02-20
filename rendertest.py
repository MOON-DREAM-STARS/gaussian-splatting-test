#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render # 需要修改
import torchvision
from utils.general_utils import safe_state # 这里面默认指定了cuda设备为device0,如果需要使用其他设备，需要修改该文件
from argparse import ArgumentParser
# ModelParams 继承自 ParamGroup 类，用于管理和加载模型相关的参数。
# pipeline 继承自 ParamGroup 类，用于管理和加载渲染管线相关的参数。或许会影响到gaussian_render/init.py/render()
# get_combined_args 用于将命令行参数和配置文件中的参数合并到一个统一的 Namespace 对象中，以便后续程序使用。
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
# SPARSE_ADAM_AVAILABLE这个是是否分离球谐计算的参数
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, train_test_exp, separate_sh):
    """
    渲染一组图像并保存渲染结果与真实图像
    参数说明:
    - model_path: 模型保存路径
    - name: 实验名称
    - iteration: 训练迭代次数
    - views: 视角列表
    - gaussians: 高斯点云模型
    - pipeline: 渲染管线
    - background: 背景颜色
    - train_test_exp: 是否使用训练好的曝光参数
    - separate_sh: 是否分离球谐函数
    """
    # 创建保存渲染结果和真实图像(ground truth)的路径
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    # 确保输出目录存在
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    # 遍历所有视角进行渲染
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        # 执行渲染操作，获取渲染结果
        rendering = render(view, gaussians, pipeline, background, 
                         use_trained_exp=train_test_exp, 
                         separate_sh=separate_sh)["render"]
        # 获取对应视角的真实图像(仅使用RGB三个通道)
        gt = view.original_image[0:3, :, :]

        # 如果使用训练好的曝光参数，则只取图像的后半部分
        if args.train_test_exp:
            rendering = rendering[..., rendering.shape[-1] // 2:]
            gt = gt[..., gt.shape[-1] // 2:]

        # 保存渲染结果和真实图像
        # 使用5位数字作为文件名，例如: 00000.png, 00001.png
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, separate_sh: bool):
    """
    批量渲染训练集和测试集图像
    参数说明:
    - dataset: 数据集参数，包含模型路径、球谐度数等信息
    - iteration: 要加载的模型迭代次数
    - pipeline: 渲染管线参数
    - skip_train: 是否跳过训练集渲染
    - skip_test: 是否跳过测试集渲染
    - separate_sh: 是否分离球谐函数计算
    """
    # 使用torch.no_grad()上下文管理器禁用梯度计算，提高渲染速度和减少内存使用
    with torch.no_grad():
        # 初始化高斯点云模型，使用数据集指定的球谐度数
        gaussians = GaussianModel(dataset.sh_degree)
        # 创建场景对象，加载指定迭代次数的模型参数，不进行数据打乱
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        # 根据数据集配置设置背景色
        # white_background为True时使用白色背景[1,1,1]，否则使用黑色背景[0,0,0]
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        # 将背景色转换为CUDA张量用于GPU计算
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # 如果不跳过训练集渲染，则渲染训练集图像
        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, 
                       scene.getTrainCameras(), gaussians, pipeline, 
                       background, dataset.train_test_exp, separate_sh)

        # 如果不跳过测试集渲染，则渲染测试集图像
        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, 
                       scene.getTestCameras(), gaussians, pipeline, 
                       background, dataset.train_test_exp, separate_sh)

def test_gaussian_loading(model_path: str):
    """
    测试高斯点云数据的加载
    参数:
    - model_path: 模型路径，包含point_cloud文件夹的目录路径
    """
    try:
        # 获取最新的迭代结果文件夹
        point_cloud_dir = os.path.join(model_path, "point_cloud")
        if not os.path.exists(point_cloud_dir):
            raise FileNotFoundError(f"点云目录不存在: {point_cloud_dir}")
            
        iterations = sorted([int(dir.split("_")[1]) 
                           for dir in os.listdir(point_cloud_dir) 
                           if dir.startswith("iteration_")])
        if not iterations:
            raise FileNotFoundError(f"未找到任何迭代结果: {point_cloud_dir}")
            
        latest_iteration = iterations[-1]
        point_cloud_path = os.path.join(point_cloud_dir, 
                                      f"iteration_{latest_iteration}", 
                                      "point_cloud.ply")
                                      
        # 初始化高斯模型并加载数据
        gaussians = GaussianModel(sh_degree=3)  # 使用默认球谐度数3
        gaussians.load_ply(point_cloud_path)
        
        print(f"成功加载高斯点云数据:")
        print(f"- 路径: {point_cloud_path}")
        print(f"- 迭代次数: {latest_iteration}")
        return True
        
    except Exception as e:
        print(f"加载高斯点云数据失败: {str(e)}")
        return False

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    # 添加测试模式参数
    parser.add_argument("-t", "--test", action="store_true",
                       help="运行测试模式")
    
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    if args.test:
        # 运行测试模式
        test_gaussian_loading(args.model_path)
    else:
        # 运行正常渲染流程
        render_sets(model.extract(args), 
                       args.iteration, 
                       pipeline.extract(args), 
                       args.skip_train, 
                       args.skip_test, 
                       SPARSE_ADAM_AVAILABLE)