import laspy
import numpy as np
from scipy.spatial import KDTree
from tqdm import tqdm
import time
import os


class TraversabilityAssessmentPointCloudSegmentation:
    """
    基于可通行评估的点云分割算法实现（含Z轴高度预过滤）

    思路参考论文:
    "Constraint-aware motion planning for vehicles with terrain traversability assessment and optimization"

    流程：
    1) 评估前高度预过滤：Z > 阈值的点直接删除
    2) 可通行评估分割：基于邻域坡度约束与松弛因子筛选可通行点
    3) 可选高度后过滤：在可通行点集合中再次按Z阈值剔除
    """

    def __init__(self):
        self.las_data = None
        self.points = None
        self.kdtree = None
        self.traversable_indices = None  # 可通行点索引（最终结果）
        self.z_max_threshold = 1.0  # 高度阈值（默认1米，可调整）

    def _pre_filter_high_z_points(self, z_max_threshold: float):
        """评估前高度预过滤：删除Z坐标超过阈值的点云"""
        if self.points is None or self.las_data is None:
            raise ValueError("请先加载LAS文件")

        self.z_max_threshold = float(z_max_threshold)
        total_points = len(self.points)

        valid_mask = self.points[:, 2] <= self.z_max_threshold
        filtered_count = total_points - int(np.count_nonzero(valid_mask))

        self.points = self.points[valid_mask]
        self.las_data.points = self.las_data.points[valid_mask]

        print(f"\n评估前高度过滤：Z > {self.z_max_threshold} 米的点将被删除")
        print(f"  过滤前点数: {total_points:,}")
        print(f"  删除点数: {filtered_count:,}")
        print(f"  过滤后点数: {len(self.points):,}")

        if len(self.points) == 0:
            raise ValueError("高度过滤后无点云可用，请调整阈值")

    def load_las_file(self, file_path: str, pre_z_max_threshold: float = 1.0):
        """加载LAS点云文件，并在可通行评估前执行高度预过滤"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件 {file_path} 不存在")

        print(f"正在加载LAS文件: {file_path}")
        self.las_data = laspy.read(file_path)
        self.points = np.vstack([self.las_data.x, self.las_data.y, self.las_data.z]).T

        if pre_z_max_threshold is not None:
            self._pre_filter_high_z_points(pre_z_max_threshold)

        print(f"点云数量: {len(self.points):,}")
        print(f"坐标范围:")
        print(f"  X: {self.points[:, 0].min():.2f} ~ {self.points[:, 0].max():.2f}")
        print(f"  Y: {self.points[:, 1].min():.2f} ~ {self.points[:, 1].max():.2f}")
        print(f"  Z: {self.points[:, 2].min():.2f} ~ {self.points[:, 2].max():.2f}")

        print("正在构建KDTree...")
        start_time = time.time()
        self.kdtree = KDTree(self.points)
        print(f"KDTree构建完成，耗时: {time.time() - start_time:.2f} 秒")

    def traversability_segmentation(self, radius: float = 1.0, max_climbing_angle: float = 15.0, relaxation_factor: float = 0.1):
        """
        可通行评估分割核心逻辑

        Args:
            radius: 邻域半径（米）
            max_climbing_angle: 最大爬坡角度（度）
            relaxation_factor: 松弛因子（0~1），表示允许超过最大坡度的邻居比例上限
        """
        if self.points is None or self.kdtree is None:
            raise ValueError("请先加载LAS文件")

        print(f"\n开始基于可通行评估的点云分割...")
        print(f"参数设置:")
        print(f"  邻域半径: {radius:.2f} 米 | 最大爬坡角度: {max_climbing_angle:.1f} 度 | 松弛因子: {relaxation_factor:.2f}")

        max_angle_rad = np.radians(max_climbing_angle)
        total_points = len(self.points)
        traversable_indices = []

        with tqdm(total=total_points, desc="可通行评估进度", unit="点") as pbar:
            for i, point in enumerate(self.points):
                indices = self.kdtree.query_ball_point(point, radius)
                if len(indices) <= 1:
                    pbar.update(1)
                    continue

                angle_count = 0
                total_neighbors = len(indices) - 1

                for idx in indices:
                    if idx == i:
                        continue
                    neighbor_point = self.points[idx]
                    horizontal_dist = np.sqrt((point[0] - neighbor_point[0]) ** 2 + (point[1] - neighbor_point[1]) ** 2)
                    if horizontal_dist < 1e-6:
                        continue

                    height_diff = abs(point[2] - neighbor_point[2])
                    angle = np.arctan(height_diff / horizontal_dist)
                    if angle > max_angle_rad:
                        angle_count += 1

                exceed_ratio = angle_count / total_neighbors if total_neighbors > 0 else 0.0
                if exceed_ratio <= relaxation_factor:
                    traversable_indices.append(i)

                pbar.update(1)
                pbar.set_postfix({
                    "可通行点": len(traversable_indices),
                    "比例": f"{len(traversable_indices) / (i + 1) * 100:.1f}%"
                })

        self.traversable_indices = np.array(traversable_indices, dtype=int)
        print(f"\n分割完成! 原始可通行点数量: {len(self.traversable_indices):,}")
        return self.traversable_indices

    def filter_high_z_points(self, z_max_threshold: float = None):
        """
        高度后过滤：将可通行点中Z坐标超标的点标记为不可通行

        Args:
            z_max_threshold: 高度阈值（默认使用self.z_max_threshold，传入则覆盖）
        """
        if self.traversable_indices is None:
            raise ValueError("请先运行可通行评估分割算法")

        if z_max_threshold is not None:
            self.z_max_threshold = float(z_max_threshold)

        print(f"\n开始高度后过滤：Z > {self.z_max_threshold} 米的点视为不可通行")

        traversable_z = self.points[self.traversable_indices, 2]
        valid_mask = traversable_z <= self.z_max_threshold
        final_traversable_indices = self.traversable_indices[valid_mask]

        filtered_count = len(self.traversable_indices) - len(final_traversable_indices)
        print("高度后过滤完成:")
        print(f"  可通行点数量(过滤前): {len(self.traversable_indices):,}")
        print(f"  过滤掉的高点数量: {filtered_count:,}")
        print(f"  可通行点数量(最终): {len(final_traversable_indices):,}")

        self.traversable_indices = final_traversable_indices

    def save_traversable_points(self, output_file: str = "traversable_points.las"):
        """保存最终可通行点（基于过滤后的索引）"""
        if self.traversable_indices is None:
            raise ValueError("请先运行可通行评估分割和高度过滤")

        print(f"\n正在保存最终可通行点到: {output_file}")
        new_las = laspy.create(point_format=self.las_data.header.point_format, file_version=self.las_data.header.version)
        new_las.header = self.las_data.header
        new_las.points = self.las_data.points[self.traversable_indices]
        new_las.write(output_file)
        print(f"保存完成! 包含 {len(new_las.points):,} 个可通行点")

    def save_untraversable_points(self, output_file: str = "untraversable_points.las"):
        """保存不可通行点（包含评估过滤+高度过滤的点）"""
        if self.traversable_indices is None:
            raise ValueError("请先运行可通行评估分割和高度过滤")

        print(f"\n正在保存不可通行点到: {output_file}")
        total_points = len(self.points)

        untraversable_mask = np.ones(total_points, dtype=bool)
        untraversable_mask[self.traversable_indices] = False
        untraversable_indices = np.nonzero(untraversable_mask)[0]

        new_las = laspy.create(point_format=self.las_data.header.point_format, file_version=self.las_data.header.version)
        new_las.header = self.las_data.header
        new_las.points = self.las_data.points[untraversable_indices]
        new_las.write(output_file)
        print(f"保存完成! 包含 {len(new_las.points):,} 个不可通行点")

    def get_statistics(self):
        """获取最终分割统计（基于高度过滤后的结果）"""
        if self.traversable_indices is None:
            raise ValueError("请先运行可通行评估分割和高度过滤")

        total_points = len(self.points)
        traversable_count = len(self.traversable_indices)
        untraversable_count = total_points - traversable_count
        traversable_ratio = traversable_count / total_points * 100

        # 可通行点Z
        traversable_z = self.points[self.traversable_indices, 2]

        # 不可通行点Z（修复原实现的广播/索引错误）
        untraversable_mask = np.ones(total_points, dtype=bool)
        untraversable_mask[self.traversable_indices] = False
        untraversable_z = self.points[untraversable_mask, 2]

        print(f"\n=== 最终分割统计信息 ===")
        print(f"总点数: {total_points:,}")
        print(f"可通行点数: {traversable_count:,} ({traversable_ratio:.1f}%)")
        print(f"不可通行点数: {untraversable_count:,} ({100 - traversable_ratio:.1f}%)")
        print(f"高度阈值: Z ≤ {self.z_max_threshold} 米")

        if traversable_count > 0:
            print(f"可通行点Z范围: {traversable_z.min():.2f} ~ {traversable_z.max():.2f}")
        else:
            print("可通行点Z范围: 无可通行点")

        if untraversable_count > 0:
            print(f"不可通行点Z范围: {untraversable_z.min():.2f} ~ {untraversable_z.max():.2f}")
        else:
            print("不可通行点Z范围: 无不可通行点")

        return {
            "total_points": total_points,
            "traversable_points": traversable_count,
            "untraversable_points": untraversable_count,
            "traversable_ratio": traversable_ratio,
            "z_threshold": self.z_max_threshold
        }


def main():
    """主函数：评估前高度预过滤 + 基于可通行评估的点云分割 + 结果输出"""
    print("=" * 70)
    print("基于可通行评估的点云分割（含评估前Z轴高度预过滤）")
    print("流程：高度预过滤（Z>阈值删除）→ 可通行评估分割 → 结果输出")
    print("=" * 70)

    try:
        segmenter = TraversabilityAssessmentPointCloudSegmentation()

        # 1. 加载LAS文件，并在评估前删除高于阈值的点云（可调整阈值）
        las_file = "scans21.las"
        segmenter.load_las_file(las_file, pre_z_max_threshold=1.0)

        # 2. 可通行评估分割参数（可调整）
        radius = 0.5                 # 邻域半径（略大于车辆外接球半径）
        max_climbing_angle = 20.0    # 最大爬坡角度（度）
        relaxation_factor = 0.4      # 松弛因子
        segmenter.traversability_segmentation(
            radius=radius,
            max_climbing_angle=max_climbing_angle,
            relaxation_factor=relaxation_factor
        )

        # （可选）如果你还希望在“可通行点”结果里再做一次高度后过滤，可打开下面这行：
        # segmenter.filter_high_z_points(z_max_threshold=1.0)

        # 3. 输出结果（文件名包含关键参数）
        def format_param(value):
            return f"{value}".replace(".", "p")

        base_name = os.path.splitext(os.path.basename(las_file))[0]
        radius_tag = format_param(radius)
        angle_tag = format_param(max_climbing_angle)
        relax_tag = format_param(relaxation_factor)
        z_tag = format_param(segmenter.z_max_threshold)

        tag = f"{base_name}_trav_r{radius_tag}_a{angle_tag}_rf{relax_tag}_z{z_tag}"

        segmenter.get_statistics()
        segmenter.save_traversable_points(f"traversable_{tag}.las")
        segmenter.save_untraversable_points(f"untraversable_{tag}.las")

        print("\n全部流程完成！")
        print("最终结果文件:")
        print(f"  - traversable_{tag}.las (可通行点)")
        print(f"  - untraversable_{tag}.las (不可通行点)")

    except FileNotFoundError as e:
        print(f"\n错误: {e}")
        print("请检查LAS文件路径是否正确")
    except Exception as e:
        print(f"\n运行错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
