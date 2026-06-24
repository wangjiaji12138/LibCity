import pandas as pd
import numpy as np
import argparse
import os
import json
from geopy.distance import geodesic

class DataProcessor:
    def __init__(self,
                city: str = "sh",
                grid_size: float = 0.05,
                time_col: str = "pickup_time",
                time_size: str = "30min",
                day_start_hour: int = 8,
                day_end_hour: int = 20,
                 ) -> None:
        self.city = city
        self.grid_size = grid_size
        self.time_col = time_col
        self.time_size = time_size
        self.day_start_hour = day_start_hour
        self.day_end_hour = day_end_hour

        # 数据路径
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_raw = "/mnt/wangjiaji/LaDe/stg_prediction/data/raw"

        self.raw_data_path = os.path.join(self.data_raw, f"pickup_{city.lower()}.csv")

        self.raw_data = pd.read_csv(self.raw_data_path)
        print(f"Data loading completed, {len(self.raw_data)} records")

        self.processed_data = None
        self.grid_idx = None
        self.gridSplitter = GridSplitter(city, grid_size, self.raw_data)
        self.gridSplitter.find_border()
        self.gridSplitter.create_grid_idx()
        self.gridSplitter.find_grid_id()  # 在 GridSplitter 内部完成 ID 重映射


    def process(self):
        self.processed_data = self.raw_data.copy()

        def add_year(time_str):
            """为缺少年份的时间字符串添加年份"""
            if pd.isna(time_str):
                return time_str
            time_str = str(time_str).strip()
            if len(time_str.split()[0].split('-')) == 2:
                month_day = time_str.split()[0]
                time_part = time_str.split()[1] if len(time_str.split()) > 1 else "00:00:00"
                return f"2022-{month_day} {time_part}"
            return time_str

        # 转换时间列
        self.processed_data['datetime'] = pd.to_datetime(
            self.processed_data[self.time_col].apply(add_year),
            errors='coerce'
        )
        self.processed_data = self.processed_data.dropna(subset=['datetime'])

        # 使用 GridSplitter 提供的连续 ID
        self.processed_data['grid_id'] = self.gridSplitter.get_mapped_grid_ids(
            self.processed_data['lng'].values,
            self.processed_data['lat'].values
        )

        # 按时间分组
        self.processed_data = self.processed_data.set_index('datetime')
        self.processed_data = self.processed_data.between_time(
            f'{self.day_start_hour:02d}:00:00',
            f'{self.day_end_hour:02d}:59:59'
        )
        groups = self.processed_data.groupby(pd.Grouper(freq=self.time_size, label='left', closed='left'))

        # 构建完整时间范围（包括没有数据的时段）
        # 使用 origin='start_day' 保持时间从每天 00:00 开始，与 groupby 对齐
        min_time = self.processed_data.index.min()
        max_time = self.processed_data.index.max()
        full_time_range = pd.date_range(
            start=min_time.floor('D') + pd.Timedelta(hours=self.day_start_hour),
            end=max_time.floor('D') + pd.Timedelta(hours=self.day_end_hour, minutes=59, seconds=59),
            freq=self.time_size
        )

        # 创建时间到索引的映射
        time_to_idx = {ts: i for i, ts in enumerate(full_time_range)}

        # 只保留白天时段
        from datetime import time as dt_time
        start_time = dt_time(hour=self.day_start_hour)
        end_time = dt_time(hour=self.day_end_hour, minute=59, second=59)
        valid_mask = [(ts.time() >= start_time and ts.time() <= end_time) for ts in full_time_range]
        full_time_range = full_time_range[valid_mask]
        time_to_idx = {ts: i for i, ts in enumerate(full_time_range)}

        # 使用连续 ID 的有效网格列表
        valid_grids = self.gridSplitter.valid_grids
        N = len(valid_grids)

        # 初始化数据集（全零）
        dataset = np.zeros(shape=(len(full_time_range), N, 1))

        # 统计每个时间段的订单数
        # valid_grids 中的 grid[0] 是映射后的连续 ID (0, 1, 2, ..., N-1)
        for ts, group in groups:
            if ts in time_to_idx:
                idx = time_to_idx[ts]
                for i, grid in enumerate(valid_grids):
                    # grid[0] 是新 ID (0 到 N-1)
                    dataset[idx, i, 0] = len(group[group.grid_id == grid[0]])

        time_feature = full_time_range.tolist()
        print('Number of time stamps:{}'.format(len(time_feature)))

        # ==================== 生成样本并过滤全零样本 ====================
        seq_len = 48
        horizon = 48
        x_offsets = np.sort(np.concatenate((np.arange(-(seq_len - 1), 1),)))
        y_offsets = np.sort(np.arange(1, horizon + 1))

        x_samples = []
        y_samples = []
        time_indices = []
        
        min_t = abs(min(x_offsets))
        max_t = len(full_time_range) - max(y_offsets)
        
        for t in range(min_t, max_t):
            x_ = dataset[t + x_offsets]  # (seq_len, N, 1)
            y_ = dataset[t + y_offsets]   # (horizon, N, 1)
            
            # 全部保留样本，不做过滤
            x_samples.append(x_)
            y_samples.append(y_)
            time_indices.append(t)

        x_all = np.stack(x_samples, axis=0)  # (N_samples, seq_len, N, 1)
        y_all = np.stack(y_samples, axis=0)   # (N_samples, horizon, N, 1)
        
        print(f"Filtered samples: {len(x_samples)} / {max_t - min_t} (removed {(max_t - min_t) - len(x_samples)} all-zero samples)")

        # ==================== 划分训练/验证/测试 ====================
        train_ratio = 0.6
        val_ratio = 0.2
        
        num_samples = len(x_samples)
        num_train = round(num_samples * train_ratio)
        num_val = round(num_samples * val_ratio)
        
        train_idx = np.arange(num_train)
        val_idx = np.arange(num_train, num_train + num_val)
        test_idx = np.arange(num_train + num_val, num_samples)
        # 直接使用原始数据（不进行节点归一化）
        self.split_data = {
            'x_train': x_all[train_idx], 'y_train': y_all[train_idx],
            'x_val': x_all[val_idx], 'y_val': y_all[val_idx],
            'x_test': x_all[test_idx], 'y_test': y_all[test_idx],
        }
        print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
        self.time_indices = {
            'train': time_indices[:num_train],
            'val': time_indices[num_train:num_train+num_val],
            'test': time_indices[num_train+num_val:],
        }

        print(dataset.shape)

        # ==================== LibCity Format Output ====================
        self._save_libcity_format(dataset, time_feature)

    def _save_libcity_format(self, dataset, time_feature):
        """生成 LibCity 格式文件: .geo, .dyna, .rel, config.json, 以及 npz 样本文件"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_name = f"LaDe_{self.city.upper()}"
        output_dir = os.path.join(script_dir, "raw_data", dataset_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        valid_grids = self.gridSplitter.valid_grids
        N = len(valid_grids)

        # ==================== 保存 npz 样本文件 ====================
        split_data = self.split_data
        for split_name in ['train', 'val', 'test']:
            npz_path = os.path.join(output_dir, f'{split_name}.npz')
            np.savez_compressed(
                npz_path,
                x=split_data[f'x_{split_name}'],
                y=split_data[f'y_{split_name}']
            )
        print(f"NPZ sample files saved (train/val/test)")

        # 1. 生成 .geo 文件
        geo_data = []
        for grid in valid_grids:
            grid_id, lng_min, lng_max, lat_min, lat_max = grid
            lng_center = (lng_min + lng_max) / 2
            lat_center = (lat_min + lat_max) / 2
            geo_data.append({
                'geo_id': grid_id,
                'type': 'Point',
                'coordinates': f"[{lng_center:.6f}, {lat_center:.6f}]"
            })
        geo_df = pd.DataFrame(geo_data)
        geo_df.to_csv(os.path.join(output_dir, f'{dataset_name}.geo'), index=False)
        print(f"LibCity .geo saved: ({N} nodes)")

        # 2. 生成 .dyna 文件
        num_times = dataset.shape[0]
        dyna_data = []
        dyna_id = 0
        for t_idx in range(num_times):
            ts = time_feature[t_idx]
            ts_str = pd.Timestamp(ts).strftime('%Y-%m-%dT%H:%M:%SZ') if pd.notna(ts) else f"2022-01-01T00:00:00Z"
            for node_idx in range(N):
                grid_id = node_idx  # 直接使用连续索引作为 ID
                package_count = float(dataset[t_idx, node_idx, 0])
                dyna_data.append({
                    'dyna_id': dyna_id,
                    'type': 'state',
                    'time': ts_str,
                    'entity_id': grid_id,
                    'package': package_count
                })
                dyna_id += 1
        dyna_df = pd.DataFrame(dyna_data)
        dyna_df.to_csv(os.path.join(output_dir, f'{dataset_name}.dyna'), index=False)
        print(f"LibCity .dyna saved: ({len(dyna_df)} records)")

        # 3. 生成 .rel 文件（基于所有grid之间的地理距离全连接图）
        rel_data = []
        rel_id = 0
        centers = []
        for grid in valid_grids:
            lng_center = (grid[1] + grid[2]) / 2
            lat_center = (grid[3] + grid[4]) / 2
            centers.append((lat_center, lng_center))

        # 全连接：计算所有grid对之间的距离（只取上三角避免重复）
        for i in range(N):
            for j in range(i + 1, N):
                dist = geodesic(centers[i], centers[j]).kilometers
                rel_data.append({
                    'rel_id': rel_id,
                    'type': 'geo',
                    'origin_id': i,
                    'destination_id': j,
                    'distance': round(dist, 6)
                })
                rel_id += 1
                # 添加反向边（无向图）
                rel_data.append({
                    'rel_id': rel_id,
                    'type': 'geo',
                    'origin_id': j,
                    'destination_id': i,
                    'distance': round(dist, 6)
                })
                rel_id += 1

        rel_df = pd.DataFrame(rel_data)
        rel_df.to_csv(os.path.join(output_dir, f'{dataset_name}.rel'), index=False)
        print(f"LibCity .rel saved: ({len(rel_df)} relations, sparse graph with {N} nodes)")
        print(f"  Graph density: {len(rel_df) / (N * (N - 1)) * 100:.2f}%")

        # 4. 生成 config.json
        time_intervals_map = {'1h': 3600, '30min': 1800, '15min': 900, '5min': 300}
        
        config = {
            "geo": {"including_types": ["Point"], "Point": {}},
            "rel": {"including_types": ["geo"], "geo": {"distance": "num"}},
            "dyna": {"including_types": ["state"], "state": {"entity_id": "geo_id", "package": "num"}},
            "info": {
                "data_col": ["package"],
                "weight_col": "distance",
                "data_files": [dataset_name],
                "geo_file": dataset_name,
                "rel_file": dataset_name,
                "output_dim": 1,
                "time_intervals": time_intervals_map.get(self.time_size, 3600),
                "init_weight_inf_or_zero": "inf",
                "set_weight_link_or_dist": "dist",
                "calculate_weight_adj": True,
                "weight_adj_epsilon": 0.1,
                "scaler": "standard"
            }
        }
        with open(os.path.join(output_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        print(f"LibCity config.json saved (null_value=-1.0)")


class GridSplitter:
    def __init__(self,
                 city: str,
                 grid_size: float,
                 raw_data: pd.DataFrame,
                 ) -> None:
        self.city = city
        self.grid_size = grid_size
        self.raw_data = raw_data
        self.border = None
        self.grids = []  # (id, lng_min, lng_max, lat_min, lat_max)
        self.valid_grids = []  # (new_id, lng_min, lng_max, lat_min, lat_max) - 连续 ID
        self.adj = None
        self._id_mapping = {}  # 原始 ID -> 新连续 ID
        self._inverse_mapping = {}  # 新连续 ID -> 原始 ID

    def find_border(self):
        lng_min = float(self.raw_data["lng"].min())
        lng_max = float(self.raw_data["lng"].max())
        lat_min = float(self.raw_data["lat"].min())
        lat_max = float(self.raw_data["lat"].max())

        self.lng_min = float(int(lng_min / self.grid_size)) * self.grid_size
        self.lng_max = (float(int(lng_max / self.grid_size)) + 1) * self.grid_size
        self.lat_min = float(int(lat_min / self.grid_size)) * self.grid_size
        self.lat_max = (float(int(lat_max / self.grid_size)) + 1) * self.grid_size

    def create_grid_idx(self):
        idx = 0
        n_lng = int((self.lng_max - self.lng_min) / self.grid_size) + 1
        n_lat = int((self.lat_max - self.lat_min) / self.grid_size) + 1

        for i in range(n_lng):
            for j in range(n_lat):
                lng_min_val = self.lng_min + i * self.grid_size
                lng_max_val = lng_min_val + self.grid_size
                lat_min_val = self.lat_min + j * self.grid_size
                lat_max_val = lat_min_val + self.grid_size
                self.grids.append((idx, lng_min_val, lng_max_val, lat_min_val, lat_max_val))
                idx += 1

    def find_grid_id(self):
        """识别有效网格并重新分配连续的 ID"""
        n_lng = int((self.lng_max - self.lng_min) / self.grid_size) + 1
        n_lat = int((self.lat_max - self.lat_min) / self.grid_size) + 1

        # 计算每个点属于哪个网格
        lng_arr = self.raw_data['lng'].values
        lat_arr = self.raw_data['lat'].values
        lng_idx = ((lng_arr - self.lng_min) / self.grid_size).astype(int)
        lat_idx = ((lat_arr - self.lat_min) / self.grid_size).astype(int)

        # 计算原始 grid_id
        original_grid_ids = (lng_idx * n_lat + lat_idx).tolist()

        # 找出所有唯一的有效网格 ID
        unique_original_ids = sorted(set(original_grid_ids))

        # 建立 ID 映射：原始 ID -> 新连续 ID
        self._id_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_original_ids)}
        self._inverse_mapping = {new_id: old_id for new_id, old_id in enumerate(unique_original_ids)}

        # 构建连续 ID 的 valid_grids
        for new_id, old_id in enumerate(unique_original_ids):
            grid = self.grids[old_id]
            self.valid_grids.append((new_id, grid[1], grid[2], grid[3], grid[4]))

        print(f"Grid ID mapping: {len(unique_original_ids)} unique grids -> continuous IDs 0-{len(unique_original_ids)-1}")

    def get_mapped_grid_ids(self, lng_arr, lat_arr):
        """获取映射后的连续网格 ID（用于数据处理）"""
        n_lng = int((self.lng_max - self.lng_min) / self.grid_size) + 1
        n_lat = int((self.lat_max - self.lat_min) / self.grid_size) + 1

        # 计算每个点属于哪个原始网格
        lng_idx = ((lng_arr - self.lng_min) / self.grid_size).astype(int)
        lat_idx = ((lat_arr - self.lat_min) / self.grid_size).astype(int)
        original_ids = lng_idx * n_lat + lat_idx

        # 映射为连续 ID
        mapped_ids = np.array([self._id_mapping.get(id, -1) for id in original_ids])
        return mapped_ids

    def create_adj(self):
        N = len(self.valid_grids)
        centers = []
        for grid in self.valid_grids:
            lng_center = (grid[1] + grid[2]) / 2
            lat_center = (grid[3] + grid[4]) / 2
            centers.append((lat_center, lng_center))

        adj = np.full((N, N), np.inf)
        for i in range(N):
            for j in range(N):
                adj[i][j] = geodesic(centers[i], centers[j]).kilometers
        distances = adj[~np.isinf(adj)].flatten()

        std = np.median(distances)

        print(distances.mean(), std, distances.max(), distances.min())

        self.adj = np.exp(-np.square(adj / std))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', type=str, help='City name (e.g., sh, cq, jl)')
    parser.add_argument('--all', action='store_true', help='Process all cities')
    parser.add_argument('--grid_size', type=float, default=0.05)
    parser.add_argument('--time_size', type=str, default="30min")
    parser.add_argument('--time_col', type=str, default='pickup_time')
    parser.add_argument('--day_start_hour', type=int, default=8)
    parser.add_argument('--day_end_hour', type=int, default=20)

    args = parser.parse_args()

    # 查找原始数据目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    possible_raw = [
        os.path.join(script_dir, "data/raw"),
        os.path.join(script_dir, "DSTGN/data/raw"),
        "/mnt/wangjiaji/LaDe/stg_prediction/data/raw",
    ]
    raw_dir = None
    for path in possible_raw:
        if os.path.isdir(path):
            raw_dir = path
            break
    if raw_dir is None:
        raise FileNotFoundError("Raw data dir not found")

    if args.all:
        cities = []
        for fn in os.listdir(raw_dir):
            if fn.startswith("pickup_") and fn.endswith(".csv"):
                city = fn[len("pickup_"):-len(".csv")]
                if city:
                    cities.append(city)
        cities = sorted(set(cities))
    elif args.city:
        cities = [args.city]
    else:
        raise ValueError("请指定 --city 或使用 --all")

    for city in cities:
        print(f"\n{'='*60}\nProcessing: {city}\n{'='*60}")
        try:
            processor = DataProcessor(
                city=city,
                grid_size=args.grid_size,
                time_col=args.time_col,
                time_size=args.time_size,
                day_start_hour=args.day_start_hour,
                day_end_hour=args.day_end_hour,
            )
            processor.process()
            print(f"✓ Done: {city}")
        except Exception as e:
            import traceback
            print(f"✗ Failed: {city}\n{traceback.format_exc()}")
