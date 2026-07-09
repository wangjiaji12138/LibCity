"""
根据输入坐标（通过 config 输入）计算最近站点信息。

本项目采用 Bigscity-LibCity 自带的 LaDe 数据集作为参考地理网格，
因此这里将“最近的加油站”映射为：距离输入坐标最近的网格站点信息。

运行方式：
    python find_nearest_station.py --config config/find_nearest_station.json
"""

import os
import json
import csv
import argparse
from math import radians, sin, cos, sqrt, atan2

# 与 LibCity 数据集对齐：.geo 坐标为 [lng, lat]，且采用 WGS84


def parse_coordinate(coordinate):
    items = coordinate.strip()[1:-1].split(',')
    return float(items[0]), float(items[1])


def haversine_distance_m(lat1, lng1, lat2, lng2):
    """计算两个 WGS84 坐标之间的球面距离，单位米。"""
    phi1, lambda1, phi2, lambda2 = radians(lat1), radians(lng1), radians(lat2), radians(lng2)
    dphi = phi2 - phi1
    dlambda = lambda2 - lambda1
    a = sin(dphi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(dlambda / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return 6371000 * c


def load_geo(geo_path):
    geo_rows = []
    with open(geo_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for parts in reader:
            if not parts or len(parts) < 3:
                continue
            geo_rows.append({
                'geo_id': parts[0].strip(),
                'type': parts[1].strip(),
                'coordinates': parts[2].strip(),
            })
    return header, geo_rows


def find_nearest(lat, lng, geo_rows, top_k=1):
    scored = []
    for row in geo_rows:
        if row['type'] != 'Point':
            continue
        lng2, lat2 = parse_coordinate(row['coordinates'])
        scored.append({
            'geo_id': row['geo_id'],
            'lat': lat2,
            'lng': lng2,
            'distance_m': haversine_distance_m(lat, lng, lat2, lng2),
        })
    scored.sort(key=lambda x: x['distance_m'])
    return scored[:top_k]


def resolve_dataset_root(dataset_name, workspace_root=None):
    if workspace_root is None:
        workspace_root = os.getcwd()
    candidate = os.path.join(workspace_root, 'raw_data', dataset_name)
    if os.path.isdir(candidate):
        return candidate
    candidate = os.path.join(workspace_root, 'raw_data', dataset_name, dataset_name)
    if os.path.isdir(candidate):
        return candidate
    raise FileNotFoundError(f'未找到数据集目录: raw_data/{dataset_name}')


def build_default_config():
    return {
        'dataset': 'LaDe_CQ',
        'workspace_root': os.getcwd(),
        'input_coordinate': [106.551557, 29.563009],
        'top_k': 5,
        'output': {
            'print': True,
            'json_path': 'output/nearest_station_result.json'
        }
    }


def parse_args():
    parser = argparse.ArgumentParser(description='Find nearest station info by config/input coordinates.')
    parser.add_argument('--config', type=str, default='config/find_nearest_station.json', help='配置文件路径')
    return parser.parse_args()


def load_config(path):
    cfg = build_default_config()
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            user = json.load(f)
        for section in ('dataset', 'workspace_root', 'input_coordinate', 'top_k', 'output'):
            if section in user:
                cfg[section] = user[section]
    return cfg


def main():
    args = parse_args()
    cfg = load_config(args.config)

    lng, lat = cfg['input_coordinate']
    if len(cfg['input_coordinate']) != 2:
        raise ValueError('input_coordinate 必须为 [lng, lat]')
    dataset_root = resolve_dataset_root(cfg['dataset'], cfg.get('workspace_root'))
    geo_path = os.path.join(dataset_root, f"{cfg['dataset']}.geo")
    if not os.path.exists(geo_path):
        raise FileNotFoundError(f'缺少 .geo 文件: {geo_path}')

    header, geo_rows = load_geo(geo_path)
    results = find_nearest(lat, lng, geo_rows, top_k=int(cfg.get('top_k', 1)))

    output = {
        'dataset': cfg['dataset'],
        'query': {'lat': lat, 'lng': lng},
        'top_k': len(results),
        'results': results
    }

    if cfg.get('output', {}).get('print', True):
        print(f"dataset: {cfg['dataset']}")
        print(f"query: lat={lat}, lng={lng}")
        print('nearest stations:')
        for idx, item in enumerate(results, 1):
            print(f"{idx}. geo_id={item['geo_id']}, lat={item['lat']}, lng={item['lng']}, distance_m={item['distance_m']:.2f}")

    out_cfg = cfg.get('output', {})
    if out_cfg.get('json_path'):
        os.makedirs(os.path.dirname(out_cfg['json_path']) or '.', exist_ok=True)
        with open(out_cfg['json_path'], 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"结果已写入: {out_cfg['json_path']}")

    return output


if __name__ == '__main__':
    main()
