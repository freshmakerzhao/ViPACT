import json

# 这是你提供的 JSON 数据的精简版（去掉了不影响逻辑的冗余字段）
json_input = """
{
  "det_debug": {
    "candidates": [
      {"index": 0, "score": 0.7002, "box_xyxy": [264.34, 174.55, 286.33, 203.41]},
      {"index": 2, "score": 0.6897, "box_xyxy": [340.22, 239.35, 366.22, 273.45]},
      {"index": 1, "score": 0.6551, "box_xyxy": [201.18, 255.40, 232.42, 289.79]},
      {"index": 3, "score": 0.6121, "box_xyxy": [201.22, 312.92, 236.31, 351.59]},
      {"index": 4, "score": 0.2531, "box_xyxy": [185.82, 1.12, 215.03, 66.69]},
      {"index": 5, "score": 0.1788, "box_xyxy": [60.73, 94.66, 612.36, 478.49]},
      {"index": 6, "score": 0.1027, "box_xyxy": [442.60, 385.44, 568.83, 478.92]},
      {"index": 7, "score": 0.0994, "box_xyxy": [199.02, 173.25, 368.62, 352.48]}
    ]
  }
}
"""

def analyze_left_to_right(json_str, score_threshold=0.5, x_tolerance=15.0):
    # 1. 加载数据
    data = json.loads(json_str)
    candidates = data["det_debug"]["candidates"]
    
    valid_cubes = []
    
    # 2. 过滤有效方块并计算中心 X 坐标
    # 观察你的数据，真正的方块 score 都在 0.6 以上，这里设 0.5 作为阈值滤除巨大背景框
    for c in candidates:
        if c["score"] > score_threshold:
            box = c["box_xyxy"]
            center_x = (box[0] + box[2]) / 2.0
            valid_cubes.append({
                "index": c["index"],
                "center_x": center_x,
                "score": c["score"],
                "box": box
            })
            
    # 3. 按 X 坐标从小到大（从左到右）排序
    valid_cubes.sort(key=lambda cube: cube["center_x"])
    
    if not valid_cubes:
        print("没有找到符合条件的方块！")
        return

    # 4. 容差聚类（解决上下两个方块属于“同一列”的问题）
    current_rank = 1
    valid_cubes[0]["rank"] = current_rank
    
    for i in range(1, len(valid_cubes)):
        prev_x = valid_cubes[i-1]["center_x"]
        curr_x = valid_cubes[i]["center_x"]
        
        # 如果当前方块和上一个方块的 X 坐标差距大于 tolerance，说明进入了新的一列
        if abs(curr_x - prev_x) > x_tolerance:
            current_rank += 1
            
        valid_cubes[i]["rank"] = current_rank

    # 5. 打印结果
    print("=== 方块空间位置分析 (基于 X 坐标从左到右) ===")
    for cube in valid_cubes:
        print(f"方块 Index: {cube['index']} "
              f"| 置信度: {cube['score']:.3f} "
              f"| 中心X坐标: {cube['center_x']:.1f} "
              f"| 结论: 左侧第 {cube['rank']} 列")

if __name__ == "__main__":
    analyze_left_to_right(json_input)