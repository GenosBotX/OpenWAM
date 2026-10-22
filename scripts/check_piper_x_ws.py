"""检查真实云端服务；默认只握手，--infer 额外执行两次合成观测推理。"""

import argparse
import base64
import io
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
from websockets.sync.client import connect


PROMPT = "A video recorded from a robot's point of view executing the following instruction: pick"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def exchange(ws, payload):
    ws.send(json.dumps(payload, allow_nan=False))
    return json.loads(ws.recv(timeout=300))


def synthetic_observation():
    # 灰图不涉及 R/B 通道差异；仅验证协议和推理返回，不评估任务效果。
    buffer = io.BytesIO()
    Image.new("RGB", (320, 240), (96, 96, 96)).save(buffer, format="PNG")
    png = base64.b64encode(buffer.getvalue()).decode("ascii")
    return {
        "type": "obs",
        "images": dict.fromkeys(("head_camera", "left_wrist_camera", "right_wrist_camera"), png),
        "prompt": PROMPT,
        "state": [0.25, 0, 0.25, 1, 0, 0, 0, 1, 0, 0.5] * 2,
    }


def check_service(url, infer=False):
    report = {"test": "synthetic_protocol_and_inference_only" if infer else "handshake_only", "url": url}
    with connect(url, open_timeout=20, max_size=32 * 1024 * 1024, ping_interval=None) as ws:
        pong = exchange(ws, {"type": "ping"})
        expected = {
            "type": "pong", "embodiment": "piper_x",
            "contract_id": "robodojo-real-native-eef20-v1",
            "state_dim": 20, "action_dim": 20,
            "model_state_dim": 80, "model_action_dim": 80,
            "action_response_mode": "chunk", "gripper_convention": "zero_closed_one_open",
            "rotation_representation": "rot6d_first_two_columns_concatenated",
            "position_units": "meters", "target_frame": "per_arm_robot_base",
            "action_target": "absolute_pose", "normalization": "server_side_min_max",
        }
        for key, value in expected.items():
            require(pong.get(key) == value, f"握手字段 {key} 不匹配：{pong.get(key)!r}，预期 {value!r}")
        num_actions = pong.get("num_actions")
        require(type(num_actions) is int and num_actions > 0, "握手缺少有效的 num_actions")
        report["pong"] = pong
        if infer:
            request = synthetic_observation()
            report["wrong_width_rejections"] = []
            for width in (16, 80):
                result = exchange(ws, {**request, "state": [0.0] * width})
                require(result.get("type") == "error" and result.get("code") == "obs_validation_error",
                        f"未拒绝错误状态宽度 {width}：{result}")
                report["wrong_width_rejections"].append(result)
            require(exchange(ws, {"type": "reset"}).get("type") == "reset_ack", "重置失败")
            report["inferences"] = []
            try:
                for index in range(2):
                    started = time.monotonic()
                    result = exchange(ws, request)
                    elapsed = time.monotonic() - started
                    require(result.get("type") == "action", f"推理失败：{result}")
                    actions = np.asarray(result.get("action"), dtype=np.float64)
                    require(actions.shape == (num_actions, 20) and np.isfinite(actions).all(),
                            f"返回动作形状或数值异常：{actions.shape}")
                    require(result.get("num_actions") == num_actions and result.get("action_dim") == 20,
                            "返回动作元数据与握手不一致")
                    for start in (3, 13):
                        first, second = actions[:, start:start + 3], actions[:, start + 3:start + 6]
                        require((np.linalg.norm(first, axis=-1) > 1e-8).all()
                                and (np.linalg.norm(np.cross(first, second), axis=-1) > 1e-8).all(),
                                "返回动作包含退化旋转")
                    report["inferences"].append({
                        "request": index + 1, "elapsed_seconds": elapsed,
                        "server_latency_ms": result.get("latency_ms"),
                        "shape": list(actions.shape), "action": actions.tolist(),
                    })
            finally:
                require(exchange(ws, {"type": "reset"}).get("type") == "reset_ack", "结束重置失败")
    report["passed"] = True
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://127.0.0.1:8848", help="真实云端服务地址")
    parser.add_argument("--infer", action="store_true", help="发送合成观测；会重置共享策略状态")
    parser.add_argument("--output", type=Path, help="保存 JSON 检查报告")
    args = parser.parse_args()
    report = check_service(args.url, args.infer)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    shapes = [item["shape"] for item in report.get("inferences", [])]
    print(f"检查通过：{args.url}，每次 {report['pong']['num_actions']} 步 × 20 维；推理返回 {shapes}")


if __name__ == "__main__":
    main()
