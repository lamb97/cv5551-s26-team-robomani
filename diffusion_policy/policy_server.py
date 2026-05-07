import argparse
import base64
import json
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import dill
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf


def _ensure_diffusion_policy_on_path(diffusion_root: Path) -> None:
    diffusion_root = diffusion_root.expanduser().resolve()
    if str(diffusion_root) not in sys.path:
        sys.path.insert(0, str(diffusion_root))


def _decode_rgb_image(image_b64: str) -> np.ndarray:
    encoded = base64.b64decode(image_b64.encode("utf-8"))
    array = np.frombuffer(encoded, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode JPEG image.")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


class DiffusionPolicyServer:
    def __init__(self, checkpoint: Path, diffusion_root: Path, device: str) -> None:
        _ensure_diffusion_policy_on_path(diffusion_root)

        from diffusion_policy.real_world.real_inference_util import get_real_obs_dict

        self._get_real_obs_dict = get_real_obs_dict
        self.checkpoint = checkpoint.expanduser().resolve()
        self.device = torch.device(device)

        payload = torch.load(self.checkpoint.open("rb"), pickle_module=dill, map_location="cpu")
        self.cfg = payload["cfg"]
        workspace_cls = hydra.utils.get_class(self.cfg._target_)
        workspace = workspace_cls(self.cfg, output_dir=str(self.checkpoint.parent.parent))
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)

        self.policy = workspace.ema_model if self.cfg.training.use_ema else workspace.model
        self.policy.to(self.device)
        self.policy.eval()

        self.shape_meta = OmegaConf.to_container(self.cfg.shape_meta, resolve=True)
        self.n_obs_steps = int(self.cfg.n_obs_steps)
        self.n_action_steps = int(self.cfg.n_action_steps)
        self.horizon = int(self.cfg.horizon)

    def reset(self) -> None:
        if hasattr(self.policy, "reset"):
            self.policy.reset()

    def predict(self, images_rgb, eef_xyz_mm):
        if len(images_rgb) != self.n_obs_steps:
            raise ValueError(f"Expected {self.n_obs_steps} images, got {len(images_rgb)}.")
        if len(eef_xyz_mm) != self.n_obs_steps:
            raise ValueError(f"Expected {self.n_obs_steps} eef_xyz entries, got {len(eef_xyz_mm)}.")

        image_stack = np.stack(images_rgb, axis=0)
        eef_xyz = np.asarray(eef_xyz_mm, dtype=np.float32)
        if eef_xyz.shape != (self.n_obs_steps, 3):
            raise ValueError(
                f"Expected eef_xyz_mm shape {(self.n_obs_steps, 3)}, got {eef_xyz.shape}."
            )

        obs_np = self._get_real_obs_dict(
            env_obs={
                "image": image_stack,
                "eef_xyz": eef_xyz,
            },
            shape_meta=self.shape_meta,
        )
        obs_torch = {
            key: torch.from_numpy(value).unsqueeze(0).to(self.device)
            for key, value in obs_np.items()
        }

        with torch.no_grad():
            result = self.policy.predict_action(obs_torch)

        action = result["action"][0].detach().to("cpu").numpy().astype(np.float64)
        action_pred = result["action_pred"][0].detach().to("cpu").numpy().astype(np.float64)
        return {
            "actions_mm": action.tolist(),
            "action_pred_mm": action_pred.tolist(),
            "n_action_steps": self.n_action_steps,
            "horizon": self.horizon,
        }


class PolicyRequestHandler(BaseHTTPRequestHandler):
    server_version = "Lite6DiffusionPolicyServer/0.1"

    @property
    def policy_server(self) -> DiffusionPolicyServer:
        return self.server.policy_server

    def _send_json(self, payload, status=HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def do_GET(self) -> None:
        if self.path != "/health":
            self._send_json({"error": "Not found."}, status=HTTPStatus.NOT_FOUND)
            return

        self._send_json(
            {
                "status": "ok",
                "checkpoint": str(self.policy_server.checkpoint),
                "device": str(self.policy_server.device),
                "n_obs_steps": self.policy_server.n_obs_steps,
                "n_action_steps": self.policy_server.n_action_steps,
                "horizon": self.policy_server.horizon,
            }
        )

    def do_POST(self) -> None:
        try:
            if self.path == "/reset":
                self.policy_server.reset()
                self._send_json({"status": "reset"})
                return

            if self.path != "/act":
                self._send_json({"error": "Not found."}, status=HTTPStatus.NOT_FOUND)
                return

            request = self._read_json()
            obs = request["obs"]
            images_rgb = [_decode_rgb_image(item) for item in obs["images_jpeg_b64"]]
            response = self.policy_server.predict(
                images_rgb=images_rgb,
                eef_xyz_mm=obs["eef_xyz_mm"],
            )
            self._send_json(response)
        except Exception as exc:
            self._send_json(
                {"error": type(exc).__name__, "message": str(exc)},
                status=HTTPStatus.BAD_REQUEST,
            )


def parse_args():
    parser = argparse.ArgumentParser(description="Serve a diffusion policy checkpoint over HTTP.")
    parser.add_argument(
        "--checkpoint",
        default="/home/rob/08.39.50_train_diffusion_unet_image_demo_xyz_image/checkpoints/latest.ckpt",
    )
    parser.add_argument("--diffusion-root", default="/home/rob/diffusion_policy")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy_server = DiffusionPolicyServer(
        checkpoint=Path(args.checkpoint),
        diffusion_root=Path(args.diffusion_root),
        device=args.device,
    )
    policy_server.reset()

    httpd = ThreadingHTTPServer((args.host, args.port), PolicyRequestHandler)
    httpd.policy_server = policy_server

    print(
        f"Serving checkpoint={policy_server.checkpoint} "
        f"on http://{args.host}:{args.port}",
        flush=True,
    )
    httpd.serve_forever()


if __name__ == "__main__":
    main()
