"""Deploy FoveaVLA-Single and FoveaVLA-Dual on DROID."""

import contextlib
import dataclasses
import datetime
import enum
import json
import pathlib
import signal
import time

import cv2
import numpy as np
from foveavla_client import foveation, image_tools
from foveavla_client import websocket_client_policy
import tyro

DROID_CONTROL_FREQUENCY = 15
_DUAL_FOVEATION_METADATA_VIEWS = ("physical_left", "physical_right")


class ActionSpace(enum.Enum):
    JOINT_POSITION = "joint_position"


@dataclasses.dataclass
class Args:
    # Serial numbers in physical-left / physical-right order. Single uses only left.
    left_camera_id: str
    instruction: str
    right_camera_id: str | None = None
    remote_host: str = "127.0.0.1"
    remote_port: int = 8000
    max_timesteps: int = 900
    open_loop_horizon: int = 8
    # Save a debug video, foveal crops, and a per-query log for each rollout, also when stopped early.
    debug: bool = False
    debug_dir: str = "outputs/rollouts"
    # Abort when the gripper does not follow an open/close command for this long.
    gripper_timeout_s: float = 2.0


@contextlib.contextmanager
def prevent_keyboard_interrupt():
    """Temporarily prevent keyboard interrupts by delaying them until after the protected code."""
    interrupted = False
    original_handler = signal.getsignal(signal.SIGINT)

    def handler(signum, frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, original_handler)
        if interrupted:
            raise KeyboardInterrupt


def _uses_fast_foundation_depth(policy_metadata: dict) -> bool:
    if not policy_metadata.get("requires_metric_depth", False):
        return False
    if not policy_metadata.get("requires_fast_foundation_stereo_pair", False):
        raise ValueError("The metric-depth policy does not declare a supported online stereo input contract")
    if policy_metadata.get("fast_foundation_online_input_contract") != (
        "rectified_rgb_pair_fx_baseline_and_direct_roi_crop_v1"
    ):
        raise ValueError("Unsupported Fast-FoundationStereo online input contract")
    if tuple(policy_metadata.get("fast_foundation_input_resolution", ())) != (640, 360):
        raise ValueError("The DROID client requires the declared Fast-FoundationStereo 640x360 input")
    return True


def _select_droid_zed_cameras(multi_camera_wrapper, camera_ids: tuple[str, ...]) -> None:
    """Limit DROID camera startup before any ZED device is opened."""
    selected = frozenset(camera_ids)
    gather_all = multi_camera_wrapper.gather_zed_cameras

    def gather_selected():
        cameras = gather_all()
        available = {camera.serial_number for camera in cameras}
        missing = selected - available
        if missing:
            raise RuntimeError(f"Configured ZED cameras are unavailable: {sorted(missing)}; found {sorted(available)}")
        return [camera for camera in cameras if camera.serial_number in selected]

    multi_camera_wrapper.gather_zed_cameras = gather_selected


def _resolve_policy_exterior_views(policy_metadata: dict, selected_exterior: str) -> tuple[str, ...]:
    declared_views = tuple(
        policy_metadata.get(
            "droid_exterior_views",
            policy_metadata.get("future_bbox_views", ("selected_exterior",)),
        )
    )
    future_bbox_views = policy_metadata.get("future_bbox_views")
    if future_bbox_views is not None and tuple(future_bbox_views) != declared_views:
        raise ValueError(
            "Policy image and future bbox view orders disagree: "
            f"droid_exterior_views={declared_views}, future_bbox_views={tuple(future_bbox_views)}"
        )
    if declared_views == ("selected_exterior",):
        return (selected_exterior,)
    if declared_views == ("physical_left",):
        return ("left",)
    if declared_views == ("physical_right",):
        return ("right",)
    if declared_views == _DUAL_FOVEATION_METADATA_VIEWS:
        return ("left", "right")
    raise ValueError(f"Unsupported policy future_bbox_views metadata: {declared_views}")


def _parse_future_boxes(prediction: dict, view_count: int) -> tuple[np.ndarray | None, ...]:
    prediction = foveation.canonicalize_fields(prediction)
    bboxes = np.asarray(prediction["future_interaction_box"], dtype=np.float32)
    valid = np.asarray(prediction.get("future_interaction_box_valid", True), dtype=bool)
    if view_count == 1:
        if bboxes.shape != (4,) or valid.ndim != 0:
            raise ValueError(f"expected bbox shape (4,) and scalar validity, got {bboxes.shape} and {valid.shape}")
        bboxes = bboxes[None]
        valid = valid[None]
    elif bboxes.shape != (view_count, 4) or valid.shape != (view_count,):
        raise ValueError(
            f"expected bbox shape ({view_count}, 4) and validity shape ({view_count},), "
            f"got {bboxes.shape} and {valid.shape}"
        )

    return tuple(
        bbox
        if is_valid
        and np.all(np.isfinite(bbox))
        and np.all((bbox >= 0.0) & (bbox <= 1.0))
        and bbox[0] < bbox[2]
        and bbox[1] < bbox[3]
        else None
        for bbox, is_valid in zip(bboxes, valid, strict=True)
    )


def _fast_foundation_camera_geometry(env, camera_id: str, source_width: int) -> tuple[float, float]:
    camera = env.camera_reader.get_camera(camera_id)
    configuration = camera._cam.get_camera_information().camera_configuration
    calibration = configuration.calibration_parameters
    native_width = int(configuration.resolution.width)
    fx = float(calibration.left_cam.fx) * source_width / native_width
    baseline_m = float(calibration.get_camera_baseline()) / 1000.0
    if fx <= 0.0 or not 0.05 <= baseline_m <= 0.2:
        raise ValueError(f"Invalid ZED stereo geometry: fx={fx}, baseline_m={baseline_m}")
    return fx, baseline_m


def _retrieve_fast_foundation_stereo_pair(env, zed_camera, camera_id: str, resolution: tuple[int, int]):
    camera = env.camera_reader.get_camera(camera_id)
    width, height = resolution
    zed_resolution = zed_camera.sl.Resolution(width, height)
    left = zed_camera.sl.Mat()
    right = zed_camera.sl.Mat()
    left_status = camera._cam.retrieve_image(left, zed_camera.sl.VIEW.LEFT, resolution=zed_resolution)
    right_status = camera._cam.retrieve_image(right, zed_camera.sl.VIEW.RIGHT, resolution=zed_resolution)
    if any(status is not None and status != zed_camera.sl.ERROR_CODE.SUCCESS for status in (left_status, right_status)):
        raise RuntimeError(f"Could not retrieve synchronized ZED stereo pair: {left_status}, {right_status}")
    left_rgb = np.asarray(left.get_data())[..., :3][..., ::-1].copy()
    right_rgb = np.asarray(right.get_data())[..., :3][..., ::-1].copy()
    expected_shape = (height, width, 3)
    if left_rgb.shape != expected_shape or right_rgb.shape != expected_shape:
        raise RuntimeError(f"Unexpected ZED stereo pair shapes: {left_rgb.shape}, {right_rgb.shape}")
    return left_rgb, right_rgb


def _prepare_action(action: np.ndarray, action_space: ActionSpace) -> np.ndarray:
    """Validate an 8-D policy action and adapt it to the DROID controller."""
    action = np.asarray(action, dtype=np.float64)
    if action.shape != (8,):
        raise RuntimeError(f"Expected one 8-D action, got {action.shape}")
    if not np.all(np.isfinite(action)):
        raise RuntimeError(f"Policy returned a non-finite action: {action}")

    joints = action[:-1]
    if action_space is not ActionSpace.JOINT_POSITION:
        raise ValueError(f"Unsupported action space: {action_space}")

    gripper = np.array([1.0 if action[-1] > 0.5 else 0.0])
    return np.concatenate([joints, gripper])


def _extract_observation(args, observation, exterior_views):
    images = observation["image"]
    result = {}
    for view in exterior_views:
        serial = args.left_camera_id if view == "left" else args.right_camera_id
        matches = [np.asarray(image) for key, image in images.items() if serial in key and "left" in key]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one rectified left image for {view} camera {serial}; found {list(images)}")
        image = matches[0]
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] not in (3, 4):
            raise RuntimeError(f"Invalid source image for {view}: {image.shape}, {image.dtype}")
        result[f"{view}_image"] = image[..., :3][..., ::-1].copy()
    state = observation["robot_state"]
    result["joint_position"] = np.asarray(state["joint_positions"])
    result["gripper_position"] = np.asarray([state["gripper_position"]])
    return result


def _build_policy_request(curr_obs, exterior_views, foveal_images, prompt, fast_foundation_inputs):
    request = {
        "observation/joint_position": curr_obs["joint_position"],
        "observation/gripper_position": curr_obs["gripper_position"],
        "prompt": prompt,
    }
    for index, (view, crop) in enumerate(zip(exterior_views, foveal_images, strict=True), start=1):
        request[f"observation/exterior_image_{index}_left"] = image_tools.resize_with_pad(
            curr_obs[f"{view}_image"], 224, 224
        )
        crop_name = f"{view}_foveal_image" if len(exterior_views) == 2 else "foveal_image"
        request[f"observation/{crop_name}"] = crop
    request.update({f"observation/fast_foundation_{key}": value for key, value in fast_foundation_inputs.items()})
    return request


def _capture_fast_foundation_inputs(env, zed_camera, args, curr_obs, exterior_views, crop_bboxes, resolution):
    inputs = {}
    for view, crop_bbox in zip(exterior_views, crop_bboxes, strict=True):
        camera_id = args.left_camera_id if view == "left" else args.right_camera_id
        left, right = _retrieve_fast_foundation_stereo_pair(env, zed_camera, camera_id, resolution)
        height, width = curr_obs[f"{view}_image"].shape[:2]
        fx, baseline_m = _fast_foundation_camera_geometry(env, camera_id, width)
        prefix = f"{view}_" if len(exterior_views) == 2 else ""
        inputs.update(
            {
                f"{prefix}{key}": value
                for key, value in {
                    "left_rgb": left,
                    "right_rgb": right,
                    "source_size": np.asarray([width, height], dtype=np.int32),
                    "fx": np.float32(fx),
                    "baseline_m": np.float32(baseline_m),
                    "crop_bbox": crop_bbox,
                }.items()
            }
        )
    return inputs


class GripperWatchdog:
    """Fail when the gripper stays fully opposite to its command, e.g. when the NUC gripper client is dead."""

    def __init__(self, timeout_s: float):
        self._timeout_s = timeout_s
        self._stuck_since = None
        self._command = None

    def record_command(self, command: float) -> None:
        """Track every executed command, including changes between observations."""
        command = float(command > 0.5)
        if command != self._command:
            self._stuck_since = None
        self._command = command

    def check_open_after_reset(self, gripper_position: float) -> None:
        if not np.isfinite(gripper_position) or gripper_position > 0.1:
            raise RuntimeError(
                f"Gripper reports position {gripper_position:.2f} after DROID opened it; "
                "check the gripper USB/power and the NUC gripper client"
            )

    def update(self, gripper_position: float) -> None:
        command = self._command
        stuck = command is not None and (
            (command > 0.5 and gripper_position < 0.05) or (command <= 0.5 and gripper_position > 0.95)
        )
        if not stuck:
            self._stuck_since = None
            return
        now = time.monotonic()
        if self._stuck_since is None:
            self._stuck_since = now
        if now - self._stuck_since > self._timeout_s:
            raise RuntimeError(
                f"Gripper did not follow command {command:.0f} for {self._timeout_s:.1f}s "
                f"(position {gripper_position:.2f}); check the gripper USB/power and the NUC gripper client"
            )


class RolloutRecorder:
    """Write one debug video frame, foveal crops, and one log record per policy query; finalized on any exit."""

    def __init__(self, debug_dir: str, args, views, depth_range_m):
        self._dir = pathlib.Path(debug_dir) / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        (self._dir / "crops").mkdir(parents=True, exist_ok=True)
        self._views = views
        self._depth_range_m = tuple(float(v) for v in depth_range_m)
        self._fps = DROID_CONTROL_FREQUENCY / args.open_loop_horizon
        self._writer = None
        self._records = []
        (self._dir / "args.json").write_text(json.dumps(dataclasses.asdict(args), indent=2))

    def _colorize_depth(self, depth_m):
        """Approximate the model's depth input: log-scaled fixed range, Turbo with near=red, invalid=black."""
        near, far = self._depth_range_m
        depth_m = np.asarray(depth_m, dtype=np.float32)
        valid = np.isfinite(depth_m) & (depth_m >= near) & (depth_m <= far)
        normalized = np.log(np.clip(depth_m, near, far) / near) / np.log(far / near)
        image = cv2.applyColorMap(((1.0 - normalized) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)[..., ::-1]
        return np.where(valid[..., None], image, 0).astype(np.uint8)

    def record(self, step, curr_obs, contexts, crop_bboxes, raw_actions, timing, metric_depth=None):
        gripper = float(curr_obs["gripper_position"][0])
        height = 360
        tiles = []
        for view, context, bbox in zip(self._views, contexts, crop_bboxes, strict=True):
            cv2.imwrite(str(self._dir / "crops" / f"step{step:04d}_{view}.png"), context[..., ::-1])
            image = curr_obs[f"{view}_image"].copy()
            if bbox is not None:
                x1, y1, x2, y2 = (int(v) for v in bbox)
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 3)
            image = cv2.resize(image, (int(image.shape[1] * height / image.shape[0]), height))
            row = [image, cv2.resize(context, (height, height))]
            prefix = f"{view}_" if len(self._views) == 2 else ""
            for kind in ("full", "crop"):
                depth = (metric_depth or {}).get(f"{prefix}{kind}")
                depth = np.zeros((height, height, 3), np.uint8) if depth is None else self._colorize_depth(depth)
                row.append(cv2.resize(depth, (height, height), interpolation=cv2.INTER_NEAREST))
            tiles.append(np.concatenate(row, axis=1))
        frame = np.ascontiguousarray(np.concatenate(tiles, axis=0)[..., ::-1])
        lines = [
            f"step {step}  infer {timing['infer_ms']:.0f} ms  obs {timing['obs_ms']:.0f} ms",
            f"gripper obs {gripper:.2f}  cmd {np.round(raw_actions[:8, -1], 2).tolist()}",
        ]
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (10, 25 + 25 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        if self._writer is None:
            self._writer = cv2.VideoWriter(
                str(self._dir / "rollout.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), self._fps, frame.shape[1::-1]
            )
        self._writer.write(frame)
        self._records.append(
            {
                "step": step,
                "joint_position": curr_obs["joint_position"].tolist(),
                "gripper_position": gripper,
                "crop_bboxes": [None if b is None else [int(v) for v in b] for b in crop_bboxes],
                "raw_actions": raw_actions.tolist(),
                **timing,
            }
        )

    def close(self):
        if self._writer is not None:
            self._writer.release()
        with open(self._dir / "queries.jsonl", "w") as f:
            f.writelines(json.dumps(record) + "\n" for record in self._records)
        print(f"Saved rollout debug video and log: {self._dir}")


def _validate_policy_metadata(metadata, args):
    metadata = foveation.canonicalize_fields(metadata)
    if not _uses_fast_foundation_depth(metadata):
        raise ValueError("The server must serve a FoveaVLA foveal RGB/depth model")
    if metadata.get("action_space") != "joint_position" or metadata.get("requires_wrist_camera") is not False:
        raise ValueError("Expected joint-position actions without a wrist camera")
    if metadata.get("foveal_context_format") != "direct_v2.2.0" or metadata.get("foveal_context_scale") != 1.5:
        raise ValueError("Expected direct-v2.2 foveal crops with context scale 1.5")
    horizon = metadata.get("future_bbox_horizon")
    if horizon != 8 or args.open_loop_horizon != horizon:
        raise ValueError("Execute exactly 8 actions between foveation queries")
    views = _resolve_policy_exterior_views(metadata, "left")
    if views not in (("left",), ("left", "right")):
        raise ValueError(f"Unsupported FoveaVLA camera order: {views}")
    if len(views) == 2 and (not args.right_camera_id or args.left_camera_id == args.right_camera_id):
        raise ValueError("FoveaVLA-Dual requires two distinct camera serial numbers")
    return views


def main(args: Args):
    if args.max_timesteps <= 0 or not args.left_camera_id or not args.instruction.strip():
        raise ValueError("Provide a camera serial, instruction, and positive max-timesteps")
    client = websocket_client_policy.WebsocketClientPolicy(args.remote_host, args.remote_port)
    metadata = client.get_server_metadata()
    views = _validate_policy_metadata(metadata, args)
    from droid.camera_utils.camera_readers import zed_camera
    from droid.camera_utils.wrappers import multi_camera_wrapper
    from droid.robot_env import RobotEnv

    sdk = zed_camera.sl.Camera.get_sdk_version()
    if sdk != "5.1.2":
        raise RuntimeError(f"Expected ZED SDK 5.1.2, got {sdk}")
    camera_ids = (args.left_camera_id,) if len(views) == 1 else (args.left_camera_id, args.right_camera_id)
    _select_droid_zed_cameras(multi_camera_wrapper, camera_ids)
    zed_camera.standard_params.update(camera_disable_self_calib=True, depth_mode=zed_camera.sl.DEPTH_MODE.NONE)
    print(f"{metadata.get('model_name', 'FoveaVLA')}: camera order {views}: {camera_ids}")
    env = RobotEnv(action_space="joint_position", gripper_action_space="position", do_reset=True)
    watchdog = GripperWatchdog(args.gripper_timeout_s)
    watchdog.check_open_after_reset(float(env.get_observation()["robot_state"]["gripper_position"]))
    recorder = RolloutRecorder(args.debug_dir, args, views, metadata["metric_depth_range_m"]) if args.debug else None
    boxes = tuple(None for _ in views)
    last_valid = tuple(None for _ in views)
    pending = None
    actions = None
    action_index = 0
    try:
        for step in range(args.max_timesteps):
            started = time.monotonic()
            # Read the robot and cameras every step, as in OpenPI's DROID client.
            curr_obs = _extract_observation(args, env.get_observation(), views)
            watchdog.update(curr_obs["gripper_position"][0])
            if actions is None or action_index >= args.open_loop_horizon:
                if pending is not None:
                    if not pending.done():
                        print("Future box is not ready at the 8-step boundary; waiting for it.")
                    try:
                        prediction = pending.result()
                    except Exception as exc:
                        boxes = tuple(None for _ in views)
                        print(f"Future box inference failed ({exc}); using the last valid box.")
                    else:
                        try:
                            boxes = _parse_future_boxes(prediction, len(views))
                        except ValueError as exc:
                            boxes = tuple(None for _ in views)
                            print(f"Policy predicted an invalid future box output ({exc}); using the last valid box.")
                        else:
                            last_valid = tuple(
                                box if box is not None else old for box, old in zip(boxes, last_valid, strict=True)
                            )
                            if any(box is None for box in boxes):
                                print("Policy predicted one or more invalid future boxes; using the last valid box.")
                contexts, crop_bboxes = [], []
                for view, box, previous in zip(views, boxes, last_valid, strict=True):
                    image = curr_obs[f"{view}_image"]
                    region = box if box is not None else previous
                    context, bbox = None, None
                    if region is not None:
                        try:
                            context, bbox = image_tools.extract_foveal_crop_direct_with_bbox(
                                image, region, context_scale=1.5
                            )
                        except ValueError as exc:
                            print(f"Invalid {view} foveal crop ({exc}); using the full image.")
                    if context is None:
                        # Bootstrap the first query before the policy has predicted a t+8 box.
                        context = image_tools.resize_with_pad(image, 224, 224)
                    contexts.append(context)
                    crop_bboxes.append(bbox)
                stereo = _capture_fast_foundation_inputs(
                    env,
                    zed_camera,
                    args,
                    curr_obs,
                    views,
                    crop_bboxes,
                    tuple(metadata["fast_foundation_input_resolution"]),
                )
                request = _build_policy_request(
                    curr_obs, views, contexts, args.instruction, stereo
                )
                if "foveal_context_format" not in metadata:
                    # Send each image once when connecting to a released server.
                    request = {foveation.FIELD_ALIASES.get(key, key): value for key, value in request.items()}
                infer_started = time.monotonic()
                with prevent_keyboard_interrupt():
                    prediction, pending = client.infer_staged(request)
                raw_actions = np.asarray(prediction["actions"])
                if raw_actions.ndim != 2 or raw_actions.shape[1] != 8 or len(raw_actions) < args.open_loop_horizon:
                    raise RuntimeError(
                        f"Expected at least 8 joint-position actions with shape [N,8], got {raw_actions.shape}"
                    )
                if recorder is not None:
                    timing = {
                        "obs_ms": (infer_started - started) * 1000,
                        "infer_ms": (time.monotonic() - infer_started) * 1000,
                    }
                    recorder.record(
                        step, curr_obs, contexts, crop_bboxes, raw_actions, timing, prediction.get("metric_depth")
                    )
                actions = np.stack([_prepare_action(action, ActionSpace.JOINT_POSITION) for action in raw_actions])
                action_index = 0
            env.step(actions[action_index])
            watchdog.record_command(actions[action_index][-1])
            action_index += 1
            time.sleep(max(0.0, 1 / DROID_CONTROL_FREQUENCY - (time.monotonic() - started)))
    except KeyboardInterrupt:
        print("Rollout stopped.")
    finally:
        if recorder is not None:
            recorder.close()
        client.close()
        close = getattr(env, "close", None)
        if callable(close):
            close()


if __name__ == "__main__":
    main(tyro.cli(Args))
