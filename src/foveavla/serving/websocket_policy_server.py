import asyncio
import http
import logging
import time
import traceback

from foveavla_client import base_policy as _base_policy
from foveavla_client import msgpack_numpy
from foveavla_client import websocket_protocol
import websockets.asyncio.server as _server
import websockets.frames

logger = logging.getLogger(__name__)


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()
        send_lock = asyncio.Lock()
        background_tasks: set[asyncio.Task] = set()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        try:
            while True:
                try:
                    start_time = time.monotonic()
                    request = msgpack_numpy.unpackb(await websocket.recv())
                    if (
                        isinstance(request, dict)
                        and request.get(websocket_protocol.REQUEST_KIND_KEY)
                        == websocket_protocol.STAGED_INFERENCE_REQUEST
                    ):
                        request_id = str(request[websocket_protocol.REQUEST_ID_KEY])
                        infer_actions = getattr(self._policy, "infer_actions_with_pending_bbox", None)
                        infer_bbox = getattr(self._policy, "infer_future_bbox", None)
                        if infer_actions is None or infer_bbox is None:
                            raise NotImplementedError("Policy does not support staged future-bbox inference")

                        infer_time = time.monotonic()
                        action, pending_bbox = infer_actions(request[websocket_protocol.OBSERVATION_KEY])
                        action["server_timing"] = {"infer_ms": (time.monotonic() - infer_time) * 1000}
                        response = {
                            websocket_protocol.RESPONSE_KIND_KEY: websocket_protocol.STAGED_INFERENCE_RESPONSE,
                            websocket_protocol.REQUEST_ID_KEY: request_id,
                            websocket_protocol.STAGE_KEY: websocket_protocol.ACTIONS_STAGE,
                            websocket_protocol.RESULT_KEY: action,
                        }
                        async with send_lock:
                            await websocket.send(packer.pack(response))

                        task = asyncio.create_task(
                            self._send_future_bbox(
                                websocket,
                                packer,
                                send_lock,
                                request_id,
                                infer_bbox,
                                pending_bbox,
                            )
                        )
                        background_tasks.add(task)

                        def finish_background_task(done_task: asyncio.Task) -> None:
                            background_tasks.discard(done_task)
                            if not done_task.cancelled():
                                done_task.exception()

                        task.add_done_callback(finish_background_task)
                        prev_total_time = time.monotonic() - start_time
                        continue

                    infer_time = time.monotonic()
                    action = self._policy.infer(request)
                    infer_time = time.monotonic() - infer_time

                    action["server_timing"] = {
                        "infer_ms": infer_time * 1000,
                    }
                    if prev_total_time is not None:
                        # We can only record the last total time since we also want to include the send time.
                        action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                    async with send_lock:
                        await websocket.send(packer.pack(action))
                    prev_total_time = time.monotonic() - start_time

                except websockets.ConnectionClosed:
                    logger.info(f"Connection from {websocket.remote_address} closed")
                    break
                except Exception:
                    async with send_lock:
                        await websocket.send(traceback.format_exc())
                    await websocket.close(
                        code=websockets.frames.CloseCode.INTERNAL_ERROR,
                        reason="Internal server error. Traceback included in previous frame.",
                    )
                    raise
        finally:
            for task in background_tasks:
                task.cancel()
            if background_tasks:
                await asyncio.gather(*background_tasks, return_exceptions=True)

    async def _send_future_bbox(
        self,
        websocket: _server.ServerConnection,
        packer,
        send_lock: asyncio.Lock,
        request_id: str,
        infer_bbox,
        pending_bbox,
    ) -> None:
        try:
            infer_time = time.monotonic()
            bbox = await asyncio.to_thread(infer_bbox, pending_bbox)
            bbox["server_timing"] = {"infer_ms": (time.monotonic() - infer_time) * 1000}
            response = {
                websocket_protocol.RESPONSE_KIND_KEY: websocket_protocol.STAGED_INFERENCE_RESPONSE,
                websocket_protocol.REQUEST_ID_KEY: request_id,
                websocket_protocol.STAGE_KEY: websocket_protocol.FUTURE_BBOX_STAGE,
                websocket_protocol.RESULT_KEY: bbox,
            }
        except Exception:
            response = {
                websocket_protocol.RESPONSE_KIND_KEY: websocket_protocol.STAGED_INFERENCE_RESPONSE,
                websocket_protocol.REQUEST_ID_KEY: request_id,
                websocket_protocol.STAGE_KEY: websocket_protocol.ERROR_STAGE,
                websocket_protocol.ERROR_KEY: traceback.format_exc(),
            }
        async with send_lock:
            await websocket.send(packer.pack(response))


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None
