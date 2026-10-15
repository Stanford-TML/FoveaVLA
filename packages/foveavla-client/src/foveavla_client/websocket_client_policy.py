from __future__ import annotations

import concurrent.futures
import itertools
import logging
import threading
import time
from typing import Dict, Optional, Tuple

from typing_extensions import override
import websockets.sync.client

from foveavla_client import base_policy as _base_policy
from foveavla_client import msgpack_numpy
from foveavla_client import websocket_protocol


class WebsocketClientPolicy(_base_policy.BasePolicy):
    """Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer for a corresponding server implementation.
    """

    def __init__(self, host: str = "0.0.0.0", port: Optional[int] = None, api_key: Optional[str] = None) -> None:
        if host.startswith("ws"):
            self._uri = host
        else:
            self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = msgpack_numpy.Packer()
        self._api_key = api_key
        self._ws, self._server_metadata = self._wait_for_server()
        self._send_lock = threading.Lock()
        self._staged_lock = threading.Lock()
        self._staged_requests: Dict[
            str, Tuple[concurrent.futures.Future, concurrent.futures.Future]
        ] = {}
        self._request_ids = itertools.count()
        self._staged_reader: threading.Thread | None = None
        self._staged_reader_error: BaseException | None = None
        self._closed = False

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logging.info(f"Waiting for server at {self._uri}...")
        while True:
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                conn = websockets.sync.client.connect(
                    self._uri, compression=None, max_size=None, additional_headers=headers
                )
                metadata = msgpack_numpy.unpackb(conn.recv())
                return conn, metadata
            except ConnectionRefusedError:
                logging.info("Still waiting for server...")
                time.sleep(5)

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        if self._staged_reader is not None:
            raise RuntimeError("Cannot mix infer() with infer_staged() on one websocket connection")
        data = self._packer.pack(obs)
        with self._send_lock:
            self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            # we're expecting bytes; if the server sends a string, it's an error.
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)

    def infer_staged(self, obs: Dict) -> Tuple[Dict, concurrent.futures.Future]:  # noqa: UP006
        """Return actions first and resolve the future-bbox result asynchronously."""
        self._start_staged_reader()
        request_id = str(next(self._request_ids))
        actions_future = concurrent.futures.Future()
        bbox_future = concurrent.futures.Future()
        with self._staged_lock:
            if self._staged_reader_error is not None:
                raise RuntimeError("Staged inference response reader has stopped") from self._staged_reader_error
            self._staged_requests[request_id] = (actions_future, bbox_future)

        request = {
            websocket_protocol.REQUEST_KIND_KEY: websocket_protocol.STAGED_INFERENCE_REQUEST,
            websocket_protocol.REQUEST_ID_KEY: request_id,
            websocket_protocol.OBSERVATION_KEY: obs,
        }
        try:
            with self._send_lock:
                self._ws.send(self._packer.pack(request))
        except Exception:
            with self._staged_lock:
                self._staged_requests.pop(request_id, None)
            raise
        return actions_future.result(), bbox_future

    def _start_staged_reader(self) -> None:
        with self._staged_lock:
            if self._staged_reader_error is not None:
                raise RuntimeError("Staged inference response reader has stopped") from self._staged_reader_error
            if self._staged_reader is not None:
                return
            self._staged_reader = threading.Thread(
                target=self._receive_staged_responses,
                name="foveavla-staged-inference",
                daemon=True,
            )
            self._staged_reader.start()

    def _receive_staged_responses(self) -> None:
        try:
            while True:
                response = self._ws.recv()
                if isinstance(response, str):
                    raise RuntimeError(f"Error in inference server:\n{response}")
                message = msgpack_numpy.unpackb(response)
                if (
                    not isinstance(message, dict)
                    or message.get(websocket_protocol.RESPONSE_KIND_KEY)
                    != websocket_protocol.STAGED_INFERENCE_RESPONSE
                ):
                    raise RuntimeError(f"Unexpected staged inference response: {message!r}")

                request_id = str(message[websocket_protocol.REQUEST_ID_KEY])
                stage = message[websocket_protocol.STAGE_KEY]
                with self._staged_lock:
                    pending = self._staged_requests.get(request_id)
                    if stage in (websocket_protocol.FUTURE_BBOX_STAGE, websocket_protocol.ERROR_STAGE):
                        self._staged_requests.pop(request_id, None)
                if pending is None:
                    raise RuntimeError(f"Received response for unknown staged request {request_id!r}")

                actions_future, bbox_future = pending
                if stage == websocket_protocol.ACTIONS_STAGE:
                    actions_future.set_result(message[websocket_protocol.RESULT_KEY])
                elif stage == websocket_protocol.FUTURE_BBOX_STAGE:
                    bbox_future.set_result(message[websocket_protocol.RESULT_KEY])
                elif stage == websocket_protocol.ERROR_STAGE:
                    error = RuntimeError(message[websocket_protocol.ERROR_KEY])
                    if not actions_future.done():
                        actions_future.set_exception(error)
                    if not bbox_future.done():
                        bbox_future.set_exception(error)
                else:
                    raise RuntimeError(f"Unknown staged inference response stage: {stage!r}")
        except BaseException as exc:
            with self._staged_lock:
                self._staged_reader_error = exc
                pending_requests = tuple(self._staged_requests.values())
                self._staged_requests.clear()
            for actions_future, bbox_future in pending_requests:
                if not actions_future.done():
                    actions_future.set_exception(exc)
                if not bbox_future.done():
                    bbox_future.set_exception(exc)

    def close(self) -> None:
        """Close the websocket and wait for the staged response reader to stop."""
        with self._send_lock:
            if self._closed:
                return
            self._closed = True
            self._ws.close()

        reader = self._staged_reader
        if reader is not None and reader is not threading.current_thread():
            reader.join()

    @override
    def reset(self) -> None:
        pass
