"""Check gripper monitoring without connecting to a robot or camera."""

import unittest
from unittest import mock

import numpy as np

from examples.droid import main as droid_client


class GripperWatchdogTest(unittest.TestCase):
    def test_healthy_motion_with_brief_closes_at_chunk_boundaries(self):
        watchdog = droid_client.GripperWatchdog(2.0)
        now, position, target = 10.0, 0.0, 0.0
        # DROID commands 50 mm/s; simulate an 85 mm gripper's normal travel.
        speed = 0.05 / 0.085

        def advance(duration):
            nonlocal now, position
            position += float(np.clip(target - position, -speed * duration, speed * duration))
            now += duration

        with mock.patch.object(droid_client.time, "monotonic", side_effect=lambda: now):
            for chunk in range(10):
                if chunk:
                    self.assertLess(position, 0.05)
                watchdog.update(position)
                advance(0.2)  # Inference latency; the previous target remains active.
                for target in [0.0] * 7 + [1.0]:
                    watchdog.record_command(target)
                    advance(1 / 15)

    def test_repeated_identical_commands_do_not_restart_failure_timeout(self):
        for command, position in ((1.0, 0.0), (0.0, 1.0)):
            with self.subTest(command=command):
                watchdog = droid_client.GripperWatchdog(2.0)
                with mock.patch.object(droid_client.time, "monotonic") as clock:
                    clock.return_value = 0.0
                    watchdog.record_command(command)
                    watchdog.update(position)
                    for step in range(32):
                        watchdog.record_command(command)
                    clock.return_value = 2.1
                    with self.assertRaisesRegex(RuntimeError, "did not follow command"):
                        watchdog.update(position)

    def test_partial_grasp_clears_failure_timeout(self):
        watchdog = droid_client.GripperWatchdog(2.0)
        watchdog.record_command(1.0)
        with mock.patch.object(droid_client.time, "monotonic") as clock:
            clock.return_value = 10.0
            watchdog.update(0.0)
            clock.return_value = 11.0
            watchdog.update(0.4)
            clock.return_value = 13.0
            watchdog.update(0.0)
            clock.return_value = 14.0
            watchdog.update(0.0)

    def test_reset_requires_open_finite_feedback(self):
        watchdog = droid_client.GripperWatchdog(2.0)
        watchdog.check_open_after_reset(0.0)
        for position in (1.0, float("nan"), float("inf")):
            with self.subTest(position=position):
                with self.assertRaisesRegex(RuntimeError, "after DROID opened it"):
                    watchdog.check_open_after_reset(position)


if __name__ == "__main__":
    unittest.main()
