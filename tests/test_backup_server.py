"""API tests for the Raspberry Pi fallback backend."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import backup_server_pi


class BackupServerApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.original_state_path = backup_server_pi.STATE_PATH
        backup_server_pi.STATE_PATH = Path(self.temp_dir.name) / "state.json"
        self.addCleanup(
            setattr,
            backup_server_pi,
            "STATE_PATH",
            self.original_state_path,
        )
        backup_server_pi.app.config.update(TESTING=True)
        self.client = backup_server_pi.app.test_client()

    def test_health_endpoint(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(200, response.status_code)
        self.assertEqual({"status": "ok"}, response.get_json())

    def test_devices_round_trip(self) -> None:
        devices = {
            "living-room-light": {"power": "on", "brightness": 75},
            "fan": {"power": "off"},
        }

        update = self.client.put("/devices", json=devices)
        self.assertEqual(200, update.status_code)
        self.assertEqual(2, update.get_json()["device_count"])

        read_back = self.client.get("/devices")
        self.assertEqual(200, read_back.status_code)
        self.assertEqual(devices, read_back.get_json())

    def test_doors_and_scenes_are_isolated(self) -> None:
        doors = {"front-door": {"locked": True}}
        scenes = {"night": {"enabled": True}}

        self.assertEqual(200, self.client.post("/doors", json=doors).status_code)
        self.assertEqual(200, self.client.post("/scenes", json=scenes).status_code)
        self.assertEqual(doors, self.client.get("/doors").get_json())
        self.assertEqual(scenes, self.client.get("/scenes").get_json())
        self.assertEqual({}, self.client.get("/devices").get_json())

    def test_corrupt_state_fails_safe(self) -> None:
        backup_server_pi.STATE_PATH.write_text("not-json", encoding="utf-8")

        self.assertEqual({}, self.client.get("/devices").get_json())
        self.assertEqual({}, self.client.get("/doors").get_json())
        self.assertEqual({}, self.client.get("/scenes").get_json())

    def test_missing_sections_are_filled(self) -> None:
        backup_server_pi.STATE_PATH.write_text(
            json.dumps({"devices": {"pump": {"power": "on"}}}),
            encoding="utf-8",
        )

        state = backup_server_pi.load_state()
        self.assertEqual({"pump": {"power": "on"}}, state["devices"])
        self.assertEqual({}, state["doors"])
        self.assertEqual({}, state["scenes"])


if __name__ == "__main__":
    unittest.main()
