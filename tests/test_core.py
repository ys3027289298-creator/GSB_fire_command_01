"""Regression tests for the fire command rescue game.

Covers the eight historical rescue-flow defects:
  1. smoke spread used the previous wind direction
  2. evacuated rooms were still counted as trapped
  3. water allocation could exceed pump capacity
  4. injured crew still counted as available
  5. failed breach attempts consumed tools
  6. alarm escalation timer advanced more than once per tick
  7. undoing a deployment did not restore the route list
  8. loading a save re-executed the previous turn's events
plus the zero-water-pressure and all-crew-injured boundaries.
"""

import json
import os
import tempfile
import unittest

from fire.core import CommandError, Game


class TempSaveMixin:
    def setUp(self):
        fd, self.save_path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        self.addCleanup(self._remove_save)

    def _remove_save(self):
        if os.path.exists(self.save_path):
            os.remove(self.save_path)


class SmokeSpreadTests(unittest.TestCase):
    """Bug 1: smoke must spread with the wind in effect this tick."""

    def test_spread_uses_current_wind_not_previous(self):
        game = Game(seed=46)
        game.ignite(1, 1)
        game.set_wind("E")
        game.tick()
        rooms = game.status()["rooms"]
        self.assertGreater(rooms["2,1"]["smoke"], 0,
                           "smoke must reach the downwind (east) room")
        self.assertEqual(rooms["1,0"]["smoke"], 0,
                         "stale north wind must not spread smoke")
        self.assertEqual(rooms["0,1"]["smoke"], 0,
                         "smoke must not spread upwind")

    def test_wind_change_between_ticks_is_honored(self):
        game = Game(seed=46)
        game.ignite(1, 1)
        game.tick()  # default wind N -> smoke at (1, 0)
        game.set_wind("S")
        game.tick()  # new wind S -> smoke at (1, 2)
        rooms = game.status()["rooms"]
        self.assertGreater(rooms["1,0"]["smoke"], 0)
        self.assertGreater(rooms["1,2"]["smoke"], 0)


class EvacuationTests(unittest.TestCase):
    """Bug 2: evacuated rooms must not count as trapped."""

    def test_evacuated_room_not_counted_as_trapped(self):
        game = Game(seed=46)
        game.set_trapped(2, 2, 5)
        self.assertEqual(game.trapped_total, 5)
        game.evacuate(2, 2)
        self.assertEqual(game.trapped_total, 0)
        self.assertEqual(game.status()["trapped_total"], 0)

    def test_evacuate_is_idempotent(self):
        game = Game(seed=46)
        game.set_trapped(1, 1, 3)
        self.assertEqual(game.evacuate(1, 1), 3)
        self.assertEqual(game.evacuate(1, 1), 0)
        self.assertEqual(game.evacuate(1, 1), 0)
        self.assertEqual(game.rescued, 3,
                         "repeated evacuations must not rescue twice")


class WaterAllocationTests(unittest.TestCase):
    """Bug 3: total water allocation must never exceed pump capacity."""

    def test_allocation_above_capacity_rejected(self):
        game = Game(seed=46, pump_capacity=10)
        with self.assertRaises(CommandError):
            game.deploy(0, 1, 1, water=11)
        self.assertEqual(game.water_used, 0)
        self.assertEqual(game.routes, [])
        self.assertIn(0, game.available_crew,
                      "rejected deploy must not consume the crew member")

    def test_cumulative_allocation_cannot_exceed_capacity(self):
        game = Game(seed=46, pump_capacity=10)
        game.deploy(0, 1, 1, water=6)
        game.deploy(1, 2, 2, water=4)
        self.assertEqual(game.water_available, 0)
        with self.assertRaises(CommandError):
            game.deploy(2, 3, 3, water=1)
        self.assertEqual(game.water_used, 10)

    def test_negative_allocation_rejected(self):
        game = Game(seed=46)
        with self.assertRaises(CommandError):
            game.deploy(0, 1, 1, water=-1)

    def test_zero_pump_capacity_boundary(self):
        game = Game(seed=46, pump_capacity=0)
        self.assertEqual(game.water_available, 0)
        game.deploy(0, 1, 1, water=0)  # zero allocation still allowed
        self.assertEqual(game.water_used, 0)
        with self.assertRaises(CommandError):
            game.deploy(1, 2, 2, water=1)
        self.assertEqual(game.water_available, 0)


class CrewAvailabilityTests(unittest.TestCase):
    """Bug 4: injured crew must not count as available."""

    def test_injured_member_leaves_available_pool(self):
        game = Game(seed=46, crew_size=3)
        game.schedule_event(1, "injury")
        game.tick()
        injured = [m["id"] for m in game.crew if m["injured"]]
        self.assertEqual(len(injured), 1)
        self.assertEqual(len(game.available_crew), 2)
        self.assertNotIn(injured[0], game.available_crew)
        with self.assertRaises(CommandError):
            game.deploy(injured[0], 1, 1)

    def test_all_crew_injured_boundary(self):
        game = Game(seed=46, crew_size=2)
        for member in game.crew:
            member["injured"] = True
        self.assertEqual(game.available_crew, [])
        self.assertEqual(game.status()["available_crew"], [])
        with self.assertRaises(CommandError):
            game.deploy(0, 1, 1)
        with self.assertRaises(CommandError):
            game.deploy(1, 1, 1)
        # Injury events with nobody healthy must be a harmless no-op.
        game.schedule_event(game.time + 1, "injury")
        game.tick()
        self.assertEqual(game.available_crew, [])


class BreachToolTests(unittest.TestCase):
    """Bug 5: failed breach attempts must not consume tools."""

    def test_failed_attempt_keeps_tool(self):
        game = Game(seed=46, breach_tools=3)
        attempts = [(x, 0) for x in range(4)] + [(x, 1) for x in range(2)]
        results = [game.breach(x, y) for x, y in attempts]
        successes = sum(1 for ok in results if ok)
        failures = sum(1 for ok in results if not ok)
        self.assertGreater(successes, 0, "seed 46 must yield a success")
        self.assertGreater(failures, 0, "seed 46 must yield a failure")
        self.assertEqual(game.breach_tools, 3 - successes,
                         "only successful breaches may consume tools")
        breached = sum(1 for room in game.rooms.values() if room["breached"])
        self.assertEqual(breached, successes)

    def test_tools_exhausted_raises(self):
        game = Game(seed=46, breach_tools=0)
        with self.assertRaises(CommandError):
            game.breach(0, 0)

    def test_breach_is_idempotent_per_room(self):
        game = Game(seed=46, breach_tools=3)
        while not game.breach(0, 0):
            pass
        tools_after_success = game.breach_tools
        with self.assertRaises(CommandError):
            game.breach(0, 0)
        self.assertEqual(game.breach_tools, tools_after_success)


class AlarmEscalationTests(unittest.TestCase):
    """Bug 6: the alarm timer advances exactly once per tick."""

    def test_alarm_escalates_once_per_threshold(self):
        game = Game(seed=46, alarm_threshold=3)
        game.ignite(0, 0)
        game.ignite(1, 1)  # multiple fires must not multiply the timer
        game.tick(2)
        self.assertEqual(game.alarm_level, 0)
        self.assertEqual(game.status()["alarm_timer"], 2)
        game.tick()
        self.assertEqual(game.alarm_level, 1)
        self.assertEqual(game.status()["alarm_timer"], 0)
        game.tick(3)
        self.assertEqual(game.alarm_level, 2)

    def test_alarm_timer_resets_when_fire_out(self):
        game = Game(seed=46, alarm_threshold=3)
        game.ignite(0, 0)
        game.tick(2)
        game.rooms[(0, 0)]["fire"] = False
        game.tick()
        self.assertEqual(game.alarm_level, 0)
        self.assertEqual(game.status()["alarm_timer"], 0)


class UndoDeploymentTests(unittest.TestCase):
    """Bug 7: undo must restore routes, water and crew together."""

    def test_undo_restores_routes_water_and_crew(self):
        game = Game(seed=46, pump_capacity=10)
        game.deploy(0, 2, 3, water=4)
        self.assertEqual(len(game.routes), 1)
        self.assertEqual(game.water_used, 4)
        game.undo()
        self.assertEqual(game.routes, [], "undo must restore the route list")
        self.assertEqual(game.water_used, 0)
        self.assertEqual(game.water_available, 10)
        self.assertIn(0, game.available_crew)
        self.assertFalse(game.crew[0]["deployed"])

    def test_undo_restores_intermediate_state(self):
        game = Game(seed=46, pump_capacity=10)
        game.deploy(0, 1, 0, water=2)
        game.deploy(1, 2, 0, water=3)
        game.undo()
        self.assertEqual(len(game.routes), 1)
        self.assertEqual(game.routes[0]["crew"], 0)
        self.assertEqual(game.water_used, 2)
        self.assertIn(1, game.available_crew)

    def test_undo_without_deploy_raises(self):
        game = Game(seed=46)
        with self.assertRaises(CommandError):
            game.undo()


class SaveLoadTests(TempSaveMixin, unittest.TestCase):
    """Bug 8: loading a save must not replay the previous turn."""

    def test_load_does_not_reexecute_applied_events(self):
        game = Game(seed=46, crew_size=4)
        game.schedule_event(1, "injury")
        game.tick()  # turn 1: injury applied
        game.save(self.save_path)
        loaded = Game.load(self.save_path)
        loaded.tick(5)
        injured = [m for m in loaded.crew if m["injured"]]
        self.assertEqual(len(injured), 1,
                         "the turn-1 injury must not be applied again")
        self.assertEqual(len(loaded.events), 1)

    def test_future_events_survive_load_exactly_once(self):
        game = Game(seed=46, crew_size=4)
        game.schedule_event(1, "injury")
        game.schedule_event(3, "injury")
        game.tick()  # turn 1 applied; turn 3 still pending
        game.save(self.save_path)
        loaded = Game.load(self.save_path)
        loaded.tick(4)
        injured = [m for m in loaded.crew if m["injured"]]
        self.assertEqual(len(injured), 2)
        self.assertEqual(len(loaded.events), 2)

    def test_save_load_roundtrip_is_idempotent(self):
        game = Game(seed=46, pump_capacity=8)
        game.ignite(1, 1)
        game.set_wind("E")
        game.set_trapped(2, 2, 4)
        game.evacuate(2, 2)
        game.deploy(0, 1, 1, water=3)
        game.schedule_event(2, "injury")
        game.tick()
        before = game.status()
        game.save(self.save_path)
        loaded = Game.load(self.save_path)
        self.assertEqual(loaded.status(), before)
        # Saving again from the loaded game yields identical JSON.
        second_path = self.save_path + ".2"
        self.addCleanup(lambda: os.path.exists(second_path)
                        and os.remove(second_path))
        loaded.save(second_path)
        with open(self.save_path, encoding="utf-8") as handle:
            first_dump = json.load(handle)
        with open(second_path, encoding="utf-8") as handle:
            second_dump = json.load(handle)
        self.assertEqual(first_dump, second_dump)

    def test_loaded_game_continues_deterministically(self):
        game = Game(seed=46)
        game.ignite(0, 0)
        game.tick(2)
        game.save(self.save_path)
        loaded = Game.load(self.save_path)
        game.tick(3)
        loaded.tick(3)
        self.assertEqual(loaded.status(), game.status())


class TerminalInterfaceTests(unittest.TestCase):
    """The terminal command interface must stay intact."""

    def test_execute_dispatches_commands(self):
        game = Game(seed=46)
        game.execute("wind E")
        game.execute("fire 1 1")
        game.execute("trapped 2 2 3")
        status = game.execute("evacuate 2 2")
        self.assertEqual(status["rescued"], 3)
        status = game.execute("deploy 0 1 1 2")
        self.assertEqual(status["water_used"], 2)
        status = game.execute("undo")
        self.assertEqual(status["routes"], [])
        status = game.execute("tick 2")
        self.assertEqual(status["time"], 2)
        game.execute("event 4 injury")
        self.assertEqual(len(game.status()["pending_events"]), 1)

    def test_unknown_command_raises(self):
        game = Game(seed=46)
        with self.assertRaises(CommandError):
            game.execute("teleport 1 1")
        with self.assertRaises(CommandError):
            game.execute("")

    def test_fixed_seed_reproducibility(self):
        first = Game(seed=46)
        second = Game(seed=46)
        for game in (first, second):
            game.ignite(1, 1)
            game.schedule_event(2, "injury")
            game.breach(0, 0)
            game.tick(3)
        self.assertEqual(first.status(), second.status())


if __name__ == "__main__":
    unittest.main()
