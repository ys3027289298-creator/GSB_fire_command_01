"""Core logic for the fire command rescue game.

Discrete-time simulation driven by a terminal command interface, with
JSON saves and a fixed random seed.  Every state transition is
observable through :meth:`Game.status` and idempotent across save/load
round trips: loading a save never replays events that were already
applied, and repeated commands such as ``evacuate`` never compound
their effects.
"""

from __future__ import annotations

import copy
import json
import random

DIRECTIONS = {
    "N": (0, -1),
    "E": (1, 0),
    "S": (0, 1),
    "W": (-1, 0),
}

BREACH_SUCCESS_RATE = 0.5


class CommandError(Exception):
    """Raised when a terminal command cannot be carried out."""


class Game:
    """A single fire-rescue scenario on a rectangular room grid."""

    def __init__(self, width=4, height=4, seed=46, pump_capacity=10,
                 crew_size=4, breach_tools=3, alarm_threshold=3):
        self.width = width
        self.height = height
        self.seed = seed
        self.rng = random.Random(seed)
        self.time = 0
        self.wind = "N"
        self.pump_capacity = pump_capacity
        self.water_used = 0
        self.breach_tools = breach_tools
        self.alarm_level = 0
        self.alarm_threshold = alarm_threshold
        self._alarm_timer = 0
        self.rescued = 0
        self.rooms = {
            (x, y): {
                "smoke": 0,
                "fire": False,
                "trapped": 0,
                "evacuated": False,
                "breached": False,
            }
            for x in range(width)
            for y in range(height)
        }
        self.crew = [
            {"id": cid, "injured": False, "deployed": False}
            for cid in range(crew_size)
        ]
        self.routes = []
        self.events = []
        self._pending_events = []
        self._history = []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _room(self, x, y):
        try:
            return self.rooms[(x, y)]
        except KeyError:
            raise CommandError(f"room ({x}, {y}) is outside the grid")

    def _crew_member(self, crew_id):
        for member in self.crew:
            if member["id"] == crew_id:
                return member
        raise CommandError(f"unknown crew id {crew_id}")

    @property
    def available_crew(self):
        """Ids of members who are neither injured nor deployed.

        Bug 4 fix: injured members no longer occupy an availability slot.
        """
        return [m["id"] for m in self.crew
                if not m["injured"] and not m["deployed"]]

    @property
    def trapped_total(self):
        """People still trapped; evacuated rooms never count (bug 2)."""
        return sum(room["trapped"] for room in self.rooms.values()
                   if not room["evacuated"])

    @property
    def water_available(self):
        return self.pump_capacity - self.water_used

    # ------------------------------------------------------------------
    # Scenario setup commands
    # ------------------------------------------------------------------

    def set_wind(self, direction):
        direction = direction.upper()
        if direction not in DIRECTIONS:
            raise CommandError(f"unknown wind direction {direction!r}")
        self.wind = direction

    def ignite(self, x, y):
        self._room(x, y)["fire"] = True

    def set_trapped(self, x, y, count):
        if count < 0:
            raise CommandError("trapped count cannot be negative")
        room = self._room(x, y)
        room["trapped"] = count
        room["evacuated"] = False

    # ------------------------------------------------------------------
    # Rescue commands
    # ------------------------------------------------------------------

    def evacuate(self, x, y):
        """Evacuate a room; repeated calls rescue nobody twice."""
        room = self._room(x, y)
        rescued = 0 if room["evacuated"] else room["trapped"]
        self.rescued += rescued
        room["trapped"] = 0
        room["evacuated"] = True
        return rescued

    def deploy(self, crew_id, x, y, water=0):
        """Deploy an available crew member along a route to (x, y)."""
        member = self._crew_member(crew_id)
        if member["injured"]:
            raise CommandError(f"crew {crew_id} is injured")
        if member["deployed"]:
            raise CommandError(f"crew {crew_id} is already deployed")
        if water < 0:
            raise CommandError("water allocation cannot be negative")
        # Bug 3 fix: reject before mutating any state so a failed deploy
        # consumes neither pump capacity nor a crew member.
        if self.water_used + water > self.pump_capacity:
            raise CommandError(
                f"water allocation exceeds pump capacity "
                f"({self.water_used}+{water}>{self.pump_capacity})")
        self._room(x, y)
        # Bug 7 fix: keep a snapshot so undo can restore the full route
        # list together with water usage and crew deployment flags.
        self._history.append({
            "crew": copy.deepcopy(self.crew),
            "water_used": self.water_used,
            "routes": copy.deepcopy(self.routes),
        })
        member["deployed"] = True
        self.water_used += water
        self.routes.append({
            "crew": crew_id,
            "target": [x, y],
            "water": water,
            "path": self._route_to(x, y),
        })

    def _route_to(self, x, y):
        path = [[0, 0]]
        path += [[step, 0] for step in range(1, x + 1)]
        path += [[x, step] for step in range(1, y + 1)]
        return path

    def undo(self):
        """Revert the most recent deploy, restoring crew, water, routes."""
        if not self._history:
            raise CommandError("nothing to undo")
        snapshot = self._history.pop()
        self.crew = snapshot["crew"]
        self.water_used = snapshot["water_used"]
        self.routes = snapshot["routes"]

    def breach(self, x, y):
        """Attempt to breach a door.

        Bug 5 fix: a failed attempt rolls no consumption, so the tool is
        returned to the inventory and the room stays unbreached.
        """
        room = self._room(x, y)
        if room["breached"]:
            raise CommandError(f"room ({x}, {y}) is already breached")
        if self.breach_tools <= 0:
            raise CommandError("no breach tools left")
        if self.rng.random() < BREACH_SUCCESS_RATE:
            room["breached"] = True
            self.breach_tools -= 1
            return True
        return False

    # ------------------------------------------------------------------
    # Events and time
    # ------------------------------------------------------------------

    def schedule_event(self, turn, kind, payload=None):
        if turn <= self.time:
            raise CommandError("events must be scheduled in the future")
        event = {"turn": turn, "kind": kind}
        if payload:
            event.update(payload)
        self._pending_events.append(event)

    def tick(self, steps=1):
        for _ in range(steps):
            self.time += 1
            self._spread_smoke()
            self._update_alarm()
            self._apply_due_events()

    def _spread_smoke(self):
        """Spread smoke using the wind in effect *this* tick (bug 1)."""
        dx, dy = DIRECTIONS[self.wind]
        delta = {}
        for (x, y), room in self.rooms.items():
            if room["fire"]:
                delta[(x, y)] = delta.get((x, y), 0) + 1
            if room["fire"] or room["smoke"] > 0:
                downwind = (x + dx, y + dy)
                if downwind in self.rooms:
                    delta[downwind] = delta.get(downwind, 0) + 1
        for pos, amount in delta.items():
            self.rooms[pos]["smoke"] += amount

    def _update_alarm(self):
        """Advance the alarm timer exactly once per tick (bug 6).

        Multiple burning rooms must not multiply the increment.
        """
        if any(room["fire"] for room in self.rooms.values()):
            self._alarm_timer += 1
            if self._alarm_timer >= self.alarm_threshold:
                self.alarm_level += 1
                self._alarm_timer = 0
        else:
            self._alarm_timer = 0

    def _apply_due_events(self):
        due = [e for e in self._pending_events if e["turn"] == self.time]
        self._pending_events = [
            e for e in self._pending_events if e["turn"] != self.time
        ]
        for event in due:
            self._apply_event(event)
            applied = dict(event)
            applied["applied_at"] = self.time
            self.events.append(applied)

    def _apply_event(self, event):
        kind = event["kind"]
        if kind == "injury":
            candidates = [m for m in self.crew if not m["injured"]]
            if candidates:
                self.rng.choice(candidates)["injured"] = True
        elif kind == "fire":
            self.ignite(*event["at"])
        else:
            raise CommandError(f"unknown event kind {kind!r}")

    # ------------------------------------------------------------------
    # Observation, save and load
    # ------------------------------------------------------------------

    def status(self):
        """A JSON-friendly snapshot of everything observable."""
        return {
            "time": self.time,
            "wind": self.wind,
            "alarm_level": self.alarm_level,
            "alarm_timer": self._alarm_timer,
            "pump_capacity": self.pump_capacity,
            "water_used": self.water_used,
            "water_available": self.water_available,
            "breach_tools": self.breach_tools,
            "rescued": self.rescued,
            "trapped_total": self.trapped_total,
            "available_crew": self.available_crew,
            "crew": copy.deepcopy(self.crew),
            "routes": copy.deepcopy(self.routes),
            "events": copy.deepcopy(self.events),
            "pending_events": copy.deepcopy(self._pending_events),
            "rooms": {
                f"{x},{y}": dict(room)
                for (x, y), room in sorted(self.rooms.items())
            },
        }

    def _state_dict(self):
        rng_version, rng_internal, rng_gauss = self.rng.getstate()
        return {
            "version": 1,
            "width": self.width,
            "height": self.height,
            "seed": self.seed,
            "pump_capacity": self.pump_capacity,
            "alarm_threshold": self.alarm_threshold,
            "time": self.time,
            "wind": self.wind,
            "water_used": self.water_used,
            "breach_tools": self.breach_tools,
            "alarm_level": self.alarm_level,
            "alarm_timer": self._alarm_timer,
            "rescued": self.rescued,
            "rooms": {f"{x},{y}": room for (x, y), room in self.rooms.items()},
            "crew": self.crew,
            "routes": self.routes,
            "events": self.events,
            "pending_events": self._pending_events,
            "rng_state": [rng_version, list(rng_internal), rng_gauss],
        }

    def save(self, path):
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self._state_dict(), handle, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as handle:
            state = json.load(handle)
        game = cls(
            width=state["width"],
            height=state["height"],
            seed=state["seed"],
            pump_capacity=state["pump_capacity"],
            crew_size=len(state["crew"]),
            alarm_threshold=state["alarm_threshold"],
        )
        game._restore(state)
        return game

    def _restore(self, state):
        self.time = state["time"]
        self.wind = state["wind"]
        self.water_used = state["water_used"]
        self.breach_tools = state["breach_tools"]
        self.alarm_level = state["alarm_level"]
        self._alarm_timer = state["alarm_timer"]
        self.rescued = state["rescued"]
        self.rooms = {}
        for key, room in state["rooms"].items():
            x, y = (int(part) for part in key.split(","))
            self.rooms[(x, y)] = dict(room)
        self.crew = [dict(member) for member in state["crew"]]
        self.routes = copy.deepcopy(state["routes"])
        self.events = copy.deepcopy(state["events"])
        # Bug 8 fix: events due at or before the saved time were already
        # applied (and recorded in ``events``); drop them on restore so a
        # loaded game never re-executes the previous turn.  Only strictly
        # future events remain pending.
        self._pending_events = [
            dict(event) for event in state["pending_events"]
            if event["turn"] > self.time
        ]
        self._history = []
        rng_version, rng_internal, rng_gauss = state["rng_state"]
        self.rng.setstate((rng_version, tuple(rng_internal), rng_gauss))

    # ------------------------------------------------------------------
    # Terminal command interface
    # ------------------------------------------------------------------

    def execute(self, line):
        """Run one terminal command and return the new status."""
        parts = line.split()
        if not parts:
            raise CommandError("empty command")
        command, args = parts[0].lower(), parts[1:]
        if command == "wind":
            self.set_wind(args[0])
        elif command == "fire":
            self.ignite(int(args[0]), int(args[1]))
        elif command == "trapped":
            self.set_trapped(int(args[0]), int(args[1]), int(args[2]))
        elif command == "evacuate":
            self.evacuate(int(args[0]), int(args[1]))
        elif command == "deploy":
            water = int(args[3]) if len(args) > 3 else 0
            self.deploy(int(args[0]), int(args[1]), int(args[2]), water)
        elif command == "breach":
            self.breach(int(args[0]), int(args[1]))
        elif command == "undo":
            self.undo()
        elif command == "tick":
            self.tick(int(args[0]) if args else 1)
        elif command == "event":
            turn, kind = int(args[0]), args[1]
            payload = (
                {"at": [int(args[2]), int(args[3])]}
                if len(args) > 2 else None
            )
            self.schedule_event(turn, kind, payload)
        elif command == "save":
            self.save(args[0])
        elif command == "load":
            other = Game.load(args[0])
            self.__dict__.update(other.__dict__)
        elif command == "status":
            pass
        else:
            raise CommandError(f"unknown command {command!r}")
        return self.status()
