"""
Behavioral correction system for RL AI.
Prevents the AI from getting stuck in repetitive patterns.
"""
from collections import deque


class BehavioralCorrector:
    """Monitors and corrects AI movement patterns."""

    def __init__(self):
        self.force_vertical_threshold = 80
        self.vertical_move_cooldown = 15
        self.last_vertical_move = 100
        self.force_horizontal_threshold = 120
        self.horizontal_move_cooldown = 15
        self.last_horizontal_move = 100
        self.movement_history = deque(maxlen=50)
        self.stuck_in_bottom_counter = 0
        self.stuck_in_side_counter = 0

    def correct_action(self, action: int, ai_pos, puck_pos, field_height: int, field_width: int) -> int:
        """
        Apply behavioral corrections to the AI's chosen action.
        Returns the (possibly corrected) action.
        """
        self.movement_history.append(action)
        self.last_vertical_move += 1
        self.last_horizontal_move += 1

        if action in (0, 1):
            self.last_vertical_move = 0
        if action in (2, 3):
            self.last_horizontal_move = 0

        corrected = action
        half_w = field_width // 2

        # Force vertical movement if puck is far vertically and AI hasn't moved vertically recently
        puck_dy = abs(puck_pos[1] - ai_pos[1])
        if puck_dy > self.force_vertical_threshold and self.last_vertical_move > self.vertical_move_cooldown:
            if puck_pos[1] < ai_pos[1]:
                corrected = 0  # Up
            else:
                corrected = 1  # Down
            self.last_vertical_move = 0

        # Force horizontal movement if puck is far horizontally
        puck_dx = abs(puck_pos[0] - ai_pos[0])
        if puck_pos[0] > half_w and puck_dx > self.force_horizontal_threshold and self.last_horizontal_move > self.horizontal_move_cooldown:
            if puck_pos[0] < ai_pos[0]:
                corrected = 2  # Left
            else:
                corrected = 3  # Right
            self.last_horizontal_move = 0

        # Detect stuck in bottom
        if ai_pos[1] > field_height * 0.85:
            self.stuck_in_bottom_counter += 1
            if self.stuck_in_bottom_counter > 30:
                corrected = 0  # Force up
                self.stuck_in_bottom_counter = 0
        else:
            self.stuck_in_bottom_counter = 0

        # Detect stuck at side
        if ai_pos[0] > field_width * 0.9:
            self.stuck_in_side_counter += 1
            if self.stuck_in_side_counter > 30:
                corrected = 2  # Force left
                self.stuck_in_side_counter = 0
        elif ai_pos[0] < half_w + 30:
            self.stuck_in_side_counter += 1
            if self.stuck_in_side_counter > 30:
                corrected = 3  # Force right
                self.stuck_in_side_counter = 0
        else:
            self.stuck_in_side_counter = 0

        return corrected

    def reset(self):
        self.movement_history.clear()
        self.stuck_in_bottom_counter = 0
        self.stuck_in_side_counter = 0
        self.last_vertical_move = 100
        self.last_horizontal_move = 100
