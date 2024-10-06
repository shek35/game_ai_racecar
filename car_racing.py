__credits__ = ["Andrea PIERRÉ"]

import math
from random import random
from typing import Optional, Union

from sklearn import preprocessing
import numpy as np

import gymnasium as gym
from gymnasium import spaces
from car_dynamics import Car
from gymnasium.error import DependencyNotInstalled, InvalidAction
from gymnasium.utils import EzPickle

try:
    import Box2D
    from Box2D.b2 import contactListener, fixtureDef, polygonShape
except ImportError as e:
    raise DependencyNotInstalled(
        "Box2D is not installed, run `pip install gymnasium[box2d]`"
    ) from e

try:
    # As pygame is necessary for using the environment (reset and step) even without a render mode
    #   therefore, pygame is a necessary import for the environment.
    import pygame
    from pygame import gfxdraw
except ImportError as e:
    raise DependencyNotInstalled(
        "pygame is not installed, run `pip install gymnasium[box2d]`"
    ) from e

STATE_W = 96  # less than Atari 160x192
STATE_H = 96
VIDEO_W = 600
VIDEO_H = 400
WINDOW_W = 1000
WINDOW_H = 800

SCALE = 6.0  # Track scale
TRACK_RAD = 900 / SCALE  # Track is heavily morphed circle with this radius
PLAYFIELD = 2000 / SCALE  # Game over boundary
FPS = 50  # Frames per second
ZOOM = 2.7  # Camera zoom
ZOOM_FOLLOW = True  # Set to False for fixed view (don't use zoom)

TRACK_DETAIL_STEP = 21 / SCALE
TRACK_TURN_RATE = 0.31
TRACK_WIDTH = 40 / SCALE
BORDER = 8 / SCALE
BORDER_MIN_COUNT = 4
GRASS_DIM = PLAYFIELD / 20.0
MAX_SHAPE_DIM = (
        max(GRASS_DIM, TRACK_WIDTH, TRACK_DETAIL_STEP) * math.sqrt(2) * ZOOM * SCALE
)

MAX_BACKWARDS_DIST = 30
MAX_FORWARDS_DIST = 200
MAX_SCALAR_SPEED = 200


class FrictionDetector(contactListener):
    def __init__(self, env, lap_complete_percent):
        contactListener.__init__(self)
        self.env = env
        self.lap_complete_percent = lap_complete_percent
        self.first_contacts = 0

    def BeginContact(self, contact):
        self._contact(contact, True)

    def EndContact(self, contact):
        self._contact(contact, False)

    def _contact(self, contact, begin):
        tile = None
        obj = None
        u1 = contact.fixtureA.body.userData
        u2 = contact.fixtureB.body.userData
        if u1 and "road_friction" in u1.__dict__:
            tile = u1
            obj = u2
        if u2 and "road_friction" in u2.__dict__:
            tile = u2
            obj = u1
        if not tile:
            return

        # inherit tile color from env
        tile.color[:] = self.env.road_color
        if not obj or "tiles" not in obj.__dict__:
            return
        if begin:
            obj.tiles.add(tile)
            if not tile.road_visited:
                tile.road_visited = True
                # Skip the first two times to start the game with 0 reward
                if self.first_contacts >= 2:
                    self.env.reward += 1000.0 / len(self.env.track)
                else:
                    self.first_contacts += 1
                self.env.tile_visited_count += 1

                # Lap is considered completed if enough % of the track was covered
                if tile.idx == 0 and self.env.tile_visited_count / len(self.env.track) > self.lap_complete_percent:
                    self.env.new_lap = True
        else:
            obj.tiles.remove(tile)


class LineSegment:
    # line is in form: Ax + By = C
    def __init__(self, A, B, C, start_x, start_y, end_x, end_y):
        self.A = A
        self.B = B
        self.C = C
        self.start_x = start_x
        self.start_y = start_y
        self.end_x = end_x
        self.end_y = end_y

    def within_segment(self, x, y):
        return within_bounds(x, self.start_x, self.end_x) and within_bounds(y, self.start_y, self.end_y)


def within_bounds(target, first, second):
    if first > second:
        return first >= target >= second
    return first <= target <= second


class CarRacing(gym.Env, EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "state_pixels",
            "none",
        ],
        "render_fps": FPS,
    }

    def __init__(
            self,
            render_mode: Optional[str] = None,
            verbose: bool = False,
            lap_complete_percent: float = 0.95,
            domain_randomize: bool = False,
            continuous: bool = True,
            consecutive_negative_terminate_threshold: int = -5,
            deterministic = True
    ):
        EzPickle.__init__(
            self,
            render_mode,
            verbose,
            lap_complete_percent,
            domain_randomize,
            continuous,
        )
        self.continuous = continuous
        self.domain_randomize = domain_randomize
        self.lap_complete_percent = lap_complete_percent
        self._init_colors()
        self.consecutive_negative_terminate_threshold = consecutive_negative_terminate_threshold
        self.deterministic = deterministic

        self.contactListener_keepref = FrictionDetector(self, self.lap_complete_percent)
        self.world = Box2D.b2World((0, 0), contactListener=self.contactListener_keepref)
        self.screen: Optional[pygame.Surface] = None
        self.surf = None
        self.clock = None
        self.isopen = True
        self.invisible_state_window = None
        self.invisible_video_window = None
        self.road = None
        self.car: Optional[Car] = None
        self.reward = 0.0
        self.prev_reward = 0.0
        self.verbose = verbose
        self.new_lap = False
        self.fd_tile = fixtureDef(
            shape=polygonShape(vertices=[(0, 0), (1, 0), (1, -1), (0, -1)])
        )
        self.dist_scaler = preprocessing.MinMaxScaler()
        self.dist_scaler.fit(np.array([0, MAX_FORWARDS_DIST]).reshape(-1, 1))
        self.speed_scaler = preprocessing.MinMaxScaler()
        self.speed_scaler.fit(np.array([0, MAX_SCALAR_SPEED]).reshape(-1, 1))
        self.angle_scaler = preprocessing.MinMaxScaler()
        self.angle_scaler.fit(np.array([-180, 180]).reshape(-1, 1))
        self.consecutive_negative_rewards = 0

        # This will throw a warning in tests/envs/test_envs in utils/env_checker.py as the space is not symmetric
        #   or normalised however this is not possible here so ignore
        if self.continuous:
            self.action_space = spaces.Box(
                np.array([-1, 0, 0]).astype(np.float32),
                np.array([+1, +1, +1]).astype(np.float32),
            )  # steer, gas, brake
        else:
            self.action_space = spaces.Discrete(5)
            # do nothing, left, right, gas, brake

        self.observation_space = spaces.Box(
            low=0, high=255, shape=(STATE_H, STATE_W, 3), dtype=np.uint8
        )

        self.render_mode = render_mode

    def _destroy(self):
        if not self.road:
            return
        for t in self.road:
            self.world.DestroyBody(t)
        self.road = []
        assert self.car is not None
        self.car.destroy()

    def _init_colors(self):
        if self.domain_randomize:
            # domain randomize the bg and grass colour
            self.road_color = self.np_random.uniform(0, 210, size=3)

            self.bg_color = self.np_random.uniform(0, 210, size=3)

            self.grass_color = np.copy(self.bg_color)
            idx = self.np_random.integers(3)
            self.grass_color[idx] += 20
        else:
            # default colours
            self.road_color = np.array([102, 102, 102])
            self.bg_color = np.array([102, 204, 102])
            self.grass_color = np.array([102, 230, 102])

    def _reinit_colors(self, randomize):
        assert (
            self.domain_randomize
        ), "domain_randomize must be True to use this function."

        if randomize:
            # domain randomize the bg and grass colour
            self.road_color = self.np_random.uniform(0, 210, size=3)

            self.bg_color = self.np_random.uniform(0, 210, size=3)

            self.grass_color = np.copy(self.bg_color)
            idx = self.np_random.integers(3)
            self.grass_color[idx] += 20

    def _create_track(self):
        CHECKPOINTS = 12

        # Create checkpoints
        checkpoints = []
        for c in range(CHECKPOINTS):
            noise = self.np_random.uniform(0, 2 * math.pi * 1 / CHECKPOINTS)
            alpha = 2 * math.pi * c / CHECKPOINTS + noise
            rad = self.np_random.uniform(TRACK_RAD / 3, TRACK_RAD)

            if c == 0:
                alpha = 0
                rad = 1.5 * TRACK_RAD
            if c == CHECKPOINTS - 1:
                alpha = 2 * math.pi * c / CHECKPOINTS
                self.start_alpha = 2 * math.pi * (-0.5) / CHECKPOINTS
                rad = 1.5 * TRACK_RAD

            checkpoints.append((alpha, rad * math.cos(alpha), rad * math.sin(alpha)))
        self.road = []

        # Go from one checkpoint to another to create track
        x, y, beta = 1.5 * TRACK_RAD, 0, 0
        dest_i = 0
        laps = 0
        track = []
        no_freeze = 2500
        visited_other_side = False
        while True:
            alpha = math.atan2(y, x)
            if visited_other_side and alpha > 0:
                laps += 1
                visited_other_side = False
            if alpha < 0:
                visited_other_side = True
                alpha += 2 * math.pi

            while True:  # Find destination from checkpoints
                failed = True

                while True:
                    dest_alpha, dest_x, dest_y = checkpoints[dest_i % len(checkpoints)]
                    if alpha <= dest_alpha:
                        failed = False
                        break
                    dest_i += 1
                    if dest_i % len(checkpoints) == 0:
                        break

                if not failed:
                    break

                alpha -= 2 * math.pi
                continue

            r1x = math.cos(beta)
            r1y = math.sin(beta)
            p1x = -r1y
            p1y = r1x
            dest_dx = dest_x - x  # vector towards destination
            dest_dy = dest_y - y
            # destination vector projected on rad:
            proj = r1x * dest_dx + r1y * dest_dy
            while beta - alpha > 1.5 * math.pi:
                beta -= 2 * math.pi
            while beta - alpha < -1.5 * math.pi:
                beta += 2 * math.pi
            prev_beta = beta
            proj *= SCALE
            if proj > 0.3:
                beta -= min(TRACK_TURN_RATE, abs(0.001 * proj))
            if proj < -0.3:
                beta += min(TRACK_TURN_RATE, abs(0.001 * proj))
            x += p1x * TRACK_DETAIL_STEP
            y += p1y * TRACK_DETAIL_STEP
            track.append((alpha, prev_beta * 0.5 + beta * 0.5, x, y))
            if laps > 4:
                break
            no_freeze -= 1
            if no_freeze == 0:
                break

        # Find closed loop range i1..i2, first loop should be ignored, second is OK
        i1, i2 = -1, -1
        i = len(track)
        while True:
            i -= 1
            if i == 0:
                return False  # Failed
            pass_through_start = track[i][0] > self.start_alpha >= track[i - 1][0]
            if pass_through_start and i2 == -1:
                i2 = i
            elif pass_through_start and i1 == -1:
                i1 = i
                break
        if self.verbose:
            print("Track generation: %i..%i -> %i-tiles track" % (i1, i2, i2 - i1))
        assert i1 != -1
        assert i2 != -1

        track = track[i1: i2 - 1]

        first_beta = track[0][1]
        first_perp_x = math.cos(first_beta)
        first_perp_y = math.sin(first_beta)
        # Length of perpendicular jump to put together head and tail
        well_glued_together = np.sqrt(
            np.square(first_perp_x * (track[0][2] - track[-1][2]))
            + np.square(first_perp_y * (track[0][3] - track[-1][3]))
        )
        if well_glued_together > TRACK_DETAIL_STEP:
            return False

        # Red-white border on hard turns
        border = [False] * len(track)
        for i in range(len(track)):
            good = True
            oneside = 0
            for neg in range(BORDER_MIN_COUNT):
                beta1 = track[i - neg - 0][1]
                beta2 = track[i - neg - 1][1]
                good &= abs(beta1 - beta2) > TRACK_TURN_RATE * 0.2
                oneside += np.sign(beta1 - beta2)
            good &= abs(oneside) == BORDER_MIN_COUNT
            border[i] = good
        for i in range(len(track)):
            for neg in range(BORDER_MIN_COUNT):
                border[i - neg] |= border[i]

        # Create tiles
        for i in range(len(track)):
            alpha1, beta1, x1, y1 = track[i]
            alpha2, beta2, x2, y2 = track[i - 1]
            road1_l = (
                x1 - TRACK_WIDTH * math.cos(beta1),
                y1 - TRACK_WIDTH * math.sin(beta1),
            )
            road1_r = (
                x1 + TRACK_WIDTH * math.cos(beta1),
                y1 + TRACK_WIDTH * math.sin(beta1),
            )
            road2_l = (
                x2 - TRACK_WIDTH * math.cos(beta2),
                y2 - TRACK_WIDTH * math.sin(beta2),
            )
            road2_r = (
                x2 + TRACK_WIDTH * math.cos(beta2),
                y2 + TRACK_WIDTH * math.sin(beta2),
            )
            vertices = [road1_l, road1_r, road2_r, road2_l]
            self.fd_tile.shape.vertices = vertices
            t = self.world.CreateStaticBody(fixtures=self.fd_tile)
            t.userData = t
            c = 0.01 * (i % 3) * 255
            t.color = self.road_color + c
            t.road_visited = False
            t.road_friction = 1.0
            t.idx = i
            t.fixtures[0].sensor = True
            self.road_poly.append(([road1_l, road1_r, road2_r, road2_l], t.color))
            self.road.append(t)
            if border[i]:
                side = np.sign(beta2 - beta1)
                b1_l = (
                    x1 + side * TRACK_WIDTH * math.cos(beta1),
                    y1 + side * TRACK_WIDTH * math.sin(beta1),
                )
                b1_r = (
                    x1 + side * (TRACK_WIDTH + BORDER) * math.cos(beta1),
                    y1 + side * (TRACK_WIDTH + BORDER) * math.sin(beta1),
                )
                b2_l = (
                    x2 + side * TRACK_WIDTH * math.cos(beta2),
                    y2 + side * TRACK_WIDTH * math.sin(beta2),
                )
                b2_r = (
                    x2 + side * (TRACK_WIDTH + BORDER) * math.cos(beta2),
                    y2 + side * (TRACK_WIDTH + BORDER) * math.sin(beta2),
                )
                self.road_poly.append(
                    (
                        [b1_l, b1_r, b2_r, b2_l],
                        (255, 255, 255) if i % 2 == 0 else (255, 0, 0),
                    )
                )
        self.track = track
        return True

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self._destroy()
        self.world.contactListener_bug_workaround = FrictionDetector(
            self, self.lap_complete_percent
        )
        self.world.contactListener = self.world.contactListener_bug_workaround
        self.reward = 0.0
        self.prev_reward = 0.0
        self.consecutive_negative_rewards = 0
        self.tile_visited_count = 0
        self.t = 0.0
        self.new_lap = False
        self.road_poly = []

        if self.domain_randomize:
            randomize = True
            if isinstance(options, dict):
                if "randomize" in options:
                    randomize = options["randomize"]

            self._reinit_colors(randomize)

        while True:
            success = self._create_track()
            if success:
                break
            if self.verbose:
                print(
                    "retry to generate track (normal if there are not many"
                    "instances of this message)"
                )
        self.car = Car(self.world, *self.track[0][1:4])

        if self.render_mode == "human":
            self.render()
        zero_step = self.step(None if self.continuous else 0)
        return zero_step[0]

    def step(self, action: Union[np.ndarray, int]):
        assert self.car is not None
        if action is not None:
            if self.continuous:
                self.car.steer(-action[0])
                self.car.gas(action[1])
                self.car.brake(action[2])
                """
                Nondeterministic environment:
                - 5% chance to steer wrong way
                - 5% chance to not brake
                Neither check whether the current value of the action is nonzero
                Commented code is for debugging/demonstration purposes
                """
                if not self.deterministic:
                    roll = random()
                    if roll - 0.05 <= 0:
                        action[0] = -action[0]
                        # print("steered wrong way")
                    elif roll - 0.1 <= 0:
                        action[2] = 0
                        # print("faulty brake")
            else:
                if not self.action_space.contains(action):
                    raise InvalidAction(
                        f"you passed the invalid action `{action}`. "
                        f"The supported action_space is `{self.action_space}`"
                    )
                self.car.steer(-1 * (action == 1) + 1 * (action == 2))
                self.car.gas(action == 3)
                self.car.brake(action == 4)

        self.car.step(1.0 / FPS)
        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
        self.t += 1.0 / FPS

        # self.car.hull.angle is pointing to the right of the car. Add pi/2 to point straight.
        forward_car_angle = self.car.hull.angle + 1.5708
        distance_to_grass = self.calc_distance_to_grass(self.car.hull.position, forward_car_angle, self.road_poly)
        angles_ahead = self.calc_angles_ahead(self.car.hull.position, forward_car_angle, self.road_poly)

        step_reward = 0
        terminated = False
        truncated = False
        if action is not None:  # First step without action, called from reset()
            self.reward -= 0.1
            # if no distances, then minus reward for being on grass
            if distance_to_grass[0] == 0 and distance_to_grass[3] == 0:
                self.reward -= 0.4
            step_reward = self.reward - self.prev_reward
            self.prev_reward = self.reward

            if step_reward < 0:
                self.consecutive_negative_rewards += step_reward
            else:
                self.consecutive_negative_rewards = 0
            if self.consecutive_negative_rewards < self.consecutive_negative_terminate_threshold:
                terminated = True

            if self.tile_visited_count == len(self.track) or self.new_lap:
                # Truncation due to finishing lap
                # This should not be treated as a failure
                # but like a timeout
                truncated = True
            x, y = self.car.hull.position
            if abs(x) > PLAYFIELD or abs(y) > PLAYFIELD:
                terminated = True
                step_reward = -100

        if self.render_mode == "human":
            self.render()
        state = []
        # distance to grass
        distance_to_grass = self.dist_scaler.transform(np.array(distance_to_grass).reshape(-1, 1))
        state.extend(distance_to_grass.reshape(1, -1)[0])
        # road segment angles ahead
        angles_ahead = self.angle_scaler.transform(np.array(angles_ahead).reshape(-1, 1))
        state.extend(angles_ahead.reshape(1, -1)[0])
        # speed
        speed = self.speed_scaler.transform(np.array([[self.get_speed(self.car)]]))
        state.append(speed[0][0])
        # wheel angle
        state.append(self.car.wheels[0].joint.angle)  # range -0.42 to 0.42 on front wheels
        return state, step_reward, terminated, truncated

    def calc_distance_to_grass(self, start_pos, forward_rad, road_segments):
        """
        Calculates the distances from the car to the grass for different angles.

        Args:
            start_pos (tuple): The starting position of the car.
            forward_rad (float): The forward direction of the car in radians.
            road_segments (list): List of road segments.

        Returns:
            list: List of distances from the car to the grass for different angles.
        """
        # get index of road segment that contains car
        road_segment_index = self.get_location_of_car(start_pos, road_segments)
        distances = []

        if road_segment_index is None:
            return np.zeros((7, 1))

        # (angle, rad) tuples
        # angles = [(-105, 1.8326), (-90, 1.5708), (-45, 0.7854), (-15, 0.2618), (-5, 0.08727), (0, 0),
        #           (5, -0.08727), (15, -0.2618), (45, -0.7854), (90, -1.5708), (105, 1.8326)]
        angles_to_calculate = \
            [(-45, 0.7854), (-15, 0.2618), (-5, 0.08727), (0, 0), (5, -0.08727), (15, -0.2618), (45, -0.7854)]

        # for each angle, create line
        for angle, rad in angles_to_calculate:
            cur_line = self.get_line_from_car(start_pos, forward_rad + rad)
            distance = self.get_distance_to_grass(cur_line, road_segment_index, road_segments)
            distances.append(distance)

        return distances

    def calc_angles_ahead(self, start_pos, forward_rad, road_segments):
        """
        Calculates the angles between the car's forward direction and the road segments ahead.

        Args:
            start_pos (tuple): The starting position of the car (x, y).
            forward_rad (float): The forward direction of the car in radians.
            road_segments (list): A list of road segments.

        Returns:
            list: A list of angles between the car's forward direction and the road segments ahead.
        """
        # get index of road segment that contains car
        road_segment_index = self.get_location_of_car(start_pos, road_segments)
        angles = []

        if road_segment_index is None:
            return np.zeros((6, 1))

        segments_to_calculate = [3, 5, 7, 10, 15, 20]

        line = self.get_line_from_car(start_pos, forward_rad)
        for segments_ahead in segments_to_calculate:
            segment = self.get_road_segment(road_segment_index, segments_ahead, road_segments)
            x, y = self.get_midpoint(segment)
            angles.append(self.get_angle([line.end_y, line.end_x], [line.start_y, line.start_x], [y, x]))

        return angles

    def get_road_segment(self, road_segment_index, segments_ahead, road_segments):
        """
        Get the road segment that is `segments_ahead` segments ahead of the given `road_segment_index`.

        Parameters:
            road_segment_index (int): The index of the current road segment.
            segments_ahead (int): The number of segments to look ahead.
            road_segments (list): A list of road segments.

        Returns:
            The road segment that is `segments_ahead` segments ahead of the given `road_segment_index`.
        """
        count = 1
        i = 1
        while count < segments_ahead:
            # don't count curbs as road segments
            if not self.is_curb(road_segments[(road_segment_index + i) % len(road_segments)]):
                count += 1
            i += 1

        return road_segments[(road_segment_index + i) % len(road_segments)]

    @staticmethod
    def get_midpoint(segment):
        """
        Calculates the midpoint of a given segment.

        Args:
            segment (list): A list representing a segment, containing four points.

        Returns:
            tuple: A tuple representing the x and y coordinates of the midpoint.
        """
        x = 0
        y = 0
        for i in range(4):
            x += segment[0][i][0]
            y += segment[0][i][1]

        return x/4, y/4

    @staticmethod
    def get_angle(a, b, c):
        """
        Calculates the angle between three points.

        Args:
            a (tuple): The coordinates of point a.
            b (tuple): The coordinates of point b.
            c (tuple): The coordinates of point c.

        Returns:
            float: The angle between the line segments formed by points a, b, and c.
        """
        angle = math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])) % 360

        # clip to +-180
        if angle > 180:
            angle -= 360
        if angle < -180:
            angle += 360

        return angle

    def get_location_of_car(self, start_pos, road_segments):
        """
        Returns the index of the road segment where the car is located based on its start position.

        Parameters:
        - start_pos (tuple): The starting position of the car.
        - road_segments (list): A list of road segments.

        Returns:
        - int or None: The index of the road segment where the car is located. Returns None if the car is not within any segment.
        """
        # enumerate road segments and find if start_pos lies within segment
        for i, segment in enumerate(road_segments):
            if self.within_road_segment(start_pos, segment):
                return i

        return None

    def get_first_segment_dist(self, line_segment, road_segment):
        """
        Calculates the distance from the start of the line segment to the first intersection with the road segment.

        Args:
            line_segment (LineSegment): The line segment to calculate the distance from.
            road_segment (RoadSegment): The road segment to check for intersections.

        Returns:
            float: The distance from the start of the line segment to the first intersection.
        """
        intersection = None
        # find first intersection
        for i in range(len(road_segment[0])):
            start = road_segment[0][i]
            end = road_segment[0][(i + 1) % len(road_segment[0])]
            line = self.get_line(start[0], start[1], end[0], end[1])
            intersection = self.get_intersection(line_segment, line)
            if intersection:
                break

        # calc distance from start to intersection
        return self.get_distance(line_segment.start_x, line_segment.start_y, intersection[0], intersection[1])

    def get_distance_to_grass(self, line, road_segment_index, road_segments):
        """
        Calculates the maximum distance from the current position to the edge of the grass.

        Args:
            line (Line): The current line representing the car's position.
            road_segment_index (int): The index of the current road segment.
            road_segments (list): A list of road segments.

        Returns:
            float: The maximum distance from the current position to the edge of the grass.
        """
        # adding len to keep road_segment_index from going negative
        road_segment_index += len(road_segments)
        # loop through forwards then backwards because some angles might point slightly backwards.
        # looking backwards might not be necessary
        forward_direction = [True, False]
        forward_and_back_distances = []

        for going_forward in forward_direction:
            # get distance from start_pos to edge of first segment in correct direction
            segment_dist = self.get_first_segment_dist(line, road_segments[road_segment_index % len(road_segments)])
            total_distance = 0
            current_road_index = road_segment_index
            # road segments also include the curbs on the side of turns
            # no_interaction_count lets the algo skip over the curbs when calculating distance
            while segment_dist != 0:
                # get next index of road segment
                current_road_index += 1 if going_forward else -1
                if self.is_curb(road_segments[current_road_index % len(road_segments)]):
                    current_road_index += 1 if going_forward else -1

                total_distance += segment_dist

                # backwards distance is limited to not incentivize driving backwards
                if not going_forward and total_distance > MAX_BACKWARDS_DIST:
                    break

                # get next segment dist
                segment_dist = self.get_intersection_distance(line,
                                                              road_segments[current_road_index % len(road_segments)])
            forward_and_back_distances.append(total_distance)

        return max(forward_and_back_distances)

    @staticmethod
    def is_curb(segment):
        """
        Check if a given segment represents a curb.

        Args:
            segment (list): A list representing a segment of a road.

        Returns:
            bool: True if the segment is a curb, False otherwise.
        """
        # is curb if longest side is less than 5
        return max(
            math.fabs(max(segment[0][0][0], segment[0][1][0], segment[0][2][0], segment[0][3][0])
                        - min(segment[0][0][0], segment[0][1][0], segment[0][2][0], segment[0][3][0])),
            math.fabs(max(segment[0][0][1], segment[0][1][1], segment[0][2][1], segment[0][3][1])
                        - min(segment[0][0][1], segment[0][1][1], segment[0][2][1], segment[0][3][1]))) \
            < 5

    def get_intersection_distance(self, line_segment, road_segment):
        """
        Calculates the distance between two intersections on a road segment.

        Args:
            line_segment (tuple): A tuple representing a line segment.
            road_segment (list): A list of points representing a road segment.

        Returns:
            float: The distance between the two intersections, or 0 if there are less than two intersections.
        """
        intersections = []
        # find first intersection
        for i in range(len(road_segment[0])):
            start = road_segment[0][i]
            end = road_segment[0][(i + 1) % len(road_segment[0])]
            line = self.get_line(start[0], start[1], end[0], end[1])
            intersection = self.get_intersection(line_segment, line)
            if intersection:
                intersections.append(intersection)

        if len(intersections) == 2:
            return self.get_distance(intersections[0][0], intersections[0][1], intersections[1][0], intersections[1][1])

        return 0

    @staticmethod
    def get_distance(first_x, first_y, second_x, second_y):
        """
        Calculate the distance between two points in a 2D plane.

        Args:
            first_x (float): The x-coordinate of the first point.
            first_y (float): The y-coordinate of the first point.
            second_x (float): The x-coordinate of the second point.
            second_y (float): The y-coordinate of the second point.

        Returns:
            float: The distance between the two points.
        """
        return math.sqrt((first_x - second_x) ** 2 + (first_y - second_y) ** 2)

    @staticmethod
    def within_road_segment(start_pos, segment):
        """
        Checks if a given start position is within a road segment.

        Parameters:
        - start_pos (Point): The starting position to check.
        - segment (list): The road segment to check.

        Returns:
        - bool: True if the start position is within the road segment, False otherwise.
        """
        i = 0
        j = 3
        within = False
        vertices = segment[0]

        # there are 4 vertices to check
        while i < 4:
            if ((vertices[i][1] > start_pos.y) != (vertices[j][1] > start_pos.y)) \
                    and (start_pos.x < ((vertices[j][0] - vertices[i][0]) * (start_pos.y - vertices[i][1])
                                        / (vertices[j][1] - vertices[i][1])
                                        + vertices[i][0])):
                within = not within
            j = i
            i += 1
        return within

    def get_line_from_car(self, start_pos, angle):
        """
        Returns a line segment from the car's starting position to a point determined by the given angle.

        Parameters:
        - start_pos (Point): The starting position of the car.
        - angle (float): The angle in radians.

        Returns:
        - line (Line): A line segment from the car's starting position to the calculated endpoint.

        """
        # get endpoint with (r * cos(a), r * sin(a))
        # just make endpoint very far away
        magnitude = 500
        end_x = start_pos.x + magnitude * math.cos(angle)
        end_y = start_pos.y + magnitude * math.sin(angle)

        return self.get_line(start_pos.x, start_pos.y, end_x, end_y)

    @staticmethod
    def get_line(start_x, start_y, end_x, end_y):
        """
        Returns a LineSegment object representing a line segment between two points.

        Parameters:
        start_x (float): The x-coordinate of the starting point.
        start_y (float): The y-coordinate of the starting point.
        end_x (float): The x-coordinate of the ending point.
        end_y (float): The y-coordinate of the ending point.

        Returns:
        LineSegment: A LineSegment object representing the line segment.
        """
        # calculate Ax + By = C
        A = end_y - start_y
        B = start_x - end_x
        C = A * start_x + B * start_y

        return LineSegment(A, B, C, start_x, start_y, end_x, end_y)

    @staticmethod
    def get_intersection(first, second):
        """
        Calculates the intersection point of two lines.

        Args:
            first (Line): The first line.
            second (Line): The second line.

        Returns:
            tuple: The coordinates of the intersection point (x, y) if the lines intersect within their segments,
                    None if the lines are parallel or do not intersect within their segments.
        """
        determinant = first.A * second.B - second.A * first.B
        if determinant == 0:
            # lines are parallel
            return None
        x = (second.B * first.C - first.B * second.C) / determinant
        y = (first.A * second.C - second.A * first.C) / determinant

        if first.within_segment(x, y) and second.within_segment(x, y):
            return x, y
        return None

    @staticmethod
    def get_speed(car):
        """
        Calculate the speed of a car.

        Parameters:
        car (object): The car object for which the speed needs to be calculated.

        Returns:
        float: The speed of the car.

        """
        return np.sqrt(
            np.square(car.hull.linearVelocity[0])
            + np.square(car.hull.linearVelocity[1])
        )

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return
        else:
            return self._render(self.render_mode)

    def _render(self, mode: str):
        assert mode in self.metadata["render_modes"]

        pygame.font.init()
        if self.screen is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        if "t" not in self.__dict__:
            return  # reset() not called yet

        self.surf = pygame.Surface((WINDOW_W, WINDOW_H))

        assert self.car is not None
        # computing transformations
        angle = -self.car.hull.angle
        # Animating first second zoom.
        zoom = 0.1 * SCALE * max(1 - self.t, 0) + ZOOM * SCALE * min(self.t, 1)
        scroll_x = -(self.car.hull.position[0]) * zoom
        scroll_y = -(self.car.hull.position[1]) * zoom
        trans = pygame.math.Vector2((scroll_x, scroll_y)).rotate_rad(angle)
        trans = (WINDOW_W / 2 + trans[0], WINDOW_H / 4 + trans[1])

        self._render_road(zoom, trans, angle)
        self.car.draw(
            self.surf,
            zoom,
            trans,
            angle,
            mode not in ["state_pixels_list", "state_pixels"],
        )

        self.surf = pygame.transform.flip(self.surf, False, True)

        # showing stats
        self._render_indicators(WINDOW_W, WINDOW_H)

        font = pygame.font.Font(pygame.font.get_default_font(), 42)
        text = font.render("%04i" % self.reward, True, (255, 255, 255), (0, 0, 0))
        text_rect = text.get_rect()
        text_rect.center = (60, WINDOW_H - WINDOW_H * 2.5 / 40.0)
        self.surf.blit(text, text_rect)

        if mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            assert self.screen is not None
            self.screen.fill(0)
            self.screen.blit(self.surf, (0, 0))
            pygame.display.flip()
        elif mode == "rgb_array":
            return self._create_image_array(self.surf, (VIDEO_W, VIDEO_H))
        elif mode == "state_pixels":
            return self._create_image_array(self.surf, (STATE_W, STATE_H))
        elif mode == "none":
            return None
        else:
            return self.isopen

    def _render_road(self, zoom, translation, angle):
        bounds = PLAYFIELD
        field = [
            (bounds, bounds),
            (bounds, -bounds),
            (-bounds, -bounds),
            (-bounds, bounds),
        ]

        # draw background
        self._draw_colored_polygon(
            self.surf, field, self.bg_color, zoom, translation, angle, clip=False
        )

        # draw grass patches
        grass = []
        for x in range(-20, 20, 2):
            for y in range(-20, 20, 2):
                grass.append(
                    [
                        (GRASS_DIM * x + GRASS_DIM, GRASS_DIM * y + 0),
                        (GRASS_DIM * x + 0, GRASS_DIM * y + 0),
                        (GRASS_DIM * x + 0, GRASS_DIM * y + GRASS_DIM),
                        (GRASS_DIM * x + GRASS_DIM, GRASS_DIM * y + GRASS_DIM),
                    ]
                )
        for poly in grass:
            self._draw_colored_polygon(
                self.surf, poly, self.grass_color, zoom, translation, angle
            )

        # draw road
        for poly, color in self.road_poly:
            # converting to pixel coordinates
            poly = [(p[0], p[1]) for p in poly]
            color = [int(c) for c in color]
            self._draw_colored_polygon(self.surf, poly, color, zoom, translation, angle)

    def _render_indicators(self, W, H):
        s = W / 40.0
        h = H / 40.0
        color = (0, 0, 0)
        polygon = [(W, H), (W, H - 5 * h), (0, H - 5 * h), (0, H)]
        pygame.draw.polygon(self.surf, color=color, points=polygon)

        def vertical_ind(place, val):
            return [
                (place * s, H - (h + h * val)),
                ((place + 1) * s, H - (h + h * val)),
                ((place + 1) * s, H - h),
                ((place + 0) * s, H - h),
            ]

        def horiz_ind(place, val):
            return [
                ((place + 0) * s, H - 4 * h),
                ((place + val) * s, H - 4 * h),
                ((place + val) * s, H - 2 * h),
                ((place + 0) * s, H - 2 * h),
            ]

        assert self.car is not None
        true_speed = self.get_speed(self.car)

        # simple wrapper to render if the indicator value is above a threshold
        def render_if_min(value, points, color):
            if abs(value) > 1e-4:
                pygame.draw.polygon(self.surf, points=points, color=color)

        render_if_min(true_speed, vertical_ind(5, 0.02 * true_speed), (255, 255, 255))
        # ABS sensors
        render_if_min(
            self.car.wheels[0].omega,
            vertical_ind(7, 0.01 * self.car.wheels[0].omega),
            (0, 0, 255),
        )
        render_if_min(
            self.car.wheels[1].omega,
            vertical_ind(8, 0.01 * self.car.wheels[1].omega),
            (0, 0, 255),
        )
        render_if_min(
            self.car.wheels[2].omega,
            vertical_ind(9, 0.01 * self.car.wheels[2].omega),
            (51, 0, 255),
        )
        render_if_min(
            self.car.wheels[3].omega,
            vertical_ind(10, 0.01 * self.car.wheels[3].omega),
            (51, 0, 255),
        )

        render_if_min(
            self.car.wheels[0].joint.angle,
            horiz_ind(20, -10.0 * self.car.wheels[0].joint.angle),
            (0, 255, 0),
        )
        render_if_min(
            self.car.hull.angularVelocity,
            horiz_ind(30, -0.8 * self.car.hull.angularVelocity),
            (255, 0, 0),
        )

    def _draw_colored_polygon(
            self, surface, poly, color, zoom, translation, angle, clip=True
    ):
        poly = [pygame.math.Vector2(c).rotate_rad(angle) for c in poly]
        poly = [
            (c[0] * zoom + translation[0], c[1] * zoom + translation[1]) for c in poly
        ]
        # This checks if the polygon is out of bounds of the screen, and we skip drawing if so.
        # Instead of calculating exactly if the polygon and screen overlap,
        # we simply check if the polygon is in a larger bounding box whose dimension
        # is greater than the screen by MAX_SHAPE_DIM, which is the maximum
        # diagonal length of an environment object
        if not clip or any(
                (-MAX_SHAPE_DIM <= coord[0] <= WINDOW_W + MAX_SHAPE_DIM)
                and (-MAX_SHAPE_DIM <= coord[1] <= WINDOW_H + MAX_SHAPE_DIM)
                for coord in poly
        ):
            gfxdraw.aapolygon(self.surf, poly, color)
            gfxdraw.filled_polygon(self.surf, poly, color)

    def _create_image_array(self, screen, size):
        scaled_screen = pygame.transform.smoothscale(screen, size)
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(scaled_screen)), axes=(1, 0, 2)
        )

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            self.isopen = False
            pygame.quit()