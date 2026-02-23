from __future__ import annotations

import heapq
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

from shapely.geometry import LineString, Point, Polygon

from src.core import MissionTask, Transition

G_MPS2 = 9.80665
R_AIR_J_PER_KG_K = 287.05
SEA_LEVEL_TEMP_K = 288.15
SEA_LEVEL_PRESSURE_PA = 101325.0
SEA_LEVEL_DENSITY_KG_M3 = 1.225
_TRANSITION_CACHE: Dict[str, Dict[Tuple[str, str], Transition]] = {}
_INFLATED_POLYGON_CACHE: Dict[Tuple[str, int], List[tuple[float, float]]] = {}


def clear_aircraft_transition_cache() -> None:
    _TRANSITION_CACHE.clear()
    _INFLATED_POLYGON_CACHE.clear()


def wrap_angle_rad(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


@dataclass
class WindModel:
    base_u_mps: float
    base_v_mps: float
    gust_mps: float
    spatial_scale_m: float
    temporal_period_s: float
    phase_u_rad: float = 0.0
    phase_v_rad: float = 0.0
    harmonic_ratio: float = 0.35

    def vector(
        self,
        x_m: float,
        y_m: float,
        t_s: float,
        phase_u_rad: float = 0.0,
        phase_v_rad: float = 0.0,
    ) -> tuple[float, float]:
        spatial_scale = max(1.0, self.spatial_scale_m)
        phase_x = (2.0 * math.pi * x_m) / spatial_scale
        phase_y = (2.0 * math.pi * y_m) / spatial_scale
        phase_t = (2.0 * math.pi * t_s) / max(1.0, self.temporal_period_s)

        pu = self.phase_u_rad + phase_u_rad
        pv = self.phase_v_rad + phase_v_rad
        hr = float(max(0.0, min(1.0, self.harmonic_ratio)))

        # Coherent wind components with secondary harmonics to avoid a simple standing wave.
        u_primary = math.sin(phase_t + phase_x + pu)
        v_primary = math.cos(phase_t + phase_y + pv)
        u_secondary = math.sin(0.37 * phase_t - 1.23 * phase_y + 0.5 * pu + 0.25 * pv)
        v_secondary = math.cos(0.41 * phase_t + 1.17 * phase_x + 0.5 * pv + 0.25 * pu)

        norm = 1.0 + hr
        gust_u = (u_primary + hr * u_secondary) / norm
        gust_v = (v_primary + hr * v_secondary) / norm
        u = self.base_u_mps + self.gust_mps * gust_u
        v = self.base_v_mps + self.gust_mps * gust_v
        return u, v


@dataclass
class AircraftDynamicsConfig:
    cruise_speed_mps: float
    min_ground_speed_mps: float
    cruise_power_w: float
    turn_power_w: float
    loiter_power_w: float
    climb_power_w_per_mps: float
    max_turn_rate_deg_s: float
    max_bank_angle_deg: float
    max_climb_rate_mps: float
    battery_capacity_wh: float
    reserve_energy_wh: float
    altitude_min_m: float
    altitude_max_m: float
    mission_horizon_s: float
    geofence_margin_m: float
    path_sample_spacing_m: float
    mass_kg: float
    wing_area_m2: float
    cd0: float
    aspect_ratio: float
    oswald_efficiency: float
    propulsive_efficiency: float
    air_density_kg_m3: float
    auxiliary_power_w: float
    propulsion_scale: float
    cl_max: float
    stall_margin: float
    max_propulsion_power_w: float
    prop_power_altitude_scale_m: float
    temp_offset_c: float
    spherical_geometry: bool = False
    earth_radius_m: float = 6371000.0

    @property
    def max_turn_rate_rad_s(self) -> float:
        return math.radians(self.max_turn_rate_deg_s)

    @property
    def max_bank_angle_rad(self) -> float:
        return math.radians(self.max_bank_angle_deg)

    def bank_limited_turn_rate_rad_s(self, airspeed_mps: float) -> float:
        if airspeed_mps <= 1e-6:
            return float("inf")
        return (G_MPS2 * math.tan(self.max_bank_angle_rad)) / airspeed_mps

    def turn_bank_angle_for_rate_rad(self, airspeed_mps: float, turn_rate_rad_s: float) -> float:
        if airspeed_mps <= 1e-6:
            return 0.0
        bank = math.atan(max(0.0, airspeed_mps * abs(turn_rate_rad_s)) / G_MPS2)
        return max(0.0, min(bank, self.max_bank_angle_rad))


@dataclass
class AircraftState:
    time_s: float
    x_m: float
    y_m: float
    alt_m: float
    heading_rad: float
    energy_wh: float
    total_energy_used_wh: float
    total_distance_m: float
    total_time_s: float
    delivered_science: float
    wind_scale: float
    wind_phase_u_rad: float
    wind_phase_v_rad: float

    @property
    def position_xy(self) -> tuple[float, float]:
        return float(self.x_m), float(self.y_m)

    @classmethod
    def from_mapping(
        cls,
        state: Dict[str, Any],
        *,
        default_x_m: float,
        default_y_m: float,
        default_alt_m: float,
        default_energy_wh: float,
    ) -> "AircraftState":
        return cls(
            time_s=float(state.get("time_s", 0.0)),
            x_m=float(state.get("x_m", default_x_m)),
            y_m=float(state.get("y_m", default_y_m)),
            alt_m=float(state.get("alt_m", default_alt_m)),
            heading_rad=float(state.get("heading_rad", 0.0)),
            energy_wh=float(state.get("energy_wh", default_energy_wh)),
            total_energy_used_wh=float(state.get("total_energy_used_wh", 0.0)),
            total_distance_m=float(state.get("total_distance_m", 0.0)),
            total_time_s=float(state.get("total_time_s", 0.0)),
            delivered_science=float(state.get("delivered_science", 0.0)),
            wind_scale=float(state.get("wind_scale", 1.0)),
            wind_phase_u_rad=float(state.get("wind_phase_u_rad", 0.0)),
            wind_phase_v_rad=float(state.get("wind_phase_v_rad", 0.0)),
        )

    def to_mapping(self) -> Dict[str, float]:
        return {
            "time_s": float(self.time_s),
            "x_m": float(self.x_m),
            "y_m": float(self.y_m),
            "alt_m": float(self.alt_m),
            "heading_rad": float(self.heading_rad),
            "energy_wh": float(self.energy_wh),
            "total_energy_used_wh": float(self.total_energy_used_wh),
            "total_distance_m": float(self.total_distance_m),
            "total_time_s": float(self.total_time_s),
            "delivered_science": float(self.delivered_science),
            "wind_scale": float(self.wind_scale),
            "wind_phase_u_rad": float(self.wind_phase_u_rad),
            "wind_phase_v_rad": float(self.wind_phase_v_rad),
        }


def isa_density_kg_m3(altitude_m: float, temp_offset_c: float = 0.0) -> float:
    h = max(-500.0, min(20000.0, float(altitude_m)))
    t0 = SEA_LEVEL_TEMP_K + float(temp_offset_c)
    p0 = SEA_LEVEL_PRESSURE_PA

    if h <= 11000.0:
        lapse = 0.0065
        t = max(170.0, t0 - lapse * h)
        p = p0 * (t / t0) ** (G_MPS2 / (R_AIR_J_PER_KG_K * lapse))
    else:
        lapse = 0.0065
        t_tropopause = max(170.0, t0 - lapse * 11000.0)
        p_tropopause = p0 * (t_tropopause / t0) ** (G_MPS2 / (R_AIR_J_PER_KG_K * lapse))
        t = t_tropopause
        p = p_tropopause * math.exp((-G_MPS2 * (h - 11000.0)) / (R_AIR_J_PER_KG_K * t))

    rho = p / max(1.0, R_AIR_J_PER_KG_K * t)
    return max(0.05, float(rho))


def _propulsive_efficiency(
    dynamics: AircraftDynamicsConfig,
    airspeed_mps: float,
) -> float:
    speed_ratio = float(airspeed_mps) / max(1.0, float(dynamics.cruise_speed_mps))
    shape = 1.0 - 0.08 * ((speed_ratio - 1.0) ** 2)
    eta = float(dynamics.propulsive_efficiency) * shape
    return max(0.35, min(0.9, eta))


def aerodynamic_performance(
    dynamics: AircraftDynamicsConfig,
    airspeed_mps: float,
    altitude_m: float,
    climb_rate_mps: float = 0.0,
    bank_angle_rad: float = 0.0,
) -> Dict[str, float | bool]:
    v = max(1.0, float(airspeed_mps))
    rho_isa = isa_density_kg_m3(altitude_m=float(altitude_m), temp_offset_c=float(dynamics.temp_offset_c))
    rho_scale = max(0.4, float(dynamics.air_density_kg_m3) / SEA_LEVEL_DENSITY_KG_M3)
    rho = max(0.05, rho_isa * rho_scale)

    s_ref = max(0.05, float(dynamics.wing_area_m2))
    cd0 = max(1e-4, float(dynamics.cd0))
    ar = max(1.0, float(dynamics.aspect_ratio))
    oswald = max(0.3, min(1.2, float(dynamics.oswald_efficiency)))
    m = max(0.1, float(dynamics.mass_kg))

    bank = max(0.0, min(float(bank_angle_rad), float(dynamics.max_bank_angle_rad)))
    n_load = 1.0 / max(1e-3, math.cos(bank))
    q = 0.5 * rho * (v ** 2)
    weight_n = m * G_MPS2
    cl = (n_load * weight_n) / max(1e-6, q * s_ref)

    cl_limit = max(0.1, float(dynamics.cl_max) / max(1.0, float(dynamics.stall_margin)))
    stall_margin_ratio = cl_limit / max(1e-6, cl)
    stall_speed_mps = math.sqrt((2.0 * n_load * weight_n) / max(1e-6, rho * s_ref * cl_limit))
    stall_feasible = bool(cl <= cl_limit and v >= stall_speed_mps)

    cdi = (cl ** 2) / max(1e-6, math.pi * ar * oswald)
    cd = cd0 + cdi
    drag_n = q * s_ref * cd

    eta = _propulsive_efficiency(dynamics, v)
    aerodynamic_w = drag_n * v
    climb_rate = float(climb_rate_mps)
    climb_w = (weight_n * max(0.0, climb_rate)) + (0.22 * weight_n * max(0.0, -climb_rate))
    shaft_power_required_w = float(dynamics.auxiliary_power_w) + (aerodynamic_w + climb_w) / max(0.3, eta)
    shaft_power_required_w = max(0.0, shaft_power_required_w)

    altitude_derate = math.exp(-max(0.0, float(altitude_m)) / max(1000.0, float(dynamics.prop_power_altitude_scale_m)))
    available_power_w = (
        max(50.0, float(dynamics.max_propulsion_power_w))
        * max(0.1, float(dynamics.propulsion_scale))
        * max(0.2, altitude_derate)
    )
    power_margin_w = available_power_w - shaft_power_required_w
    power_feasible = bool(power_margin_w >= -1e-6)

    return {
        "feasible": bool(stall_feasible and power_feasible),
        "stall_feasible": bool(stall_feasible),
        "power_feasible": bool(power_feasible),
        "rho_kg_m3": rho,
        "cl": cl,
        "cd": cd,
        "load_factor": n_load,
        "stall_speed_mps": stall_speed_mps,
        "stall_margin_ratio": stall_margin_ratio,
        "shaft_power_required_w": shaft_power_required_w,
        "available_power_w": available_power_w,
        "power_margin_w": power_margin_w,
        "drag_n": drag_n,
    }


def aerodynamic_power_required_w(
    dynamics: AircraftDynamicsConfig,
    airspeed_mps: float,
    climb_rate_mps: float = 0.0,
    bank_angle_rad: float = 0.0,
    altitude_m: float = 0.0,
) -> float:
    perf = aerodynamic_performance(
        dynamics=dynamics,
        airspeed_mps=airspeed_mps,
        altitude_m=altitude_m,
        climb_rate_mps=climb_rate_mps,
        bank_angle_rad=bank_angle_rad,
    )
    if not bool(perf["feasible"]):
        return float("inf")
    return float(perf["shaft_power_required_w"])


def build_polygons(polygons_xy: Sequence[Sequence[Sequence[float]]]) -> List[Polygon]:
    result: List[Polygon] = []
    for points in polygons_xy:
        poly = Polygon(points)
        if not poly.is_valid:
            poly = poly.buffer(0)
        result.append(poly)
    return result


def bearing_rad(from_xy: tuple[float, float], to_xy: tuple[float, float], spherical: bool = False) -> float:
    if spherical:
        # Haversine-based bearing if inputs are interpreted as deg lon/lat
        lon1, lat1 = math.radians(from_xy[0]), math.radians(from_xy[1])
        lon2, lat2 = math.radians(to_xy[0]), math.radians(to_xy[1])
        dlon = lon2 - lon1
        y = math.sin(dlon) * math.cos(lat2)
        x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
        return math.atan2(y, x)
    dx = float(to_xy[0]) - float(from_xy[0])
    dy = float(to_xy[1]) - float(from_xy[1])
    return math.atan2(dy, dx)


def _dist(a: tuple[float, float], b: tuple[float, float], spherical_r_m: float | None = None) -> float:
    if spherical_r_m is not None:
        # Haversine distance
        lon1, lat1 = math.radians(a[0]), math.radians(a[1])
        lon2, lat2 = math.radians(b[0]), math.radians(b[1])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        u = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
        return 2.0 * spherical_r_m * math.asin(math.sqrt(u))
    return math.hypot(b[0] - a[0], b[1] - a[1])


def _point_close(a: tuple[float, float], b: tuple[float, float], tol: float = 1e-6) -> bool:
    return _dist(a, b) <= tol


def _segment_clear(
    a: tuple[float, float],
    b: tuple[float, float],
    geofences: Sequence[Polygon],
    margin_m: float,
    spherical_r_m: float | None = None,
) -> bool:
    line = LineString([a, b])
    for poly in geofences:
        # Note: Shapely operations remain in Cartesian. 
        # For true spherical geofencing at 800km, we'd need a spherical poly library.
        # This implementation scales the segment clearing for large-scale routing.
        if line.crosses(poly) or line.within(poly):
            return False

        if line.intersects(poly):
            intersection = line.intersection(poly)
            if intersection.is_empty:
                pass
            elif intersection.geom_type == "Point":
                p = (float(intersection.x), float(intersection.y))
                if not (_point_close(p, a) or _point_close(p, b)):
                    return False
            elif intersection.geom_type == "MultiPoint":
                for geom in intersection.geoms:
                    p = (float(geom.x), float(geom.y))
                    if not (_point_close(p, a) or _point_close(p, b)):
                        return False
            else:
                return False

        if line.distance(poly) < margin_m:
            return False

    return True


def _inflated_polygon_vertices(poly: Polygon, margin_m: float) -> List[tuple[float, float]]:
    inflate_m = max(1.0, margin_m + 5.0)
    cache_key = (poly.wkb_hex, int(round(inflate_m * 1000.0)))
    cached = _INFLATED_POLYGON_CACHE.get(cache_key)
    if cached is not None:
        return list(cached)
    try:
        buffered = poly.buffer(inflate_m, join_style=2)
        if buffered.geom_type == "Polygon":
            vertices = [(float(x), float(y)) for x, y in buffered.exterior.coords][:-1]
            _INFLATED_POLYGON_CACHE[cache_key] = vertices
            return list(vertices)
        elif buffered.geom_type == "MultiPolygon":
            coords = []
            for geom in buffered.geoms:
                coords.extend([(float(x), float(y)) for x, y in geom.exterior.coords][:-1])
            _INFLATED_POLYGON_CACHE[cache_key] = list(coords)
            return list(coords)
    except Exception:
        pass

    centroid = poly.centroid
    cx = float(centroid.x)
    cy = float(centroid.y)

    vertices: List[tuple[float, float]] = []
    coords = list(poly.exterior.coords)[:-1]
    for vx, vy in coords:
        dx = float(vx) - cx
        dy = float(vy) - cy
        norm = math.hypot(dx, dy)
        if norm <= 1e-9:
            continue
        scale = inflate_m / norm
        vertices.append((float(vx) + dx * scale, float(vy) + dy * scale))
    _INFLATED_POLYGON_CACHE[cache_key] = list(vertices)
    return list(vertices)


def _visibility_route(
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    geofences: Sequence[Polygon],
    margin_m: float,
) -> List[tuple[float, float]] | None:
    if _segment_clear(start_xy, goal_xy, geofences, margin_m):
        return [start_xy, goal_xy]

    nodes: List[tuple[float, float]] = [start_xy, goal_xy]
    for poly in geofences:
        nodes.extend(_inflated_polygon_vertices(poly, margin_m))

    n = len(nodes)
    if n < 2:
        return None

    adjacency: List[List[tuple[int, float]]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            a = nodes[i]
            b = nodes[j]
            if _segment_clear(a, b, geofences, margin_m):
                d = _dist(a, b)
                adjacency[i].append((j, d))
                adjacency[j].append((i, d))

    inf = float("inf")
    dist_cost = [inf] * n
    prev = [-1] * n
    dist_cost[0] = 0.0
    pq: list[tuple[float, int]] = [(0.0, 0)]

    while pq:
        cost, u = heapq.heappop(pq)
        if cost > dist_cost[u]:
            continue
        if u == 1:
            break
        for v, w in adjacency[u]:
            nc = cost + w
            if nc < dist_cost[v]:
                dist_cost[v] = nc
                prev[v] = u
                heapq.heappush(pq, (nc, v))

    if not math.isfinite(dist_cost[1]):
        return _grid_route(start_xy, goal_xy, geofences, margin_m)

    path_idx: List[int] = []
    cur = 1
    while cur != -1:
        path_idx.append(cur)
        cur = prev[cur]
    path_idx.reverse()

    return [nodes[i] for i in path_idx]


def _nearest_free_cell(
    cell: tuple[int, int],
    blocked: List[List[bool]],
) -> tuple[int, int] | None:
    max_i = len(blocked)
    max_j = len(blocked[0]) if max_i > 0 else 0
    ci, cj = cell
    if 0 <= ci < max_i and 0 <= cj < max_j and not blocked[ci][cj]:
        return (ci, cj)

    max_radius = max(max_i, max_j)
    for radius in range(1, max_radius):
        for di in range(-radius, radius + 1):
            for dj in range(-radius, radius + 1):
                ni = ci + di
                nj = cj + dj
                if 0 <= ni < max_i and 0 <= nj < max_j and not blocked[ni][nj]:
                    return (ni, nj)
    return None


def _grid_route(
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    geofences: Sequence[Polygon],
    margin_m: float,
) -> List[tuple[float, float]] | None:
    step_m = max(600.0, margin_m * 1.2)
    xs = [start_xy[0], goal_xy[0]]
    ys = [start_xy[1], goal_xy[1]]
    for poly in geofences:
        minx, miny, maxx, maxy = poly.bounds
        xs.extend([minx, maxx])
        ys.extend([miny, maxy])

    pad = 2.0 * step_m + margin_m
    min_x = min(xs) - pad
    max_x = max(xs) + pad
    min_y = min(ys) - pad
    max_y = max(ys) + pad

    nx = int(math.ceil((max_x - min_x) / step_m)) + 1
    ny = int(math.ceil((max_y - min_y) / step_m)) + 1
    if nx <= 0 or ny <= 0 or nx * ny > 200000:
        return None

    blocked: List[List[bool]] = [[False for _ in range(ny)] for _ in range(nx)]
    for i in range(nx):
        x = min_x + i * step_m
        for j in range(ny):
            y = min_y + j * step_m
            p = Point(x, y)
            for poly in geofences:
                if poly.contains(p) or poly.distance(p) < margin_m:
                    blocked[i][j] = True
                    break

    def to_cell(x: float, y: float) -> tuple[int, int]:
        i = int(round((x - min_x) / step_m))
        j = int(round((y - min_y) / step_m))
        return (max(0, min(nx - 1, i)), max(0, min(ny - 1, j)))

    def to_xy(i: int, j: int) -> tuple[float, float]:
        return (min_x + i * step_m, min_y + j * step_m)

    start_cell = _nearest_free_cell(to_cell(start_xy[0], start_xy[1]), blocked)
    goal_cell = _nearest_free_cell(to_cell(goal_xy[0], goal_xy[1]), blocked)
    if start_cell is None or goal_cell is None:
        return None

    open_heap: List[tuple[float, tuple[int, int]]] = []
    heapq.heappush(open_heap, (0.0, start_cell))
    g_cost: Dict[tuple[int, int], float] = {start_cell: 0.0}
    prev: Dict[tuple[int, int], tuple[int, int]] = {}

    neighbors = [
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, math.sqrt(2.0)),
        (-1, 1, math.sqrt(2.0)),
        (1, -1, math.sqrt(2.0)),
        (1, 1, math.sqrt(2.0)),
    ]

    def heuristic(c: tuple[int, int]) -> float:
        return _dist(to_xy(c[0], c[1]), to_xy(goal_cell[0], goal_cell[1]))

    visited: set[tuple[int, int]] = set()
    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current in visited:
            continue
        visited.add(current)
        if current == goal_cell:
            break

        ci, cj = current
        for di, dj, w in neighbors:
            ni, nj = ci + di, cj + dj
            if not (0 <= ni < nx and 0 <= nj < ny):
                continue
            if blocked[ni][nj]:
                continue
            next_cell = (ni, nj)
            seg = LineString([to_xy(ci, cj), to_xy(ni, nj)])
            blocked_seg = False
            for poly in geofences:
                if seg.crosses(poly) or seg.within(poly) or seg.distance(poly) < margin_m:
                    blocked_seg = True
                    break
            if blocked_seg:
                continue

            tentative = g_cost[current] + w * step_m
            if tentative < g_cost.get(next_cell, float("inf")):
                g_cost[next_cell] = tentative
                prev[next_cell] = current
                f = tentative + heuristic(next_cell)
                heapq.heappush(open_heap, (f, next_cell))

    if goal_cell not in prev and goal_cell != start_cell:
        return None

    cells: List[tuple[int, int]] = [goal_cell]
    cur = goal_cell
    while cur != start_cell:
        cur = prev[cur]
        cells.append(cur)
    cells.reverse()

    path = [start_xy] + [to_xy(i, j) for (i, j) in cells[1:-1]] + [goal_xy]
    simplified: List[tuple[float, float]] = [path[0]]
    for idx in range(1, len(path) - 1):
        a = simplified[-1]
        b = path[idx]
        c = path[idx + 1]
        ab = (b[0] - a[0], b[1] - a[1])
        bc = (c[0] - b[0], c[1] - b[1])
        cross = abs(ab[0] * bc[1] - ab[1] * bc[0])
        if cross <= 1e-3:
            continue
        simplified.append(b)
    simplified.append(path[-1])
    return simplified


def _polyline_length(path_xy: Sequence[tuple[float, float]]) -> float:
    if len(path_xy) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(path_xy)):
        total += _dist(path_xy[i - 1], path_xy[i])
    return total


def _estimate_leg_time_with_wind_s(
    p0: tuple[float, float],
    p1: tuple[float, float],
    start_time_s: float,
    dynamics: AircraftDynamicsConfig,
    wind_model: WindModel,
    sample_spacing_m: float,
) -> float:
    leg_dist_m = _dist(p0, p1)
    if leg_dist_m <= 1e-9:
        return 0.0
    heading = bearing_rad(p0, p1)
    n_steps = max(1, int(math.ceil(leg_dist_m / max(50.0, sample_spacing_m))))
    ds_m = leg_dist_m / n_steps

    leg_time_s = 0.0
    t_s = start_time_s
    spherical_r = dynamics.earth_radius_m if dynamics.spherical_geometry else None
    for idx in range(n_steps):
        frac = (idx + 0.5) / n_steps
        x = p0[0] + frac * (p1[0] - p0[0])
        y = p0[1] + frac * (p1[1] - p0[1])
        wind_u, wind_v = wind_model.vector(x, y, t_s)
        h = bearing_rad(p0, p1, spherical=dynamics.spherical_geometry)
        wind_along = wind_u * math.cos(h) + wind_v * math.sin(h)
        ground_speed_mps = max(dynamics.min_ground_speed_mps, dynamics.cruise_speed_mps + wind_along)
        dt_s = (leg_dist_m / n_steps) / max(1e-6, ground_speed_mps)
        leg_time_s += dt_s
        t_s += dt_s
    return leg_time_s


def _path_geofence_clear(
    path_xy: Sequence[tuple[float, float]],
    geofences: Sequence[Polygon],
    margin_m: float,
) -> bool:
    for i in range(1, len(path_xy)):
        if not _segment_clear(path_xy[i - 1], path_xy[i], geofences, margin_m):
            return False
    return True


def build_aircraft_tasks(config: Dict[str, Any]) -> Dict[str, MissionTask]:
    horizon = float(config["mission_horizon_s"])
    tasks: Dict[str, MissionTask] = {}

    base = config["base"]
    tasks["BASE_START"] = MissionTask(
        task_id="BASE_START",
        domain="aircraft",
        task_type="start",
        window_start_s=0.0,
        window_end_s=horizon,
        duration_s=0.0,
        value=0.0,
        required=False,
        metadata={"x_m": base["x_m"], "y_m": base["y_m"], "alt_m": base["alt_m"]},
    )

    for wp in config["waypoints"]:
        task_id = str(wp["id"])
        tasks[task_id] = MissionTask(
            task_id=task_id,
            domain="aircraft",
            task_type="waypoint",
            window_start_s=0.0,
            window_end_s=horizon,
            duration_s=float(wp.get("dwell_s", 30.0)),
            value=float(wp.get("value", 1.0)),
            required=True,
            metadata={"x_m": wp["x_m"], "y_m": wp["y_m"], "alt_m": wp.get("alt_m", base["alt_m"])},
        )

    tasks["BASE_END"] = MissionTask(
        task_id="BASE_END",
        domain="aircraft",
        task_type="end",
        window_start_s=0.0,
        window_end_s=horizon,
        duration_s=0.0,
        value=0.0,
        required=False,
        metadata={"x_m": base["x_m"], "y_m": base["y_m"], "alt_m": base["alt_m"]},
    )

    return tasks


def _transition_signature(
    tasks: Dict[str, MissionTask],
    dynamics: AircraftDynamicsConfig,
    geofences: Sequence[Polygon],
    wind_model: WindModel | None,
) -> str:
    task_data = []
    for tid in sorted(tasks.keys()):
        t = tasks[tid]
        task_data.append(
            {
                "id": tid,
                "x_m": float(t.metadata.get("x_m", 0.0)),
                "y_m": float(t.metadata.get("y_m", 0.0)),
                "alt_m": float(t.metadata.get("alt_m", 0.0)),
            }
        )

    geofence_data = []
    for poly in geofences:
        coords = []
        for x, y in list(poly.exterior.coords):
            coords.append([round(float(x), 3), round(float(y), 3)])
        geofence_data.append(coords)

    signature_obj: Dict[str, Any] = {
        "tasks": task_data,
        "geofences": geofence_data,
        "dynamics": {
            "cruise_speed_mps": dynamics.cruise_speed_mps,
            "min_ground_speed_mps": dynamics.min_ground_speed_mps,
            "mass_kg": dynamics.mass_kg,
            "wing_area_m2": dynamics.wing_area_m2,
            "cd0": dynamics.cd0,
            "aspect_ratio": dynamics.aspect_ratio,
            "oswald_efficiency": dynamics.oswald_efficiency,
            "propulsive_efficiency": dynamics.propulsive_efficiency,
            "air_density_kg_m3": dynamics.air_density_kg_m3,
            "auxiliary_power_w": dynamics.auxiliary_power_w,
            "propulsion_scale": dynamics.propulsion_scale,
            "cl_max": dynamics.cl_max,
            "stall_margin": dynamics.stall_margin,
            "max_propulsion_power_w": dynamics.max_propulsion_power_w,
            "prop_power_altitude_scale_m": dynamics.prop_power_altitude_scale_m,
            "temp_offset_c": dynamics.temp_offset_c,
            "geofence_margin_m": dynamics.geofence_margin_m,
            "path_sample_spacing_m": dynamics.path_sample_spacing_m,
        },
    }
    if wind_model is not None:
        signature_obj["wind_model"] = {
            "base_u_mps": wind_model.base_u_mps,
            "base_v_mps": wind_model.base_v_mps,
            "gust_mps": wind_model.gust_mps,
            "spatial_scale_m": wind_model.spatial_scale_m,
            "temporal_period_s": wind_model.temporal_period_s,
            "phase_u_rad": wind_model.phase_u_rad,
            "phase_v_rad": wind_model.phase_v_rad,
            "harmonic_ratio": wind_model.harmonic_ratio,
        }
    else:
        signature_obj["wind_model"] = None

    return json.dumps(signature_obj, sort_keys=True)


def build_aircraft_transitions(
    tasks: Dict[str, MissionTask],
    dynamics: AircraftDynamicsConfig,
    geofences: Sequence[Polygon],
    wind_model: WindModel | None = None,
) -> Dict[Tuple[str, str], Transition]:
    cache_key = _transition_signature(tasks, dynamics, geofences, wind_model)
    cached = _TRANSITION_CACHE.get(cache_key)
    if cached is not None:
        return dict(cached)

    transitions: Dict[Tuple[str, str], Transition] = {}
    task_ids = list(tasks.keys())

    for from_id in task_ids:
        for to_id in task_ids:
            if from_id == to_id:
                continue

            from_task = tasks[from_id]
            to_task = tasks[to_id]
            from_xy = (float(from_task.metadata["x_m"]), float(from_task.metadata["y_m"]))
            to_xy = (float(to_task.metadata["x_m"]), float(to_task.metadata["y_m"]))

            route_xy = _visibility_route(from_xy, to_xy, geofences, dynamics.geofence_margin_m)
            if route_xy is None:
                transitions[(from_id, to_id)] = Transition(
                    from_task_id=from_id,
                    to_task_id=to_id,
                    travel_time_s=float("inf"),
                    energy_cost_wh=float("inf"),
                    feasible=False,
                    metadata={"reason": "no_feasible_route"},
                )
                continue

            distance_m = _polyline_length(route_xy)
            nominal_time_s = 0.0
            if wind_model is None:
                nominal_time_s = distance_m / max(0.1, dynamics.cruise_speed_mps)
            else:
                t_s = 0.0
                for i in range(1, len(route_xy)):
                    leg_time_s = _estimate_leg_time_with_wind_s(
                        route_xy[i - 1],
                        route_xy[i],
                        t_s,
                        dynamics,
                        wind_model,
                        dynamics.path_sample_spacing_m,
                    )
                    nominal_time_s += leg_time_s
                    t_s += leg_time_s
            from_alt = float(from_task.metadata.get("alt_m", 0.0))
            to_alt = float(to_task.metadata.get("alt_m", from_alt))
            signed_climb_rate = (to_alt - from_alt) / max(1e-6, nominal_time_s)
            required_climb_rate = min(abs(signed_climb_rate), dynamics.max_climb_rate_mps)
            avg_alt_m = 0.5 * (from_alt + to_alt)
            perf = aerodynamic_performance(
                dynamics=dynamics,
                airspeed_mps=dynamics.cruise_speed_mps,
                altitude_m=avg_alt_m,
                climb_rate_mps=signed_climb_rate,
                bank_angle_rad=0.0,
            )
            if not bool(perf["feasible"]):
                transitions[(from_id, to_id)] = Transition(
                    from_task_id=from_id,
                    to_task_id=to_id,
                    travel_time_s=float("inf"),
                    energy_cost_wh=float("inf"),
                    feasible=False,
                    metadata={
                        "reason": "aerodynamic_feasibility",
                        "required_climb_rate_mps": signed_climb_rate,
                        "stall_margin_ratio": float(perf["stall_margin_ratio"]),
                        "power_margin_w": float(perf["power_margin_w"]),
                    },
                )
                continue

            nominal_power_w = float(perf["shaft_power_required_w"])
            nominal_energy_wh = nominal_power_w * (nominal_time_s / 3600.0)

            transitions[(from_id, to_id)] = Transition(
                from_task_id=from_id,
                to_task_id=to_id,
                travel_time_s=nominal_time_s,
                energy_cost_wh=nominal_energy_wh,
                feasible=True,
                metadata={
                    "distance_m": distance_m,
                    "path_xy": route_xy,
                    "n_legs": max(0, len(route_xy) - 1),
                    "nominal_bearing_rad": bearing_rad(from_xy, to_xy),
                    "nominal_required_climb_rate_mps": required_climb_rate,
                    "nominal_stall_margin_ratio": float(perf["stall_margin_ratio"]),
                    "nominal_power_margin_w": float(perf["power_margin_w"]),
                },
            )

    _TRANSITION_CACHE[cache_key] = dict(transitions)
    return transitions


def simulate_aircraft_step(
    state: Dict[str, Any],
    from_task: MissionTask,
    to_task: MissionTask,
    transition: Transition,
    dynamics: AircraftDynamicsConfig,
    wind_model: WindModel,
    geofences: Sequence[Polygon],
) -> tuple[Dict[str, Any], Dict[str, Any]] | None:
    if not transition.feasible:
        return None

    typed_state = AircraftState.from_mapping(
        state,
        default_x_m=float(from_task.metadata["x_m"]),
        default_y_m=float(from_task.metadata["y_m"]),
        default_alt_m=float(from_task.metadata["alt_m"]),
        default_energy_wh=float(dynamics.battery_capacity_wh),
    )

    speed_scale = float(transition.metadata.get("control_speed_scale", 1.0))
    speed_scale = max(0.6, min(1.4, speed_scale))
    commanded_speed_mps = dynamics.cruise_speed_mps * speed_scale

    path_xy = transition.metadata.get("path_xy")
    if not path_xy or len(path_xy) < 2:
        path_xy = [
            typed_state.position_xy,
            (float(to_task.metadata["x_m"]), float(to_task.metadata["y_m"])),
        ]

    now_s = float(typed_state.time_s)
    from_alt = float(typed_state.alt_m)
    to_alt = float(to_task.metadata.get("alt_m", from_alt))
    estimated_transition_time_s = max(1e-6, float(transition.travel_time_s))
    estimated_climb_rate_signed_mps = (to_alt - from_alt) / estimated_transition_time_s
    provisional_turn_rate = min(
        dynamics.max_turn_rate_rad_s,
        dynamics.bank_limited_turn_rate_rad_s(commanded_speed_mps),
    )
    provisional_turn_bank_rad = dynamics.turn_bank_angle_for_rate_rad(commanded_speed_mps, provisional_turn_rate)

    avg_alt_m = 0.5 * (from_alt + to_alt)
    cruise_perf_nominal = aerodynamic_performance(
        dynamics=dynamics,
        airspeed_mps=commanded_speed_mps,
        altitude_m=avg_alt_m,
        climb_rate_mps=estimated_climb_rate_signed_mps,
        bank_angle_rad=0.0,
    )
    turn_perf_nominal = aerodynamic_performance(
        dynamics=dynamics,
        airspeed_mps=commanded_speed_mps,
        altitude_m=avg_alt_m,
        climb_rate_mps=estimated_climb_rate_signed_mps,
        bank_angle_rad=provisional_turn_bank_rad,
    )
    if not bool(cruise_perf_nominal["feasible"]) or not bool(turn_perf_nominal["feasible"]):
        return None

    current_heading = float(typed_state.heading_rad)
    current_xy = (float(path_xy[0][0]), float(path_xy[0][1]))
    transition_route_distance_m = max(
        1e-6,
        float(transition.metadata.get("distance_m", _polyline_length(path_xy))),
    )
    route_distance_m = 0.0
    total_distance_m = 0.0
    turn_time_total_s = 0.0
    cruise_time_total_s = 0.0
    turn_energy_wh = 0.0
    cruise_energy_wh = 0.0
    max_turn_rate_used = 0.0
    min_stall_margin_ratio = float("inf")
    min_power_margin_w = float("inf")
    avg_density_kg_m3_acc = 0.0
    density_samples = 0
    wind_u_acc = 0.0
    wind_v_acc = 0.0
    wind_samples = 0

    samples: List[Dict[str, Any]] = []
    samples.append(
        {
            "time_s": now_s,
            "x_m": current_xy[0],
            "y_m": current_xy[1],
            "alt_m": from_alt,
            "heading_rad": current_heading,
            "energy_wh": float(typed_state.energy_wh),
        }
    )

    start_energy_wh = float(typed_state.energy_wh)

    for i in range(1, len(path_xy)):
        leg_target_xy = (float(path_xy[i][0]), float(path_xy[i][1]))
        leg_heading = bearing_rad(current_xy, leg_target_xy)
        delta_heading = wrap_angle_rad(leg_heading - current_heading)

        bank_limited_rate = dynamics.bank_limited_turn_rate_rad_s(commanded_speed_mps)
        max_turn_rate = min(dynamics.max_turn_rate_rad_s, bank_limited_rate)
        if max_turn_rate <= 1e-9:
            return None

        leg_turn_time_s = abs(delta_heading) / max_turn_rate
        progress_before_turn = min(1.0, route_distance_m / transition_route_distance_m)
        leg_mid_alt_m = from_alt + (to_alt - from_alt) * (
            progress_before_turn
        )
        leg_turn_bank_rad = dynamics.turn_bank_angle_for_rate_rad(commanded_speed_mps, max_turn_rate)
        leg_turn_perf = aerodynamic_performance(
            dynamics=dynamics,
            airspeed_mps=commanded_speed_mps,
            altitude_m=leg_mid_alt_m,
            climb_rate_mps=estimated_climb_rate_signed_mps,
            bank_angle_rad=leg_turn_bank_rad,
        )
        if not bool(leg_turn_perf["feasible"]):
            return None

        leg_turn_power_w = float(leg_turn_perf["shaft_power_required_w"])
        min_stall_margin_ratio = min(min_stall_margin_ratio, float(leg_turn_perf["stall_margin_ratio"]))
        min_power_margin_w = min(min_power_margin_w, float(leg_turn_perf["power_margin_w"]))
        avg_density_kg_m3_acc += float(leg_turn_perf["rho_kg_m3"])
        density_samples += 1

        if leg_turn_time_s > 1e-9:
            turn_radius_m = commanded_speed_mps / max_turn_rate
            turn_sign = 1.0 if delta_heading >= 0.0 else -1.0
            cx = current_xy[0] + turn_sign * (-math.sin(current_heading)) * turn_radius_m
            cy = current_xy[1] + turn_sign * (math.cos(current_heading)) * turn_radius_m
            start_theta = math.atan2(current_xy[1] - cy, current_xy[0] - cx)
            turn_sample_dt = max(0.5, dynamics.path_sample_spacing_m / max(1.0, commanded_speed_mps))
            turn_samples = max(1, int(math.ceil(leg_turn_time_s / turn_sample_dt)))
            dt_turn_s = leg_turn_time_s / turn_samples

            for turn_idx in range(1, turn_samples + 1):
                frac_turn = turn_idx / turn_samples
                theta = start_theta + delta_heading * frac_turn
                x_turn = cx + turn_radius_m * math.cos(theta)
                y_turn = cy + turn_radius_m * math.sin(theta)
                heading_turn = wrap_angle_rad(current_heading + delta_heading * frac_turn)

                now_s += dt_turn_s
                turn_time_total_s += dt_turn_s
                turn_distance_m = commanded_speed_mps * dt_turn_s
                total_distance_m += turn_distance_m
                turn_energy_wh += leg_turn_power_w * (dt_turn_s / 3600.0)
                max_turn_rate_used = max(max_turn_rate_used, abs(delta_heading) / max(1e-9, leg_turn_time_s))

                progress_turn = min(1.0, route_distance_m / transition_route_distance_m)
                alt_turn = from_alt + progress_turn * (to_alt - from_alt)
                energy_left = start_energy_wh - (turn_energy_wh + cruise_energy_wh)
                samples.append(
                    {
                        "time_s": now_s,
                        "x_m": x_turn,
                        "y_m": y_turn,
                        "alt_m": alt_turn,
                        "heading_rad": heading_turn,
                        "energy_wh": energy_left,
                    }
                )

            current_xy = (x_turn, y_turn)

        current_heading = leg_heading

        leg_dist_m = _dist(current_xy, leg_target_xy)
        if leg_dist_m <= 1e-9:
            continue

        n_steps = max(1, int(math.ceil(leg_dist_m / max(1.0, dynamics.path_sample_spacing_m))))
        ds_m = leg_dist_m / n_steps

        for step_idx in range(1, n_steps + 1):
            frac = step_idx / n_steps
            x = current_xy[0] + frac * (leg_target_xy[0] - current_xy[0])
            y = current_xy[1] + frac * (leg_target_xy[1] - current_xy[1])

            wind_scale = float(typed_state.wind_scale)
            wind_phase_u = float(typed_state.wind_phase_u_rad)
            wind_phase_v = float(typed_state.wind_phase_v_rad)
            wind_u_base, wind_v_base = wind_model.vector(
                x,
                y,
                now_s,
                phase_u_rad=wind_phase_u,
                phase_v_rad=wind_phase_v,
            )
            wind_u = wind_u_base * wind_scale
            wind_v = wind_v_base * wind_scale
            wind_u_acc += wind_u
            wind_v_acc += wind_v
            wind_samples += 1
            wind_along = wind_u * math.cos(leg_heading) + wind_v * math.sin(leg_heading)
            ground_speed_mps = max(dynamics.min_ground_speed_mps, commanded_speed_mps + wind_along)

            dt_s = ds_m / max(1e-6, ground_speed_mps)
            progress = min(1.0, (route_distance_m + ds_m) / transition_route_distance_m)
            alt = from_alt + progress * (to_alt - from_alt)
            cruise_perf = aerodynamic_performance(
                dynamics=dynamics,
                airspeed_mps=commanded_speed_mps,
                altitude_m=alt,
                climb_rate_mps=estimated_climb_rate_signed_mps,
                bank_angle_rad=0.0,
            )
            if not bool(cruise_perf["feasible"]):
                return None

            now_s += dt_s
            cruise_time_total_s += dt_s
            cruise_energy_wh += float(cruise_perf["shaft_power_required_w"]) * (dt_s / 3600.0)
            min_stall_margin_ratio = min(min_stall_margin_ratio, float(cruise_perf["stall_margin_ratio"]))
            min_power_margin_w = min(min_power_margin_w, float(cruise_perf["power_margin_w"]))
            avg_density_kg_m3_acc += float(cruise_perf["rho_kg_m3"])
            density_samples += 1
            total_distance_m += ds_m
            route_distance_m += ds_m

            energy_left = start_energy_wh - (turn_energy_wh + cruise_energy_wh)
            samples.append(
                {
                    "time_s": now_s,
                    "x_m": x,
                    "y_m": y,
                    "alt_m": alt,
                    "heading_rad": leg_heading,
                    "energy_wh": energy_left,
                }
            )

        current_xy = leg_target_xy

    flight_time_s = turn_time_total_s + cruise_time_total_s
    climb_m = abs(to_alt - from_alt)
    required_climb_rate_mps = climb_m / max(1e-6, flight_time_s)
    if required_climb_rate_mps > dynamics.max_climb_rate_mps + 1e-9:
        return None

    task_duration_s = float(to_task.duration_s)
    loiter_perf = aerodynamic_performance(
        dynamics=dynamics,
        airspeed_mps=max(8.0, 0.45 * commanded_speed_mps),
        altitude_m=to_alt,
        climb_rate_mps=0.0,
        bank_angle_rad=0.0,
    )
    if bool(loiter_perf["feasible"]):
        dwell_power_w = max(dynamics.loiter_power_w, float(loiter_perf["shaft_power_required_w"]))
        min_stall_margin_ratio = min(min_stall_margin_ratio, float(loiter_perf["stall_margin_ratio"]))
        min_power_margin_w = min(min_power_margin_w, float(loiter_perf["power_margin_w"]))
        avg_density_kg_m3_acc += float(loiter_perf["rho_kg_m3"])
        density_samples += 1
    else:
        dwell_power_w = dynamics.loiter_power_w

    dwell_energy_wh = dwell_power_w * (task_duration_s / 3600.0)

    energy_used_wh = turn_energy_wh + cruise_energy_wh + dwell_energy_wh
    remaining_wh = start_energy_wh - energy_used_wh

    task_start_s = now_s
    task_end_s = task_start_s + task_duration_s

    path_xy_samples = [(float(s["x_m"]), float(s["y_m"])) for s in samples]
    geofence_clear = _path_geofence_clear(path_xy_samples, geofences, dynamics.geofence_margin_m)

    next_typed_state = AircraftState(
        time_s=float(task_end_s),
        x_m=float(to_task.metadata["x_m"]),
        y_m=float(to_task.metadata["y_m"]),
        alt_m=float(to_alt),
        heading_rad=float(current_heading),
        energy_wh=float(remaining_wh),
        total_energy_used_wh=float(typed_state.total_energy_used_wh + energy_used_wh),
        total_distance_m=float(typed_state.total_distance_m + total_distance_m),
        total_time_s=float(task_end_s),
        delivered_science=float(typed_state.delivered_science),
        wind_scale=float(typed_state.wind_scale),
        wind_phase_u_rad=float(typed_state.wind_phase_u_rad),
        wind_phase_v_rad=float(typed_state.wind_phase_v_rad),
    )
    state_after = dict(state)
    state_after.update(next_typed_state.to_mapping())

    samples.append(
        {
            "time_s": task_end_s,
            "x_m": float(to_task.metadata["x_m"]),
            "y_m": float(to_task.metadata["y_m"]),
            "alt_m": to_alt,
            "heading_rad": current_heading,
            "energy_wh": remaining_wh,
        }
    )

    avg_ground_speed = total_distance_m / max(1e-6, flight_time_s)
    avg_density_kg_m3 = avg_density_kg_m3_acc / max(1, density_samples)
    avg_wind_u_mps = wind_u_acc / max(1, wind_samples)
    avg_wind_v_mps = wind_v_acc / max(1, wind_samples)
    if not math.isfinite(min_stall_margin_ratio):
        min_stall_margin_ratio = float("inf")
    if not math.isfinite(min_power_margin_w):
        min_power_margin_w = float("inf")

    step_meta: Dict[str, Any] = {
        "task_start_s": task_start_s,
        "task_end_s": task_end_s,
        "transition_time_s": flight_time_s,
        "task_duration_s": task_duration_s,
        "distance_m": total_distance_m,
        "route_points": path_xy,
        "from_x_m": float(path_xy[0][0]),
        "from_y_m": float(path_xy[0][1]),
        "to_x_m": float(path_xy[-1][0]),
        "to_y_m": float(path_xy[-1][1]),
        "from_alt_m": from_alt,
        "to_alt_m": to_alt,
        "ground_speed_mps": avg_ground_speed,
        "wind_u_mps": avg_wind_u_mps,
        "wind_v_mps": avg_wind_v_mps,
        "commanded_speed_mps": commanded_speed_mps,
        "control_speed_scale": speed_scale,
        "turn_time_s": turn_time_total_s,
        "cruise_time_s": cruise_time_total_s,
        "turn_rate_used_rad_s": max_turn_rate_used,
        "required_climb_rate_mps": required_climb_rate_mps,
        "signed_climb_rate_mps": estimated_climb_rate_signed_mps,
        "min_stall_margin_ratio": min_stall_margin_ratio,
        "min_power_margin_w": min_power_margin_w,
        "avg_density_kg_m3": avg_density_kg_m3,
        "aero_feasible": bool(min_stall_margin_ratio >= 1.0 and min_power_margin_w >= -1e-6),
        "geofence_clear": geofence_clear,
        "samples": samples,
    }

    return state_after, step_meta


def build_initial_state(config: Dict[str, Any], battery_scale: float = 1.0) -> Dict[str, Any]:
    base = config["base"]
    capacity = float(config["battery_capacity_wh"]) * battery_scale
    return AircraftState(
        time_s=0.0,
        x_m=float(base["x_m"]),
        y_m=float(base["y_m"]),
        alt_m=float(base["alt_m"]),
        heading_rad=0.0,
        energy_wh=capacity,
        total_energy_used_wh=0.0,
        total_distance_m=0.0,
        total_time_s=0.0,
        delivered_science=0.0,
        wind_scale=1.0,
        wind_phase_u_rad=0.0,
        wind_phase_v_rad=0.0,
    ).to_mapping()
