import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from numba import jit, prange

# from scipy.integrate import solve_ivp
from tqdm import tqdm

from configs import Config


class Object:
    def __init__(self, pos, vel, mass, color):
        self.x = pos[0]
        self.y = pos[1]
        self.z = pos[2]
        self.vx = vel[0]
        self.vy = vel[1]
        self.vz = vel[2]
        self.mass = mass
        self.color = color
        self.trail = []

    def init_plot(self, ax):
        (self.trace,) = ax.plot([], [], [], color=self.color, lw=1)
        (self.body,) = ax.plot([], [], [], "o", color=self.color, ms=8)

    def draw(self, i, x, y, z):
        self.trace.set_data(x[:i], y[:i])
        self.trace.set_3d_properties(z[:i])
        self.body.set_data([x[i]], [y[i]])
        self.body.set_3d_properties([z[i]])

        return self.trace, self.body


@jit(nopython=False)
def system_odes(t, S, mass):
    m1, m2, m3 = mass
    p1, p2, p3 = S[0:3], S[3:6], S[6:9]
    v1, v2, v3 = S[9:12], S[12:15], S[15:18]

    F = np.zeros(18, dtype=np.float64)

    F[0:3] = v1
    F[3:6] = v2
    F[6:9] = v3

    F[9:12] = (
        m3 * (p3 - p1) / np.linalg.norm(p3 - p1) ** 3
        + m2 * (p2 - p1) / np.linalg.norm(p2 - p1) ** 3
    )

    F[12:15] = (
        m3 * (p3 - p2) / np.linalg.norm(p3 - p2) ** 3
        + m1 * (p1 - p2) / np.linalg.norm(p1 - p2) ** 3
    )

    F[15:18] = (
        m1 * (p1 - p3) / np.linalg.norm(p1 - p3) ** 3
        + m2 * (p2 - p3) / np.linalg.norm(p2 - p3) ** 3
    )

    return F


@jit(nopython=False)
def rk4(t, S, dt, mass):
    k1 = system_odes(t, S, mass)
    k2 = system_odes(t + dt / 2, S + (dt / 2) * k1, mass)
    k3 = system_odes(t + dt / 2, S + (dt / 2) * k2, mass)
    k4 = system_odes(t + dt, S + dt * k3, mass)

    return (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def params_out(filename, planets):
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Input file: {path}")

    with open(filename, "w", newline="") as f:
        f.write("---------Initial condition---------\n")
        for planet in planets:
            pos = planets[planet]["pos"]
            vel = planets[planet]["velo"]

            x, y, z = pos
            vx, vy, vz = vel

            f.write(f"{planet:8s}\n")
            f.write("x        y        z\n")
            f.write(f"{x:.4f} {y:.4f} {z:.4f}\n")
            f.write("vx       vy       vz\n")
            f.write(f"{vx:.4f} {vy:.4f} {vz:.4f}\n")
    return True


def save_results(sol_a, sol_b, t_points, ln_deltas, lamb, eps):

    with (
        open(f"./sol_ic_A.txt", "w", newline="") as fa,
        open(f"./sol_ic_B.txt", "w", newline="") as fb,
    ):
        header = [
            "t",
            "ln_delta",
            "lambda",
            "x_1",
            "y_1",
            "z_1",
            "x_2",
            "y_2",
            "z_2",
            "x_3",
            "y_3",
            "z_3",
        ]

        writer_fa = csv.DictWriter(fa, fieldnames=header, lineterminator="\n")
        writer_fb = csv.DictWriter(fb, fieldnames=header, lineterminator="\n")

        writer_fa.writeheader()
        writer_fb.writeheader()

        for i in tqdm(range(1, len(t_points)), desc="Write file"):
            t = t_points[i]

            row_a = {
                "t": t,
                "ln_delta": ln_deltas[i - 1],
                "lambda": lamb[i - 1],
                "x_1": sol_a[0, i],
                "y_1": sol_a[1, i],
                "z_1": sol_a[2, i],
                "x_2": sol_a[3, i],
                "y_2": sol_a[4, i],
                "z_2": sol_a[5, i],
                "x_3": sol_a[6, i],
                "y_3": sol_a[7, i],
                "z_3": sol_a[8, i],
            }

            row_b = {
                "t": t,
                "ln_delta": ln_deltas[i - 1],
                "lambda": lamb[i - 1],
                "x_1": sol_b[0, i],
                "y_1": sol_b[1, i],
                "z_1": sol_b[2, i],
                "x_2": sol_b[3, i],
                "y_2": sol_b[4, i],
                "z_2": sol_b[5, i],
                "x_3": sol_b[6, i],
                "y_3": sol_b[7, i],
                "z_3": sol_b[8, i],
            }

            writer_fa.writerow(row_a)
            writer_fb.writerow(row_b)
    return True


def return_proximity_function(X, X0):

    return np.linalg.norm(X - X0)


@jit(nopython=False)
def solve_eq(cond_a, cond_b, n_steps, dt, t_start, masses):
    sol_a = np.zeros((18, n_steps))
    sol_b = np.zeros((18, n_steps))
    t_points = np.zeros(n_steps)
    for i in range(n_steps):
        t_points[i] = t_start + i * dt
    sol_a[:, 0] = cond_a
    sol_b[:, 0] = cond_b
    X0 = cond_a.copy()
    sa = cond_a.copy()
    sb = cond_b.copy()
    delta0 = np.linalg.norm(sa - sb)
    S_sum = 0.0
    ln_deltas = np.zeros(n_steps - 1)
    lambdas = np.zeros(n_steps - 1)
    proximity_list = np.zeros(n_steps - 1)
    for i in range(1, n_steps):
        t = t_points[i - 1]
        Xi = sa.copy()
        sa += rk4(t, sa, dt, masses)
        sb += rk4(t, sb, dt, masses)
        Xi1 = sa.copy()

        # Khoảng cách điểm đến đoạn thẳng Xi -> Xi1
        v = Xi1 - Xi  # X_{i+1} - X_i
        w = X0 - Xi  # X_0 - X_i
        v_norm_sq = np.dot(v, v)
        if v_norm_sq > 1e-20:
            a = np.dot(v, w) / v_norm_sq
            if 0.0 < a < 1.0:
                d = np.linalg.norm(w - a * v)
            else:
                d = np.linalg.norm(w)
        else:
            d = np.linalg.norm(w)
        proximity_list[i - 1] = d

        delta_d = np.linalg.norm(sa - sb)
        S_sum += np.log((delta_d + 1e-20) / (delta0 + 1e-20))
        sol_a[:, i] = sa
        sol_b[:, i] = sb
        ln_deltas[i - 1] = np.log(delta_d + 1e-20)
        lambdas[i - 1] = S_sum / t_points[i]

    return sol_a, sol_b, t_points, ln_deltas, lambdas, proximity_list


@jit(parallel=True, nopython=True)
def compute_heatmap_matrix(vx1_range, vy1_range, n_steps, dt, t_start, masses_arr):
    nx = len(vx1_range)
    ny = len(vy1_range)
    result_matrix = np.zeros((nx, ny))

    pos = np.empty(9)
    pos[0] = 1.0
    pos[1] = 0.0
    pos[2] = 0.0
    pos[3] = -1.0
    pos[4] = 0.0
    pos[5] = 0.0
    pos[6] = 0.0
    pos[7] = 0.0
    pos[8] = 0.0

    for i in prange(nx):
        vx1 = vx1_range[i]
        for j in range(ny):
            vy1 = vy1_range[j]

            ca = np.zeros(18)
            cb = ca.copy()
            cb[0] += 1e-10
            ca[0:9] = pos
            ca[9], ca[10] = vx1, vy1  # v1
            ca[12], ca[13] = vx1, vy1  # v2
            ca[15], ca[16] = -2.0 * vx1, -2.0 * vy1  # v3

            _, _, _, _, _, proximity_list = solve_eq(
                ca, cb, n_steps, dt, t_start, masses_arr
            )

            d_min = np.min(proximity_list[1:])
            result_matrix[i, j] = -np.log10(d_min + 1e-15)

    return result_matrix


def generate_proximity_heatmap(cfg):
    vx1_range = np.linspace(0.20, 0.46, 130)
    vy1_range = np.linspace(0.51, 0.56, 1000)
    masses_arr = np.array(cfg.masses, dtype=np.float64)

    print("Starting Parallel Scan...")
    heatmap_data = compute_heatmap_matrix(
        vx1_range, vy1_range, cfg.n_steps, cfg.dt, cfg.t_start, masses_arr
    )

    with open("heatmap.csv", "w", newline="") as fa:
        header = ["vx", "vy", "log_d"]
        writer_fa = csv.DictWriter(fa, fieldnames=header)
        writer_fa.writeheader()

        for i, vx in enumerate(vx1_range):
            for j, vy in enumerate(vy1_range):
                writer_fa.writerow({"vx": vx, "vy": vy, "log_d": heatmap_data[i, j]})
            fa.write("\n")
    return True


def main():
    cfg = Config()
    cfg.load_from_txt("./IC/ic1.txt")
    cond_a, cond_b, planets_info = cfg.setup_systems()

    masses_arr = np.array(cfg.masses, dtype=np.float64)
    sol_a, sol_b, t_points, ln_deltas, lambdas, _ = solve_eq(
        cond_a, cond_b, cfg.n_steps, cfg.dt, cfg.t_start, masses_arr
    )
    generate_proximity_heatmap(cfg)
    # --- File Output ---
    save_results(sol_a, sol_b, t_points, ln_deltas, lambdas, cfg.eps)

    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")

    bodies_a = [
        Object(
            sol_a[i * 3 : i * 3 + 3, 0],
            sol_a[9 + i * 3 : 12 + i * 3, 0],
            cfg.masses[i],
            c,
        )
        for i, c in enumerate(["red", "green", "blue"])
    ]
    bodies_b = [
        Object(
            sol_b[i * 3 : i * 3 + 3, 0],
            sol_b[9 + i * 3 : 12 + i * 3, 0],
            cfg.masses[i],
            c,
        )
        for i, c in enumerate(["red", "green", "blue"])
    ]

    for b in bodies_a:
        b.init_plot(ax1)
    for b in bodies_b:
        b.init_plot(ax2)

    def animate(i):
        all_plots = []
        curr_x = sol_a[0, i]
        curr_y = sol_a[1, i]
        curr_z = sol_a[2, i]

        for j, b in enumerate(bodies_a):
            all_plots += b.draw(i, sol_a[j * 3], sol_a[j * 3 + 1], sol_a[j * 3 + 2])
        for j, b in enumerate(bodies_b):
            all_plots += b.draw(i, sol_b[j * 3], sol_b[j * 3 + 1], sol_b[j * 3 + 2])

        print(f"\rFrame: {i} | Sim Time = {t_points[i]:.2f}", end="")
        r = 4

        for ax in [ax1, ax2]:
            ax.set_xlim(-r, r)
            ax.set_ylim(-r, r)
            ax.set_zlim(-r, r)
            # ax.set_xlim(curr_x - r, curr_x + r)
            # ax.set_ylim(curr_y - r, curr_y + r)
            # ax.set_zlim(curr_z - r, curr_z + r)

        return all_plots

    # ani = FuncAnimation(fig, animate, frames=cfg.n_steps, interval=20)
    # plt.show()


if __name__ == "__main__":
    main()
