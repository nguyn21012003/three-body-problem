import numpy as np
import pandas as pd


class Config:
    def __init__(self):
        self.t_start = 0
        self.t_end = 100
        self.n_steps = 2000
        self.eps = 1e-1

        self.masses = [0, 0, 0]
        self.ip1 = None
        self.ip2 = None
        self.ip3 = None
        self.iv1 = None
        self.iv2 = None
        self.iv3 = None

    def load_from_txt(self, filename):
        df = pd.read_csv(filename)

        m1 = float(df.iloc[0]["mass"])
        m2 = float(df.iloc[1]["mass"])
        m3 = float(df.iloc[2]["mass"])
        self.masses = (m1, m2, m3)
        self.ip1 = df.iloc[0][["px", "py", "pz"]].values.astype(float)
        self.ip2 = df.iloc[1][["px", "py", "pz"]].values.astype(float)
        self.ip3 = df.iloc[2][["px", "py", "pz"]].values.astype(float)

        self.iv1 = df.iloc[0][["vx", "vy", "vz"]].values.astype(float)
        self.iv2 = df.iloc[1][["vx", "vy", "vz"]].values.astype(float)
        self.iv3 = df.iloc[2][["vx", "vy", "vz"]].values.astype(float)

        print(f"Data loaded successfully from {filename}")

    @property
    def dt(self):
        return (self.t_end - self.t_start) / (self.n_steps - 1)

    def setup_systems(self):
        cond_a = np.array(
            [self.ip1, self.ip2, self.ip3, self.iv1, self.iv2, self.iv3]
        ).ravel()
        ip1b = self.ip1 + np.array([self.eps, 0.0, 0.0])
        cond_b = np.array(
            [ip1b, self.ip2, self.ip3, self.iv1, self.iv2, self.iv3]
        ).ravel()

        planets_info = {
            "Object 1": {"pos": self.ip1, "velo": self.iv1},
            "Object 2": {"pos": self.ip2, "velo": self.iv2},
            "Object 3": {"pos": self.ip3, "velo": self.iv3},
        }
        return cond_a, cond_b, planets_info
