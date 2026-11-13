import numpy as np
import time
from collections import deque
from dataclasses import dataclass

from ipycanvas import Canvas, hold_canvas


@dataclass
class Color:
    fill: str
    outline: str


@dataclass
class Molecule:
    name: str
    reaction_partner: str
    activation_energy: float
    color: Color
    radius: int
    mass: float = None


oxygen = Molecule(
    "oxygen", "nitrogen", activation_energy=200, color=Color(fill="#fdd835", outline="#ff8f00"), radius=16
)
nitrogen = Molecule(
    "nitrogen", "oxygen", activation_energy=200, color=Color(fill="#1e88e5", outline="#0d47a1"), radius=14
)
nitric_oxide = Molecule(
    "NO", None, activation_energy=0, color=Color(fill="#1abc9c", outline="#148f77"), radius=22
)


class Sphere:
    """A class to represent a sphere in a 2D space."""

    def __init__(self, x, y, vx, vy, radius, color: Color, mass=None, gravity=0.001):
        """Initialize the sphere with position, velocity, radius, color, mass, and gravity."""
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.radius = radius
        self.color = color
        self.mass = radius**2 if mass is None else mass
        self.gravity = gravity

    def move(self):
        self.vy += self.gravity  # Apply gravity to vertical velocity
        self.x += self.vx
        self.y += self.vy

    def check_wall_collision(self, width, height):
        if self.x - self.radius < 0 or self.x + self.radius > width:
            self.vx = -self.vx
        if self.y - self.radius < 0 or self.y + self.radius > height:
            self.vy = -self.vy

    def check_sphere_collision(self, other):
        dx = self.x - other.x
        dy = self.y - other.y
        distance = np.sqrt(dx**2 + dy**2)
        if distance < self.radius + other.radius:
            # Calculate new velocities based on conservation of momentum and kinetic energy
            nx = dx / distance
            ny = dy / distance
            p = 2 * (self.vx * nx + self.vy * ny - other.vx * nx - other.vy * ny) / (self.mass + other.mass)
            self.vx -= p * other.mass * nx
            self.vy -= p * other.mass * ny
            other.vx += p * self.mass * nx
            other.vy += p * self.mass * ny

            # Separate the spheres to prevent overlap
            overlap = 0.5 * (self.radius + other.radius - distance + 1)
            self.x += overlap * nx
            self.y += overlap * ny
            other.x -= overlap * nx
            other.y -= overlap * ny
            return "collision"
        return None

    def equilibriate_temperature(self, temperature):
        # on average kb*T energy per degree of freedom
        kb = 1
        v_equilibrium = np.sqrt(2 * kb * temperature / self.mass)
        alpha = 0.01
        direction = np.arctan2(self.vy, self.vx)
        self.vx = alpha * v_equilibrium * np.cos(direction) + self.vx * (1 - alpha)
        self.vy = alpha * v_equilibrium * np.sin(direction) + self.vy * (1 - alpha)


class ChemicalSphere(Sphere):
    """Extends sphere, so that it can react with other spheres."""

    def __init__(self, x, y, vx, vy, atom: Molecule, gravity=0.001):
        self.atom = atom
        super().__init__(x, y, vx, vy, atom.radius, atom.color, atom.mass, gravity)

    def check_sphere_collision(self, other):
        """Returns "collision", "reaction", or None."""
        if (
            not isinstance(other, ChemicalSphere)
            or self.atom == other.atom
            or not self.atom.reaction_partner == other.atom.name
        ):
            return super().check_sphere_collision(other)

        # calculate total kinetic energy
        self_kinetic_energy = 0.5 * self.mass * (self.vx**2 + self.vy**2)
        other_kinetic_energy = 0.5 * other.mass * (other.vx**2 + other.vy**2)
        total_kinetic_energy = self_kinetic_energy + other_kinetic_energy

        if total_kinetic_energy < self.atom.activation_energy:
            return super().check_sphere_collision(other)

        # determine if the spheres collide and react
        dx = self.x - other.x
        dy = self.y - other.y
        distance = np.sqrt(dx**2 + dy**2)
        if distance < self.radius + other.radius:
            self.atom = nitric_oxide
            self.x = (self.x + other.x) / 2
            self.y = (self.y + other.y) / 2
            self.vx = self.vx / 2
            self.vy = self.vy / 2
            self.radius = self.atom.radius
            self.color = self.atom.color
            self.mass = self.radius**2 if self.atom.mass is None else self.atom.mass

            other.x = -1000
            other.y = -1000
            return "reaction"
        return None


class Wall:
    """A class to represent a wall on our canvas."""

    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def check_collision(self, sphere):
        # We only reverse directions if a collision occured and not
        # if we are already moving away from the wall.

        # Check for vertical walls
        if self.x1 == self.x2:
            if self.x1 - sphere.radius < sphere.x < self.x1 + sphere.radius:
                if self.y1 < sphere.y < self.y2 or self.y2 < sphere.y < self.y1:
                    if (sphere.x < self.x1) and sphere.vx > 0:
                        sphere.vx = -sphere.vx
                    elif (sphere.x > self.x1) and sphere.vx < 0:
                        sphere.vx = -sphere.vx

        # Check for horizontal walls
        elif self.y1 == self.y2:
            if self.y1 - sphere.radius < sphere.y < self.y1 + sphere.radius:
                if self.x1 < sphere.x < self.x2 or self.x2 < sphere.x < self.x1:
                    if (sphere.y < self.y1) and sphere.vy > 0:
                        sphere.vy = -sphere.vy
                    elif (sphere.y > self.y1) and sphere.vy < 0:
                        sphere.vy = -sphere.vy


def wall_with_holes(x1, y1, x2, y2, holes):
    """Build a straight wall from (x1, y1) to (x2, y2), leaving gaps.

    holes: list of (start, end) positions along the wall, measured from (x1, y1).
    Returns a list of Wall pieces.
    """
    length = abs(x2 - x1) + abs(y2 - y1)  # walls are either horizontal or vertical
    ux = np.sign(x2 - x1)
    uy = np.sign(y2 - y1)

    pieces = []
    position = 0
    for start, end in sorted(holes):
        if start > position:
            pieces.append((position, start))
        position = max(position, end)
    if position < length:
        pieces.append((position, length))

    return [Wall(x1 + ux * a, y1 + uy * a, x1 + ux * b, y1 + uy * b) for a, b in pieces]


VESSELS = ["box", "divider", "divider_with_hole", "leaky_box"]


class Simulation:
    """The main simulation class that contains the canvas, spheres, and walls."""

    def __init__(self, canvas, width, height, num_spheres, temperature=100,
                 vessel="box", hole_size=80, chart_width=360, history_length=400):
        if vessel not in VESSELS:
            raise ValueError(f"vessel must be one of {VESSELS}, not {vessel!r}")

        self.width = width
        self.height = height
        self.temperature = temperature
        self.vessel = vessel
        self.hole_size = hole_size

        # one canvas for the molecules, one for the charts
        self.canvas = canvas(width=width, height=height)
        self.chart_canvas = canvas(width=chart_width, height=height)
        self.chart_width = chart_width

        self.walls = self.create_walls()
        self.spheres = self.create_spheres(num_spheres)

        # counters and their history for the live charts
        self.step = 0
        self.collisions = 0
        self.reactions = 0
        self.history = {
            "temperature": deque(maxlen=history_length),
            "collisions": deque(maxlen=history_length),
            "reactions": deque(maxlen=history_length),
            "product": deque(maxlen=history_length),
        }

    def create_spheres(self, num_spheres):
        spheres = []
        divided = self.vessel in ("divider", "divider_with_hole")
        middle = self.width / 2
        for _ in range(num_spheres):
            atom = np.random.choice([oxygen, nitrogen])
            if divided:
                # oxygen starts on the left, nitrogen on the right
                if atom is oxygen:
                    x = np.random.randint(20, int(middle) - 25)
                else:
                    x = np.random.randint(int(middle) + 25, self.width - 20)
            else:
                x = np.random.randint(20, self.width - 20)
            y = np.random.randint(20, self.height - 20)
            vx = np.sqrt(self.temperature) / 20 * np.random.uniform(-0.5, 0.5)
            vy = np.sqrt(self.temperature) / 20 * np.random.uniform(-0.5, 0.5)
            spheres.append(ChemicalSphere(x, y, vx, vy, atom))
        return spheres

    def create_side_walls(self):
        walls = [
            Wall(0, 0, self.width, 0),  # Top wall
            Wall(0, self.height, self.width, self.height),  # Bottom wall
            Wall(0, 0, 0, self.height),  # Left wall
            Wall(self.width, 0, self.width, self.height),  # Right wall
        ]
        return walls

    def create_walls(self):
        """Build the walls for the chosen reaction vessel."""
        w, h = self.width, self.height
        middle = w / 2
        hole = self.hole_size

        if self.vessel == "box":
            return self.create_side_walls()

        if self.vessel == "divider":
            return self.create_side_walls() + [Wall(middle, 0, middle, h)]

        if self.vessel == "divider_with_hole":
            centre = h / 2
            return self.create_side_walls() + wall_with_holes(
                middle, 0, middle, h, holes=[(centre - hole / 2, centre + hole / 2)]
            )

        if self.vessel == "leaky_box":
            # one hole in the middle of the top, left and right walls
            return (
                wall_with_holes(0, 0, w, 0, holes=[(w / 2 - hole / 2, w / 2 + hole / 2)])
                + [Wall(0, h, w, h)]
                + wall_with_holes(0, 0, 0, h, holes=[(h / 2 - hole / 2, h / 2 + hole / 2)])
                + wall_with_holes(w, 0, w, h, holes=[(h / 2 - hole / 2, h / 2 + hole / 2)])
            )

    def update(self):
        for sphere in self.spheres:
            sphere.move()
            for wall in self.walls:
                wall.check_collision(sphere)
            for other in self.spheres:
                if sphere != other:
                    event = sphere.check_sphere_collision(other)
                    if event == "collision":
                        self.collisions += 1
                    elif event == "reaction":
                        self.reactions += 1

            sphere.equilibriate_temperature(self.temperature)

        # remove spheres that are out of bounds (inplace, https://stackoverflow.com/questions/1207406)
        self.spheres[:] = [
            sp for sp in self.spheres if not (abs(sp.x) > self.width * 2 or abs(sp.y) > self.height * 2)
        ]

        self.step += 1
        self.record()

    # ---------------------------------------------------- values for the charts

    def in_vessel(self, sphere):
        return 0 <= sphere.x <= self.width and 0 <= sphere.y <= self.height

    def measured_temperature(self):
        """Mean kinetic energy per molecule (kb = 1, 2 degrees of freedom -> <E> = kb*T)."""
        inside = [sp for sp in self.spheres if self.in_vessel(sp)]
        if not inside:
            return 0.0
        return float(np.mean([0.5 * sp.mass * (sp.vx**2 + sp.vy**2) for sp in inside]))

    def product_count(self):
        return sum(1 for sp in self.spheres if self.in_vessel(sp) and sp.atom is nitric_oxide)

    def record(self):
        self.history["temperature"].append(self.measured_temperature())
        self.history["collisions"].append(self.collisions)
        self.history["reactions"].append(self.reactions)
        self.history["product"].append(self.product_count())

    # ----------------------------------------------------------------- drawing

    def draw(self):
        with hold_canvas():
            self.canvas.clear()
            for sphere in self.spheres:
                self.canvas.fill_style = sphere.color.fill
                self.canvas.fill_circle(sphere.x, sphere.y, sphere.radius)
                self.canvas.stroke_style = sphere.color.outline
                self.canvas.stroke_circle(sphere.x, sphere.y, sphere.radius)

            self.canvas.stroke_style = "black"
            self.canvas.line_width = 2
            for wall in self.walls:
                self.canvas.stroke_line(wall.x1, wall.y1, wall.x2, wall.y2)

    def draw_charts(self):
        """Draw four small live line charts on the chart canvas."""
        panels = [
            ("Temperature", "temperature", "#e53935"),
            ("Collisions (total)", "collisions", "#5e35b1"),
            ("Reactions (total)", "reactions", "#fb8c00"),
            ("Product NO (in vessel)", "product", nitric_oxide.color.outline),
        ]
        c = self.chart_canvas
        panel_height = self.height / len(panels)

        with hold_canvas():
            c.clear()
            c.font = "12px sans-serif"
            for i, (title, key, color) in enumerate(panels):
                top = i * panel_height
                values = list(self.history[key])
                current = values[-1] if values else 0

                # title with the current value
                c.fill_style = "black"
                c.fill_text(f"{title}: {current:.0f}", 10, top + 15)

                # chart frame
                x0, x1 = 10, self.chart_width - 10
                y0, y1 = top + 22, top + panel_height - 8
                c.stroke_style = "#cccccc"
                c.line_width = 1
                c.stroke_rect(x0, y0, x1 - x0, y1 - y0)

                # the line
                if len(values) < 2:
                    continue
                v_max = max(values) or 1
                n = len(values)
                points = [
                    (x0 + (x1 - x0) * j / (n - 1), y1 - (y1 - y0) * v / v_max)
                    for j, v in enumerate(values)
                ]
                c.stroke_style = color
                c.line_width = 2
                c.stroke_lines(points)

    def run(self, steps=None, chart_every=10):
        """Run the simulation. steps=None runs until you press the notebook's stop button."""
        count = 0
        while steps is None or count < steps:
            self.update()
            self.draw()
            if self.step % chart_every == 0:
                self.draw_charts()
            time.sleep(0.002)
            count += 1
