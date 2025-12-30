"""
2D Steady-State Heat Equation Solver for PCB Thermal Simulation

Uses finite difference method to solve:
    ∇·(k∇T) + Q = 0

Where:
    k = thermal conductivity (varies spatially for copper/FR4)
    T = temperature
    Q = heat source (power dissipation from components)

Boundary conditions:
    - Convective: -k∂T/∂n = h(T - T_ambient) on exposed surfaces
    - Adiabatic: ∂T/∂n = 0 on board edges (optional)
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class ThermalProperties:
    """Material thermal properties"""
    k_copper: float = 385.0      # W/(m·K) - copper thermal conductivity
    k_fr4: float = 0.3           # W/(m·K) - FR4 thermal conductivity
    h_conv: float = 10.0         # W/(m²·K) - natural convection coefficient
    t_ambient: float = 25.0      # °C - ambient temperature
    board_thickness: float = 1.6e-3  # m - typical PCB thickness


class HeatEquationSolver:
    """
    Finite difference solver for 2D steady-state heat equation on PCB.
    
    The PCB is modeled as a 2D domain with:
    - Variable thermal conductivity (copper traces vs FR4 substrate)
    - Localized heat sources (components)
    - Convective boundary conditions (top and bottom surfaces)
    """
    
    def __init__(
        self,
        grid_size: Tuple[int, int] = (128, 128),
        physical_size: Tuple[float, float] = (0.1, 0.1),  # 10cm x 10cm
        properties: Optional[ThermalProperties] = None
    ):
        """
        Initialize the solver.
        
        Args:
            grid_size: (ny, nx) number of grid points
            physical_size: (height, width) in meters
            properties: Thermal properties dataclass
        """
        self.ny, self.nx = grid_size
        self.height, self.width = physical_size
        self.props = properties or ThermalProperties()
        
        # Grid spacing
        self.dx = self.width / (self.nx - 1)
        self.dy = self.height / (self.ny - 1)
        
        # Initialize fields
        self.conductivity = np.ones((self.ny, self.nx)) * self.props.k_fr4
        self.heat_source = np.zeros((self.ny, self.nx))
        
    def set_copper_regions(self, copper_mask: np.ndarray):
        """
        Set regions with copper (high thermal conductivity).
        
        Args:
            copper_mask: Boolean array (ny, nx) where True = copper
        """
        assert copper_mask.shape == (self.ny, self.nx), \
            f"Mask shape {copper_mask.shape} doesn't match grid {(self.ny, self.nx)}"
        
        # Effective conductivity considering copper layer thickness
        # Simplified: assume copper is distributed through thickness
        copper_fraction = 0.035e-3 / self.props.board_thickness  # 35μm copper / 1.6mm board
        k_effective = (
            copper_mask * (self.props.k_copper * copper_fraction + 
                          self.props.k_fr4 * (1 - copper_fraction)) +
            ~copper_mask * self.props.k_fr4
        )
        self.conductivity = k_effective
        
    def set_copper_density(self, density_map: np.ndarray):
        """
        Set copper density (0-1) for each cell.
        More realistic than binary copper mask.
        
        Args:
            density_map: Array (ny, nx) with values 0-1 representing copper fill
        """
        assert density_map.shape == (self.ny, self.nx)
        density_map = np.clip(density_map, 0, 1)
        
        copper_fraction = 0.035e-3 / self.props.board_thickness
        k_effective = (
            density_map * (self.props.k_copper * copper_fraction) +
            (1 - density_map * copper_fraction) * self.props.k_fr4
        )
        self.conductivity = k_effective
        
    def add_heat_source(
        self,
        center: Tuple[int, int],
        size: Tuple[int, int],
        power: float
    ):
        """
        Add a rectangular heat source (component).
        
        Args:
            center: (y, x) center position in grid coordinates
            size: (height, width) in grid cells
            power: Total power dissipation in Watts
        """
        cy, cx = center
        h, w = size
        
        # Calculate bounds
        y1 = max(0, cy - h // 2)
        y2 = min(self.ny, cy + h // 2 + 1)
        x1 = max(0, cx - w // 2)
        x2 = min(self.nx, cx + w // 2 + 1)
        
        # Distribute power over area
        area = (y2 - y1) * (x2 - x1) * self.dx * self.dy
        if area > 0:
            # Q is in W/m³, so divide by thickness
            q_volumetric = power / (area * self.props.board_thickness)
            self.heat_source[y1:y2, x1:x2] += q_volumetric
            
    def clear_heat_sources(self):
        """Reset all heat sources"""
        self.heat_source = np.zeros((self.ny, self.nx))
        
    def _build_system_matrix(self) -> Tuple[sparse.csr_matrix, np.ndarray]:
        """
        Build the sparse system matrix for the finite difference equations.
        
        Uses 5-point stencil for Laplacian with variable conductivity:
            ∂/∂x(k ∂T/∂x) + ∂/∂y(k ∂T/∂y) + Q = h_eff * (T - T_amb)
        
        The convection term is treated as a volumetric sink.
        """
        n = self.ny * self.nx
        
        # Effective convection coefficient (both surfaces)
        h_eff = 2 * self.props.h_conv / self.props.board_thickness
        
        # Build sparse matrix using COO format for efficiency
        rows = []
        cols = []
        data = []
        rhs = np.zeros(n)
        
        def idx(i, j):
            """Convert 2D index to 1D"""
            return i * self.nx + j
        
        for i in range(self.ny):
            for j in range(self.nx):
                node = idx(i, j)
                
                # Conductivity at interfaces (harmonic mean for stability)
                k_e = 2 * self.conductivity[i, j] * self.conductivity[i, min(j+1, self.nx-1)] / \
                      (self.conductivity[i, j] + self.conductivity[i, min(j+1, self.nx-1)] + 1e-10)
                k_w = 2 * self.conductivity[i, j] * self.conductivity[i, max(j-1, 0)] / \
                      (self.conductivity[i, j] + self.conductivity[i, max(j-1, 0)] + 1e-10)
                k_n = 2 * self.conductivity[i, j] * self.conductivity[min(i+1, self.ny-1), j] / \
                      (self.conductivity[i, j] + self.conductivity[min(i+1, self.ny-1), j] + 1e-10)
                k_s = 2 * self.conductivity[i, j] * self.conductivity[max(i-1, 0), j] / \
                      (self.conductivity[i, j] + self.conductivity[max(i-1, 0), j] + 1e-10)
                
                # Coefficients
                a_e = k_e / self.dx**2
                a_w = k_w / self.dx**2
                a_n = k_n / self.dy**2
                a_s = k_s / self.dy**2
                
                # Center coefficient (includes convection sink)
                a_p = a_e + a_w + a_n + a_s + h_eff
                
                # Handle boundaries (Neumann: zero flux at edges)
                if i == 0:  # South boundary
                    a_s = 0
                if i == self.ny - 1:  # North boundary
                    a_n = 0
                if j == 0:  # West boundary
                    a_w = 0
                if j == self.nx - 1:  # East boundary
                    a_e = 0
                
                # Recalculate center coefficient for boundaries
                a_p = a_e + a_w + a_n + a_s + h_eff
                
                # Fill matrix
                rows.append(node)
                cols.append(node)
                data.append(a_p)
                
                if j < self.nx - 1:  # East neighbor
                    rows.append(node)
                    cols.append(idx(i, j + 1))
                    data.append(-a_e)
                    
                if j > 0:  # West neighbor
                    rows.append(node)
                    cols.append(idx(i, j - 1))
                    data.append(-a_w)
                    
                if i < self.ny - 1:  # North neighbor
                    rows.append(node)
                    cols.append(idx(i + 1, j))
                    data.append(-a_n)
                    
                if i > 0:  # South neighbor
                    rows.append(node)
                    cols.append(idx(i - 1, j))
                    data.append(-a_s)
                
                # RHS: heat source + convection to ambient
                rhs[node] = self.heat_source[i, j] + h_eff * self.props.t_ambient
        
        A = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
        return A, rhs
    
    def solve(self) -> np.ndarray:
        """
        Solve the heat equation and return temperature field.
        
        Returns:
            temperature: Array (ny, nx) of temperatures in °C
        """
        A, rhs = self._build_system_matrix()
        T_flat = spsolve(A, rhs)
        temperature = T_flat.reshape((self.ny, self.nx))
        return temperature
    
    def get_hotspot_info(self, temperature: np.ndarray) -> dict:
        """
        Extract hotspot information from temperature field.
        
        Returns:
            dict with max_temp, hotspot_location, mean_temp, etc.
        """
        max_temp = float(np.max(temperature))
        max_idx = np.unravel_index(np.argmax(temperature), temperature.shape)
        
        return {
            'max_temp': max_temp,
            'hotspot_y': max_idx[0],
            'hotspot_x': max_idx[1],
            'mean_temp': float(np.mean(temperature)),
            'min_temp': float(np.min(temperature)),
            'temp_range': max_temp - float(np.min(temperature))
        }


def demo():
    """Quick demonstration of the solver"""
    # Create solver
    solver = HeatEquationSolver(
        grid_size=(64, 64),
        physical_size=(0.05, 0.05)  # 5cm x 5cm board
    )
    
    # Create simple copper pattern (ground plane with cutout)
    copper = np.ones((64, 64), dtype=bool)
    copper[20:44, 20:44] = False  # Cutout in center
    solver.set_copper_regions(copper)
    
    # Add heat source (component in center)
    solver.add_heat_source(center=(32, 32), size=(8, 8), power=1.0)  # 1W
    
    # Solve
    temperature = solver.solve()
    
    # Get hotspot info
    info = solver.get_hotspot_info(temperature)
    print(f"Max temperature: {info['max_temp']:.1f}°C")
    print(f"Mean temperature: {info['mean_temp']:.1f}°C")
    print(f"Hotspot at: ({info['hotspot_y']}, {info['hotspot_x']})")
    
    return temperature


if __name__ == "__main__":
    demo()
