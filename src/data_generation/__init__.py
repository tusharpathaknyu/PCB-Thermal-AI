# Make data_generation a package
from .thermal_solver import HeatEquationSolver, ThermalProperties
from .pcb_generator import PCBGenerator
from .dataset_generator import DatasetGenerator, DatasetConfig, quick_generate

__all__ = [
    'HeatEquationSolver',
    'ThermalProperties', 
    'PCBGenerator',
    'DatasetGenerator',
    'DatasetConfig',
    'quick_generate'
]
