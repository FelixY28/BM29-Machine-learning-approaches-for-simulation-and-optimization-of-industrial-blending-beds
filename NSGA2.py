import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
from typing import Optional, Dict
from bmh.benchmark.material_deposition import MaterialDeposition, Material, Deposition
from bmh.simulation.bsl_blending_simulator import BslBlendingSimulator
from bmh.helpers.math import stdev, weighted_avg_and_std
from bmh.helpers.stockpile_math import get_stockpile_height, get_stockpile_slice_volume
from scipy.ndimage import gaussian_filter
from pymoo.visualization.scatter import Scatter
import matplotlib.pyplot as plt

# Define the original function for computing f1 and f2
class F1F2Problem(Problem):
    def __init__(self, bed_size_x, bed_size_z, x_min, x_max, initial_values):
        initial_values = np.asarray(initial_values)
        if initial_values.shape[0] != 50:
            raise ValueError("initial_values must have exactly 50 elements.")
        xl_x1_x50 = initial_values
        xu_x1_x50 = initial_values

        # Fix X1-X50 to the initial values by setting xl and xu to the same values for these variables
        # Allow X51-X70 to vary between 10 and 49
                # Allow X51-X70 to vary between 10 and 49
        xl_x51_x70 = np.full(20, 10)
        xu_x51_x70 = np.full(20, 49)

        # Concatenate the bounds for all variables
        xl = np.concatenate((xl_x1_x50, xl_x51_x70))
        xu = np.concatenate((xu_x1_x50, xu_x51_x70))

        # Initialize the problem with the defined bounds
        super().__init__(n_var=70, n_obj=2, n_constr=0, xl=xl, xu=xu)
        
        # Store additional problem-specific parameters
        self.bed_size_x = bed_size_x
        self.bed_size_z = bed_size_z
        self.x_min = x_min
        self.x_max = x_max

    def _evaluate(self, X, out, *args, **kwargs):
        f1_values = []
        f2_values = []
        
        for i in range(X.shape[0]):
            quality_values = X[i, :50]
            x_positions = X[i, 50:70]

            # Generate material and deposition data
            material_timestamps = np.linspace(0, 100, 50)
            volume_values = np.ones(50) * 50

            material = Material.from_data(pd.DataFrame({
                'timestamp': material_timestamps,
                'volume': volume_values,
                'quality': quality_values
            }))

            deposition_timestamps = np.linspace(0, 100, 20)
            deposition = Deposition.from_data(
                data=pd.DataFrame({
                    'timestamp': deposition_timestamps,
                    'x': x_positions,
                    'z': [0.5 * self.bed_size_z] * len(x_positions)
                }),
                bed_size_x=self.bed_size_x,
                bed_size_z=self.bed_size_z,
                reclaim_x_per_s=6
            )

            material_deposition = MaterialDeposition(material=material, deposition=deposition)

            sim = BslBlendingSimulator(bed_size_x=self.bed_size_x, bed_size_z=self.bed_size_z)
            reclaimed_material = sim.stack_reclaim(material_deposition)

            # Calculate f1 and f2
            reclaimed_quality = reclaimed_material.data['quality']
            reclaimed_volume = reclaimed_material.data['volume']
            f1 = weighted_avg_and_std(reclaimed_quality, reclaimed_volume)[1]

            evaluator = ReclaimedMaterialEvaluator(reclaimed=reclaimed_material, x_min=self.x_min, x_max=self.x_max)
            f2 = evaluator.get_volume_stdev()

            # Debugging checks
            if np.isnan(f1) or np.isnan(f2) or np.isinf(f1) or np.isinf(f2):
                print(f"Invalid f1 or f2 detected: f1={f1}, f2={f2}")
                f1 = np.inf
                f2 = np.inf

            f1_values.append(f1)
            f2_values.append(f2)

        out["F"] = np.column_stack([f1_values, f2_values])

class ReclaimedMaterialEvaluator:
    def __init__(self, reclaimed: Material, x_min: Optional[float] = None, x_max: Optional[float] = None):
        self.reclaimed = reclaimed
        self.x_min = x_min
        self.x_max = x_max

        # Caches
        self._parameter_stdev: Optional[Dict[str, float]] = None
        self._volume_stdev: Optional[float] = None

    def get_volume_stdev(self) -> float:
        if self._volume_stdev is None:
            ideal_df = self.reclaimed.data.copy()
            ideal_height = get_stockpile_height(volume=ideal_df['volume'].sum(), core_length=self.x_max - self.x_min)
            ideal_df['x_diff'] = (ideal_df['x'] - ideal_df['x'].shift(1)).fillna(0.0)
            ideal_df['volume'] = ideal_df.apply(
                lambda row: get_stockpile_slice_volume(
                    x=row['x'],
                    core_length=self.x_max - self.x_min,
                    height=ideal_height,
                    x_min=self.x_min,
                    x_diff=row['x_diff']
                ), axis=1
            )

            self._volume_stdev = stdev((ideal_df['volume'] - self.reclaimed.data['volume']).values)

        return self._volume_stdev

# Parameters for the problem
BED_SIZE_X = 59
BED_SIZE_Z = 20
x_min = BED_SIZE_Z * 0.5
x_max = BED_SIZE_X - BED_SIZE_Z * 0.5

# Instantiate the problem with the original objective functions
initial_values = np.array([4.793837218, 4.767604423, 5.105982752, 5.577101236, 5.623442779, 5.205666619,
                           4.804107196, 4.74872685, 4.789729315, 4.548653149, 4.278148707, 4.338010075,
                           4.693351334, 5.032509274, 5.180082922, 5.389772449, 5.614655856, 5.47497079,
                           5.028901924, 4.711435281, 4.719157143, 4.677107986, 4.354806347, 3.976992475,
                           3.669054807, 3.462731499, 3.461861142, 3.533878255, 3.807587018, 4.291268453,
                           4.752757253, 5.026002025, 5.005269285, 4.887497353, 4.786402493, 4.982608236,
                           5.43795932, 5.727856293, 5.625004022, 5.295989211, 5.075470806, 5.155016123,
                           5.507488769, 5.714822865, 5.457396549, 5.171083733, 5.332959438, 5.773380216,
                           6.149479502, 6.403548406])
problem = F1F2Problem(bed_size_x=BED_SIZE_X, bed_size_z=BED_SIZE_Z, x_min=x_min, x_max=x_max, initial_values=initial_values)

# Define the NSGA-II Algorithm
algorithm = NSGA2(
    pop_size=100,  # Further increase population size
    sampling=FloatRandomSampling(),
    crossover=SBX(prob=0.95, eta=10),  # Increase crossover probability and decrease eta
    mutation=PolynomialMutation(prob=0.25, eta=25)  # Increase mutation rate
)

# Set up colors for the different runs
colors = ['blue', 'green', 'red', 'orange', 'purple']

# Set a fixed seed for all runs to make them identical
fixed_seed = 1

# Create a figure for the combined plot
plt.figure(figsize=(10, 7))

# Example of plotting with labels for the legend
for i in range(5):
    # Assume res.F contains the Pareto front results from an optimization
    res = minimize(problem, algorithm, ('n_gen', 100), seed=fixed_seed + i, save_history=True, verbose=True)
    
    # Plot the Pareto front with a label for the legend
    plt.scatter(res.F[:, 0], res.F[:, 1], label=f'Run {i+1}')

# Add labels and legend
plt.xlabel('Objective 1 (f1)')
plt.ylabel('Objective 2 (f2)')
plt.title('Pareto Fronts from 5 Runs of NSGA-II')

# Ensure the legend is only created if there are valid labels
if plt.gca().get_legend_handles_labels()[1]:  # Check if there are labels
    plt.legend()

# Save the plot as a PNG file
plt.savefig("pareto_frontier_combined.png", dpi=300)