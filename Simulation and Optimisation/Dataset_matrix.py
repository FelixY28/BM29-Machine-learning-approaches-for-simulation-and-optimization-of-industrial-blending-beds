import pandas as pd
import numpy as np
import math
from scipy.ndimage import gaussian_filter
from typing import Optional, Dict

# Assume the classes are already correctly imported
from bmh.benchmark.material_deposition import MaterialDeposition, Material, Deposition
from bmh.simulation.bsl_blending_simulator import BslBlendingSimulator
from bmh.helpers.math import stdev, weighted_avg_and_std
from bmh.helpers.stockpile_math import get_stockpile_height, get_stockpile_slice_volume
from bmh.benchmark.material_deposition import Material

# Function for a random walk
def random_walk(n, start=5, step_size=0.1):
    walk = [start]
    for _ in range(n - 1):
        step = np.random.uniform(-step_size, step_size)
        next_value = max(0, walk[-1] + step)
        walk.append(next_value)
    return np.array(walk)

# Function to calculate weighted average and standard deviation
def weighted_avg_and_std(values, weights):
    average = np.average(values, weights=weights)
    variance = np.average((values - average) ** 2, weights=weights)
    return average, math.sqrt(variance)

# ReclaimedMaterialEvaluator class
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

# Function to generate dataset with F1 and F2
def generate_dataset_with_f1_f2(file_id, output_folder):
    BED_SIZE_X = 59
    BED_SIZE_Z = 20

    deposition_timestamps = np.linspace(0, 100, 20)
    material_timestamps = np.linspace(0, 100, 50)
    x_min = BED_SIZE_Z * 0.5
    x_max = BED_SIZE_X - BED_SIZE_Z * 0.5
    x_positions = np.random.uniform(x_min,x_max,20)

    

    quality_values = random_walk(50, start=5, step_size=1)
    quality_values = gaussian_filter(quality_values, sigma=1)
    volume_values = np.ones(50) * 50

    material = Material.from_data(pd.DataFrame({
        'timestamp': material_timestamps,
        'volume': volume_values,
        'quality': quality_values
    }))

    deposition = Deposition.from_data(
        data=pd.DataFrame({
            'timestamp': deposition_timestamps,
            'x': x_positions,
            'z': [0.5 * BED_SIZE_Z] * len(x_positions)
        }),
        bed_size_x=BED_SIZE_X,
        bed_size_z=BED_SIZE_Z,
        reclaim_x_per_s=6
    )

    material_deposition = MaterialDeposition(material=material, deposition=deposition)

    sim = BslBlendingSimulator(bed_size_x=BED_SIZE_X, bed_size_z=BED_SIZE_Z)
    reclaimed_material = sim.stack_reclaim(material_deposition)

    reclaimed_quality = reclaimed_material.data['quality']
    reclaimed_volume = reclaimed_material.data['volume']

    standard_deviation_f1 = weighted_avg_and_std(reclaimed_quality, reclaimed_volume)[1]

    evaluator = ReclaimedMaterialEvaluator(reclaimed=reclaimed_material, x_min = x_min, x_max= x_max)
    standard_deviation_f2 = evaluator.get_volume_stdev()

    output_data = pd.DataFrame({
        'y1': [standard_deviation_f1],
        'y2': [standard_deviation_f2],
        **{f'x{i+1}': [quality] for i, quality in enumerate(material.data['quality'])},
        **{f'x{i+51}': [x] for i, x in enumerate(deposition.data['x'])}
    })

    return output_data

# Generate 200,000 datasets and concatenate them
file_id = 1  
output_folder = '/mnt/d/UoM/DATA72000_ERP/Code V2/blending-evaluation-master'
combined_data = pd.DataFrame()

for _ in range(200000):
    dataset = generate_dataset_with_f1_f2(file_id=file_id, output_folder=output_folder)
    combined_data = pd.concat([combined_data, dataset], ignore_index=True)

# Save the combined dataset
combined_file_path = f'{output_folder}/matrix_f1_f2_200,000.csv'
combined_data.to_csv(combined_file_path, index=False)

print(f"Matrix dataset saved to {combined_file_path}")
