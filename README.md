Welcome to the GitHub repository for 'Machine learning approaches for simulation and optimisation of industrial blending beds', where you will find all the necessary resources to generate simulated datasets, train models, and perform optimization processes effectively. This repository is structured into two main sections: Environment Setup and Packages, and Simulation and Optimisation. Each section is designed to ensure that users can reproduce our research and extend the functionalities based on their specific requirements.

Environment Setup and Packages:

This section includes detailed instructions and the necessary files to set up the environment and install the required packages to run the project code.

Files and Their Functions:
README.md: Contains environment setup instructions and sample code to generate datasets. The actual datasets used in this project were built upon these samples, enhanced with additional data analysis techniques and tailored to specific project objectives.
requirements-dev.txt: Lists all pre-packaged libraries and dependencies required to run the dataset generation code. Apply the commands in this file directly to install the necessary packages.
BlendingSimulatorLibPython.cpp: Used in dataset generation, this file computes reclaimed_quality from input material_quality.
Additional Python files such as bsl_blending_simulator.py, homogenization_problem.py, material_deposition.py, and reclaimed_material_evaluator.py are crucial for generating deposition data, and calculating target variables F1 and F2.

Simulation and Optimisation:

This folder includes scripts for generating simulation data, training models, and conducting optimizations.

Key Scripts:
Dataset_matrix.py: Generates data necessary for model training and converts variables like y1, y2, material quality, and deposition_x into a matrix format, facilitating easy model training. This script allows for data volume expansion as needed through random walks and loops.
Modeling_y1.ipynb: Contains exploratory analysis on the effectiveness of the dataset and focuses on modeling with y1. Different models are tried and evaluated, with a detailed analysis provided in the methodology and results sections of the report.
NSGAII.py: Performs optimization on the original objective functions without embedding any model. This script allows optimization computations to be carried out without an actual dataset.
model_y1y2NSGA_II_ipynb.ipynb: Trains and visualizes the effects of the four selected models (Lasso, XGBoost, MLP, LSTM) and integrates them into the NSGA-II optimisation process. The outcomes of different optimisation runs are displayed through visualization, with the detailed results discussed in the report.
