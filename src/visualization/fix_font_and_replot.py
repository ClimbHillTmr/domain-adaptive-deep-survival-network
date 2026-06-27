import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import matplotlib.pyplot as plt
import matplotlib as mpl

# Override Type 42 fonts globally to prevent PDF corruption issues on some Linux systems
mpl.rcParams['pdf.fonttype'] = 3
mpl.rcParams['ps.fonttype'] = 3

from src.visualization.generate_architecture import generate_architecture_diagram
from src.visualization.generate_phenotype_transition import generate_phenotype_transition
from src.evaluate.generate_clinical_plots import main as generate_clinical

print("Regenerating Architecture Diagram...")
generate_architecture_diagram()

print("Regenerating Phenotype Transition Diagram...")
generate_phenotype_transition()

print("Regenerating Clinical Plots...")
generate_clinical()

print("All plots regenerated with compatible font settings.")
