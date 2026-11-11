"""Hyper-parameters in one place.

Values marked [paper] come from Table 1 of Ramya et al. (2025).
Values marked [ours] are not specified in the paper (or are unusable as printed),
so they are our own documented choices.
"""

# ----------------------------------------------------------------- data
TARGET = "DBO-S"            # effluent BOD (mg/L)
N_FEATURES = 6              # [paper] 6 inputs survive feature selection (600 MF params = 6 x 50 rules x 2)
SPLIT = (0.70, 0.15, 0.15)  # [paper] train / validation / test
IQR_K = 1.5                 # [ours] Tukey fence used instead of the AutoGAN anomaly handling

# ----------------------------------------------------------------- ANFIS
SIGMA_MIN = 0.05            # [paper] minimum Gaussian width
UNIFORM_CENTER = (0.1, 0.9) # [paper] range of MF means for the non-SOM ("uniform") initialisation
UNIFORM_SIGMA = (0.1, 0.5)  # [paper] range of MF std for the uniform initialisation
SINGLE_GRID = (3, 3)        # [ours] single ANFIS uses a 3x3 SOM = 9 rules (paper is ambiguous: 9 / 10 / 25)
PANFIS_GRIDS = ((3, 3), (4, 4), (5, 5))   # [paper] 9, 16 and 25 rules
RIDGE = 0.3                 # [ours] ridge term of the least-squares consequent estimate used inside the
                            #        metaheuristic fitness; picked on validation data of development seeds
                            #        100-104 (experiments/baseline/select_design.py), never on test data

# ----------------------------------------------------------------- SOM
SOM_SIGMA = 0.3             # [paper] neighbourhood radius
SOM_LR = 0.5                # [paper] learning rate
SOM_ITER = 100              # [paper] iterations (we read it as passes over the training set)

# ----------------------------------------------------------------- metaheuristics
PSO = dict(n_particles=50, n_iter=40,        # [paper]
           w=0.7, c1=1.5, c2=1.5)            # [ours] inertia / cognitive / social (not given in the paper)
GA = dict(pop_size=50, n_gen=40, alpha=0.5,  # [paper] blend crossover alpha
          p_mut=0.1, tournament=3,           # [paper]
          sigma_mut=0.1)                     # [ours] paper prints sigma=1, meaningless for parameters in [0,1]
ACO = dict(n_ants=10, n_iter=50, rho=0.5,    # [paper]
           tau0=0.1)                         # [ours] initial pheromone (= initial step size)
INIT_SPREAD = 0.05          # [ours] std of the noise that spreads the initial swarm around the SOM solution

# ----------------------------------------------------------------- gradient training
ADAM_LR = 0.01              # [paper]
MAX_EPOCHS = 1000           # [paper]
PATIENCE = 50               # [ours] early-stopping patience (paper: "early stopping", no value)
LR_PATIENCE = 20            # [ours] ReduceLROnPlateau patience (paper: "learning rate scheduler")
ATT_HIDDEN = 16             # [ours] hidden units of each attention network
