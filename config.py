import math
# =====================================================================
# CURRICULUM / RESET PARAMETERS
# =====================================================================
INIT_NOISE_DEFAULT = 0.01
MAX_INIT_NOISE_DEFAULT = math.pi
VELOCITY_NOISE_RATIO = 0.2
# =====================================================================
# PHYSICAL PARAMETERS (Matches the smaller Baek et al. style setup)
# =====================================================================
CART_MASS = 1.0
M1 = 0.2       # bob-1 mass (kg)
M2 = 0.2       # bob-2 mass (kg)
M3 = 0.2       # bob-3 mass (kg)
L1 = 0.4      # rod-1 length (m)
L2 = 0.4       # rod-2 length (m)
L3 = 0.4       # rod-3 length (m)
DAMPING = 0.5  # cart friction
GRAVITY = 9.81

# =====================================================================
# ENVIRONMENT LIMITS
# =====================================================================
DT = 1.0 / 120.0
MAX_FORCE = 50.0

MAX_STEPS = 1500

FORCE_LIMIT = 50.0
X_LIMIT     = 10.0   
SWINGUP_OFF_TRACK_PENALTY = X_LIMIT 
X_THRESHOLD = X_LIMIT 

BALANCE_INIT_NOISE_START = 0.01   # near-upright, easy
BALANCE_INIT_NOISE_MAX = 0.25     # hardest spawn condition you're targeting
BALANCE_INIT_X_NOISE = 0.5
BALANCE_INIT_VEL_NOISE = 0.5

# ====================================================================
# PHASE-AWARE SWING-UP REWARD TUNING
# =====================================================================
SWINGUP_HEIGHT_WEIGHT_1 = 1.0   
SWINGUP_HEIGHT_WEIGHT_2 = 1.5    
SWINGUP_HEIGHT_WEIGHT_3 = 2.0    

SWINGUP_POSITION_PENALTY_COEF = 0.05
SWINGUP_EFFORT_PENALTY_COEF = 0.1  # Increased from 0.0005
SWINGUP_OFF_TRACK_PENALTY = X_LIMIT 
SWINGUP_CART_VELOCITY_PENALTY_COEF = 0.05 

# Stabilization (only active near top)
SWINGUP_STILLNESS_WEIGHT = 20.0 # was 2
SWINGUP_STAB_HEIGHT_START = 3.0   # near_top ramps from 0.0 here...
SWINGUP_STAB_HEIGHT_END = 4.5     # ...to 1.0 here (max height)

# Strict Success Criteria
SUCCESS_ANGLE_TOL = 0.15          # ~8.6 degrees
SUCCESS_OMEGA_TOL = 0.5           # rad/s
SUCCESS_X_TOL = 1.0               # meters
SUCCESS_XD_TOL = 0.5              # m/s
SUCCESS_STEPS_REQ = 50            # ~0.42 seconds at 120Hz
SUCCESS_BONUS = 5.0

# Hard Angular Velocity Limit
SWINGUP_MAX_OMEGA = 40.0
SWINGUP_MAX_OMEGA_PENALTY = 1.0  # Massive negative hit
# Hard Energy Limit
SWINGUP_ENERGY_LIMIT_MULT = 2.5
SWINGUP_ENERGY_EXCESS_PENALTY = 0.02

# Link Alignment Reward (Positive reward for being straight near the top)
ALIGNMENT_REWARD_WEIGHT = 1.0  # Max reward per pair if perfectly aligned
ALIGNMENT_THRESHOLD_DEG = 10.0
ALIGNMENT_THRESHOLD_RAD = math.radians(ALIGNMENT_THRESHOLD_DEG)

# Link Alignment Activation (Radians away from hanging down)
ALIGNMENT_ACTIVATION_THRESHOLD_DEG = 50.0
ALIGNMENT_ACTIVATION_THRESHOLD_RAD = math.radians(ALIGNMENT_ACTIVATION_THRESHOLD_DEG)

# =====================================================================
# DERIVED CONSTANTS (Calculated automatically based on above)
# =====================================================================
E_TARGET = GRAVITY * (M1 * L1 + M2 * (L1 + L2) + M3 * (L1 + L2 + L3))
Y_MAX = L1 + L2 + L3
Y_MIN = -(L1 + L2 + L3)