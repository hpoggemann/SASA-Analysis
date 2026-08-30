# convert kcal/mole to eV
KCAL_TO_EV = 0.04336  

# File names
FN_IN_PRE = "in.pre"
FN_IN_TEMPLATE = "in.template"
FN_ELEM_LIBRARY = "element_library.txt"
FN_FF_PARAMS = "ff_params.dat"
FN_DUMP_COM = "dump_com.dat"
FN_SPEC = "spec.xyz"
FN_SASA_XYZ = "sasa.xyz"
FN_RESIDUE_LIST = "residuelist.txt"
FN_ETOT = "etot"
FN_TRAJ = "traj.lmp"
FN_THERMOLOG = "thermolog1"
DATA_FILE = "data.macromol"

# Fixed seed for reproducibility in the C extension
SAS_SEED = 38572111

# VdW radii in Angstroms - hierarchical selection: Bondi > Batsanov > Alvarez
RADII_MAP = {
    # Period 1
    'H': 1.20,   # Hydrogen (Bondi)
    'He': 1.40,  # Helium (Bondi)

    # Period 2
    'Li': 1.81,  # Lithium (Bondi)
    'Be': 1.9,   # Beryllium (Batsanov)
    'B': 1.8,    # Boron (Batsanov)
    'C': 1.70,   # Carbon (Bondi)
    'N': 1.55,   # Nitrogen (Bondi)
    'O': 1.52,   # Oxygen (Bondi)
    'F': 1.47,   # Fluorine (Bondi)
    'Ne': 1.54,  # Neon (Bondi)

    # Period 3
    'Na': 2.27,  # Sodium (Bondi)
    'Mg': 1.73,  # Magnesium (Bondi)
    'Al': 2.1,   # Aluminum (Batsanov)
    'Si': 2.10,  # Silicon (Batsanov)
    'P': 1.80,   # Phosphorus (Bondi)
    'S': 1.80,   # Sulfur (Bondi)
    'Cl': 1.75,  # Chlorine (Bondi)
    'Ar': 1.88,  # Argon (Bondi)

    # Period 4
    'K': 2.75,   # Potassium (Bondi)
    'Ca': 2.4,   # Calcium (Batsanov)
    'Sc': 2.3,   # Scandium (Batsanov)
    'Ti': 2.15,  # Titanium (Batsanov)
    'V': 2.05,   # Vanadium (Batsanov)
    'Cr': 2.05,  # Chromium (Batsanov)
    'Mn': 2.05,  # Manganese (Batsanov)
    'Fe': 2.05,  # Iron (Batsanov)
    'Co': 2.0,   # Cobalt (Batsanov)
    'Ni': 1.63,  # Nickel (Bondi)
    'Cu': 1.40,  # Copper (Bondi)
    'Zn': 1.39,  # Zinc (Bondi)
    'Ga': 1.87,  # Gallium (Bondi)
    'Ge': 2.1,   # Germanium (Batsanov)
    'As': 1.85,  # Arsenic (Bondi)
    'Se': 1.90,  # Selenium (Bondi)
    'Br': 1.85,  # Bromine (Bondi)
    'Kr': 2.02,  # Krypton (Bondi)

    # Period 5
    'Rb': 2.9,   # Rubidium (Batsanov)
    'Sr': 2.55,  # Strontium (Batsanov)
    'Y': 2.4,    # Yttrium (Batsanov)
    'Zr': 2.3,   # Zirconium (Batsanov)
    'Nb': 2.15,  # Niobium (Batsanov)
    'Mo': 2.1,   # Molybdenum (Batsanov)
    'Tc': 2.05,  # Technetium (Batsanov)
    'Ru': 2.05,  # Ruthenium (Batsanov)
    'Rh': 2.0,   # Rhodium (Batsanov)
    'Pd': 1.63,  # Palladium (Bondi)
    'Ag': 1.72,  # Silver (Bondi)
    'Cd': 1.58,  # Cadmium (Bondi)
    'In': 1.93,  # Indium (Bondi)
    'Sn': 2.17,  # Tin (Bondi)
    'Sb': 2.2,   # Antimony (Batsanov)
    'Te': 2.06,  # Tellurium (Bondi)
    'I': 1.98,   # Iodine (Bondi)
    'Xe': 2.16,  # Xenon (Bondi)

    # Period 6
    'Cs': 3.0,   # Cesium (Batsanov)
    'Ba': 2.7,   # Barium (Batsanov)
    'La': 2.5,   # Lanthanum (Batsanov)
    'Ce': 2.88,  # Cerium (Alvarez)
    'Pr': 2.92,  # Praseodymium (Alvarez)
    'Nd': 2.95,  # Neodymium (Alvarez)
    'Pm': 2.90,  # Promethium (Alvarez)
    'Sm': 2.90,  # Samarium (Alvarez)
    'Eu': 2.87,  # Europium (Alvarez)
    'Gd': 2.83,  # Gadolinium (Alvarez)
    'Tb': 2.79,  # Terbium (Alvarez)
    'Dy': 2.87,  # Dysprosium (Alvarez)
    'Ho': 2.81,  # Holmium (Alvarez)
    'Er': 2.83,  # Erbium (Alvarez)
    'Tm': 2.79,  # Thulium (Alvarez)
    'Yb': 2.80,  # Ytterbium (Alvarez)
    'Lu': 2.74,  # Lutetium (Alvarez)
    'Hf': 2.25,  # Hafnium (Batsanov)
    'Ta': 2.2,   # Tantalum (Batsanov)
    'W': 2.1,    # Tungsten (Batsanov)
    'Re': 2.05,  # Rhenium (Batsanov)
    'Os': 2.0,   # Osmium (Batsanov)
    'Ir': 2.0,   # Iridium (Batsanov)
    'Pt': 1.72,  # Platinum (Bondi)
    'Au': 1.66,  # Gold (Bondi)
    'Hg': 1.55,  # Mercury (Bondi)
    'Tl': 1.96,  # Thallium (Bondi)
    'Pb': 2.02,  # Lead (Bondi)
    'Bi': 2.3,   # Bismuth (Batsanov)
    'Po': 1.97,  # Polonium (Alvarez)
    'At': 2.02,  # Astatine (Alvarez)
    'Rn': 2.20,  # Radon (Alvarez)

    # Period 7 - Actinides (mostly Alvarez, Th from Batsanov)
    'Fr': 3.48,  # Francium (Alvarez)
    'Ra': 2.83,  # Radium (Alvarez)
    'Ac': 2.8,   # Actinium (Alvarez)
    'Th': 2.4,   # Thorium (Batsanov)
    'Pa': 2.88,  # Protactinium (Alvarez)
    'U': 1.86,   # Uranium (Bondi)
    'Np': 2.82,  # Neptunium (Alvarez)
    'Pu': 2.81,  # Plutonium (Alvarez)
    'Am': 2.83,  # Americium (Alvarez)
    'Cm': 3.05,  # Curium (Alvarez)
    'Bk': 3.4,   # Berkelium (Alvarez)
    'Cf': 3.05,  # Californium (Alvarez)
    'Es': 2.7,   # Einsteinium (Alvarez)

    # Superheavy elements (estimated)
    'Fm': 2.45,  # Fermium (estimated)
    'Md': 2.46,  # Mendelevium (estimated)
    'No': 2.46,  # Nobelium (estimated)
    'Lr': 2.46,  # Lawrencium (estimated)
    'Rf': 2.30,  # Rutherfordium (estimated)
    'Db': 2.30,  # Dubnium (estimated)
    'Sg': 2.30,  # Seaborgium (estimated)
    'Bh': 2.30,  # Bohrium (estimated)
    'Hs': 2.30,  # Hassium (estimated)
    'Mt': 2.30,  # Meitnerium (estimated)
    'Ds': 2.30,  # Darmstadtium (estimated)
    'Rg': 2.30,  # Roentgenium (estimated)
    'Cn': 2.30,  # Copernicium (estimated)
    'Nh': 2.30,  # Nihonium (estimated)
    'Fl': 2.30,  # Flerovium (estimated)
    'Mc': 2.30,  # Moscovium (estimated)
    'Lv': 2.30,  # Livermorium (estimated)
    'Ts': 2.30,  # Tennessine (estimated)
    'Og': 2.30,  # Oganesson (estimated)
}