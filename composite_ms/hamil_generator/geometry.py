"""
Molecular Geometry Generators

Functions to produce geometry objects for molecular Hamiltonian generation.
"""

from math import sin, cos, pi


def get_H2_geo(bond_len):
    """Generate H2 molecule geometry."""
    atom_1 = 'H'
    atom_2 = 'H'
    coordinate_1 = (0.0, 0.0, 0.0)
    coordinate_2 = (bond_len, 0.0, 0.0)
    geometry = [(atom_1, coordinate_1), (atom_2, coordinate_2)]
    return geometry


def get_H6_geo(bond_len):
    """Generate H6 chain molecule geometry."""
    atoms = ['H'] * 6
    coordinates = [(bond_len * i, 0.0, 0.0) for i in range(6)]
    geometry = list(zip(atoms, coordinates))
    return geometry


def get_H4_geo(bond_len):
    """Generate H4 chain molecule geometry."""
    atoms = ['H'] * 4
    coordinates = [(bond_len * i, 0.0, 0.0) for i in range(4)]
    geometry = list(zip(atoms, coordinates))
    return geometry


def get_LiH_geo(bond_len):
    """Generate LiH molecule geometry."""
    atom_1 = 'H'
    atom_2 = 'Li'
    coordinate_1 = (0.0, 0.0, 0.0)
    coordinate_2 = (bond_len, 0.0, 0.0)
    geometry = [(atom_1, coordinate_1), (atom_2, coordinate_2)]
    return geometry


def get_N2_geo(bond_len):
    """Generate N2 molecule geometry."""
    atom_1 = 'N'
    atom_2 = 'N'
    coordinate_1 = (0.0, 0.0, 0.0)
    coordinate_2 = (bond_len, 0.0, 0.0)
    geometry = [(atom_1, coordinate_1), (atom_2, coordinate_2)]
    return geometry


def get_H2O_geo(bond_len):
    """Generate H2O molecule geometry."""
    atom_1 = 'H'
    atom_2 = 'H'
    atom_3 = 'O'
    coordinate_1 = (0.0, 0.0, 0.0)
    coordinate_2 = (2 * cos(104.5 / 180 * pi) * bond_len, 0.0, 0.0)
    coordinate_3 = (1 * cos(104.5 / 180 * pi) * bond_len,
                    sin(104.5 / 180 * pi) * bond_len, 0.0)
    geometry = [(atom_1, coordinate_1), (atom_2, coordinate_2),
                (atom_3, coordinate_3)]
    return geometry


def get_H2S_geo(bond_len):
    """Generate H2S molecule geometry."""
    atom_1 = 'H'
    atom_2 = 'H'
    atom_3 = 'S'
    coordinate_1 = (0.0, 0.0, 0.0)
    coordinate_2 = (2 * cos(104.5 / 180 * pi) * bond_len, 0.0, 0.0)
    coordinate_3 = (1 * cos(104.5 / 180 * pi) * bond_len,
                    sin(104.5 / 180 * pi) * bond_len, 0.0)
    geometry = [(atom_1, coordinate_1), (atom_2, coordinate_2),
                (atom_3, coordinate_3)]
    return geometry


def get_CH2_geo(bond_len):
    """Generate CH2 molecule geometry."""
    atom_1 = 'H'
    atom_2 = 'H'
    atom_3 = 'C'
    coordinate_1 = (0.0, 0.0, 0.0)
    coordinate_2 = (2 * cos(104.5 / 180 * pi) * bond_len, 0.0, 0.0)
    coordinate_3 = (1 * cos(104.5 / 180 * pi) * bond_len,
                    sin(104.5 / 180 * pi) * bond_len, 0.0)
    geometry = [(atom_1, coordinate_1), (atom_2, coordinate_2),
                (atom_3, coordinate_3)]
    return geometry


def get_C2H2_geo(args=None):
    """
    Generate C2H2 (acetylene) molecule geometry.
    
    Reference: https://cccbdb.nist.gov/exp2x.asp?casno=74862&charge=0
    """
    atom_1 = 'H'
    atom_2 = 'C'
    atom_3 = 'C'
    atom_4 = 'H'
    coordinate_1 = (1.6644, 0.0, 0.0)
    coordinate_2 = (0.6013, 0.0, 0.0)
    coordinate_3 = (-0.6013, 0.0, 0.0)
    coordinate_4 = (-1.6644, 0.0, 0.0)
    geometry = [(atom_1, coordinate_1), (atom_2, coordinate_2),
                (atom_3, coordinate_3), (atom_4, coordinate_4)]
    return geometry


def get_C2H4_geo(args=None):
    """
    Generate C2H4 (ethylene) molecule geometry.
    
    Reference: https://cccbdb.nist.gov/expgeom2x.asp
    """
    C1 = ('C', (0.0, 0.0, 0.6695))
    C2 = ('C', (0.0, 0.0, -0.6695))
    H3 = ('H', (0.0, 0.9289, 1.2321))
    H4 = ('H', (0.0, -0.9289, 1.2321))
    H5 = ('H', (0.0, 0.9289, -1.2321))
    H6 = ('H', (0.0, -0.9289, -1.2321))
    geometry = [C1, C2, H3, H4, H5, H6]
    return geometry


def get_CO2_geo(bond_length):
    """
    Generate CO2 molecule geometry.
    
    Reference: https://cccbdb.nist.gov/expgeom2x.asp
    """
    C1 = ('C', (0.0, 0.0, 0.0))
    O2 = ('O', (0.0, 0.0, bond_length))
    O3 = ('O', (0.0, 0.0, -bond_length))
    geometry = [C1, O2, O3]
    return geometry


def get_NH3_geo(position_of_N):
    """
    Generate NH3 (ammonia) molecule geometry.
    
    Reference: https://cccbdb.nist.gov/expgeom2.asp?casno=7664417&charge=0
    """
    N1 = (0.0000, 0.0000, position_of_N)
    H2 = (0.0000, -0.9377, -0.3816)
    H3 = (0.8121, 0.4689, -0.3816)
    H4 = (-0.8121, 0.4689, -0.3816)
    geometry = [("N", N1), ("H", H2),
                ("H", H3), ("H", H4)]
    return geometry


def get_CH3OH_geo():
    """
    Generate CH3OH (methanol) molecule geometry.
    
    Methanol is a 6-atom molecule (C, O, 4H) that produces
    Hamiltonians with approximately 30-40 qubits depending on
    the basis set and transformation.
    """
    geometry = [
        ('C', (0.0416, 0.0000, 0.0000)),      # Carbon
        ('O', (1.4080, 0.0000, 0.0000)),      # Oxygen
        ('H', (2.0000, 0.0000, 0.0000)),      # Hydroxyl H
        ('H', (-0.3584, -0.5150, 0.8900)),    # Methyl H1
        ('H', (-0.3584, -0.5150, -0.8900)),   # Methyl H2
        ('H', (-0.3584, 1.0300, 0.0000))      # Methyl H3
    ]
    return geometry


def get_C2H5OH_geo():
    """
    Generate C2H5OH (ethanol) molecule geometry.
    
    Ethanol is a 9-atom molecule that produces larger Hamiltonians
    with 40-50+ qubits.
    """
    geometry = [
        ('C', (-0.7479, 0.0564, 0.0000)),     # C1
        ('C', (0.7479, -0.0564, 0.0000)),     # C2
        ('O', (1.2479, -1.2000, 0.0000)),     # O
        ('H', (-1.1479, 1.0564, 0.0000)),     # H on C1
        ('H', (-1.1479, -0.4436, 0.8900)),    # H on C1
        ('H', (-1.1479, -0.4436, -0.8900)),   # H on C1
        ('H', (1.1479, 0.4436, 0.8900)),      # H on C2
        ('H', (1.1479, 0.4436, -0.8900)),     # H on C2
        ('H', (2.2000, -1.2000, 0.0000))      # H on O
    ]
    return geometry


def get_benzene_geo():
    """
    Generate C6H6 (benzene) molecule geometry.
    
    Benzene is a 12-atom aromatic molecule that produces large
    Hamiltonians with 50+ qubits.
    """
    import math
    # Benzene ring with C-C bond length of 1.39 Angstroms
    r_cc = 1.39
    r_ch = 1.08
    
    geometry = []
    # Carbon ring
    for i in range(6):
        angle = i * math.pi / 3
        x = r_cc * math.cos(angle)
        y = r_cc * math.sin(angle)
        geometry.append(('C', (x, y, 0.0)))
    
    # Hydrogen atoms (one per carbon, pointing outward)
    for i in range(6):
        angle = i * math.pi / 3
        r_total = r_cc + r_ch
        x = r_total * math.cos(angle)
        y = r_total * math.sin(angle)
        geometry.append(('H', (x, y, 0.0)))
    
    return geometry


# Equilibrium bond lengths/positions for various molecules (in Angstroms)
equilibrium_geometry_dict = {
    "H2": 0.74,
    "H6": 1.0,
    "H4": 1.0,
    "LiH": 1.4,
    "H2O": 0.96,
    "N2": 1.09,
    "NH3": 0.0,
    "C2H2": None,
    "C2H4": None,
    "CO2": 1.1621,
    "CH3OH": None,      # Methanol (pre-optimized geometry)
    "C2H5OH": None,     # Ethanol (pre-optimized geometry)
    "C6H6": None        # Benzene (pre-optimized geometry)
}

# Mapping from molecule name to geometry generator function
geometry_generator_dict = {
    "H2": get_H2_geo,
    "H6": get_H6_geo,
    "H4": get_H4_geo,
    "LiH": get_LiH_geo,
    "H2O": get_H2O_geo,
    "N2": get_N2_geo,
    "NH3": get_NH3_geo,
    "C2H2": get_C2H2_geo,
    "C2H4": get_C2H4_geo,
    "CO2": get_CO2_geo,
    "CH3OH": get_CH3OH_geo,
    "C2H5OH": get_C2H5OH_geo,
    "C6H6": get_benzene_geo
}

# Approximate qubit counts for various molecules (with sto-3g basis)
molecule_qubit_counts = {
    "H2": "4 qubits",
    "H4": "8 qubits",
    "H6": "12 qubits",
    "LiH": "12 qubits",
    "H2O": "14 qubits",
    "NH3": "16 qubits",
    "N2": "20 qubits",
    "C2H2": "24 qubits",
    "H2O (6-31g)": "26 qubits",
    "C2H4": "28 qubits",
    "CO2": "30 qubits",
    "NH3 (6-31g)": "30 qubits",
    "CH3OH": "30-38 qubits (basis dependent)",
    "C2H5OH": "40-50 qubits (basis dependent)",
    "C6H6": "50+ qubits (basis dependent)"
}
