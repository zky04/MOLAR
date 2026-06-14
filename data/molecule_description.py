"""
Generate text descriptions for molecules from SMILES using RDKit.
Used as the text modality for multimodal learning.
"""

from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, Lipinski


def smiles_to_description(smiles: str) -> str:
    """Convert a SMILES string to a natural language description."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "Unknown molecule."

    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    rot_bonds = Lipinski.NumRotatableBonds(mol)
    rings = Lipinski.RingCount(mol)
    atoms = mol.GetNumAtoms()
    heavy_atoms = mol.GetNumHeavyAtoms()

                              
    patterns = {
        "aromatic ring": Chem.MolFromSmarts("a1aaaaa1"),
        "carboxylic acid": Chem.MolFromSmarts("C(=O)O"),
        "primary amine": Chem.MolFromSmarts("[NH2]"),
        "amide": Chem.MolFromSmarts("C(=O)N"),
        "hydroxyl": Chem.MolFromSmarts("[OH]"),
        "ketone": Chem.MolFromSmarts("[#6]C(=O)[#6]"),
        "ether": Chem.MolFromSmarts("[#6]O[#6]"),
        "halogen": Chem.MolFromSmarts("[F,Cl,Br,I]"),
        "nitro": Chem.MolFromSmarts("[N+](=O)[O-]"),
        "sulfonamide": Chem.MolFromSmarts("S(=O)(=O)N"),
        "ester": Chem.MolFromSmarts("C(=O)O[#6]"),
    }

    present = []
    for name, smarts in patterns.items():
        if mol.HasSubstructMatch(smarts):
            present.append(name)

                       
    parts = [f"The molecule has {heavy_atoms} heavy atoms ({atoms} total atoms), "
             f"a molecular weight of {mw:.1f} Da, "
             f"LogP of {logp:.1f}, "
             f"{hbd} hydrogen bond donors, {hba} hydrogen bond acceptors, "
             f"{rot_bonds} rotatable bonds, and {rings} rings."]

    if present:
        parts.append(f" It contains: {', '.join(present)}.")

    return " ".join(parts)


def batch_smiles_to_descriptions(smiles_list: list) -> list:
    """Convert a batch of SMILES to descriptions."""
    return [smiles_to_description(s) for s in smiles_list]
