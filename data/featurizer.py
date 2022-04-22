import torch

allowable_features = {
    'possible_atomic_num_list' : list(range(1, 119)) + ['misc'],
    'possible_chirality_list' : [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER'
    ],
    'possible_degree_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
    'possible_formal_charge_list' : [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc'],
    'possible_numH_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    'possible_number_radical_e_list': [0, 1, 2, 3, 4, 'misc'],
    'possible_hybridization_list' : [
        'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'misc'
        ],
    'possible_is_aromatic_list': [False, True],
    'possible_is_in_ring_list': [False, True],
    'possible_bond_type_list' : [
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'AROMATIC',
        'misc'
    ],
    'possible_bond_stereo_list': [
        'STEREONONE',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
        'STEREOANY',
    ], 
    'possible_is_conjugated_list': [False, True],
}

def safe_index(l, e):
    """
    Return index of element e in list l. If e is not present, return the last index
    """
    try:
        return l.index(e)
    except:
        return len(l) - 1

class OGBAtomEncoder(torch.nn.Module):
    def __init__(self, emb_dim, full_atom_feature_dims):
        super(OGBAtomEncoder, self).__init__()
        
        self.atom_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(full_atom_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        for i in range(x.shape[1]):
            x_embedding += self.atom_embedding_list[i](x[:,i])

        return x_embedding

class OGBBondEncoder(torch.nn.Module):
    def __init__(self, emb_dim, full_bond_feature_dims):
        super(OGBBondEncoder, self).__init__()
        
        self.bond_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(full_bond_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, edge_attr):
        bond_embedding = 0
        for i in range(edge_attr.shape[1]):
            bond_embedding += self.bond_embedding_list[i](edge_attr[:,i])

        return bond_embedding 

class OGBFeaturizer:

    def __init__(self):
        super()

    def atom_to_feature_vector(self,atom):
        """
        Converts rdkit atom object to feature list of indices
        :param mol: rdkit atom object
        :return: list
        """
        atom_feature = [
                safe_index(allowable_features['possible_atomic_num_list'], atom.GetAtomicNum()),
                allowable_features['possible_chirality_list'].index(str(atom.GetChiralTag())),
                safe_index(allowable_features['possible_degree_list'], atom.GetTotalDegree()),
                safe_index(allowable_features['possible_formal_charge_list'], atom.GetFormalCharge()),
                safe_index(allowable_features['possible_numH_list'], atom.GetTotalNumHs()),
                safe_index(allowable_features['possible_number_radical_e_list'], atom.GetNumRadicalElectrons()),
                safe_index(allowable_features['possible_hybridization_list'], str(atom.GetHybridization())),
                allowable_features['possible_is_aromatic_list'].index(atom.GetIsAromatic()),
                allowable_features['possible_is_in_ring_list'].index(atom.IsInRing()),
                ]
        return atom_feature

    def bond_to_feature_vector(self,bond):
        """
        Converts rdkit bond object to feature list of indices
        :param mol: rdkit bond object
        :return: list
        """
        bond_feature = [
                    safe_index(allowable_features['possible_bond_type_list'], str(bond.GetBondType())),
                    allowable_features['possible_bond_stereo_list'].index(str(bond.GetStereo())),
                    allowable_features['possible_is_conjugated_list'].index(bond.GetIsConjugated()),
                ]
        return bond_feature

    @classmethod
    def get_atom_encoder(cls,emb_dim):
        return OGBAtomEncoder(emb_dim, cls.get_atom_feature_dims())

    @classmethod
    def get_bond_encoder(cls,emb_dim):
        return OGBBondEncoder(emb_dim, cls.get_bond_feature_dims())
    
    @staticmethod
    def get_atom_feature_dims():
        return list(map(len, [
            allowable_features['possible_atomic_num_list'],
            allowable_features['possible_chirality_list'],
            allowable_features['possible_degree_list'],
            allowable_features['possible_formal_charge_list'],
            allowable_features['possible_numH_list'],
            allowable_features['possible_number_radical_e_list'],
            allowable_features['possible_hybridization_list'],
            allowable_features['possible_is_aromatic_list'],
            allowable_features['possible_is_in_ring_list']
        ]))
    
    @staticmethod
    def get_bond_feature_dims():
        return list(map(len, [
            allowable_features['possible_bond_type_list'],
            allowable_features['possible_bond_stereo_list'],
            allowable_features['possible_is_conjugated_list']
        ]))

FEATURIZER = {
    'ogb': OGBFeaturizer,
}

def get_featurizer(name):
    if name not in FEATURIZER:
        raise ValueError("Featurizer name not recognized or featurizer is not supported.")
    else:
        return FEATURIZER[name]