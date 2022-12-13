import sys
sys.path.append('../')
from model.model import HyperNeRFResNetSymmLocal


model_dict = {
    'hypernerf_symm_local': HyperNeRFResNetSymmLocal
}
