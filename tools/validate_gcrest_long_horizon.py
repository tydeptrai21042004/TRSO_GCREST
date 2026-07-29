from __future__ import annotations
import argparse, copy, json, random
import numpy as np
import torch
from torch import nn
from models.tuning_modules.mdl_tangent_core import calibrate_mdl_tangent_core
from tools.validate_cvest_controlled import TinyCNN, accuracy, attach_fixed_reference, freeze_except_head, loaders, train

def run(seed:int):
    hard=True
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    source,_,warm,test=loaders(seed,hard); classes=8
    factory=lambda:TinyCNN(classes,widths=(10,16,24))
    base=factory(); train(base,source,6,2e-3); freeze_except_head(base); train(base,warm,2,5e-3)
    state=copy.deepcopy(base.state_dict())
    fixed=factory(); fixed.load_state_dict(state); freeze_except_head(fixed)
    fixed_params=attach_fixed_reference(fixed,loaders(seed,hard)[2],rank=3,batches=6)
    train(fixed,loaders(seed,hard)[2],30,8e-3,freeze_bn=True)
    method=factory(); method.load_state_dict(state); freeze_except_head(method)
    report=calibrate_mdl_tangent_core(method,loaders(seed,hard)[2],nn.CrossEntropyLoss(),device='cpu')
    train(method,loaders(seed,hard)[2],30,8e-3,freeze_bn=True)
    fa=accuracy(fixed,test); ma=accuracy(method,test)
    return {'seed':seed,'epochs':30,'fixed_accuracy':fa,'gcrest_accuracy':ma,'gain':ma-fa,'fixed_adapter_parameters':fixed_params,'gcrest_adapter_parameters':report.adapter_parameters,'selected_tensors':report.selected_tensors}
if __name__=='__main__':
    torch.set_num_threads(8)
    ap=argparse.ArgumentParser(); ap.add_argument('--seeds',default='0'); ap.add_argument('--output',default=''); a=ap.parse_args()
    rows=[run(int(v)) for v in a.seeds.split(',') if v.strip()]
    print(json.dumps(rows,indent=2))
    if a.output: open(a.output,'w').write(json.dumps(rows,indent=2)+'\n')
