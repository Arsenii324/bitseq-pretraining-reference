"""Offline full terminal-weight × conditional-KL-direction matrix.

No production training/evaluator edits. Uses already saved primitive probabilities;
does not infer causal effects from differences between descriptive populations.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from bitseq.artifacts import write_json
from bitseq.evaluate import _propagate, _validate
from bitseq.oracle import FiniteGraph, task_distribution


def path_matrix(graph,current,reference):
    current,reference=_validate(graph,current),_validate(graph,reference)
    now=_propagate(graph,current,"random",reference)
    old=_propagate(graph,reference,"random",current)
    forward=now["logpath_mean"]-now["cross_mean"]-now["logp"]+old["logp"]
    reverse=old["logpath_mean"]-old["cross_mean"]-old["logp"]+now["logp"]
    if min(forward.min(),reverse.min()) < -1e-9:
        raise ValueError("Negative conditionalKL beyond rounding")
    summary={}
    for name,p in (("reference_weighted",old["p"]),("current_weighted",now["p"])):
        summary[name]=dict(kl_ref_to_current=float(p@reverse),
                           kl_current_to_ref=float(p@forward),
                           current_conditional_entropy=float(p@now["conditional_entropy"]))
    return dict(summary=summary,p_reference=old["p"],p_current=now["p"],
                kl_ref_to_current=reverse,kl_current_to_ref=forward,
                h_reference=old["conditional_entropy"],h_current=now["conditional_entropy"])


def analyze_checkpoint(directory,step=1500):
    directory=Path(directory)
    manifest=json.loads((directory/"manifest.json").read_text())
    table=np.load(directory/f"table_{step:06d}.npz")["logwords"]
    ref=np.load(manifest["base_reference_table"]["path"])["logwords"]
    if manifest["reference_table_max_logp_deviation"]!=0:
        raise ValueError("Offline source table differs from runtime reference; reconstruct that law first")
    graph=FiniteGraph()
    result=path_matrix(graph,table,ref)
    oldeval=json.loads((directory/f"eval_{step:06d}.json").read_text())["decoders"]["random"]
    for field,weight in (("path_kl_ref_to_current","reference_weighted"),("path_kl_current_to_ref","current_weighted")):
        key=field.removeprefix("path_")
        if abs(result["summary"][weight][key]-oldeval[field])>1e-10:
            raise ValueError("Offline matrix disagrees with actual saved evaluation")
    prior=task_distribution(graph,12)
    subsets={"exact_peaks":prior["distance"]==0}
    subsets.update({f"basin_radius2_{j}":b for j,b in enumerate(prior["basin_masks"])})
    conditioned={}
    for label,mask in subsets.items():
        rows={}
        for name,p in (("reference_weighted",result["p_reference"]),("current_weighted",result["p_current"])):
            mass=float(p[mask].sum())
            rows[name]=dict(subset_mass=mass,renormalized_within_subset=True,
                            kl_ref_to_current=None if mass==0 else float(p[mask]@result["kl_ref_to_current"][mask]/mass),
                            kl_current_to_ref=None if mass==0 else float(p[mask]@result["kl_current_to_ref"][mask]/mass),
                            current_conditional_entropy=None if mass==0 else float(p[mask]@result["h_current"][mask]/mass))
        conditioned[label]=rows
    record=dict(step=step,method=manifest["method"],seed=manifest["seed"],beta=manifest["beta"],
                base_checkpoint_sha256=manifest["base_checkpoint_sha256"],
                weights=manifest["base_reference_table"],matrix=result["summary"],
                posthoc_descriptive_conditioned_views=conditioned)
    out=directory/f"path_matrix_{step:06d}.json"
    write_json(out,record)
    np.savez_compressed(directory/f"path_by_y_{step:06d}.npz",
                        **{k:v for k,v in result.items() if k!="summary"})
    return record


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--run",required=True)
    parser.add_argument("--step",type=int,default=1500)
    args=parser.parse_args()
    print(json.dumps(analyze_checkpoint(args.run,args.step),indent=2))
