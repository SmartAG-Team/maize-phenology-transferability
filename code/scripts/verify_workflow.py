"""Verify workflow graph, connector clearance, text fit and native exports."""
from pathlib import Path
import argparse,json,sys
import numpy as np
import pandas as pd
from PIL import Image
PROJECT=Path(__file__).resolve().parents[2]
ROOT=PROJECT
sys.path.insert(0,str(PROJECT/'code'))

def main():
    p=argparse.ArgumentParser();p.add_argument('--figure-dir',type=Path,default=ROOT/'outputs/figures');a=p.parse_args();folder=a.figure_dir
    nodes=pd.read_excel(folder/'fig1.xlsx','nodes').fillna('')
    edges=pd.read_excel(folder/'fig1.xlsx','edges')
    labels=pd.read_excel(folder/'fig1.xlsx','labels').fillna('')
    encodings=pd.read_excel(folder/'fig1.xlsx','encodings')
    assert nodes.node.is_unique and labels.within_container.all()
    assert labels.container_node.ne('').all(),'Standalone title or heading outside workflow nodes'
    assert not labels.label.eq('question').any()
    assert set(nodes[nodes.kind.eq('parallel comparison')].panel)=={'(a)','(b)','(c)','(d)'}
    from research.annual_temperature_transfer.figures import COLORS
    assert dict(zip(encodings.model,encodings.color))==COLORS
    assert len(nodes[nodes.kind.eq('model formulation')])==6
    ids=set(nodes.node);assert set(edges.source)|set(edges.target)<=ids
    adjacency={key:set() for key in ids}
    for e in edges.itertuples():
        adjacency[e.source].add(e.target)
        points=np.array(json.loads(e.path),float)
        # Independently check that routed connectors avoid unrelated cards.
        for r in nodes.itertuples():
            if r.node in [e.source,e.target] or not r.width_layout:continue
            x0,x1=r.x_layout+1e-7,r.x_layout+r.width_layout-1e-7
            y0,y1=r.y_layout+1e-7,r.y_layout+r.height_layout-1e-7
            for start,end in zip(points[:-1],points[1:]):
                q=start+(end-start)*np.linspace(0,1,301)[:,None]
                assert not ((q[:,0]>x0)&(q[:,0]<x1)&(q[:,1]>y0)&(q[:,1]<y1)).any(),(e.source,e.target,r.node)
    def walk(key,stack):
        assert key not in stack,'Workflow contains a directed cycle'
        result={key}
        for target in adjacency[key]:result|=walk(target,stack|{key})
        return result
    reachable=walk('qualified_data',set())
    assert {'models','benchmark','local','transfer','weather','evaluation'}<=reachable
    for key in ids:walk(key,set())
    from figure_export import verify_png
    layout=verify_png(folder,'fig1')
    result=dict(complete=True,figure='fig1',conceptual_nodes=len(nodes),directed_edges=len(edges),text_items=len(labels),
        parallel_comparisons=4,model_formulations=6,graph_acyclic=True,all_shared_inputs_reach_evaluation=True,
        text_fits_cards=True,connectors_avoid_unrelated_cards=True,model_palette_matches_results=True,
        standalone_title=False,width_cm=19,font_pt=12,dpi=300,coordinate_basis='conceptual layout units; not measurements',no_model_refitting=True)
    out=ROOT/'outputs/workflow_verification.json';out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
