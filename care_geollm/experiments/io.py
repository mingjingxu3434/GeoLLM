from pathlib import Path
import json
import pandas as pd
from ..data import load_graph

def load_graph_dir(path):
    return [load_graph(str(p)) for p in sorted(Path(path).glob('*.pt'))]

def save_metrics(metrics, path, extra=None):
    obj={'metrics':metrics}; obj.update(extra or {})
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')

def append_summary_row(row,path):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    df=pd.DataFrame([row]); df.to_csv(p,mode='a',header=not p.exists(),index=False)
