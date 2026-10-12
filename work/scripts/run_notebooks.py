"""Execute these Python-only notebooks cell-by-cell and save genuine text outputs.

This small offline runner requires no Jupyter installation. The same cells can
also be executed with Colab/Jupyter Run All. No output is synthesized.
"""
import contextlib, io, json, os, traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def run_one(path):
    n=json.loads(path.read_text()); context={'__name__':'__main__'}; count=0
    for cell in n['cells']:
        if cell['cell_type']!='code':continue
        src=cell['source']; src=''.join(src) if isinstance(src,list) else src
        count+=1; cell['execution_count']=count; cell['outputs']=[]
        capture=io.StringIO()
        try:
            with contextlib.redirect_stdout(capture),contextlib.redirect_stderr(capture):
                exec(compile(src,str(path)+f':cell{count}','exec'),context)
        except Exception:
            trace=traceback.format_exc(); cell['outputs']=[{'output_type':'error','ename':'ExecutionError','evalue':trace.splitlines()[-1],'traceback':trace.splitlines()}]
            path.write_text(json.dumps(n,indent=1)+'\n')
            raise
        text=capture.getvalue()
        if text:cell['outputs']=[{'output_type':'stream','name':'stdout','text':text}]
    path.write_text(json.dumps(n,indent=1)+'\n')
    print(f'PASS {path.name}: {count} code cells, executed in order')

def main():
    os.environ['CAPSTONE_OFFLINE']='1'; os.chdir(ROOT)
    for path in sorted((ROOT/'work/notebooks').glob('*.ipynb')):run_one(path)
if __name__=='__main__':main()
