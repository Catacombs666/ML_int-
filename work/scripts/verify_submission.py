"""Verify genuine notebook outputs, model receipts, public-safe files and paper links."""
import argparse, json, re
from pathlib import Path
from html.parser import HTMLParser
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
class PaperParser(HTMLParser):
    def __init__(self):super().__init__();self.sections=[];self.images=[];self.links=[]
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if tag=='section':self.sections.append(a.get('id'))
        if tag=='img':self.images.append(a)
        if tag=='a':self.links.append(a.get('href',''))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--allow-unpublished',action='store_true');ap.add_argument('--expected');args=ap.parse_args()
    r=json.loads((ROOT/'work/outputs/ctr_metrics.json').read_text())
    assert r['status']=='measured' and r['data']['eligible_rows']==16726
    assert sum(v['rows'] for v in r['splits'].values())==16726
    assert all(r['leakage_checks'][k]==0 for k in ['client_overlap','content_overlap','target_or_identifier_features'])
    assert r['selected_method']==min(r['validation'],key=lambda k:r['validation'][k]['weighted_mae_pp'])
    if args.expected:
        e=json.loads(Path(args.expected).read_text())
        assert e['data']['sha256']==r['data']['sha256']
        for group in ['test','validation']:
            for model in e[group]:
                for key in e[group][model]:assert np.isclose(e[group][model][key],r[group][model][key],rtol=1e-9,atol=1e-10), (group,model,key)
    notebooks=sorted((ROOT/'work/notebooks').glob('*.ipynb'));assert len(notebooks)==10
    for p in notebooks:
        n=json.loads(p.read_text());cells=[c for c in n['cells'] if c['cell_type']=='code'];assert cells
        assert all(c['execution_count'] is not None for c in cells),p
        assert any(c['outputs'] for c in cells),p
        assert not any(o['output_type']=='error' for c in cells for o in c['outputs']),p
    paper=ROOT/'work/paper/index.html';text=paper.read_text();parser=PaperParser();parser.feed(text)
    assert parser.sections==['abstract','problem','data','methods','results','limits','recommendations','reproducibility','credit']
    assert 'https://flyrank.ai' in parser.links
    assert len(parser.images)==4
    for image in parser.images:
        assert image.get('alt') and not image['src'].startswith(('/', 'http'))
        assert (paper.parent/image['src']).is_file()
    assert r['data']['sha256'] in text
    for name in r['test']:assert f"{r['test'][name]['weighted_mae_pp']:.4f}" in text
    for path in list((ROOT/'work/paper').rglob('*'))+list((ROOT/'work/notebooks').glob('*.ipynb'))+[ROOT/'work/outputs/ctr_metrics.json']:
        if path.is_file():
            s=path.read_text()
            assert not re.search(r'hf_[A-Za-z0-9]{25,}|gh[pousr]_[A-Za-z0-9]{25,}',s),path
            assert not re.search(r'(?:client_[0-9a-f]{10}|content_[0-9a-f]{12})',s),path
    url_file=ROOT/'submission/paper_url.txt'
    if not args.allow_unpublished:
        lines=url_file.read_text().splitlines();assert len(lines)==1 and re.fullmatch(r'https://[^\s]+',lines[0])
        assert 'PASTE' not in lines[0]
    print('PASS: measured receipts, selection rule, overlap guards, 10 executed notebooks, 9 paper sections, 4 local images, public-output scan and submission format.')
if __name__=='__main__':main()
