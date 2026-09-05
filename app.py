"""Review Analyst — FastAPI app.

ML part: 3-class sentiment (negative/neutral/positive) with the best pipeline
(LinearSVC + TF-IDF bigram). AI part: an LLM (Qwen 27B via an OpenAI-compatible
API) turns the negative reviews into a business report.

Run:  uvicorn app:app --reload --port 8000
"""
import os
from pathlib import Path

import joblib
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from openai import OpenAI
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

MODEL_PATH = BASE_DIR / "model" / "best_pipeline.pkl"
LLM_MODEL = os.environ.get("LLM_MODEL", "unknown")

app = FastAPI(title="Review Analyst", version="1.0.0")

model = None
llm_client = None


@app.on_event("startup")
def _load():
    """Load the trained ML pipeline and the LLM client once at startup."""
    global model, llm_client
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model not found at {MODEL_PATH}. Run notebooks/analysis.ipynb first.")
    model = joblib.load(MODEL_PATH)
    llm_client = OpenAI(base_url=os.environ["LLM_API_BASE"], api_key=os.environ["LLM_API_KEY"])


MAX_REVIEWS = 200
MAX_REVIEW_CHARS = 5_000
MAX_TOTAL_CHARS = 500_000


class AnalyzeRequest(BaseModel):
    reviews: list[str] = Field(..., min_length=1, max_length=MAX_REVIEWS)
    generate_report: bool = True


def _llm_report(negative_reviews: list[str], max_reviews: int = 15) -> str:
    """Ask the LLM to turn negative reviews into a Russian business report.

    The model is a *thinking* model, so we must disable thinking
    (``enable_thinking: False``) and give a generous ``max_tokens`` —
    otherwise the tokens go to ``reasoning_content`` and ``content`` comes back empty.
    """
    sample = "\n\n".join(f"[{i+1}] {t[:600]}" for i, t in enumerate(negative_reviews[:max_reviews]))
    prompt = (
        "Ты — аналитик для бизнеса. Ниже приведены негативные отзывы клиентов. "
        "Напиши краткий отчёт на русском языке СТРОГО в формате Markdown: "
        "начни с заголовка '## Отчёт по негативным отзывам', затем раздел "
        "'### Основные жалобы' (нумерованный список, 3-5 пунктов) и раздел "
        "'### Рекомендации' (маркированный список, 3-5 пунктов). "
        "Важные слова выделяй **жирным**, ссылайся на отзывы по номеру (отзыв N). "
        "Не более 250 слов. Без преамбул и заключений.\n\n"
        f"Отзывы:\n{sample}"
    )
    resp = llm_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1600,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return resp.choices[0].message.content


@app.get("/health")
def health():
    """Liveness probe: confirms the app is up and the model is loaded."""
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    """Classify reviews with the ML model and (optionally) build an LLM report.

    Returns per-review sentiment + probabilities, a class summary, and the
    Markdown LLM report generated from the negative reviews.
    """
    texts = [r.strip() for r in req.reviews if r.strip()]
    if not texts:
        raise HTTPException(status_code=422, detail="No non-empty reviews provided.")
    total_chars = sum(len(t) for t in texts)
    if total_chars > MAX_TOTAL_CHARS:
        raise HTTPException(
            status_code=413,
            detail=f"Total text is {total_chars} chars, max is {MAX_TOTAL_CHARS}. "
                   f"Upload fewer reviews.",
        )
    too_long = [i + 1 for i, t in enumerate(texts) if len(t) > MAX_REVIEW_CHARS]
    if too_long:
        raise HTTPException(
            status_code=422,
            detail=f"Review(s) {', '.join(map(str, too_long[:5]))} exceed "
                   f"{MAX_REVIEW_CHARS} chars (max per review).",
        )

    # ML part: predict class + calibrated probabilities for every review
    labels = model.predict(texts)
    proba = model.predict_proba(texts)
    classes = list(model.classes_)

    results = []
    for text, label, p in zip(texts, labels, proba):
        results.append({
            "review": text[:300] + ("…" if len(text) > 300 else ""),
            "sentiment": str(label),
            "probabilities": {str(c): round(float(pc), 4) for c, pc in zip(classes, p)},
        })

    # class counts + the negative subset that feeds the LLM
    summary = {c: int(sum(1 for r in results if r["sentiment"] == c)) for c in classes}
    negative = [r["review"] for r in results if r["sentiment"] == "negative"]

    # AI part: only run the LLM if there is something negative to report on
    report = None
    if req.generate_report and negative:
        try:
            report = _llm_report(negative)
        except Exception as e:  # LLM is a bonus; don't fail the whole request
            report = f"(LLM report unavailable: {e})"

    return {"results": results, "summary": summary, "llm_report": report}


@app.get("/", response_class=HTMLResponse)
def index():
    """Serve the single-page web UI (HTML/CSS/JS inlined, no build step)."""
    return """<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Review Analyst — ML + AI</title>
<script src="https://cdn.jsdelivr.net/npm/marked@12.0.2/marked.min.js"></script>
<style>
 :root{--bg:#f4f6fb;--card:#fff;--ink:#171c2b;--muted:#6b7280;--line:#e5e8f0;
       --acc:#4f46e5;--acc2:#7c3aed;--neg:#e74c3c;--neu:#f1c40f;--pos:#2ecc71}
 *{box-sizing:border-box}
 body{font-family:system-ui,'Segoe UI',Roboto,sans-serif;margin:0;background:var(--bg);color:var(--ink)}
 .wrap{max-width:960px;margin:0 auto;padding:28px 20px 60px}
 .hero{background:linear-gradient(120deg,var(--acc),var(--acc2));border-radius:16px;color:#fff;
       padding:26px 28px;margin-bottom:22px;box-shadow:0 8px 24px rgba(79,70,229,.25)}
 .hero h1{margin:0 0 6px;font-size:24px;letter-spacing:.2px}
 .hero p{margin:0;opacity:.92;font-size:14.5px;line-height:1.5}
 .hero .badges{margin-top:12px;display:flex;gap:8px;flex-wrap:wrap}
 .hero .badge{background:rgba(255,255,255,.16);border:1px solid rgba(255,255,255,.3);
       padding:3px 12px;border-radius:999px;font-size:12px;font-weight:600}
 .panel{background:var(--card);border:1px solid var(--line);border-radius:14px;
        padding:18px 20px;margin-bottom:18px;box-shadow:0 1px 3px rgba(20,20,43,.06)}
 .panel h2{margin:0 0 12px;font-size:15px;text-transform:uppercase;letter-spacing:.6px;color:var(--muted)}
 .drop{border:2px dashed #c7cbe0;border-radius:12px;padding:22px;text-align:center;color:var(--muted);
       cursor:pointer;transition:.15s;font-size:14px}
 .drop:hover,.drop.over{border-color:var(--acc);background:#eef0ff;color:var(--acc)}
 .drop b{color:var(--acc)}
 .drop .hint{font-size:12px;margin-top:6px;opacity:.8}
 input[type=file]{display:none}
 .files{margin-top:10px;display:flex;gap:8px;flex-wrap:wrap}
 .chip{background:#eef0ff;color:var(--acc);border-radius:999px;padding:4px 12px;font-size:12.5px;font-weight:600}
 textarea{width:100%;height:170px;padding:12px 14px;border:1px solid var(--line);border-radius:10px;
          font-size:14px;font-family:inherit;resize:vertical;margin-top:12px;background:#fbfbfe}
 textarea:focus{outline:2px solid var(--acc);border-color:transparent}
 .row{display:flex;align-items:center;gap:14px;margin-top:14px;flex-wrap:wrap}
 button#go{padding:11px 26px;font-size:15px;font-weight:600;border:0;border-radius:10px;cursor:pointer;
           background:linear-gradient(120deg,var(--acc),var(--acc2));color:#fff;box-shadow:0 4px 14px rgba(79,70,229,.3)}
 button#go:disabled{opacity:.55;cursor:default}
 .count{font-size:13px;color:var(--muted)}
 #status{font-size:13.5px;color:var(--muted);min-height:18px}
 .spinner{display:inline-block;width:14px;height:14px;border:2px solid #c7cbe0;border-top-color:var(--acc);
          border-radius:50%;animation:spin .8s linear infinite;vertical-align:-2px;margin-right:6px}
 @keyframes spin{to{transform:rotate(360deg)}}
 .summary{display:flex;gap:12px;margin-bottom:16px;flex-wrap:wrap}
 .stat{flex:1;min-width:120px;background:var(--card);border:1px solid var(--line);border-radius:12px;
       padding:14px 16px;text-align:center;box-shadow:0 1px 3px rgba(20,20,43,.06)}
 .stat .n{font-size:26px;font-weight:700}
 .stat .l{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.5px;margin-top:2px}
 .stat.neg .n{color:var(--neg)}.stat.neu .n{color:#d4a90a}.stat.pos .n{color:var(--pos)}
 .card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px;
       margin-bottom:10px;box-shadow:0 1px 3px rgba(20,20,43,.05)}
 .tag{display:inline-block;padding:3px 11px;border-radius:999px;font-size:11.5px;font-weight:700;
      color:#fff;text-transform:uppercase;letter-spacing:.4px}
 .neg{background:var(--neg)}.neu{background:var(--neu);color:#4a3b00}.pos{background:var(--pos)}
 .bar{height:6px;background:#edeff7;border-radius:99px;margin-top:8px;overflow:hidden}
 .bar i{display:block;height:100%;border-radius:99px}
 .bar.neg i{background:var(--neg)}.bar.neu i{background:var(--neu)}.bar.pos i{background:var(--pos)}
 .rev{margin-top:8px;font-size:14px;line-height:1.5;color:#333}
 .pct{font-size:12px;color:var(--muted);margin-left:8px}
 .report-card{margin-top:20px}
 .report-card .head{display:flex;align-items:center;gap:10px;margin-bottom:10px}
 .report-card .head .ic{width:34px;height:34px;border-radius:9px;background:linear-gradient(120deg,var(--acc),var(--acc2));
       display:flex;align-items:center;justify-content:center;font-size:17px}
 .report-card .head b{font-size:15px}
 .md{font-size:14.5px;line-height:1.65}
 .md h2{font-size:17px;margin:14px 0 8px}
 .md h3{font-size:15px;margin:14px 0 6px;color:var(--acc)}
 .md ul,.md ol{padding-left:22px;margin:6px 0}
 .md li{margin:4px 0}
 .md strong{color:var(--ink)}
 .md p{margin:8px 0}
 .err{background:#fdecea;border:1px solid #f5c6c0;color:#b03a2e;border-radius:10px;padding:12px 14px;font-size:14px}
 .err-inline{color:#b03a2e;font-weight:600}
 .chip.err{background:#fdecea;color:#b03a2e}
 .chip.warn{background:#fff8e1;color:#8a6d00}
 footer{margin-top:26px;text-align:center;font-size:12px;color:#9aa0b4}
</style></head><body><div class="wrap">

<div class="hero">
  <h1>Review Analyst</h1>
  <p>ML-модель (LinearSVC + TF-IDF, F1 macro 0.74) определяет тональность отзывов,
     а LLM (Qwen 27B) превращает негатив в готовый бизнес-отчёт с рекомендациями.</p>
  <div class="badges"><span class="badge">ML: 3-классовая классификация</span>
       <span class="badge">AI: LLM-отчёт</span><span class="badge">FastAPI</span></div>
</div>

<div class="panel">
  <h2>1 · Источник отзывов</h2>
  <div class="drop" id="drop" onclick="document.getElementById('file').click()">
    <b>Перетащите файл сюда</b> или нажмите, чтобы выбрать
    <div class="hint">.txt / .csv / .jsonl / .json · до 5 МБ на файл · до 200 отзывов · до 5000 символов на отзыв</div>
  </div>
  <input type="file" id="file" multiple accept=".txt,.csv,.jsonl,.json,.text">
  <div class="files" id="files"></div>
  <textarea id="in" placeholder="…или вставьте отзывы вручную, по одному в строке&#10;Отличный сервис, всё понравилось!&#10;Менеджер был груб, ждать пришлось час…&#10;Нормально, без претензий"></textarea>
  <div class="row">
    <button id="go" onclick="run()">Проанализировать</button>
    <span class="count" id="count">0 отзывов</span>
    <span id="status"></span>
  </div>
</div>

<div id="out"></div>
<footer>Review Analyst · гибридный проект ML + AI · данные: Yelp reviews (HuggingFace)</footer>
</div>

<script>
const $=id=>document.getElementById(id);
const clsOf={negative:'neg',neutral:'neu',positive:'pos'};
const nameOf={negative:'Негатив',neutral:'Нейтрал',positive:'Позитив'};
const ALLOWED_EXT=['txt','csv','jsonl','json','text'];
const MAX_FILE_SIZE=5*1024*1024;   // 5 MB
const MAX_REVIEWS=200;
const MAX_REVIEW_LEN=5000;

function showStatus(msg,isErr){
  const el=$('status');
  el.innerHTML=isErr?'<span class="err-inline">⚠ '+escapeHtml(msg)+'</span>':escapeHtml(msg);
}
function addChip(text,kind){
  const c=document.createElement('span');
  c.className='chip'+(kind?' '+kind:'');
  c.textContent=text;
  $('files').appendChild(c);
}

function escapeHtml(s){return s.replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
function md(s){
  if(window.marked){try{return marked.parse(s);}catch(e){}}
  return '<pre style="white-space:pre-wrap">'+escapeHtml(s)+'</pre>';
}
function updateCount(){
  const n=$('in').value.split('\\n').map(s=>s.trim()).filter(Boolean).length;
  $('count').textContent=n+' '+(n%10===1&&n%100!==11?'отзыв':(n%10>=2&&n%10<=4&&(n%100<10||n%100>=20)?'отзыва':'отзывов'));
}
$('in').addEventListener('input',updateCount);

/* ---------- file parsing ---------- */
function pickText(obj){
  const pref=['review','text','comment','feedback','content','message','отзыв','текст'];
  for(const k of pref){const v=obj[k];if(typeof v==='string'&&v.trim())return v;}
  let best='';
  for(const v of Object.values(obj))if(typeof v==='string'&&v.length>best.length)best=v;
  return best;
}
function parseCsv(text){
  const rows=[];let row=[],cur='',q=false;
  for(let i=0;i<text.length;i++){const c=text[i];
    if(q){if(c==='"'){if(text[i+1]==='"'){cur+='"';i++;}else q=false;}else cur+=c;}
    else if(c==='"')q=true;
    else if(c===','){row.push(cur);cur='';}
    else if(c==='\\n'||c==='\\r'){if(c==='\\r'&&text[i+1]==='\\n')i++;row.push(cur);cur='';
      if(row.some(x=>x!==''))rows.push(row);row=[];}
    else cur+=c;}
  if(cur!==''||row.length){row.push(cur);if(row.some(x=>x!==''))rows.push(row);}
  if(!rows.length)return[];
  const HEADER_WORDS=['review','text','comment','feedback','content','message','отзыв','текст'];
  const head=rows[0].map(h=>h.toLowerCase().trim());
  const headerCol=head.findIndex(h=>HEADER_WORDS.includes(h));
  const hasHeader=headerCol>=0;
  const dataRows=hasHeader?rows.slice(1):rows;
  let col;
  if(hasHeader){col=headerCol;}
  else{ // no header: pick the column with the longest average text
    col=0;let bestLen=-1;
    for(let c=0;c<rows[0].length;c++){
      const avg=dataRows.reduce((s,r)=>s+((r[c]||'').length),0)/Math.max(dataRows.length,1);
      if(avg>bestLen){bestLen=avg;col=c;}}
  }
  return dataRows.map(r=>(r[col]||'').trim()).filter(Boolean);
}
function parseFile(name,text){
  const ext=name.toLowerCase().split('.').pop();
  if(ext==='csv')return parseCsv(text);
  if(ext==='jsonl')return text.split('\\n').map(l=>l.trim()).filter(Boolean)
      .map(l=>{try{const o=JSON.parse(l);return typeof o==='string'?o:pickText(o);}catch(e){return l;}}).filter(Boolean);
  if(ext==='json'){try{const o=JSON.parse(text);
      if(Array.isArray(o))return o.map(x=>typeof x==='string'?x:pickText(x)).filter(Boolean);
      return [pickText(o)].filter(Boolean);}catch(e){return text.split('\\n').filter(Boolean);}}
  return text.split('\\n').map(s=>s.trim()).filter(Boolean);
}
function addFiles(list){
  let errors=0;
  [...list].forEach(f=>{
    const ext=f.name.toLowerCase().split('.').pop();
    if(!ALLOWED_EXT.includes(ext)){
      addChip(f.name+' — неподдерживаемый формат','err');errors++;
      return;
    }
    if(f.size>MAX_FILE_SIZE){
      addChip(f.name+' — больше 5 МБ','err');errors++;
      return;
    }
    const rd=new FileReader();
    rd.onerror=()=>{addChip(f.name+' — не удалось прочитать','err');errors++;};
    rd.onload=()=>{
      const reviews=parseFile(f.name,String(rd.result));
      if(!reviews.length){
        addChip(f.name+' — отзывов не найдено','err');errors++;
        return;
      }
      const ta=$('in');
      ta.value=(ta.value?ta.value+'\\n':'')+reviews.join('\\n');
      updateCount();
      addChip(f.name+' · '+reviews.length);
    };
    rd.readAsText(f);
  });
  if(errors)showStatus('Некоторые файлы отклонены — см. метки выше',true);
}
$('file').addEventListener('change',e=>{addFiles(e.target.files);e.target.value='';});
const drop=$('drop');
['dragenter','dragover'].forEach(ev=>drop.addEventListener(ev,e=>{e.preventDefault();drop.classList.add('over');}));
['dragleave','drop'].forEach(ev=>drop.addEventListener(ev,e=>{e.preventDefault();drop.classList.remove('over');}));
drop.addEventListener('drop',e=>addFiles(e.dataTransfer.files));

/* ---------- analyze ---------- */
async function run(){
  let reviews=$('in').value.split('\\n').map(s=>s.trim()).filter(Boolean);
  if(!reviews.length){showStatus('Введите или загрузите хотя бы один отзыв',true);return;}
  if(reviews.length>MAX_REVIEWS){
    showStatus('Слишком много отзывов: '+reviews.length+', максимум '+MAX_REVIEWS,true);return;
  }
  let truncated=0;
  reviews=reviews.map(r=>{
    if(r.length>MAX_REVIEW_LEN){truncated++;return r.slice(0,MAX_REVIEW_LEN)+'…';}
    return r;
  });
  const btn=$('go');btn.disabled=true;
  $('status').innerHTML='<span class="spinner"></span>Анализирую… LLM-отчёт может занять 10–30 сек';
  $('out').innerHTML='';
  try{
    const r=await fetch('/analyze',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({reviews:reviews,generate_report:true})});
    const d=await r.json();
    if(!r.ok)throw new Error(d.detail||r.statusText);
    let html='<div class="summary">';
    for(const k of ['negative','neutral','positive'])
      html+=`<div class="stat ${clsOf[k]}"><div class="n">${d.summary[k]||0}</div><div class="l">${nameOf[k]}</div></div>`;
    html+='</div>';
    for(const x of d.results){
      const p=(x.probabilities[x.sentiment]||0)*100;
      html+=`<div class="card"><span class="tag ${clsOf[x.sentiment]}">${nameOf[x.sentiment]}</span>`+
            `<span class="pct">${p.toFixed(1)}%</span>`+
            `<div class="bar ${clsOf[x.sentiment]}"><i style="width:${p.toFixed(1)}%"></i></div>`+
            `<div class="rev">${escapeHtml(x.review)}</div></div>`;
    }
    if(d.llm_report)
      html+=`<div class="panel report-card"><div class="head"><div class="ic">📊</div><b>AI-отчёт (LLM)</b></div>`+
            `<div class="md">${md(d.llm_report)}</div></div>`;
    $('out').innerHTML=html;
    showStatus(truncated?('Готово. Отзывов обрезано до '+MAX_REVIEW_LEN+' символов: '+truncated):'Готово.');
  }catch(e){
    $('out').innerHTML=`<div class="err">Ошибка: ${escapeHtml(e.message)}</div>`;
    $('status').innerHTML='';
  }
  btn.disabled=false;
}
updateCount();
</script></body></html>"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
