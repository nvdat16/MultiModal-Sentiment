// URL của backend FastAPI
const BACKEND_API = "http://localhost:8000/predict";

let imgFile = null; // Lưu trữ file gốc thay vì base64 để gửi lên server
let historyData = [];
let accHistory = {text:[], image:[], multi:[]};

// --- Initialization ---
document.getElementById('img-input').addEventListener('change', handleImage);
document.getElementById('analyze-btn').addEventListener('click', analyze);
document.getElementById('sample-btn').addEventListener('click', loadSample);
document.getElementById('clear-btn').addEventListener('click', clearImage);

// --- Image Handlers ---
function handleImage(e){
  const file = e.target.files[0]; if(!file) return;
  imgFile = file; // Lưu file để gửi đi
  
  const reader = new FileReader();
  reader.onload = ev => {
    const prev = document.getElementById('img-preview');
    prev.src = ev.target.result; prev.style.display='block';
    document.getElementById('upload-icon').style.display='none';
    document.getElementById('upload-label').style.display='none';
    document.getElementById('upload-zone').classList.add('has-image');
    document.getElementById('input-status').textContent = '✓ Ảnh đã tải · ' + file.name;
    
    document.getElementById('att-img').src = ev.target.result;
    document.getElementById('att-img').style.display='block';
    document.getElementById('att-placeholder').style.display='none';
  };
  reader.readAsDataURL(file);
}

function clearImage(){
  imgFile = null;
  document.getElementById('img-preview').style.display='none';
  document.getElementById('img-preview').src='';
  document.getElementById('upload-icon').style.display='';
  document.getElementById('upload-label').style.display='';
  document.getElementById('upload-zone').classList.remove('has-image');
  document.getElementById('att-img').style.display='none';
  document.getElementById('att-placeholder').style.display='';
  document.getElementById('input-status').textContent='Nhập text và/hoặc upload ảnh';
  clearCanvas();
}

async function loadSample(){
  // Tạo blob từ ảnh mẫu nếu cần, ở đây giả lập bằng cách fetch một ảnh hoặc dùng placeholder
  const response = await fetch('https://picsum.photos/400/300');
  const blob = await response.blob();
  imgFile = new File([blob], "sample.jpg", { type: "image/jpeg" });
  
  const prev = document.getElementById('img-preview');
  prev.src = URL.createObjectURL(blob); prev.style.display='block';
  document.getElementById('upload-icon').style.display='none';
  document.getElementById('upload-label').style.display='none';
  document.getElementById('upload-zone').classList.add('has-image');
  document.getElementById('att-img').src = prev.src;
  document.getElementById('att-img').style.display='block';
  document.getElementById('att-placeholder').style.display='none';
}

// --- Analysis Logic ---
async function analyze(){
  const text = document.getElementById('txt-input').value.trim();
  const mode = document.getElementById('mode-sel').value;
  
  if(!text && !imgFile){ 
      alert('Vui lòng nhập text hoặc upload ảnh'); 
      return; 
  }
  
  setBtnState(true);
  document.getElementById('result-mode-tag').textContent = mode==='multi'?'Multimodal':mode==='text'?'Text Only':'Image Only';

  // Chuẩn bị dữ liệu gửi lên FastAPI
  const formData = new FormData();
  if (text) formData.append("text", text);
  if (imgFile && mode !== 'text') formData.append("file", imgFile);

  try {
    const response = await fetch(BACKEND_API, {
      method: 'POST',
      body: formData // Fetch tự động set Content-Type là multipart/form-data
    });

    if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
    }

    const result = await response.json();

    // Hiển thị kết quả
    updateMainResult(result);
    updateComparison(result);
    renderAttention(text, result);
    addHistory(text, result);

  } catch(err){
    console.error(err);
    alert('Lỗi kết nối tới Backend: ' + err.message);
  }
  setBtnState(false);
}

// --- UI Update Helpers ---
function updateMainResult(result){
  const sl = document.getElementById('sent-label');
  sl.textContent = result.sentiment;
  sl.className = 'sent-badge ' + getSentClass(result.sentiment);
  
  const bars = ['pos', 'neu', 'neg'];
  bars.forEach(b => {
    const key = b==='pos'?'positive':b==='neu'?'neutral':'negative';
    const val = result[key] || 0;
    document.getElementById(`bar-${b}`).style.width = val + '%';
    document.getElementById(`val-${b}`).textContent = val + '%';
  });

  const rb = document.getElementById('reasoning-box');
  rb.textContent = result.reasoning || "Không có giải thích từ model.";
  rb.style.display='block';
}

function updateComparison(result){
  updateModelCard('text', result.text_only_sentiment, result.text_only_confidence);
  updateModelCard('image', result.image_only_sentiment, result.image_only_confidence);
  updateModelCard('multi', result.sentiment, result.confidence);
  
  highlightBest([result.text_only_confidence || 0, result.image_only_confidence || 0, result.confidence || 0]);
  
  accHistory.text.push(result.text_only_confidence || 0);
  accHistory.image.push(result.image_only_confidence || 0);
  accHistory.multi.push(result.confidence || 0);
  drawAccChart();
}

function renderAttention(text, result){
  if(result.text_attention) renderTextAttention(text, result.text_attention);
  
  const canvas = document.getElementById('att-canvas');
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0,0,canvas.width,canvas.height);

  if(imgFile && result.image_regions){
    renderImageAttention(result.image_regions);
    const ib = document.getElementById('img-desc-box');
    ib.textContent = '📷 ' + (result.image_description || 'Phân tích vùng ảnh thành công.');
    ib.style.display='block';
  }
}

// Các hàm vẽ Canvas và giao diện giữ nguyên logic như cũ...
function renderTextAttention(text, attentionData){
  const attMap = {};
  attentionData.forEach(a=>{ attMap[a.word.toLowerCase()] = a.score; });
  const words = text.split(/(\s+)/);
  const container = document.getElementById('text-attention');
  container.innerHTML = '';

  words.forEach(part=>{
    if(/^\s+$/.test(part)){ container.appendChild(document.createTextNode(part)); return; }
    const clean = part.replace(/[^\w]/g,'').toLowerCase();
    const score = attMap[clean] || 0;
    const span = document.createElement('span');
    span.className = 'word-token';
    span.textContent = part;
    if(score > 0){
      const alpha = score/100;
      span.style.background = `rgba(74,222,128,${alpha*0.35})`;
      span.title = `Attention: ${score}%`;
    }
    container.appendChild(span);
  });
}

function renderImageAttention(regions){
  const canvas = document.getElementById('att-canvas');
  const wrap = document.getElementById('img-att-wrap');
  canvas.width = wrap.offsetWidth; canvas.height = wrap.offsetHeight;
  const ctx = canvas.getContext('2d');

  regions.forEach(r=>{
    const x=r.x*canvas.width, y=r.y*canvas.height, w=r.w*canvas.width, h=r.h*canvas.height;
    ctx.strokeStyle = `rgba(129,140,248,0.8)`;
    ctx.lineWidth = 2;
    ctx.strokeRect(x,y,w,h);
    ctx.fillStyle='rgba(129,140,248,0.9)';
    ctx.font='10px DM Mono';
    ctx.fillText(`${r.label} ${r.score}%`, x+3, y > 15 ? y-5 : y+15);
  });
}

function drawAccChart(){
  const canvas = document.getElementById('acc-chart');
  const W = canvas.parentElement.offsetWidth;
  canvas.width = W; canvas.height = 60;
  const ctx = canvas.getContext('2d');
  const colors = ['#94a3b8', '#818cf8', '#4ade80'];
  const keys = ['text', 'image', 'multi'];

  keys.forEach((k, idx) => {
    const data = accHistory[k];
    if(data.length < 1) return;
    ctx.beginPath();
    ctx.strokeStyle = colors[idx];
    ctx.lineWidth = 2;
    data.forEach((v, i) => {
      const x = (i/(Math.max(data.length-1, 1))) * (W-20) + 10;
      const y = 55 - (v/100)*50;
      i === 0 ? ctx.moveTo(x,y) : ctx.lineTo(x,y);
    });
    ctx.stroke();
  });
}

function addHistory(text, result){
  historyData.unshift({text, result});
  const tbody = document.getElementById('hist-body');
  tbody.innerHTML = '';
  historyData.slice(0, 10).forEach((h, i) => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${historyData.length-i}</td>
      <td>${h.text ? h.text.slice(0,30) + '...' : 'Chỉ ảnh'}</td>
      <td>${imgFile?'✓':''}</td>
      <td><span class="pill ${getPillClass(h.result.text_only_sentiment)}">${h.result.text_only_sentiment || '—'}</span></td>
      <td><span class="pill ${getPillClass(h.result.image_only_sentiment)}">${h.result.image_only_sentiment || '—'}</span></td>
      <td><span class="pill ${getPillClass(h.result.sentiment)}">${h.result.sentiment}</span></td>
      <td>${h.result.confidence}%</td>
    `;
    tbody.appendChild(tr);
  });
  document.getElementById('hist-count').textContent = historyData.length + ' lần';
}

function updateModelCard(id, sent, conf){
  const el = document.getElementById('mc-'+id);
  el.querySelector('.model-sent').textContent = sent || '—';
  el.querySelector('.model-sent').className = 'model-sent ' + getSentClass(sent);
  el.querySelector('.model-acc').textContent = (conf || 0) + '%';
  el.querySelector('.model-bar').style.width = (conf || 0) + '%';
}

function highlightBest(arr){
  const max = Math.max(...arr);
  ['text','image','multi'].forEach((id, i) => {
    const el = document.getElementById('mc-'+id);
    el.classList.toggle('best', arr[i] === max && max > 0);
  });
}

function getSentClass(s){
  if(!s) return '';
  const low = s.toLowerCase();
  return low.includes('pos') ? 'sent-pos' : low.includes('neg') ? 'sent-neg' : 'sent-neu';
}

function getPillClass(s){
  if(!s) return 'pill-neu';
  const low = s.toLowerCase();
  return low.includes('pos') ? 'pill-pos' : low.includes('neg') ? 'pill-neg' : 'pill-neu';
}

function setBtnState(loading){
  const btn = document.getElementById('analyze-btn');
  btn.disabled = loading;
  btn.innerHTML = loading ? '<span class="spinner"></span> Đang xử lý...' : 'Phân tích →';
}

function clearCanvas(){
  const canvas = document.getElementById('att-canvas');
  if(canvas) canvas.getContext('2d').clearRect(0,0,canvas.width,canvas.height);
}