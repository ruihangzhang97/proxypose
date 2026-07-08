const img    = document.getElementById('frame');
const canvas = document.getElementById('overlay');
const ctx    = canvas.getContext('2d');
let points   = [];

function syncCanvas() {
  canvas.width  = img.offsetWidth;
  canvas.height = img.offsetHeight;
  canvas.style.width  = img.offsetWidth  + 'px';
  canvas.style.height = img.offsetHeight + 'px';
  redraw();
}
img.addEventListener('load', syncCanvas);
window.addEventListener('resize', () => syncCanvas());
if (img.complete) syncCanvas();

function toNatural(dx, dy) {
  return [
    Math.round(dx * img.naturalWidth  / img.offsetWidth),
    Math.round(dy * img.naturalHeight / img.offsetHeight),
  ];
}

function toDisplay(px, py) {
  return [
    px * img.offsetWidth  / img.naturalWidth,
    py * img.offsetHeight / img.naturalHeight,
  ];
}

function redraw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  points.forEach(([px, py]) => {
    const [dx, dy] = toDisplay(px, py);
    ctx.beginPath();
    ctx.arc(dx, dy, 6, 0, Math.PI * 2);
    ctx.fillStyle = '#50dc50';
    ctx.fill();
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = '#fff';
    ctx.stroke();
  });
}

document.getElementById('container').addEventListener('click', e => {
  const r = img.getBoundingClientRect();
  const [px, py] = toNatural(e.clientX - r.left, e.clientY - r.top);
  points.push([px, py]);
  redraw();
});

function undo() {
  if (points.length) points.pop();
  redraw();
}

function clearPts() {
  points = [];
  redraw();
}

async function save() {
  if (!points.length) {
    document.getElementById('status').textContent = 'No points — click on the target first.';
    return;
  }
  document.getElementById('status').textContent = 'Saving…';
  const resp = await fetch('/save', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ points, frame_index: 0 }),
  });
  document.getElementById('status').textContent = await resp.text();
  document.getElementById('saveBtn').disabled = true;
}

document.addEventListener('keydown', e => {
  if      (e.key === 'z')     undo();
  else if (e.key === 'c')     clearPts();
  else if (e.key === 'Enter') save();
});
