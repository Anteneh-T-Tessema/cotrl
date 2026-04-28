/* Game of 24 — frontend logic */

// ── State ──────────────────────────────────────────────────────────
let currentNumbers = [3, 8, 8, 3];
let solving = false;

// ── Elements ───────────────────────────────────────────────────────
const inputs       = Array.from(document.querySelectorAll('.number-input'));
const btnSolve     = document.getElementById('btn-solve');
const btnRandom    = document.getElementById('btn-random');
const btnVerify    = document.getElementById('btn-verify');
const iterSlider   = document.getElementById('iter-slider');
const iterLabel    = document.getElementById('iter-label');
const progressWrap = document.getElementById('progress-wrap');
const progressBar  = document.getElementById('progress-bar');
const progressPct  = document.getElementById('progress-pct');
const progressLbl  = document.getElementById('progress-label');
const progressSub  = document.getElementById('progress-sub');
const resultWrap   = document.getElementById('result-wrap');
const resultSolved = document.getElementById('result-solved');
const resultFailed = document.getElementById('result-failed');
const resultExpr   = document.getElementById('result-expr');
const resultMeta   = document.getElementById('result-meta');
const verifyInput  = document.getElementById('verify-input');
const verifyResult = document.getElementById('verify-result');
const solverCard   = document.querySelector('.solver-card');

// ── Sync inputs → state ─────────────────────────────────────────────
inputs.forEach((inp, idx) => {
  inp.addEventListener('input', () => {
    const v = parseInt(inp.value, 10);
    if (!isNaN(v) && v >= 1 && v <= 13) {
      currentNumbers[idx] = v;
    }
  });
  inp.addEventListener('focus', () => inp.select());
});

// ── Iteration slider ────────────────────────────────────────────────
iterSlider.addEventListener('input', () => {
  iterLabel.textContent = iterSlider.value;
});

// ── Random puzzle ───────────────────────────────────────────────────
btnRandom.addEventListener('click', async () => {
  btnRandom.disabled = true;
  try {
    const res = await fetch('/api/random');
    const data = await res.json();
    currentNumbers = data.numbers;
    inputs.forEach((inp, i) => {
      inp.value = currentNumbers[i];
      inp.closest('.number-card').style.animation = 'none';
      requestAnimationFrame(() => {
        inp.closest('.number-card').style.animation = 'fade-in 0.3s ease';
      });
    });
    resetResult();
  } catch (e) {
    console.error(e);
  } finally {
    btnRandom.disabled = false;
  }
});

// ── Solve via SSE ───────────────────────────────────────────────────
btnSolve.addEventListener('click', async () => {
  if (solving) return;
  // Sync numbers from inputs in case user typed
  inputs.forEach((inp, i) => {
    const v = parseInt(inp.value, 10);
    if (!isNaN(v) && v >= 1 && v <= 13) currentNumbers[i] = v;
  });

  solving = true;
  solverCard.classList.add('solving');
  btnSolve.disabled = true;
  resetResult();
  showProgress(true);

  const iterations = parseInt(iterSlider.value, 10);

  try {
    const res = await fetch('/api/solve', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ numbers: currentNumbers, iterations }),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Server error');
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const lines = buffer.split('\n');
      buffer = lines.pop() ?? '';

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const json = line.slice(6).trim();
        if (!json || json === '{"type": "end"}') continue;
        try {
          const event = JSON.parse(json);
          handleEvent(event, iterations);
        } catch (_) { /* partial JSON, ignore */ }
      }
    }
  } catch (e) {
    progressLbl.textContent = 'Error: ' + e.message;
    progressBar.style.width = '0%';
  } finally {
    solving = false;
    solverCard.classList.remove('solving');
    btnSolve.disabled = false;
  }
});

function handleEvent(event, totalIterations) {
  if (event.type === 'progress') {
    const pct = Math.round((event.iteration / totalIterations) * 100);
    progressBar.style.width = pct + '%';
    progressPct.textContent = pct + '%';
    if (event.solved) {
      progressLbl.textContent = 'Solution found! Finishing…';
    } else {
      progressLbl.textContent = `Searching… iteration ${event.iteration} / ${totalIterations}`;
    }
    if (event.best_expr) {
      progressSub.textContent = 'Best so far: ' + event.best_expr;
    }
  } else if (event.type === 'done') {
    showProgress(false);
    showResult(event);
  }
}

function showResult(event) {
  resultWrap.hidden = false;
  if (event.solved) {
    resultSolved.hidden = false;
    resultFailed.hidden = true;
    resultExpr.textContent = event.expression;
    resultMeta.textContent =
      `Verified: ${event.expression} = 24 ✓  ·  ${event.elapsed_ms}ms  ·  ${event.iterations} iterations`;
  } else {
    resultSolved.hidden = true;
    resultFailed.hidden = false;
  }
}

function showProgress(show) {
  progressWrap.hidden = !show;
  if (show) {
    progressBar.style.width = '0%';
    progressPct.textContent = '0%';
    progressLbl.textContent = 'Starting MCTS search…';
    progressSub.textContent = '';
  }
}

function resetResult() {
  resultWrap.hidden = true;
  resultSolved.hidden = true;
  resultFailed.hidden = true;
  verifyResult.hidden = true;
}

// ── Verify custom expression ────────────────────────────────────────
btnVerify.addEventListener('click', async () => {
  const expr = verifyInput.value.trim();
  if (!expr) return;

  inputs.forEach((inp, i) => {
    const v = parseInt(inp.value, 10);
    if (!isNaN(v) && v >= 1 && v <= 13) currentNumbers[i] = v;
  });

  verifyResult.hidden = true;
  verifyResult.className = 'verify-result';

  try {
    const res = await fetch('/api/verify', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ numbers: currentNumbers, expression: expr }),
    });
    const data = await res.json();
    verifyResult.hidden = false;
    if (data.valid) {
      verifyResult.classList.add('ok');
      verifyResult.textContent = `✓  ${expr} = 24  (reward = ${data.reward.toFixed(2)})`;
    } else {
      verifyResult.classList.add('err');
      verifyResult.textContent = `✕  ${data.error || 'Invalid expression'}`;
    }
  } catch (e) {
    verifyResult.hidden = false;
    verifyResult.classList.add('err');
    verifyResult.textContent = 'Network error: ' + e.message;
  }
});

verifyInput.addEventListener('keydown', e => {
  if (e.key === 'Enter') btnVerify.click();
});

// ── Animate benchmark bars on scroll ───────────────────────────────
const benchRows = document.querySelectorAll('.bench-bar');
const observer = new IntersectionObserver(entries => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      // bars already have inline width set; just trigger a reflow to animate
      const bar = entry.target;
      const target = bar.style.width;
      bar.style.width = '0%';
      requestAnimationFrame(() => {
        requestAnimationFrame(() => { bar.style.width = target; });
      });
      observer.unobserve(bar);
    }
  });
}, { threshold: 0.3 });

benchRows.forEach(b => observer.observe(b));
