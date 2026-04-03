(function () {
  const lossCanvas = document.getElementById('lossChart');
  const aucCanvas = document.getElementById('aucChart');
  const perClassCanvas = document.getElementById('perClassAucChart');

  const metricBestAuc = document.getElementById('metric-best-auc');
  const metricBestF1 = document.getElementById('metric-best-f1');
  const metricEpochs = document.getElementById('metric-epochs');
  const metricTime = document.getElementById('metric-time');

  if (!lossCanvas || !aucCanvas || !perClassCanvas) {
    return; // Not on the dashboard page
  }

  function safeNumber(v) {
    const x = parseFloat(v);
    return Number.isFinite(x) ? x : null;
  }

  async function fetchJson(url) {
    const res = await fetch(url);
    if (!res.ok) {
      throw new Error('Request failed: ' + res.status);
    }
    return res.json();
  }

  async function init() {
    try {
      const [logData, metrics] = await Promise.all([
        fetchJson('/api/training-log'),
        fetchJson('/api/metrics').catch(() => ({})),
      ]);

      const epochs = (logData.epochs || []).map((row) => parseInt(row.epoch, 10));
      const trainLoss = (logData.epochs || []).map((row) => safeNumber(row.train_loss));
      const valLoss = (logData.epochs || []).map((row) => safeNumber(row.val_loss));
      const valAuc = (logData.epochs || []).map((row) => safeNumber(row.val_auc));

      if (metricEpochs) {
        metricEpochs.textContent = epochs.length ? String(epochs.length) : '–';
      }

      if (metricTime) {
        const last = (logData.epochs || [])[epochs.length - 1];
        metricTime.textContent = last && last.time ? last.time : '–';
      }

      if (valAuc.length) {
        const bestAuc = Math.max(...valAuc.filter((v) => v != null));
        if (metricBestAuc && Number.isFinite(bestAuc)) {
          metricBestAuc.textContent = bestAuc.toFixed(3);
        }
      }

      if (metrics && typeof metrics === 'object') {
        if (metrics.macro_f1 != null && metricBestF1) {
          metricBestF1.textContent = safeNumber(metrics.macro_f1)?.toFixed(3) || '–';
        }
      }

      // Loss chart
      if (epochs.length && trainLoss.length && valLoss.length) {
        new Chart(lossCanvas.getContext('2d'), {
          type: 'line',
          data: {
            labels: epochs,
            datasets: [
              {
                label: 'Train loss',
                data: trainLoss,
                borderColor: '#00ff88',
                tension: 0.2,
              },
              {
                label: 'Val loss',
                data: valLoss,
                borderColor: '#f97316',
                tension: 0.2,
              },
            ],
          },
          options: {
            responsive: true,
            plugins: {
              legend: { labels: { color: '#c9d1d9' } },
            },
            scales: {
              x: { ticks: { color: '#8b949e' } },
              y: { ticks: { color: '#8b949e' } },
            },
          },
        });
      }

      // AUC chart
      if (epochs.length && valAuc.length) {
        new Chart(aucCanvas.getContext('2d'), {
          type: 'line',
          data: {
            labels: epochs,
            datasets: [
              {
                label: 'Val AUC',
                data: valAuc,
                borderColor: '#00d4ff',
                tension: 0.2,
              },
            ],
          },
          options: {
            responsive: true,
            plugins: {
              legend: { labels: { color: '#c9d1d9' } },
            },
            scales: {
              x: { ticks: { color: '#8b949e' } },
              y: { ticks: { color: '#8b949e' } },
            },
          },
        });
      }

      // Per-class AUC bar chart
      if (metrics && metrics.per_class) {
        const labels = Object.keys(metrics.per_class);
        const aucValues = labels.map((k) => safeNumber(metrics.per_class[k].auc));

        new Chart(perClassCanvas.getContext('2d'), {
          type: 'bar',
          data: {
            labels,
            datasets: [
              {
                label: 'AUC',
                data: aucValues,
                backgroundColor: '#00ff88',
              },
            ],
          },
          options: {
            indexAxis: 'y',
            responsive: true,
            plugins: {
              legend: { display: false },
            },
            scales: {
              x: {
                ticks: { color: '#8b949e' },
                min: 0,
                max: 1,
              },
              y: { ticks: { color: '#8b949e' } },
            },
          },
        });
      }
    } catch (err) {
      // Dashboard is best-effort; log to console only.
      console.error('Failed to initialise dashboard:', err);
    }
  }

  document.addEventListener('DOMContentLoaded', init);
})();
