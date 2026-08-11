// Transformer Dynamics Lab - Clean Research Frontend Client JS

const API_BASE = window.location.origin + '/api';

// Global Application State
let appState = {
    baselineData: null,
    modifiedData: null,
    benchmarkData: null,
    currentLayer: 0,
    theme: localStorage.getItem('theme') || 'dark'
};

document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    initThemeToggle();
    initControls();
    runObservation();
});

function initThemeToggle() {
    const themeBtn = document.getElementById('btn-theme-toggle');
    if (!themeBtn) return;

    applyTheme(appState.theme);

    themeBtn.addEventListener('click', () => {
        appState.theme = appState.theme === 'dark' ? 'light' : 'dark';
        localStorage.setItem('theme', appState.theme);
        applyTheme(appState.theme);
        
        // Re-render charts with updated theme color schemes
        if (appState.baselineData) {
            renderObservationCharts(appState.baselineData);
            updateLayerExplorer();
        }
        if (appState.benchmarkData) {
            renderBenchmarkPlots(appState.benchmarkData);
        }
    });
}

function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    const icon = document.getElementById('theme-icon');
    const label = document.getElementById('theme-label');
    if (theme === 'light') {
        if (icon) icon.innerText = '☀️';
        if (label) label.innerText = 'Light Mode';
    } else {
        if (icon) icon.innerText = '🌙';
        if (label) label.innerText = 'Dark Mode';
    }
}

function getPlotThemeColors() {
    const isLight = appState.theme === 'light';
    return {
        fontColor: isLight ? '#475569' : '#94A3B8',
        gridColor: isLight ? '#E2E8F0' : '#1E293B'
    };
}

// Tab Navigation
function initTabs() {
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabPanes = document.querySelectorAll('.tab-pane');

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            tabBtns.forEach(b => b.classList.remove('active'));
            tabPanes.forEach(p => p.classList.remove('active'));

            btn.classList.add('active');
            const tabId = btn.getAttribute('data-tab');
            document.getElementById(tabId).classList.add('active');
            
            // Resize Plotly charts when tab switches
            window.dispatchEvent(new Event('resize'));
        });
    });
}

function initControls() {
    document.getElementById('btn-analyze').addEventListener('click', () => {
        runObservation();
    });

    document.getElementById('btn-run-experiment').addEventListener('click', () => {
        runExperiment();
    });

    const layerSlider = document.getElementById('layer-range');
    layerSlider.addEventListener('input', (e) => {
        appState.currentLayer = parseInt(e.target.value);
        document.getElementById('layer-display').innerText = `Layer ${appState.currentLayer}`;
        updateLayerExplorer();
    });
}

// Stage 1: Observation & Dynamics
async function runObservation() {
    const text = document.getElementById('input-text').value;
    const model = document.getElementById('select-model').value;

    try {
        const resp = await fetch(`${API_BASE}/observe`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                text: text,
                model_name: model,
                pe_mode: 'default'
            })
        });

        const result = await resp.json();
        if (result.status === 'success') {
            appState.baselineData = result.live_data;
            appState.benchmarkData = result.benchmark_data;
            
            updateMetricSummaryCards(result.live_data.metrics);
            renderObservationCharts(result.live_data);
            renderObservationNotes(model, result.live_data);
            
            // Sentence Evolution Flow (Slide 14 Demo)
            renderSentenceEvolution(result.live_data.sentence_evolution);
            
            // Render Paper Visuals directly inside Observation
            renderBenchmarkPlots(result.benchmark_data);
            
            // Update Layer Slider limits
            const numLayers = result.live_data.num_layers;
            const slider = document.getElementById('layer-range');
            slider.max = numLayers;
            if (appState.currentLayer > numLayers) appState.currentLayer = 0;
            slider.value = appState.currentLayer;
            document.getElementById('layer-display').innerText = `Layer ${appState.currentLayer}`;
            
            updateLayerExplorer();
        }
    } catch (err) {
        console.error('Observation error:', err);
    }
}

function updateMetricSummaryCards(metrics) {
    const lastIdx = metrics.distance.length - 1;
    document.getElementById('val-distance').innerText = metrics.distance[lastIdx].toFixed(2);
    document.getElementById('val-similarity').innerText = metrics.cosine_similarity[lastIdx].toFixed(3);
    document.getElementById('val-rank').innerText = metrics.effective_rank[lastIdx].toFixed(2);
    
    const maxNsi = Math.max(...metrics.nsi);
    document.getElementById('val-nsi').innerText = maxNsi.toFixed(3);
}

function renderSentenceEvolution(evolution) {
    const container = document.getElementById('sentence-evolution-container');
    if (!container || !evolution) return;

    let html = '';
    evolution.forEach((step, idx) => {
        const isCollapsed = step.text.includes('collapsed');
        html += `
            <div class="evolution-node ${isCollapsed ? 'collapsed' : ''}">
                <span class="node-tag">${step.label}</span>
                <span class="node-text">${step.text}</span>
                <span class="node-status">${step.status}</span>
            </div>
        `;
        if (idx < evolution.length - 1) {
            html += `<div class="evolution-arrow">↓</div>`;
        }
    });

    container.innerHTML = html;
}

function renderObservationCharts(data) {
    const layers = Array.from({ length: data.metrics.distance.length }, (_, i) => `L${i}`);
    const themeCols = getPlotThemeColors();
    
    // Core Metrics Dynamics Plot
    const traceDist = { x: layers, y: data.metrics.distance, name: 'Pairwise Distance', type: 'scatter', mode: 'lines+markers', line: { color: '#38BDF8', width: 2.5 } };
    const traceSim = { x: layers, y: data.metrics.cosine_similarity, name: 'Cosine Sim (SAM)', type: 'scatter', mode: 'lines+markers', yaxis: 'y2', line: { color: '#818CF8', width: 2.5 } };
    const traceRank = { x: layers, y: data.metrics.effective_rank, name: 'Effective Rank', type: 'scatter', mode: 'lines+markers', line: { color: '#34D399', width: 2.5 } };
    const traceNsi = { x: layers, y: data.metrics.nsi, name: 'NSI (Stiffness)', type: 'scatter', mode: 'lines+markers', line: { color: '#FBBF24', width: 2.5 } };

    const layoutMetrics = {
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: { color: themeCols.fontColor, family: 'Inter' },
        margin: { t: 20, b: 40, l: 40, r: 40 },
        legend: { orientation: 'h', y: 1.15 },
        xaxis: { title: 'Layer Index', gridcolor: themeCols.gridColor },
        yaxis: { title: 'Distance / Rank / NSI', gridcolor: themeCols.gridColor },
        yaxis2: { title: 'Cosine Similarity', overlaying: 'y', side: 'right', range: [0, 1.05] }
    };

    Plotly.newPlot('plot-metrics-overview', [traceDist, traceSim, traceRank, traceNsi], layoutMetrics, { responsive: true, displayModeBar: false });

    // 2D Token Trajectory Projection
    const trajTraces = data.trajectories.token_trajectories.map((t, idx) => ({
        x: t.x,
        y: t.y,
        name: data.tokens[idx] || `Token_${idx}`,
        type: 'scatter',
        mode: 'lines+markers',
        marker: { size: 5 }
    }));

    const layoutTraj = {
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: { color: themeCols.fontColor, family: 'Inter' },
        margin: { t: 20, b: 40, l: 40, r: 40 },
        xaxis: { title: 'PCA Component 1', gridcolor: themeCols.gridColor },
        yaxis: { title: 'PCA Component 2', gridcolor: themeCols.gridColor }
    };

    Plotly.newPlot('plot-trajectories', trajTraces, layoutTraj, { responsive: true, displayModeBar: false });
}

function renderObservationNotes(model, data) {
    const container = document.getElementById('observation-notes-container');
    if (!container) return;

    let notesHtml = '';
    if (model.includes('gpt2')) {
        notesHtml = `
            <li><strong>Pairwise Distance:</strong> Distance decreases across deeper layers → token representations move toward a unified state space.</li>
            <li><strong>Cosine Similarity:</strong> Similarity increases steadily toward ~0.95+ → strong representation merging.</li>
            <li><strong>Effective Rank:</strong> Rank reduces gradually → tokens occupy a lower-dimensional subspace (Representation Collapse).</li>
            <li><strong>Neural Stiffness Index (NSI):</strong> High NSI in early layers (L0-L2) → heavy initial contextualization followed by layer stabilization.</li>
            <li style="color: var(--accent-cyan); font-weight: 600;">Empirical Conclusion: Autoregressive models (GPT-2) demonstrate strong representation collapse and layer convergence.</li>
        `;
    } else if (model.includes('bert') || model.includes('roberta')) {
        notesHtml = `
            <li><strong>Pairwise Distance:</strong> Distance remains stable across layers → tokens are refined while maintaining spatial separation.</li>
            <li><strong>Cosine Similarity:</strong> Values are varied (0.1 to 0.6) → preserves contextual fine-grained differences.</li>
            <li><strong>Effective Rank:</strong> Rank remains high and flat → structural representation diversity is preserved.</li>
            <li><strong>Neural Stiffness Index (NSI):</strong> Smooth and consistent NSI across all layers → steady layer-by-layer refinement.</li>
            <li style="color: var(--accent-emerald); font-weight: 600;">Empirical Conclusion: Bidirectional models (BERT/RoBERTa) maintain structural representation diversity across all layers.</li>
        `;
    } else {
        notesHtml = `
            <li><strong>Consensus & Collapse:</strong> Self-attention ODE dynamics pull token vectors toward equilibrium on the representation sphere.</li>
            <li><strong>Multi-stability & Clustering:</strong> Non-identity value matrices V produce multiple attractor basins, inducing token clustering.</li>
            <li><strong>Stiffness Profile:</strong> Early NSI spike marks rapid alignment before reaching steady-state trajectories.</li>
        `;
    }

    container.innerHTML = notesHtml;
}

// Stage 2: Understanding & Per-Layer AI Insights
async function updateLayerExplorer() {
    if (!appState.baselineData) return;
    const l = appState.currentLayer;
    const model = document.getElementById('select-model').value;
    const data = appState.baselineData;

    // Fetch AI insights for selected layer
    try {
        const resp = await fetch(`${API_BASE}/understanding/layer?model_name=${model}&layer=${l}&total_layers=${data.num_layers}`);
        const res = await resp.json();
        if (res.status === 'success') {
            const ins = res.insights;
            
            document.getElementById('mech-stage-title').innerText = `Layer ${l}: ${ins.stage_name}`;
            document.getElementById('mech-stage-body').innerHTML = ins.role_description;
            document.getElementById('mech-stiffness-body').innerHTML = ins.stiffness_diagnosis;
            
            const comps = ins.components;
            document.getElementById('mech-components-body').innerHTML = `
                • Self-Attention: ${comps.self_attention}<br>
                • MLP Sub-layer: ${comps.mlp_sublayer}<br>
                • Residual Stream: ${comps.residual_stream}<br>
                • LayerNorm: ${comps.layernorm}
            `;
        }
    } catch (err) {
        console.error('Layer understanding error:', err);
    }

    // Attention Heatmap
    const attns = data.attentions;
    if (attns && attns[l]) {
        const matrix = attns[l].matrix;
        const heatTrace = {
            z: matrix,
            x: data.tokens,
            y: data.tokens,
            type: 'heatmap',
            colorscale: 'Viridis'
        };

        const layoutHeat = {
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            font: { color: '#94A3B8', family: 'Inter' },
            margin: { t: 20, b: 80, l: 90, r: 20 },
            xaxis: { tickangle: -45, automargin: true },
            yaxis: { automargin: true }
        };

        Plotly.newPlot('plot-attention-heatmap', [heatTrace], layoutHeat, { responsive: true, displayModeBar: false });
    } else {
        document.getElementById('plot-attention-heatmap').innerHTML = '<div style="color:#64748B; padding:40px; text-align:center;">Attention weights uniform for ODE mode</div>';
    }

    // Token Vector Movement & Norms
    const summary = data.hidden_states_summary[l];
    if (summary) {
        const traceNorms = { x: data.tokens, y: summary.token_norms, name: 'Norm ||X||', type: 'bar', marker: { color: '#38BDF8' } };
        const traceMove = { x: data.tokens, y: summary.movement, name: 'Shift ||Delta X||', type: 'bar', marker: { color: '#FBBF24' } };

        const layoutMove = {
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            font: { color: '#94A3B8', family: 'Inter' },
            margin: { t: 20, b: 80, l: 50, r: 20 },
            xaxis: { tickangle: -45, automargin: true },
            barmode: 'group'
        };

        Plotly.newPlot('plot-token-movement', [traceNorms, traceMove], layoutMove, { responsive: true, displayModeBar: false });
    }
}

// Stage 3 & 4: Controlled Experimentation & Results
async function runExperiment() {
    const text = document.getElementById('input-text').value;
    const baselineModel = document.getElementById('select-model').value;
    
    const peMode = document.getElementById('exp-pe-mode').value;
    
    const skipVal = document.getElementById('exp-skip-layer').value;
    const skipLayer = skipVal === 'none' ? null : parseInt(skipVal);
    
    const rangeVal = document.getElementById('exp-skip-range').value;
    let skipRange = null;
    if (rangeVal !== 'none') {
        const [start, end] = rangeVal.split('-').map(Number);
        skipRange = Array.from({ length: end - start + 1 }, (_, i) => start + i);
    }
    
    const disableVal = document.getElementById('exp-disable-comp').value;
    const disableComp = disableVal === 'none' ? null : disableVal;
    
    const attnScale = parseFloat(document.getElementById('exp-attn-scale').value);

    try {
        const resp = await fetch(`${API_BASE}/experiment`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                text: text,
                baseline_model: baselineModel,
                pe_mode: peMode,
                skip_layer: skipLayer,
                skip_range: skipRange,
                disable_component: disableComp,
                attn_scale: attnScale
            })
        });

        const result = await resp.json();
        if (result.status === 'success') {
            appState.modifiedData = result.modified;
            
            // Switch to Stage 4 Results Tab
            document.querySelector('[data-tab="tab-results"]').click();
            
            renderExperimentResults(result.baseline, result.modified, result.comparison);
        }
    } catch (err) {
        console.error('Experiment error:', err);
    }
}

function renderExperimentResults(base, mod, comp) {
    const layers = Array.from({ length: base.metrics.distance.length }, (_, i) => `L${i}`);
    
    const baseDist = { x: layers, y: base.metrics.distance, name: 'Baseline Distance', type: 'scatter', mode: 'lines', line: { color: '#38BDF8', width: 2, dash: 'solid' } };
    const modDist = { x: layers, y: mod.metrics.distance, name: 'Modified Distance', type: 'scatter', mode: 'lines+markers', line: { color: '#F87171', width: 2.5 } };
    
    const baseRank = { x: layers, y: base.metrics.effective_rank, name: 'Baseline Rank', type: 'scatter', mode: 'lines', line: { color: '#34D399', width: 2, dash: 'solid' } };
    const modRank = { x: layers, y: mod.metrics.effective_rank, name: 'Modified Rank', type: 'scatter', mode: 'lines+markers', line: { color: '#FBBF24', width: 2.5 } };

    const layoutResults = {
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: { color: '#94A3B8', family: 'Inter' },
        margin: { t: 20, b: 40, l: 40, r: 40 },
        legend: { orientation: 'h', x: 0.5, xanchor: 'center', y: 1.08 },
        xaxis: { title: 'Layer Index', gridcolor: '#1E293B' },
        yaxis: { title: 'Metric Values', gridcolor: '#1E293B' }
    };

    Plotly.newPlot('plot-results-comparison', [baseDist, modDist, baseRank, modRank], layoutResults, { responsive: true, displayModeBar: false });

    // Render Research Conclusions & Suggestions
    const container = document.getElementById('insights-container');
    const delta = comp.metrics_delta;
    
    let html = `
        <li style="font-weight:600; color:#F1F5F9;">Experiment Run: ${comp.modification_description}</li>
        <li>
            Shift Summary: 
            Rank <span class="badge-delta ${delta.avg_rank_shift_pct >= 0 ? 'delta-pos' : 'delta-neg'}">${delta.avg_rank_shift_pct}%</span> | 
            NSI <span class="badge-delta ${delta.avg_nsi_shift_pct >= 0 ? 'delta-pos' : 'delta-neg'}">${delta.avg_nsi_shift_pct}%</span> | 
            Distance <span class="badge-delta ${delta.avg_distance_shift_pct >= 0 ? 'delta-pos' : 'delta-neg'}">${delta.avg_distance_shift_pct}%</span>
        </li>
    `;

    comp.conclusions.forEach(c => {
        html += `<li><strong>Research Analysis:</strong> ${c}</li>`;
    });

    comp.suggestions.forEach(s => {
        html += `<li style="color: var(--accent-cyan);"><strong>Architectural Suggestion:</strong> ${s}</li>`;
    });

    container.innerHTML = html;
}

// Render Paper Empirical Benchmark Figures
function renderBenchmarkPlots(bench) {
    if (!bench) return;

    const l13 = bench.layers_13;
    const l12 = bench.layers_12;
    const themeCols = getPlotThemeColors();

    const commonLayout = {
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: { color: themeCols.fontColor, family: 'Inter' },
        margin: { t: 15, b: 35, l: 45, r: 20 },
        legend: { orientation: 'h', x: 0.5, xanchor: 'center', y: 1.08 }
    };

    // 1. Distance
    const traceDistGpt2 = { x: l13, y: bench.metrics.distance.gpt2, name: 'GPT-2', type: 'scatter', mode: 'lines', line: { color: '#38BDF8', width: 2.5 } };
    const traceDistBert = { x: l13, y: bench.metrics.distance.bert, name: 'BERT', type: 'scatter', mode: 'lines', line: { color: '#FBBF24', width: 2.5 } };
    Plotly.newPlot('bench-plot-distance', [traceDistGpt2, traceDistBert], {
        ...commonLayout,
        xaxis: { gridcolor: themeCols.gridColor }, yaxis: { gridcolor: themeCols.gridColor }
    }, { responsive: true, displayModeBar: false });

    // 2. Cosine Similarity
    const traceSimGpt2 = { x: l12, y: bench.metrics.cosine_similarity.gpt2, name: 'GPT-2', type: 'scatter', mode: 'lines', line: { color: '#38BDF8', width: 2.5 } };
    const traceSimBert = { x: l13, y: bench.metrics.cosine_similarity.bert, name: 'BERT', type: 'scatter', mode: 'lines', line: { color: '#FBBF24', width: 2.5 } };
    Plotly.newPlot('bench-plot-similarity', [traceSimGpt2, traceSimBert], {
        ...commonLayout,
        xaxis: { gridcolor: themeCols.gridColor }, yaxis: { gridcolor: themeCols.gridColor, range: [0.2, 1.0] }
    }, { responsive: true, displayModeBar: false });

    // 3. Rank
    const traceRankGpt2 = { x: l13, y: bench.metrics.effective_rank.gpt2, name: 'GPT-2', type: 'scatter', mode: 'lines', line: { color: '#38BDF8', width: 2.5 } };
    const traceRankBert = { x: l13, y: bench.metrics.effective_rank.bert, name: 'BERT', type: 'scatter', mode: 'lines', line: { color: '#FBBF24', width: 2.5 } };
    Plotly.newPlot('bench-plot-rank', [traceRankGpt2, traceRankBert], {
        ...commonLayout,
        xaxis: { gridcolor: themeCols.gridColor }, yaxis: { gridcolor: themeCols.gridColor, range: [4.9, 6.1] }
    }, { responsive: true, displayModeBar: false });

    // 4. NSI
    const traceNsiGpt2 = { x: l12, y: bench.metrics.nsi.gpt2, name: 'GPT-2', type: 'scatter', mode: 'lines', line: { color: '#38BDF8', width: 2.5 } };
    const traceNsiBert = { x: l12, y: bench.metrics.nsi.bert, name: 'BERT', type: 'scatter', mode: 'lines', line: { color: '#FBBF24', width: 2.5 } };
    Plotly.newPlot('bench-plot-nsi', [traceNsiGpt2, traceNsiBert], {
        ...commonLayout,
        xaxis: { title: 'Layer Index', gridcolor: themeCols.gridColor }, yaxis: { title: 'NSI', gridcolor: themeCols.gridColor }
    }, { responsive: true, displayModeBar: false });

    // 5. Heatmaps
    const gpt2Heat = bench.similarity_heatmaps.gpt2;
    const bertHeat = bench.similarity_heatmaps.bert;

    const traceGpt2Heat = {
        z: gpt2Heat.matrix, x: gpt2Heat.tokens, y: gpt2Heat.tokens,
        type: 'heatmap', colorscale: 'Coolwarm', zmin: 0.91, zmax: 1.0
    };
    Plotly.newPlot('bench-heatmap-gpt2', [traceGpt2Heat], {
        ...commonLayout,
        margin: { t: 20, b: 60, l: 80, r: 40 }
    }, { responsive: true, displayModeBar: false });

    const traceBertHeat = {
        z: bertHeat.matrix, x: bertHeat.tokens, y: bertHeat.tokens,
        type: 'heatmap', colorscale: 'Coolwarm', zmin: -0.12, zmax: 1.0
    };
    Plotly.newPlot('bench-heatmap-bert', [traceBertHeat], {
        ...commonLayout,
        margin: { t: 20, b: 60, l: 80, r: 40 }
    }, { responsive: true, displayModeBar: false });

    // 6. Word Movement Bar Charts
    const gpt2Mov = bench.word_movement.gpt2;
    const bertMov = bench.word_movement.bert;

    const traceGpt2Bar = { x: gpt2Mov.tokens, y: gpt2Mov.movement, type: 'bar', marker: { color: '#38BDF8' } };
    Plotly.newPlot('bench-bar-gpt2', [traceGpt2Bar], {
        ...commonLayout,
        xaxis: { title: 'Tokens', gridcolor: themeCols.gridColor }, yaxis: { title: 'Movement (L2 Norm)', gridcolor: themeCols.gridColor }
    }, { responsive: true, displayModeBar: false });

    const traceBertBar = { x: bertMov.tokens, y: bertMov.movement, type: 'bar', marker: { color: '#38BDF8' } };
    Plotly.newPlot('bench-bar-bert', [traceBertBar], {
        ...commonLayout,
        xaxis: { title: 'Tokens', gridcolor: themeCols.gridColor }, yaxis: { title: 'Movement (L2 Norm)', gridcolor: themeCols.gridColor }
    }, { responsive: true, displayModeBar: false });
}
