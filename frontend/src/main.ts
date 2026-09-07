/**
 * Main TypeScript application entrypoint for Enterprise AI Agent.
 * Coordinates Aurora Glassmorphism UI state, tab routing, and API events.
 */

import { api } from "./api";
import type { CitationItem, SearchResultItem, WorkspaceTab } from "./types";

document.addEventListener("DOMContentLoaded", () => {
  initTheme();
  initTabs();
  initChat();
  initSearch();
  initSQL();
  initGuardrails();
  initTelemetry();
  checkSystemHealth();
});

// ============================================================================
// 0. Apple Theme Manager (Dark & Light Modes)
// ============================================================================
function initTheme(): void {
  const themeToggleBtn = document.getElementById("theme-toggle-btn") as HTMLButtonElement | null;
  const themeIcon = document.getElementById("theme-icon") as HTMLElement | null;

  const savedTheme = localStorage.getItem("ea-theme");
  const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  const initialTheme = savedTheme || (prefersDark ? "dark" : "light");

  applyTheme(initialTheme);

  themeToggleBtn?.addEventListener("click", () => {
    const currentTheme = document.documentElement.getAttribute("data-theme") || "dark";
    const nextTheme = currentTheme === "dark" ? "light" : "dark";
    applyTheme(nextTheme);
    localStorage.setItem("ea-theme", nextTheme);
  });

  function applyTheme(theme: string): void {
    document.documentElement.setAttribute("data-theme", theme);
    if (!themeIcon) return;

    if (theme === "dark") {
      // Show Sun icon (indicates clicking will switch to light)
      themeIcon.innerHTML = `
        <circle cx="12" cy="12" r="5"></circle>
        <line x1="12" y1="1" x2="12" y2="3"></line>
        <line x1="12" y1="21" x2="12" y2="23"></line>
        <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line>
        <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line>
        <line x1="1" y1="12" x2="3" y2="12"></line>
        <line x1="21" y1="12" x2="23" y2="12"></line>
        <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line>
        <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line>
      `;
    } else {
      // Show Moon icon (indicates clicking will switch to dark)
      themeIcon.innerHTML = `
        <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path>
      `;
    }
  }
}


// ============================================================================
// 1. Tab Routing & Navigation
// ============================================================================
function initTabs(): void {
  const tabButtons = document.querySelectorAll<HTMLButtonElement>(".tab-btn");
  const panels = document.querySelectorAll<HTMLElement>(".workspace-panel");

  tabButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const targetTab = btn.getAttribute("data-tab") as WorkspaceTab | null;
      if (!targetTab) return;

      tabButtons.forEach((b) => b.classList.remove("active"));
      panels.forEach((p) => p.classList.remove("active"));

      btn.classList.add("active");
      const targetPanel = document.getElementById(`panel-${targetTab}`);
      if (targetPanel) {
        targetPanel.classList.add("active");
      }

      if (targetTab === "telemetry") {
        fetchTelemetryStats();
      }
    });
  });
}

// ============================================================================
// 2. Omni-Chat & Citations
// ============================================================================
function initChat(): void {
  const chatForm = document.getElementById("chat-form") as HTMLFormElement | null;
  const chatInput = document.getElementById("chat-input") as HTMLInputElement | null;
  const chatHistory = document.getElementById("chat-history") as HTMLElement | null;
  const chatMode = document.getElementById("chat-mode-select") as HTMLSelectElement | null;
  const citationsList = document.getElementById("citations-list") as HTMLElement | null;
  const promptChips = document.querySelectorAll<HTMLButtonElement>(".prompt-chip");

  promptChips.forEach((chip) => {
    chip.addEventListener("click", () => {
      const prompt = chip.getAttribute("data-prompt");
      if (prompt && chatInput) {
        chatInput.value = prompt;
        chatInput.focus();
      }
    });
  });

  if (!chatForm || !chatInput || !chatHistory) return;

  chatForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const query = chatInput.value.trim();
    if (!query) return;

    chatInput.value = "";

    // 1. Append User Bubble
    appendMessage("user", query);

    // 2. Loading indicator
    const loadingId = appendLoadingMessage();

    try {
      const startTime = performance.now();
      const mode = chatMode?.value || "rag";

      if (mode === "rag") {
        const res = await api.sendRAGQuery(query);
        const elapsed = Math.round(performance.now() - startTime);
        removeMessage(loadingId);

        appendMessage("assistant", res.answer, {
          model: res.model,
          latency: res.latency_ms || elapsed,
        });

        renderCitations(res.sources);
      } else {
        const res = await api.sendDirectChat(query);
        const elapsed = Math.round(performance.now() - startTime);
        removeMessage(loadingId);

        appendMessage("assistant", res.answer, {
          model: res.model,
          latency: res.latency_ms || elapsed,
        });

        if (citationsList) {
          citationsList.innerHTML = `<div style="color: var(--text-muted); font-size: 0.8rem; text-align: center; padding: 2rem 0;">Direct generation mode (no document citations).</div>`;
        }
      }
    } catch (err: unknown) {
      removeMessage(loadingId);
      const errMsg = err instanceof Error ? err.message : String(err);
      appendMessage("assistant", `⚠️ **Error processing request**: ${errMsg}`, {
        isError: true,
      });
    }
  });

  function appendMessage(
    role: "user" | "assistant",
    text: string,
    meta?: { model?: string; latency?: number; isError?: boolean }
  ): string {
    const msgId = `msg-${Date.now()}-${Math.random().toString(36).substring(2, 7)}`;
    const bubble = document.createElement("div");
    bubble.id = msgId;
    bubble.className = `message-bubble ${role === "user" ? "message-user" : "message-assistant"}`;
    if (meta?.isError) {
      bubble.style.borderColor = "var(--aurora-rose)";
      bubble.style.background = "rgba(244, 63, 94, 0.1)";
    }

    const contentDiv = document.createElement("div");
    contentDiv.innerHTML = formatMarkdown(text);
    bubble.appendChild(contentDiv);

    if (role === "assistant") {
      const metaDiv = document.createElement("div");
      metaDiv.className = "message-meta";
      metaDiv.innerHTML = `<span>${meta?.model || "Enterprise AI"}</span> • <span class="latency-tag">${meta?.latency || 0}ms</span>`;
      bubble.appendChild(metaDiv);
    }

    chatHistory?.appendChild(bubble);
    chatHistory?.scrollTo({ top: chatHistory.scrollHeight, behavior: "smooth" });
    return msgId;
  }

  function appendLoadingMessage(): string {
    const msgId = `msg-loading-${Date.now()}`;
    const bubble = document.createElement("div");
    bubble.id = msgId;
    bubble.className = "message-bubble message-assistant";
    bubble.innerHTML = `<div style="display: flex; gap: 0.4rem; align-items: center; color: var(--text-muted);">
      <span class="beacon-dot" style="width: 6px; height: 6px;"></span> Retrieving corporate knowledge & synthesizing answer...
    </div>`;
    chatHistory?.appendChild(bubble);
    chatHistory?.scrollTo({ top: chatHistory.scrollHeight, behavior: "smooth" });
    return msgId;
  }

  function removeMessage(id: string): void {
    const el = document.getElementById(id);
    if (el) el.remove();
  }

  function renderCitations(sources: CitationItem[] | undefined): void {
    if (!citationsList) return;

    if (!sources || sources.length === 0) {
      citationsList.innerHTML = `<div style="color: var(--text-muted); font-size: 0.8rem; text-align: center; padding: 2rem 0;">No citations were retrieved for this question.</div>`;
      return;
    }

    citationsList.innerHTML = "";
    sources.forEach((s, idx) => {
      const card = document.createElement("div");
      card.className = "citation-card";
      const scoreStr = s.score !== undefined ? (s.score > 0 ? `+${s.score.toFixed(3)}` : s.score.toFixed(3)) : "N/A";
      const excerpt = s.excerpt || (s as unknown as { content: string }).content || "";

      card.innerHTML = `
        <div class="citation-title">
          <span>[${idx + 1}] ${s.title || s.source || "Corporate Document"}</span>
          <span class="citation-score">Score: ${scoreStr}</span>
        </div>
        <div class="citation-snippet">${excerpt.substring(0, 220)}${excerpt.length > 220 ? "..." : ""}</div>
      `;
      citationsList.appendChild(card);
    });
  }
}

// ============================================================================
// 3. Hybrid Search Explorer
// ============================================================================
function initSearch(): void {
  const form = document.getElementById("search-form") as HTMLFormElement | null;
  const input = document.getElementById("search-input") as HTMLInputElement | null;
  const slider = document.getElementById("search-alpha-slider") as HTMLInputElement | null;
  const alphaVal = document.getElementById("alpha-val") as HTMLElement | null;
  const sparseVal = document.getElementById("sparse-val") as HTMLElement | null;
  const topKSelect = document.getElementById("search-top-k") as HTMLSelectElement | null;
  const resultsContainer = document.getElementById("search-results-list") as HTMLElement | null;

  slider?.addEventListener("input", () => {
    const val = parseFloat(slider.value);
    if (alphaVal) alphaVal.textContent = val.toFixed(2);
    if (sparseVal) sparseVal.textContent = (1.0 - val).toFixed(2);
  });

  form?.addEventListener("submit", async (e) => {
    e.preventDefault();
    const query = input?.value.trim();
    if (!query || !resultsContainer) return;

    const alpha = slider ? parseFloat(slider.value) : 0.5;
    const topK = topKSelect ? parseInt(topKSelect.value, 10) : 5;

    resultsContainer.innerHTML = `<div style="color: var(--text-muted); text-align: center; padding: 2rem;">Executing dual-retrieval (Dense Cosine + Sparse BM25)...</div>`;

    try {
      const res = await api.executeHybridSearch(query, topK, alpha);
      if (!res.results || res.results.length === 0) {
        resultsContainer.innerHTML = `<div style="color: var(--text-muted); text-align: center; padding: 2rem;">No matching document chunks found for query.</div>`;
        return;
      }

      resultsContainer.innerHTML = "";
      res.results.forEach((item: SearchResultItem, idx: number) => {
        const card = document.createElement("div");
        card.className = "result-card";
        const scoreStr = item.score.toFixed(3);
        const text = item.text || (item as unknown as { content: string }).content || "";

        card.innerHTML = `
          <div class="result-header">
            <div class="result-title">#${idx + 1} ${item.title || "Policy Chunk"}</div>
            <span class="score-badge">Fusion Score: ${scoreStr}</span>
          </div>
          <div style="font-size: 0.85rem; color: var(--text-secondary); line-height: 1.55;">${text}</div>
          <div style="margin-top: 0.5rem; font-size: 0.72rem; color: var(--text-muted); font-family: var(--font-mono);">
            Chunk ID: ${item.chunk_id || "N/A"}
          </div>
        `;
        resultsContainer.appendChild(card);
      });
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      resultsContainer.innerHTML = `<div style="color: var(--aurora-rose); padding: 1.5rem; text-align: center;">Error performing hybrid search: ${msg}</div>`;
    }
  });
}

// ============================================================================
// 4. Safe SQL Studio
// ============================================================================
function initSQL(): void {
  const sqlInput = document.getElementById("sql-query-input") as HTMLTextAreaElement | null;
  const runBtn = document.getElementById("sql-run-btn") as HTMLButtonElement | null;
  const sampleBtns = document.querySelectorAll<HTMLButtonElement>(".sql-sample-btn");
  const astBadge = document.getElementById("ast-status-badge") as HTMLElement | null;
  const resultWrapper = document.getElementById("sql-result-wrapper") as HTMLElement | null;

  sampleBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      const query = btn.getAttribute("data-sql");
      if (query && sqlInput) {
        sqlInput.value = query;
        updateASTBadge(query);
      }
    });
  });

  sqlInput?.addEventListener("input", () => {
    updateASTBadge(sqlInput.value);
  });

  runBtn?.addEventListener("click", async () => {
    const query = sqlInput?.value.trim();
    if (!query || !resultWrapper) return;

    resultWrapper.innerHTML = `<div style="text-align: center; color: var(--text-muted); padding: 2rem;">Validating AST syntax tree and querying read-only database...</div>`;

    try {
      const res = await api.executeSQL(query);
      renderSQLTable(res);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      resultWrapper.innerHTML = `
        <div style="background: rgba(244, 63, 94, 0.12); border: 1px solid rgba(244, 63, 94, 0.35); border-radius: var(--radius-md); padding: 1.25rem; margin-top: 1.5rem;">
          <div style="font-weight: 600; color: var(--aurora-rose); display: flex; align-items: center; gap: 0.5rem;">
            <span>🛡️</span> AST Security Interception / Query Error
          </div>
          <div style="font-size: 0.85rem; color: #fff; margin-top: 0.5rem; font-family: var(--font-mono);">${msg}</div>
        </div>
      `;
    }
  });

  function updateASTBadge(sql: string): void {
    if (!astBadge) return;
    const isMutation = /\b(DROP|UPDATE|INSERT|DELETE|ALTER|CREATE|TRUNCATE)\b/i.test(sql);
    if (isMutation) {
      astBadge.className = "ast-badge ast-danger";
      astBadge.innerHTML = `<span>⚠️</span> Mutating Statement Detected (Will be Blocked)`;
    } else {
      astBadge.className = "ast-badge ast-safe";
      astBadge.innerHTML = `<span>🛡️</span> AST Guard Active: Read-Only (SELECT Only)`;
    }
  }

  function renderSQLTable(data: { columns: string[]; rows: Array<Record<string, unknown>>; execution_time_ms: number }): void {
    if (!resultWrapper) return;

    if (!data.rows || data.rows.length === 0) {
      resultWrapper.innerHTML = `<div style="text-align: center; color: var(--text-muted); padding: 2rem;">Query executed successfully. 0 rows returned (${data.execution_time_ms.toFixed(2)}ms).</div>`;
      return;
    }

    const cols = data.columns && data.columns.length > 0 ? data.columns : Object.keys(data.rows[0]);

    let html = `
      <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 1rem; margin-bottom: 0.5rem;">
        <span style="font-size: 0.8rem; color: var(--text-muted);">Returned <strong>${data.rows.length}</strong> rows</span>
        <span style="font-size: 0.78rem; color: var(--aurora-emerald); font-family: var(--font-mono);">${data.execution_time_ms.toFixed(2)} ms</span>
      </div>
      <div class="data-table-container">
        <table class="data-table">
          <thead>
            <tr>${cols.map((c) => `<th>${c}</th>`).join("")}</tr>
          </thead>
          <tbody>
    `;

    data.rows.forEach((row) => {
      html += `<tr>${cols.map((c) => `<td>${row[c] !== null && row[c] !== undefined ? String(row[c]) : '<span style="color: var(--text-muted);">NULL</span>'}</td>`).join("")}</tr>`;
    });

    html += `</tbody></table></div>`;
    resultWrapper.innerHTML = html;
  }
}

// ============================================================================
// 5. Guardrails & PII Sandbox
// ============================================================================
function initGuardrails(): void {
  const input = document.getElementById("guardrails-input") as HTMLTextAreaElement | null;
  const validateBtn = document.getElementById("guardrails-validate-btn") as HTMLButtonElement | null;
  const redactBtn = document.getElementById("guardrails-redact-btn") as HTMLButtonElement | null;
  const outputBox = document.getElementById("guardrails-output-box") as HTMLElement | null;
  const presets = document.querySelectorAll<HTMLButtonElement>(".guardrails-preset");

  presets.forEach((p) => {
    p.addEventListener("click", () => {
      const text = p.getAttribute("data-text");
      if (text && input) {
        input.value = text;
      }
    });
  });

  validateBtn?.addEventListener("click", async () => {
    const text = input?.value.trim();
    if (!text || !outputBox) return;

    outputBox.innerHTML = `<span style="color: var(--text-muted);">Inspecting text across prompt injection and PII detection layers...</span>`;

    try {
      const res = await api.validateGuardrails(text);
      let html = "";

      if (!res.passed) {
        html += `<div style="color: var(--aurora-rose); font-weight: 600; margin-bottom: 0.75rem;">🛡️ VIOLATION DETECTED [Action: ${res.action_taken.toUpperCase()}]</div>`;
        if (res.violations && res.violations.length > 0) {
          html += `<ul style="margin-left: 1.2rem; margin-bottom: 0.75rem;">`;
          res.violations.forEach((v) => {
            html += `<li><strong style="color: var(--aurora-pink);">${v.violation_type}</strong> (${v.severity}): ${v.message}</li>`;
          });
          html += `</ul>`;
        }
      } else {
        html += `<div style="color: var(--aurora-emerald); font-weight: 600; margin-bottom: 0.75rem;">✅ VERDICT: SAFE (All Security Checks Passed)</div>`;
      }

      html += `<div style="margin-top: 0.5rem; color: var(--text-muted); font-size: 0.75rem;">Sanitized / Redacted Result:</div>`;
      html += `<div style="color: var(--text-primary); margin-top: 0.25rem;">${escapeHtml(res.sanitized_text || text)}</div>`;

      outputBox.innerHTML = html;
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      outputBox.innerHTML = `<div style="color: var(--aurora-rose);">Error: ${msg}</div>`;
    }
  });

  redactBtn?.addEventListener("click", async () => {
    const text = input?.value.trim();
    if (!text || !outputBox) return;

    outputBox.innerHTML = `<span style="color: var(--text-muted);">Executing Luhn Mod-10 algorithmic PII redaction...</span>`;

    try {
      const res = await api.redactPII(text);
      let html = `<div style="color: var(--aurora-cyan); font-weight: 600; margin-bottom: 0.75rem;">Masked ${res.total_entities} sensitive entities:</div>`;
      html += `<div style="color: var(--text-primary);">${escapeHtml(res.redacted_text)}</div>`;
      outputBox.innerHTML = html;
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      outputBox.innerHTML = `<div style="color: var(--aurora-rose);">Error: ${msg}</div>`;
    }
  });
}

// ============================================================================
// 6. Telemetry & Performance
// ============================================================================
function initTelemetry(): void {
  const purgeBtn = document.getElementById("purge-cache-btn") as HTMLButtonElement | null;

  purgeBtn?.addEventListener("click", async () => {
    try {
      await api.resetPerformanceCache();
      fetchTelemetryStats();
      alert("Performance caches successfully cleared!");
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      alert(`Failed to clear caches: ${msg}`);
    }
  });
}

async function fetchTelemetryStats(): Promise<void> {
  const hitRatio = document.getElementById("metric-hit-ratio");
  const totalReqs = document.getElementById("metric-total-requests");
  const totalHits = document.getElementById("metric-total-hits");
  const cacheSize = document.getElementById("metric-cache-size");

  try {
    const stats = await api.getPerformanceStats();
    if (hitRatio) hitRatio.textContent = `${stats.overall_hit_ratio_pct}%`;
    if (totalReqs) totalReqs.textContent = String(stats.total_requests);
    if (totalHits) totalHits.textContent = String(stats.total_hits);
    if (cacheSize) {
      const embSize = stats.embedding_cache?.size || 0;
      const retSize = stats.retrieval_cache?.size || 0;
      cacheSize.textContent = String(embSize + retSize);
    }
  } catch {
    // ignore telemetry polling error
  }
}

async function checkSystemHealth(): Promise<void> {
  const statusText = document.getElementById("status-text");
  try {
    const health = await api.getHealth();
    if (statusText) {
      statusText.textContent = `Online • ${health.environment || "ready"}`;
    }
  } catch {
    if (statusText) {
      statusText.textContent = "Connecting...";
    }
  }
}

// ============================================================================
// Utilities
// ============================================================================
function formatMarkdown(text: string): string {
  let formatted = escapeHtml(text);
  formatted = formatted.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
  formatted = formatted.replace(/\*(.*?)\*/g, "<em>$1</em>");
  formatted = formatted.replace(/`([^`]+)`/g, '<code style="font-family: var(--font-mono); background: rgba(255,255,255,0.08); padding: 0.1rem 0.35rem; border-radius: 4px;">$1</code>');
  formatted = formatted.replace(/\[Source\s+(\d+)\]/g, '<span style="color: var(--aurora-cyan); font-weight: 600; cursor: pointer;">[Source $1]</span>');
  formatted = formatted.replace(/\n/g, "<br/>");
  return formatted;
}

function escapeHtml(text: string): string {
  const map: Record<string, string> = {
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#039;",
  };
  return text.replace(/[&<>"']/g, (m) => map[m]);
}
