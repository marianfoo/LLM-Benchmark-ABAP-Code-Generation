const DASHBOARD_URL = "data/dashboard.json";

const state = {
  raw: null,
  sortKey: "Success_R5_pct",
  sortDirection: "desc",
  filterText: "",
};

function asNumber(value) {
  if (value === null || value === undefined || value === "") return NaN;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : NaN;
}

function formatValue(value, column) {
  if (value === null || value === undefined || value === "") return "—";
  if (column.type === "percent") {
    const num = asNumber(value);
    if (Number.isNaN(num)) return "—";
    return `${num.toFixed(2)}%`;
  }
  if (column.type === "number") {
    const num = asNumber(value);
    if (Number.isNaN(num)) return "—";
    const decimals = column.decimals ?? 0;
    return num.toFixed(decimals);
  }
  return String(value);
}

function compareRows(a, b, column) {
  const va = a[column.key];
  const vb = b[column.key];

  if (column.type === "text") {
    return String(va ?? "").localeCompare(String(vb ?? ""), undefined, { sensitivity: "base" });
  }

  const na = asNumber(va);
  const nb = asNumber(vb);
  if (Number.isNaN(na) && Number.isNaN(nb)) return 0;
  if (Number.isNaN(na)) return 1;
  if (Number.isNaN(nb)) return -1;
  return na - nb;
}

function getActiveColumn(columns) {
  return columns.find((column) => column.key === state.sortKey) ?? columns[0];
}

function renderSummary(data) {
  const summary = data.summary;
  const container = document.getElementById("summaryCards");
  container.innerHTML = "";

  const generatedLocal = new Date(data.generated_at_utc).toLocaleString();
  const totalModels = summary.total_models_count ?? summary.models_count;
  const cards = [
    { label: "Displayed Models", value: summary.models_count },
    { label: "Total Models (Raw)", value: totalModels },
    { label: "Fully Evaluated", value: summary.fully_evaluated_models_count },
    { label: "Top Final Success", value: summary.top_by_final_success.join(", ") },
    { label: "Top First-Try", value: summary.top_by_first_try_success.join(", ") },
    { label: "Generated", value: generatedLocal },
  ];

  for (const card of cards) {
    const element = document.createElement("div");
    element.className = "summary-pill";
    element.innerHTML = `<div class="label">${card.label}</div><div class="value">${card.value}</div>`;
    container.appendChild(element);
  }
}

function renderTable(tableId, columns, rows, sortable = false) {
  const table = document.getElementById(tableId);
  const thead = table.querySelector("thead");
  const tbody = table.querySelector("tbody");

  thead.innerHTML = "";
  tbody.innerHTML = "";

  const headerRow = document.createElement("tr");
  const activeColumn = getActiveColumn(columns);

  for (const column of columns) {
    const th = document.createElement("th");
    if (sortable) {
      th.classList.add("sortable");
      th.addEventListener("click", () => {
        if (state.sortKey === column.key) {
          state.sortDirection = state.sortDirection === "asc" ? "desc" : "asc";
        } else {
          state.sortKey = column.key;
          state.sortDirection = column.default_sort === "asc" ? "asc" : "desc";
        }
        renderMainTable();
      });
      const isActive = column.key === activeColumn.key;
      const icon = !isActive ? "↕" : state.sortDirection === "asc" ? "↑" : "↓";
      th.innerHTML = `${column.label}<span class="sort-indicator">${icon}</span>`;
    } else {
      th.textContent = column.label;
    }
    headerRow.appendChild(th);
  }
  thead.appendChild(headerRow);

  for (const row of rows) {
    const tr = document.createElement("tr");
    for (const column of columns) {
      const td = document.createElement("td");
      const rendered = formatValue(row[column.key], column);
      td.textContent = rendered;
      if (rendered === "—") td.className = "muted";
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
  }
}

function getFilteredRows() {
  const rows = [...state.raw.main_table.rows];
  const filter = state.filterText.trim().toLowerCase();

  return rows.filter((row) => {
    const modelText = String(row.Model_Display ?? "").toLowerCase();
    return filter === "" || modelText.includes(filter);
  });
}

function renderMainTable() {
  const columns = state.raw.main_table.columns;
  const activeColumn = getActiveColumn(columns);
  let rows = getFilteredRows();

  rows.sort((a, b) => compareRows(a, b, activeColumn));
  if (state.sortDirection === "desc") rows.reverse();

  renderTable("mainTable", columns, rows, true);
}

function renderSecondaryTables() {
  renderTable("roundTable", state.raw.round_table.columns, state.raw.round_table.rows);
  renderTable("categoryTable", state.raw.category_table.columns, state.raw.category_table.rows);
}

function renderPlots() {
  const grid = document.getElementById("plotsGrid");
  grid.innerHTML = "";

  const template = document.getElementById("plotCardTemplate");
  for (const plot of state.raw.plots) {
    const clone = template.content.cloneNode(true);
    const link = clone.querySelector(".plot-image-link");
    const img = clone.querySelector("img");
    const title = clone.querySelector("h3");
    const desc = clone.querySelector("p");

    const plotUrl = `assets/plots/${plot.file}`;
    link.href = plotUrl;
    img.src = plotUrl;
    img.alt = plot.title;
    title.textContent = plot.title;
    desc.textContent = plot.description;

    grid.appendChild(clone);
  }
}

function bindControls() {
  const modelFilter = document.getElementById("modelFilter");
  const resetFilters = document.getElementById("resetFilters");

  modelFilter.addEventListener("input", (event) => {
    state.filterText = event.target.value || "";
    renderMainTable();
  });

  resetFilters.addEventListener("click", () => {
    state.filterText = "";
    state.sortKey = "Success_R5_pct";
    state.sortDirection = "desc";
    modelFilter.value = "";
    renderMainTable();
  });
}

function renderLoadError(message) {
  const root = document.querySelector(".layout");
  const errorBox = document.createElement("section");
  errorBox.className = "card error-box";
  errorBox.innerHTML = `
    <strong>Failed to load dashboard data.</strong><br />
    ${message}
  `;
  root.prepend(errorBox);
}

async function bootstrap() {
  bindControls();
  try {
    const response = await fetch(DASHBOARD_URL);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    state.raw = await response.json();
    renderSummary(state.raw);
    renderMainTable();
    renderSecondaryTables();
    renderPlots();
  } catch (error) {
    renderLoadError(String(error));
  }
}

bootstrap();
