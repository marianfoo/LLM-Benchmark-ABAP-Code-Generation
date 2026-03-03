const DASHBOARD_URL = "data/dashboard.json";

const state = {
  raw: null,
  sortKey: "Success_R5_pct",
  sortDirection: "desc",
  filterText: "",
  understandingSortKey: "AUC_Success_pct",
  understandingSortDirection: "desc",
  roundSortKey: "Success_R5_pct",
  roundSortDirection: "desc",
  categorySortKey: "Success_R5_StringHandling_pct",
  categorySortDirection: "desc",
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

function renderTable(tableId, columns, rows, sortable = false, activeSortKey = null, activeSortDir = "desc", onSort = null) {
  const table = document.getElementById(tableId);
  if (!table) return;
  const thead = table.querySelector("thead");
  const tbody = table.querySelector("tbody");
  if (!thead || !tbody) return;

  thead.innerHTML = "";
  tbody.innerHTML = "";

  const headerRow = document.createElement("tr");

  for (const column of columns) {
    const th = document.createElement("th");
    // Ensure headers are never narrower than their label – prevents character stacking on mobile
    th.style.minWidth = column.type === "text" ? "120px" : "80px";
    if (sortable) {
      th.classList.add("sortable");
      th.addEventListener("click", () => {
        if (onSort) onSort(column);
      });
      const isActive = column.key === activeSortKey;
      const icon = !isActive ? "↕" : activeSortDir === "asc" ? "↑" : "↓";
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

  renderTable("mainTable", columns, rows, true, state.sortKey, state.sortDirection, (column) => {
    if (state.sortKey === column.key) {
      state.sortDirection = state.sortDirection === "asc" ? "desc" : "asc";
    } else {
      state.sortKey = column.key;
      state.sortDirection = column.default_sort === "asc" ? "asc" : "desc";
    }
    renderMainTable();
  });
}

function renderRoundTable() {
  const columns = state.raw.round_table.columns;
  let rows = [...state.raw.round_table.rows];
  const activeCol = columns.find((c) => c.key === state.roundSortKey) ?? columns[0];
  rows.sort((a, b) => compareRows(a, b, activeCol));
  if (state.roundSortDirection === "desc") rows.reverse();
  renderTable("roundTable", columns, rows, true, state.roundSortKey, state.roundSortDirection, (column) => {
    if (state.roundSortKey === column.key) {
      state.roundSortDirection = state.roundSortDirection === "asc" ? "desc" : "asc";
    } else {
      state.roundSortKey = column.key;
      state.roundSortDirection = column.default_sort === "asc" ? "asc" : "desc";
    }
    renderRoundTable();
  });
}

function renderCategoryTable() {
  const columns = state.raw.category_table.columns;
  let rows = [...state.raw.category_table.rows];
  const activeCol = columns.find((c) => c.key === state.categorySortKey) ?? columns[0];
  rows.sort((a, b) => compareRows(a, b, activeCol));
  if (state.categorySortDirection === "desc") rows.reverse();
  renderTable("categoryTable", columns, rows, true, state.categorySortKey, state.categorySortDirection, (column) => {
    if (state.categorySortKey === column.key) {
      state.categorySortDirection = state.categorySortDirection === "asc" ? "desc" : "asc";
    } else {
      state.categorySortKey = column.key;
      state.categorySortDirection = column.default_sort === "asc" ? "asc" : "desc";
    }
    renderCategoryTable();
  });
}

function renderSecondaryTables() {
  renderRoundTable();
  renderCategoryTable();
}

function renderUnderstandingTable() {
  const u = state.raw.understanding;
  if (!u) return;
  const columns = u.table.columns;
  let rows = [...u.table.rows];

  const activeCol = columns.find((c) => c.key === state.understandingSortKey) ?? columns[0];
  rows.sort((a, b) => compareRows(a, b, activeCol));
  if (state.understandingSortDirection === "desc") rows.reverse();

  renderTable(
    "understandingTable",
    columns,
    rows,
    true,
    state.understandingSortKey,
    state.understandingSortDirection,
    (column) => {
      if (state.understandingSortKey === column.key) {
        state.understandingSortDirection = state.understandingSortDirection === "asc" ? "desc" : "asc";
      } else {
        state.understandingSortKey = column.key;
        state.understandingSortDirection = column.default_sort === "asc" ? "asc" : "desc";
      }
      renderUnderstandingTable();
    },
  );
}

function renderPlots() {
  const grid = document.getElementById("plotsGrid");
  if (!grid) return;
  grid.innerHTML = "";

  const template = document.getElementById("plotCardTemplate");
  if (!template) return;
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
  if (!modelFilter || !resetFilters) return;

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

function renderUnderstandingSection() {
  const u = state.raw.understanding;
  if (!u) return;
  renderUnderstandingTable();
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
    renderMainTable();
    renderSecondaryTables();
    renderPlots();
    renderUnderstandingSection();
  } catch (error) {
    renderLoadError(String(error));
  }
}

bootstrap();
