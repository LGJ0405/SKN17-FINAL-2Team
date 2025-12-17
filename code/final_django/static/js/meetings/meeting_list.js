document.addEventListener("DOMContentLoaded", function () {
  const tbody = document.getElementById("meeting-table-body");
  if (!tbody) return;

  // 모든 회의 row (DOM 에서 한 번만 수집)
  const allRows = Array.from(tbody.querySelectorAll(".meeting-row"));

  const searchInput = document.getElementById("search-input");
  const searchClearBtn = document.getElementById("search-clear-btn");
  const searchFilterToggle = document.getElementById("search-filter-toggle");
  const searchFilterDropdown = document.getElementById("search-filter-dropdown");
  const searchFilterOptions = searchFilterDropdown
    ? Array.from(searchFilterDropdown.querySelectorAll(".search-filter-option"))
    : [];
  const searchFilterLabel = document.getElementById("search-filter-label");

  const readFilterToggle = document.getElementById("read-filter-toggle");
  const readFilterDropdown = document.getElementById("read-filter-dropdown");
  const readFilterOptions = readFilterDropdown
    ? Array.from(readFilterDropdown.querySelectorAll(".read-filter-option"))
    : [];

  const sortDateButton = document.getElementById("sort-date-button");
  const sortTitleButton = document.getElementById("sort-title-button");
  const sortDateIcon = document.getElementById("sort-date-icon");
  const sortTitleIcon = document.getElementById("sort-title-icon");

  const paginationContainer = document.getElementById("meeting-pagination");
  const pageNumbersWrap = document.getElementById("page-numbers");
  const pagePrevBtn = document.getElementById("page-prev");
  const pageNextBtn = document.getElementById("page-next");

  // ===== 상태값 =====
  let currentSearchField = "title"; // title | host | title_host | datetime
  let currentSearchKeyword = "";

  let currentReadFilter = "all"; // all | read | unread

  // 정렬 상태: 기본은 일시 기준 최신순 (내림차순)
  let currentSortField = "datetime"; // 'datetime' | 'title'
  let currentSortDirection = "desc"; // 'asc' | 'desc'

  // 페이지네이션
  const pageSize = 10;
  let currentPage = 1;
  let totalPages = 1;
  let filteredRows = []; // 필터 + 정렬 후의 row 목록
  
  function updateSearchPlaceholder() {
    if (!searchInput) return;

    switch (currentSearchField) {
      case "title":
        searchInput.placeholder = "제목을 입력해 주세요";
        break;
      case "host":
        searchInput.placeholder = "주최자를 입력해 주세요";
        break;
      case "title_host":
        searchInput.placeholder = "제목 또는 주최자를 입력해 주세요";
        break;
      case "datetime":
        searchInput.placeholder = "회의 일시를 입력해 주세요";
        break;
      default:
        searchInput.placeholder = "검색어를 입력해 주세요";
    }
  }

  // ===== 유틸: 문자열 포함 여부 (대소문자 무시) =====
  function containsIgnoreCase(target, keyword) {
    if (!keyword) return true;
    if (!target) return false;
    return target.toString().toLowerCase().indexOf(keyword.toLowerCase()) !== -1;
  }

  // ===== 필터 & 정렬 재계산 =====
  function recomputeFilteredAndSorted(resetPage = true) {
    // 1. 필터링
    filteredRows = [];

    allRows.forEach((row) => {
      const title = row.dataset.title || "";
      const host = row.dataset.host || "";

      const datetimeDisplay = row.dataset.datetimeDisplay || "";
      const readStatus = row.dataset.read || "unread";

      // 검색 필터
      let matchedSearch = true;
      if (currentSearchKeyword) {
        if (currentSearchField === "title") {
          matchedSearch = containsIgnoreCase(title, currentSearchKeyword);
        } else if (currentSearchField === "host") {
          matchedSearch = containsIgnoreCase(host, currentSearchKeyword);
        } else if (currentSearchField === "title_host") {
          matchedSearch =
            containsIgnoreCase(title, currentSearchKeyword) ||
            containsIgnoreCase(host, currentSearchKeyword);
        } else if (currentSearchField === "datetime") {
          matchedSearch = containsIgnoreCase(
            datetimeDisplay,
            currentSearchKeyword
          );
        }
      }

      

      // 열람 필터
      let matchedRead = true;
      if (currentReadFilter === "read") {
        matchedRead = readStatus === "read";
      } else if (currentReadFilter === "unread") {
        matchedRead = readStatus === "unread";
      }

      if (matchedSearch && matchedRead) {
        filteredRows.push(row);
      }
    });

    

    // 2. 정렬
    filteredRows.sort((a, b) => {
      if (currentSortField === "datetime") {
        const aKey = a.dataset.datetime || "";
        const bKey = b.dataset.datetime || "";
        if (aKey === bKey) return 0;
        if (currentSortDirection === "asc") {
          return aKey < bKey ? -1 : 1;
        } else {
          return aKey > bKey ? -1 : 1;
        }
      } else if (currentSortField === "title") {
        const aKey = (a.dataset.title || "").toLowerCase();
        const bKey = (b.dataset.title || "").toLowerCase();
        if (currentSortDirection === "asc") {
          return aKey.localeCompare(bKey);
        } else {
          return bKey.localeCompare(aKey);
        }
      }
      return 0;
    });

    // 3. 페이지 수 계산
    totalPages = Math.max(1, Math.ceil(filteredRows.length / pageSize));
    if (resetPage) {
      currentPage = 1;
    } else {
      if (currentPage > totalPages) currentPage = totalPages;
    }
  }

  // ===== 현재 페이지의 row만 테이블에 렌더링 =====
  function renderPage() {
    tbody.innerHTML = "";

    if (filteredRows.length === 0) {
      // 필터 결과가 없으면 빈 테이블
      return;
    }

    const startIdx = (currentPage - 1) * pageSize;
    const endIdx = startIdx + pageSize;

    const pageRows = filteredRows.slice(startIdx, endIdx);
    pageRows.forEach((row) => {
      row.style.display = "";
      tbody.appendChild(row);
    });
  }

  // ===== 페이지네이션 UI 렌더링 =====
  function renderPagination() {
    if (!paginationContainer || !pageNumbersWrap) return;

    pageNumbersWrap.innerHTML = "";

    // 페이지 번호 버튼들
    for (let i = 1; i <= totalPages; i++) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "page-number";
      if (i === currentPage) {
        btn.classList.add("is-active");
      }
      btn.textContent = i.toString();
      btn.addEventListener("click", function () {
        if (currentPage === i) return;
        currentPage = i;
        renderPage();
        renderPagination();
      });
      pageNumbersWrap.appendChild(btn);
    }

    // 이전/다음 버튼 활성/비활성
    if (pagePrevBtn) {
      pagePrevBtn.disabled = currentPage <= 1;
      pagePrevBtn.onclick = function () {
        if (currentPage > 1) {
          currentPage -= 1;
          renderPage();
          renderPagination();
        }
      };
    }

    if (pageNextBtn) {
      pageNextBtn.disabled = currentPage >= totalPages;
      pageNextBtn.onclick = function () {
        if (currentPage < totalPages) {
          currentPage += 1;
          renderPage();
          renderPagination();
        }
      };
    }

    // 전체 데이터가 1페이지 이하면, 그냥 버튼만 최소 상태로 유지
  }

  // ===== 필터 + 정렬 + 페이지네이션 한 번에 적용 =====
  function applyFiltersAndSort(resetPage = true) {
    recomputeFilteredAndSorted(resetPage);
    renderPage();
    renderPagination();
  }

  // ===== 검색어 입력 =====
  if (searchInput) {
    searchInput.addEventListener("input", function (e) {
      currentSearchKeyword = e.target.value.trim();

      // 검색어가 있으면 초기화 버튼 표시, 없으면 숨김
      if (searchClearBtn) {
        searchClearBtn.style.display = currentSearchKeyword ? "flex" : "none";
      }

      applyFiltersAndSort(true); // 검색 바꾸면 항상 1페이지로
    });
  }

  // ===== 검색어 초기화 버튼 =====
  if (searchClearBtn && searchInput) {
    searchClearBtn.addEventListener("click", function () {
      searchInput.value = "";
      currentSearchKeyword = "";
      searchClearBtn.style.display = "none";
      applyFiltersAndSort(true); // 검색 초기화 시 1페이지로
    });
  }

  // ===== 검색 조건 드롭다운 =====
  if (searchFilterToggle && searchFilterDropdown) {
    searchFilterToggle.addEventListener("click", function (e) {
      e.stopPropagation();
      searchFilterDropdown.classList.toggle("is-open");
      const expanded = searchFilterDropdown.classList.contains("is-open");
      searchFilterToggle.setAttribute("aria-expanded", expanded ? "true" : "false");
    });

    searchFilterOptions.forEach((btn) => {
      btn.addEventListener("click", function () {
        searchFilterOptions.forEach((b) => b.classList.remove("is-active"));
        btn.classList.add("is-active");
        currentSearchField = btn.dataset.field || "title";
        if (searchFilterLabel) {
          searchFilterLabel.textContent = btn.textContent.trim();
        }
        searchFilterDropdown.classList.remove("is-open");
        searchFilterToggle.setAttribute("aria-expanded", "false");
        applyFiltersAndSort(true);
      });
    });
  }

  // ===== 열람 필터 드롭다운 =====
  if (readFilterToggle && readFilterDropdown) {
    readFilterToggle.addEventListener("click", function (e) {
      e.stopPropagation();
      readFilterDropdown.classList.toggle("is-open");
      const expanded = readFilterDropdown.classList.contains("is-open");
      readFilterToggle.setAttribute("aria-expanded", expanded ? "true" : "false");
    });

    readFilterOptions.forEach((btn) => {
      btn.addEventListener("click", function () {
        readFilterOptions.forEach((b) => b.classList.remove("is-active"));
        btn.classList.add("is-active");
        currentReadFilter = btn.dataset.read || "all";
        readFilterDropdown.classList.remove("is-open");
        readFilterToggle.setAttribute("aria-expanded", "false");
        applyFiltersAndSort(true);
      });
    });
  }

  // ===== 정렬 버튼: 일시 =====
  if (sortDateButton && sortDateIcon) {
    sortDateButton.addEventListener("click", function () {
      currentSortField = "datetime";
      // 방향 토글
      currentSortDirection = currentSortDirection === "desc" ? "asc" : "desc";

      // 아이콘 상태 갱신
      sortDateIcon.classList.remove(
        "sort-icon-up",
        "sort-icon-down",
        "sort-icon-none"
      );
      sortTitleIcon.classList.remove("sort-icon-up", "sort-icon-down");
      sortTitleIcon.classList.add("sort-icon-none");

      if (currentSortDirection === "desc") {
        sortDateIcon.classList.add("sort-icon-down");
      } else {
        sortDateIcon.classList.add("sort-icon-up");
      }

      applyFiltersAndSort(true); // 같은 필터에서 정렬만 바뀌면 현재 페이지 유지
    });
  }

  // ===== 정렬 버튼: 제목 =====
  if (sortTitleButton && sortTitleIcon) {
    sortTitleButton.addEventListener("click", function () {
      currentSortField = "title";

      if (
        !sortTitleIcon.classList.contains("sort-icon-up") &&
        !sortTitleIcon.classList.contains("sort-icon-down")
      ) {
        currentSortDirection = "asc"; // 처음 클릭: 오름차순
      } else {
        currentSortDirection = currentSortDirection === "asc" ? "desc" : "asc";
      }

      // 아이콘 상태 업데이트
      sortTitleIcon.classList.remove(
        "sort-icon-up",
        "sort-icon-down",
        "sort-icon-none"
      );
      sortDateIcon.classList.remove("sort-icon-up", "sort-icon-down");
      sortDateIcon.classList.add("sort-icon-down"); // 일시는 기본 ↓ 표시

      if (currentSortDirection === "asc") {
        sortTitleIcon.classList.add("sort-icon-up");
      } else {
        sortTitleIcon.classList.add("sort-icon-down");
      }

      applyFiltersAndSort(true);
    });
  }

  // ===== 바깥 클릭 시 드롭다운 닫기 =====
  document.addEventListener("click", function () {
    if (searchFilterDropdown) {
      searchFilterDropdown.classList.remove("is-open");
      searchFilterToggle &&
        searchFilterToggle.setAttribute("aria-expanded", "false");
    }
    if (readFilterDropdown) {
      readFilterDropdown.classList.remove("is-open");
      readFilterToggle &&
        readFilterToggle.setAttribute("aria-expanded", "false");
    }
  });

  // ===== 초기 적용 (기본: 일시 기준 최신순, 페이지 1) =====
  applyFiltersAndSort(true);
});