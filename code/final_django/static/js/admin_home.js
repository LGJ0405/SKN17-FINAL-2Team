// static/js/admin_home.js

// ========== 부서 검색 기능 ==========
document.addEventListener("DOMContentLoaded", function () {
  const searchInput = document.getElementById("admin-user-search");
  const deptItems = document.querySelectorAll(".dept-item");

  if (searchInput && deptItems.length > 0) {
    searchInput.addEventListener("input", function () {
      const searchTerm = this.value.toLowerCase().trim();

      deptItems.forEach(function (item) {
        const deptName = item.textContent.toLowerCase();
        if (deptName.includes(searchTerm)) {
          item.style.display = "";
        } else {
          item.style.display = "none";
        }
      });
    });
  }
});

// ========== 부서 클릭 및 필터링 ==========
document.addEventListener("DOMContentLoaded", function () {
  // 1) 사이드바 부서 노드들(앞에서 dept-item 클래스 붙여둔 것)
  const deptItems = document.querySelectorAll(".dept-item");

  // 2) 가운데 상단 부서 제목 영역
  const deptTitleEl = document.getElementById("admin-dept-title");

  // 3) 직원 테이블 tbody
  const memberTbody = document.getElementById("admin-member-tbody");

  if (!memberTbody) {
    // 예상치 못한 경우를 대비한 방어 코드
    return;
  }

  // 4) 처음 로딩 시점의 모든 직원 행을 메모리에 저장해 둔다.
  const allRows = Array.from(
    memberTbody.querySelectorAll(".admin-member-row")
  );

  /**
   * 선택된 부서 ID에 따라 테이블을 다시 그리는 함수
   * @param {string|null} deptId - 부서 ID (null이면 전체 부서)
   */
  function renderRowsByDept(deptId) {
    // tbody 비우기
    memberTbody.innerHTML = "";

    let rowsToRender;

    if (deptId === null) {
      // 전체 부서: 모든 행
      rowsToRender = allRows;
    } else {
      // 특정 부서만 필터링
      rowsToRender = allRows.filter(function (row) {
        return row.dataset.deptId === String(deptId);
      });
    }

    if (rowsToRender.length === 0) {
      // 해당 부서에 직원이 없으면 "현재 표시할 데이터가 없습니다." 행 추가
      const emptyTr = document.createElement("tr");
      emptyTr.innerHTML =
        '<td colspan="5" class="admin-table-empty">현재 표시할 데이터가 없습니다.</td>';
      memberTbody.appendChild(emptyTr);
      return;
    }

    // 필터링된 행들을 tbody에 다시 붙인다.
    rowsToRender.forEach(function (row, index) {
      // 동일한 tr을 여러 번 이동시키면 allRows에서 사라지므로, clone해서 사용
      const clone = row.cloneNode(true);
      // 번호 컬럼 업데이트 (1부터 시작)
      const noCell = clone.querySelector(".col-no");
      if (noCell) {
        noCell.textContent = index + 1;
      }
      memberTbody.appendChild(clone);
    });
  }

  /**
   * 사이드바에서 선택된 부서 하이라이트(class 토글)
   */
  function clearActiveDept() {
    deptItems.forEach(function (item) {
      item.classList.remove("dept-item-active");
    });
  }

  // 5) 사이드바의 각 부서 노드에 클릭 이벤트 연결
  deptItems.forEach(function (item) {
    item.addEventListener("click", function () {
      const deptId = this.dataset.deptId;            // 숫자 또는 문자열
      const deptName = this.textContent.trim();      // 화면에 보이는 부서명

      // 사이드바 하이라이트
      clearActiveDept();
      this.classList.add("dept-item-active");

      // 가운데 제목 바꾸기
      if (deptTitleEl) {
        deptTitleEl.textContent = deptName;
      }

      // 테이블 필터링
      renderRowsByDept(deptId);
    });
  });

  // 6) 페이지 로드 시 첫 번째 부서를 자동으로 선택
  if (deptItems.length > 0) {
    const firstDept = deptItems[0];
    const firstDeptId = firstDept.dataset.deptId;
    const firstDeptName = firstDept.textContent.trim();

    // 첫 번째 부서 하이라이트
    firstDept.classList.add("dept-item-active");

    // 제목 업데이트
    if (deptTitleEl) {
      deptTitleEl.textContent = firstDeptName;
    }

    // 해당 부서의 부서원만 표시
    renderRowsByDept(firstDeptId);
  }
});

// ========== 부서원 추가/수정 폼 관리 ==========
document.addEventListener("DOMContentLoaded", function () {
  const addBtn = document.getElementById("btn-admin-add");
  const rightPanel = document.getElementById("admin-panel-right");
  const panelToggle = document.getElementById("btn-panel-toggle");
  const closeBtn = document.getElementById("btn-admin-close");
  const resetPwBtn = document.getElementById("btn-admin-reset-pw");
  const form = document.getElementById("admin-employee-form");
  const formTitle = document.getElementById("form-title");
  const submitBtn = document.getElementById("btn-admin-submit");
  const formMode = document.getElementById("form-mode");
  const formUserId = document.getElementById("form-user-id");
  const empIdField = document.getElementById("emp-id");
  const empBirthField = document.getElementById("emp-birth");
  const empAdminField = document.getElementById("emp-admin");

  let currentSelectedDeptId = null; // 현재 선택된 부서 ID 추적

  if (!addBtn || !rightPanel || !form) {
    return;
  }

  // 패널 토글 버튼
  if (panelToggle) {
    panelToggle.addEventListener("click", function () {
      rightPanel.classList.toggle("collapsed");
    });
  }

  // 생년월일 유효성 검사 함수
  function validateBirthDate(birthDate) {
    if (!/^\d{8}$/.test(birthDate)) {
      return "생년월일은 8자리 숫자로 입력해주세요.";
    }

    const year = parseInt(birthDate.substring(0, 4));
    const month = parseInt(birthDate.substring(4, 6));
    const day = parseInt(birthDate.substring(6, 8));

    const birthDateObj = new Date(year, month - 1, day);
    if (
      birthDateObj.getFullYear() !== year ||
      birthDateObj.getMonth() !== month - 1 ||
      birthDateObj.getDate() !== day
    ) {
      return "유효하지 않은 날짜입니다.";
    }

    const today = new Date();
    const hundredYearsAgo = new Date(today.getFullYear() - 100, today.getMonth(), today.getDate());
    const nineteenYearsAgo = new Date(today.getFullYear() - 19, today.getMonth(), today.getDate());

    if (birthDateObj < hundredYearsAgo) {
      return "생년월일은 현재로부터 100년 이내여야 합니다.";
    }

    if (birthDateObj > nineteenYearsAgo) {
      return "만 19세 이상만 등록 가능합니다.";
    }

    return null;
  }

  // 폼 초기화 함수
  function resetForm() {
    form.reset();
    formMode.value = "create";
    formUserId.value = "";
    formTitle.style.display = "none";
    formTitle.textContent = "";
    submitBtn.textContent = "등록";
    closeBtn.textContent = "취소";
    resetPwBtn.style.display = "none";
    closeBtn.onclick = function() {
      rightPanel.classList.add("collapsed");
      resetForm();
    };
    empIdField.readOnly = false;
    empAdminField.checked = false;

    // 현재 선택된 부서가 있으면 자동 선택
    if (currentSelectedDeptId) {
      document.getElementById("emp-dept").value = currentSelectedDeptId;
    }
  }

  // 추가 버튼 클릭
  addBtn.addEventListener("click", function () {
    resetForm();
    rightPanel.classList.remove("collapsed");
  });

  // 닫기 버튼은 onclick으로만 제어 (addEventListener 제거)

  // 폼 제출
  form.addEventListener("submit", async function (e) {
    e.preventDefault();

    const mode = formMode.value;
    const birthDate = empBirthField.value.trim();

    // 생년월일 유효성 검사
    const birthDateError = validateBirthDate(birthDate);
    if (birthDateError) {
      alert(birthDateError);
      return;
    }

    const formData = {
      name: document.getElementById("emp-name").value.trim(),
      user_id: document.getElementById("emp-id").value.trim(),
      birth_date: birthDate,
      dept_id: parseInt(document.getElementById("emp-dept").value),
      work_part: document.getElementById("emp-role").value.trim(),
      admin_yn: empAdminField.checked,
    };

    try {
      let url, method;
      if (mode === "create") {
        url = "/admin/member/create/";
        method = "POST";
      } else {
        url = `/admin/member/${formUserId.value}/update/`;
        method = "PUT";
      }

      const response = await fetch(url, {
        method: method,
        headers: {
          "Content-Type": "application/json",
          "X-CSRFToken": getCookie("csrftoken"),
        },
        body: JSON.stringify(formData),
      });

      const result = await response.json();

      if (result.success) {
        alert(result.message);
        rightPanel.classList.add("collapsed");
        resetForm();
        // 페이지 새로고침하여 목록 갱신
        window.location.reload();
      } else {
        alert(result.message || "오류가 발생했습니다.");
      }
    } catch (error) {
      console.error("Error:", error);
      alert("서버와 통신 중 오류가 발생했습니다.");
    }
  });

  // 비밀번호 초기화 버튼 클릭
  if (resetPwBtn) {
    resetPwBtn.addEventListener("click", async function () {
      if (!confirm("비밀번호를 생년월일로 초기화하시겠습니까?")) {
        return;
      }

      try {
        const response = await fetch(`/admin/member/${formUserId.value}/reset-password/`, {
          method: "POST",
          headers: {
            "X-CSRFToken": getCookie("csrftoken"),
          },
        });

        const result = await response.json();

        if (result.success) {
          alert(result.message);
        } else {
          alert(result.message || "비밀번호 초기화 중 오류가 발생했습니다.");
        }
      } catch (error) {
        console.error("Error:", error);
        alert("서버와 통신 중 오류가 발생했습니다.");
      }
    });
  }

  // 부서원 행 클릭 이벤트
  document.addEventListener("click", function (e) {
    const row = e.target.closest(".admin-member-row");
    if (row) {
      // 🔥 [추가됨] 기존 모든 행의 선택 제거
      document.querySelectorAll(".admin-member-row").forEach(r => {
      r.classList.remove("selected");
    });

    // 🔥 [추가됨] 현재 클릭한 행 선택 스타일 적용
    row.classList.add("selected");
      // 행 데이터 추출
      const cells = row.querySelectorAll("td");
      const userId = cells[2].textContent.trim(); // ID 컬럼
      const name = cells[1].textContent.trim(); // 성명 컬럼
      const deptName = cells[3].textContent.trim(); // 부서 컬럼
      const birthDate = cells[4].textContent.trim(); // 생년월일 컬럼
      const deptId = row.dataset.deptId;
      const workPart = row.dataset.workPart || ""; // 업무 정보

      // 폼을 수정 모드로 설정
      formMode.value = "edit";
      formUserId.value = userId;
      formTitle.style.display = "block";
      formTitle.textContent = name;  // 사용자 이름을 제목으로
      submitBtn.textContent = "수정";
      closeBtn.textContent = "삭제";
      resetPwBtn.style.display = "block";

      // 닫기 버튼을 삭제 기능으로 변경
      closeBtn.onclick = async function() {
        const realDelete = confirm("정말 이 부서원을 삭제하시겠습니까?\n\n확인: 비활성화 (복구 가능)\n취소: 취소");
        if (!realDelete) {
          return;
        }

        // 진짜 삭제할지 추가 확인
        const permanentDelete = confirm("⚠️ 완전히 삭제하시겠습니까?\n\n확인: 영구 삭제 (복구 불가)\n취소: 비활성화 (복구 가능)");

        try {
          const url = `/admin/member/${formUserId.value}/delete/`;
          const body = permanentDelete ? JSON.stringify({ permanent: true }) : null;

          const response = await fetch(url, {
            method: "DELETE",
            headers: {
              "X-CSRFToken": getCookie("csrftoken"),
              "Content-Type": "application/json",
            },
            body: body,
          });

          const result = await response.json();

          if (result.success) {
            alert(result.message);
            rightPanel.classList.add("collapsed");
            resetForm();
            window.location.reload();
          } else {
            alert(result.message || "삭제 중 오류가 발생했습니다.");
          }
        } catch (error) {
          console.error("Error:", error);
          alert("서버와 통신 중 오류가 발생했습니다.");
        }

        
      };

      empIdField.readOnly = true;

      // 폼 필드 채우기 (관리자 권한 정보는 data 속성에서 가져와야 함)
      document.getElementById("emp-name").value = name;
      document.getElementById("emp-id").value = userId;
      // 생년월일 YYYY.MM.DD -> YYYYMMDD 형식으로 변환
      document.getElementById("emp-birth").value = birthDate.replace(/\./g, "");
      document.getElementById("emp-dept").value = deptId;
      document.getElementById("emp-role").value = workPart;

      // 관리자 권한 체크 (data 속성에서 가져오기)
      const isAdmin = row.dataset.adminYn === "True" || row.dataset.adminYn === "true";
      empAdminField.checked = isAdmin;

      // 오른쪽 패널 열기
      rightPanel.classList.remove("collapsed");
    }
  });

  // CSRF 토큰 가져오기
  function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== "") {
      const cookies = document.cookie.split(";");
      for (let i = 0; i < cookies.length; i++) {
        const cookie = cookies[i].trim();
        if (cookie.substring(0, name.length + 1) === name + "=") {
          cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
          break;
        }
      }
    }
    return cookieValue;
  }

  // 부서 클릭 시 현재 선택된 부서 ID 추적
  document.querySelectorAll(".dept-item").forEach(function (item) {
    item.addEventListener("click", function () {
      currentSelectedDeptId = this.dataset.deptId;
    });
  });

  // 초기 로드 시 첫 번째 부서를 currentSelectedDeptId로 설정
  const allDeptItems = document.querySelectorAll(".dept-item");
  if (allDeptItems.length > 0) {
    currentSelectedDeptId = allDeptItems[0].dataset.deptId;
  }
});
