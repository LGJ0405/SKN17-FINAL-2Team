document.addEventListener("DOMContentLoaded", function () {
  const modifyPwLink = document.getElementById("modify-pw-link");
  const pwModal = document.getElementById("pw-modal");
  const pwCancel = document.getElementById("pw-cancel");
  const pwForm = document.getElementById("pw-form");

  const curErr = document.getElementById("current_pw_error"); // 1번 칸
  const curOk  = document.getElementById("current_pw_ok");
  const newPwErr = document.getElementById("new_pw_error");   // 2번 칸
  const newPw2Err = document.getElementById("new_pw2_error");   // 3번 칸
  const commonErr = document.getElementById("pw_common_error"); // 버튼 아래

  const collapse = document.querySelector(".sidebar-collapse");
  const nested = document.querySelector(".sidebar-menu-nested");

  // 로그아웃 안내 후 진행
  const logoutLinks = document.querySelectorAll('a.sidebar-link-footer[href*="logout"]');
  logoutLinks.forEach((link) => {
    link.addEventListener("click", (e) => {
      e.preventDefault();
      const ok = window.confirm("로그아웃하시겠습니까?");
      if (ok) {
        window.location.href = link.href;
      }
    });
  });

  // 회의 목록 접기 & 펼치기
  if (collapse && nested) {
    collapse.addEventListener("click", function (e) {
        // 화살표 아이콘을 클릭한 경우에만 토글
        if (e.target.classList.contains("sidebar-collapse-icon") ||
            e.target.closest(".sidebar-collapse-icon")) {
          nested.classList.toggle("is-collapsed");
          collapse.classList.toggle("collapsed");
        } else if (e.target.closest(".sidebar-collapse-left") ||
                   e.target.classList.contains("sidebar-section-title") ||
                   e.target.classList.contains("sidebar-icon-list")) {
          // 텍스트나 아이콘 부분을 클릭하면 페이지 이동
          window.location.href = collapse.dataset.url || "/meetings/list/all/";
        }
        // 빈 공간 클릭 시에는 아무 동작도 하지 않음
    });
  }

  // 사이드바 토글 기능 (일반 사이드바)
  const mainSidebar = document.getElementById("main-sidebar");
  const sidebarToggle = document.getElementById("sidebar-toggle");

  if (mainSidebar && sidebarToggle) {
    // localStorage에서 저장된 상태 불러오기
    const savedState = localStorage.getItem("sidebarCollapsed");
    if (savedState === "true") {
      mainSidebar.classList.add("collapsed");
      sidebarToggle.style.left = "56px";
    }

    sidebarToggle.addEventListener("click", function () {
      mainSidebar.classList.toggle("collapsed");
      // 상태 저장
      const isCollapsed = mainSidebar.classList.contains("collapsed");
      localStorage.setItem("sidebarCollapsed", isCollapsed);

      // 토글 버튼 위치 업데이트
      sidebarToggle.style.left = isCollapsed ? "56px" : "246px";
    });
  }

  // 관리자 사이드바 토글 기능
  const adminSidebar = document.getElementById("admin-sidebar");
  const adminSidebarToggle = document.getElementById("admin-sidebar-toggle");
  const adminSearchInput = document.getElementById("admin-user-search");

  // 접힘 상태 검색 버튼 (템플릿에 존재)
  const adminSearchToggleBtn = document.getElementById("admin-sidebar-search-toggle");

  if (adminSidebar && adminSidebarToggle) {
    // localStorage에서 저장된 상태 불러오기
    const savedState = localStorage.getItem("adminSidebarCollapsed");
    if (savedState === "true") {
      adminSidebar.classList.add("collapsed");
      adminSidebarToggle.style.left = "56px";
    }

    adminSidebarToggle.addEventListener("click", function () {
      adminSidebar.classList.toggle("collapsed");
      // 상태 저장
      const isCollapsed = adminSidebar.classList.contains("collapsed");
      localStorage.setItem("adminSidebarCollapsed", isCollapsed);

      // 토글 버튼 위치 업데이트
      adminSidebarToggle.style.left = isCollapsed ? "56px" : "246px";
    });
  }

  // 접힘 상태 검색 버튼 클릭 시 사이드바 열고 포커스
  if (adminSidebar && adminSidebarToggle && adminSearchToggleBtn) {
    adminSearchToggleBtn.addEventListener("click", () => {
      adminSidebar.classList.remove("collapsed");
      adminSidebarToggle.style.left = "246px";
      localStorage.setItem("adminSidebarCollapsed", "false");
      if (adminSearchInput) {
        setTimeout(() => adminSearchInput.focus(), 50);
      }
    });
  }

  function openModal(modal) {
    if (modal) modal.classList.remove("hidden");
  }

  function closeModal(modal) {
    if (modal) modal.classList.add("hidden");
  }

  // 모달 초기 상태는 숨김
  if (pwModal) {
    pwModal.classList.add("hidden");
  }

  // 사이드바 "비밀번호 변경" 클릭 → 모달 오픈
  if (modifyPwLink && pwModal) {
    modifyPwLink.addEventListener("click", function (e) {
      e.preventDefault();
      openModal(pwModal);
    });
  }

  // 취소 버튼 → 값/에러 초기화 + 모달 닫기
  if (pwCancel && pwModal) {
    pwCancel.addEventListener("click", function () {
      const curInput = document.getElementById("current_pw");
      const n1Input = document.getElementById("new_pw");
      const n2Input = document.getElementById("new_pw2");
      const curOk = document.getElementById("current_pw_ok");
      const newPwOk = document.getElementById("new_pw_ok");
      const newPw2Ok = document.getElementById("new_pw2_ok");

      if (curInput) curInput.value = "";
      if (n1Input) n1Input.value = "";
      if (n2Input) n2Input.value = "";

      [curErr, newPwErr, newPw2Err, commonErr, curOk, newPwOk, newPw2Ok].forEach((el) => {
        if (el) {
          el.classList.remove("visible");
          if (el === commonErr) el.textContent = "";
        }
      });

      closeModal(pwModal);
    });
  }

  // =========================
  // 기존 비밀번호 실시간 체크
  // =========================
  const curInput = document.getElementById("current_pw");
  const newPwInput = document.getElementById("new_pw");
  const newPw2Input = document.getElementById("new_pw2");
  const newPw2Ok = document.getElementById("new_pw2_ok"); // 파란 안내 span
  const newPwOk = document.getElementById("new_pw_ok");
  let curCheckTimer = null;

  if (curInput && curOk) {
    curInput.addEventListener("input", function () {
      // 입력이 바뀔 때마다 타이머 초기화 (디바운스)
      if (curCheckTimer) {
        clearTimeout(curCheckTimer);
      }

      const value = curInput.value;

      // 입력이 비어 있으면 모든 메세지 숨김
      if (!value) {
        curOk.classList.remove("visible");
        if (curErr) curErr.classList.remove("visible");
        return;
      }

      const checkUrl = curInput.dataset.checkUrl;
      const csrftoken = document.querySelector(
        "input[name='csrfmiddlewaretoken']"
      )?.value;

      // 400ms 동안 추가 타이핑 없을 때만 서버 호출
      curCheckTimer = setTimeout(async () => {
        try {
          const formData = new FormData();
          formData.append("old_password", value);

          const res = await fetch(checkUrl, {
            method: "POST",
            headers: {
              "X-CSRFToken": csrftoken,
              "X-Requested-With": "XMLHttpRequest",
            },
            body: formData,
          });

          const data = await res.json();

          // 서버 기준으로 일치 여부 판단
          if (data.ok) {
            // 일치 → 파란 메세지 표시, 빨간 에러는 숨김
            curOk.classList.add("visible");
            if (curErr) curErr.classList.remove("visible");
          } else {
            // 불일치 → 파란 메세지 숨기고 (여기서는 에러는 띄우지 않음)
            curOk.classList.remove("visible");
            if (curErr) curErr.classList.add("visible");
          }
        } catch (err) {
          console.error("기존 비밀번호 확인 중 오류:", err);
          // 오류 시에는 그냥 아무 메세지도 띄우지 않음
        }
      }, 400);
    });
  }
  
  // =========================
  // 새 비밀번호칸 실시간 형식 체크
  // =========================

  if (newPwInput && newPwOk && newPwErr) {
    newPwInput.addEventListener("input", function () {
      const n1 = newPwInput.value.trim();

      // 일단 파란/빨간 둘 다 숨김으로 시작
      newPwOk.classList.remove("visible");
      newPwErr.classList.remove("visible");

      // 아무것도 안 썼으면 메시지 없음
      if (!n1) {
        return;
      }

      // 비밀번호 정책 (8~15자, 영문 + 숫자)
      const pwPattern = /^(?=.*[A-Za-z])(?=.*\d)[A-Za-z\d]{8,15}$/;

      if (pwPattern.test(n1)) {
        // 형식에 맞으면 → 파란 "올바른 형식입니다."
        newPwOk.textContent = "올바른 형식입니다.";
        newPwOk.classList.add("visible");
      } else {
        // 형식에 맞지 않으면 → 빨간 "올바르지 않은 형식입니다."
        newPwErr.textContent = "올바르지 않은 형식입니다.";
        newPwErr.classList.add("visible");
      }
    });
  }

  // =========================
  // 새 비밀번호 확인 칸 실시간 일치 여부 체크
  // =========================

  if (newPwInput && newPw2Input && newPw2Ok && newPw2Err) {
    newPw2Input.addEventListener("input", function () {
      const n1 = newPwInput.value.trim();   // 새 비밀번호
      const n2 = newPw2Input.value.trim();  // 새 비밀번호 확인

      // 1) 입력 바뀔 때마다 둘 다 먼저 끔 (중복 방지 핵심)
      newPw2Ok.classList.remove("visible");
      newPw2Err.classList.remove("visible");

      // 2) 아무 것도 안 썼으면 메시지 없음
      if (!n2) {
        return;
      }

      // 3) 둘이 완전히 같으면 파란 OK만
      if (n1 === n2) {
        newPw2Ok.textContent = "비밀번호가 일치합니다.";
        newPw2Ok.classList.add("visible");
        // 에러는 이미 위에서 꺼놨음
      } else {
        // 4) 다르면 빨간 에러만
        newPw2Err.textContent = "비밀번호를 확인해주세요.";
        newPw2Err.classList.add("visible");
        // 파란 OK는 이미 꺼져 있음
      }
    });
  }
  
  // =========================
  // 아래에는 기존 submit 검증 로직 그대로…
  // =========================

  // 비밀번호 변경 폼 제출 로직
  if (pwForm) {
    pwForm.addEventListener("submit", async function (e) {
      e.preventDefault(); // 항상 우리가 제어

      const curInput = document.getElementById("current_pw");
      const n1Input = document.getElementById("new_pw");
      const n2Input = document.getElementById("new_pw2");

      const cur = curInput.value.trim();
      const n1 = n1Input.value.trim();
      const n2 = n2Input.value.trim();

      // 에러 초기화
      [curErr, newPwErr, newPw2Err, commonErr, newPwOk].forEach((el) => {
        if (el) {
          el.classList.remove("visible");
          if (el === commonErr) el.textContent = "";
        }
      });

      // ① 하나라도 비어 있으면 → 무조건 이 조건만 탄다
      if (!cur || !n1 || !n2) {
        if (commonErr) {
          commonErr.textContent = "모든 항목을 입력해 주세요.";
          commonErr.classList.add("visible");
        }
        return;
      }

      // 여기까지 왔다는 것은 1번 조건(모든 칸 입력) 통과

      const csrftoken = document.querySelector(
        "input[name='csrfmiddlewaretoken']"
      ).value;
      const formData = new FormData(pwForm);

      try {
        const res = await fetch(pwForm.action, {
          method: "POST",
          headers: {
            "X-CSRFToken": csrftoken,
            "X-Requested-With": "XMLHttpRequest",
          },
          body: formData,
        });

        const data = await res.json();

        if (!data.ok) {
          const field = data.field;
          const message = data.message || "비밀번호를 확인해주세요.";

          // 에러가 난 순간, 모든 파란 OK 문구는 우선 숨긴다
          [curOk, newPwOk, newPw2Ok].forEach((el) => {
            if (el) el.classList.remove("visible");
          });

          // 서버에서 어디가 틀렸다고 알려줬는지에 따라 해당 위치에 표시
          if (field === "old_password" && curErr) {
            // 조건 4: 기존 비밀번호 불일치 → 1번 입력칸 아래
            curErr.textContent = message;
            curErr.classList.add("visible");
          } else if (field === "new_password" && newPwErr) {
            // 조건 2: 형식 위반 → 2번 입력칸 아래
            newPwErr.textContent = message;
            newPwErr.classList.add("visible");
          } else if (field === "new_password2" && newPw2Err) {
            // 조건 3: 새 비밀번호 불일치 → 3번 입력칸 아래
            newPw2Err.textContent = message;
            newPw2Err.classList.add("visible");
          } else if (commonErr) {
            // 예외적인 경우 공통 에러로
            commonErr.textContent = message;
            commonErr.classList.add("visible");
          }
          return;
        }

        // 성공: 모달 닫고 새로고침(또는 원하는 동작)
        [curErr, newPwErr, newPw2Err, commonErr].forEach((el) => {
        if (el) {
          el.classList.remove("visible");
          if (el === commonErr) el.textContent = "";
        }
      });
        closeModal(pwModal);
        window.location.reload();
      } catch (err) {
        console.error("비밀번호 변경 중 오류:", err);
        if (commonErr) {
          commonErr.textContent = "비밀번호 변경 중 오류가 발생했습니다. 다시 시도해 주세요.";
          commonErr.classList.add("visible");
        }
      }
    });
  }
});
