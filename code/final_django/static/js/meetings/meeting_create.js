document.addEventListener("DOMContentLoaded", function () {
  /* ===== 특화 도메인: 라디오 버튼으로 1개만 선택 ===== */
  // 라디오 버튼은 HTML의 name 속성으로 자동으로 단일 선택됨
  // 별도의 JavaScript 처리 불필요

  /* ===== 도메인 파일 업로드: 파일명 표시 ===== */
  const domainFileInput = document.getElementById("id_domain_file");
  const domainFileNameSpan = document.getElementById("domain-file-name");

  if (domainFileInput && domainFileNameSpan) {
    domainFileInput.addEventListener("change", function () {
      if (domainFileInput.files && domainFileInput.files.length > 0) {
        domainFileNameSpan.textContent = domainFileInput.files[0].name;
      } else {
        domainFileNameSpan.textContent = "지원 형식: TXT / CSV / XLSX";
      }
    });
  }

  /* ===== 날짜/시간 분리 입력 처리 ===== */
  const meetDateInput = document.getElementById("id_meet_date");
  const meetTimeInput = document.getElementById("id_meet_time");
  const meetDateTimeHidden = document.getElementById("id_meet_date_time");

  // 현재 날짜/시간으로 초기화
  if (meetDateInput && meetTimeInput) {
    const now = new Date();
    const pad = (n) => String(n).padStart(2, "0");
    const dateStr = `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())}`;
    const timeStr = `${pad(now.getHours())}:${pad(now.getMinutes())}`;

    meetDateInput.value = dateStr;
    meetTimeInput.value = timeStr;

    // hidden input에 합친 값 설정
    if (meetDateTimeHidden) {
      meetDateTimeHidden.value = `${dateStr}T${timeStr}`;
    }

    // 날짜 또는 시간 변경 시 hidden input 업데이트
    const updateDateTime = () => {
      if (meetDateInput.value && meetTimeInput.value && meetDateTimeHidden) {
        meetDateTimeHidden.value = `${meetDateInput.value}T${meetTimeInput.value}`;
      }
    };

    meetDateInput.addEventListener("change", updateDateTime);
    meetTimeInput.addEventListener("change", updateDateTime);
  }

  /* ===== 참석자 선택 ===== */
  // 부서 토글(열기/닫기) 처리: .tree-label을 클릭하면 해당 부서의 사용자 리스트를 숨김/표시
  const deptNodes = document.querySelectorAll('.attendee-tree .tree-root li > ul > li');
  deptNodes.forEach(function(deptLi){
    const label = deptLi.querySelector('.tree-label');
    if (!label) return;
    label.classList.add('dept-toggle');
    // 초기 상태: 접힘 (collapsed)으로 시작
    deptLi.classList.add('dept-collapsed');
    label.setAttribute('aria-expanded', 'false');
    label.addEventListener('click', function(e){
      // 토글 클래스
      const collapsed = deptLi.classList.toggle('dept-collapsed');
      label.setAttribute('aria-expanded', collapsed ? 'false' : 'true');
    });
  });

  const treeUsers = document.querySelectorAll(".tree-user");
  const selectedList = document.getElementById("selected-attendee-list");
  const attendeeCount = document.getElementById("attendee-count");
  const loginUserIdInput = document.getElementById("login-user-id");
  const loginUserNameInput = document.getElementById("login-user-name");
  const loginUserId = loginUserIdInput?.value || null;
  const loginUserName = loginUserNameInput?.value || "";
  const clearBtn = document.getElementById("btn-clear-attendees");

  function refreshCount() {
    if (!attendeeCount || !selectedList) return;
    const items = selectedList.querySelectorAll(".selected-attendee-item");
    const totalCount = items.length;
    attendeeCount.textContent = `${totalCount}명`;
  }

  function createHiddenInput(userId) {
    const input = document.createElement("input");
    input.type = "hidden";
    input.name = "attendees";      // 서버에서 getlist("attendees") 로 받는 값
    input.value = userId;
    return input;
  }

  function handleUserClick(e) {
    const el = e.currentTarget;
    const userId = el.dataset.userId;
    const userName = el.dataset.userName || el.textContent.trim();

    // 주최자는 트리에서 선택할 수 없음 (이미 기본 참석)
    if (userId === loginUserId) {
      return;
    }

    if (!selectedList) return;

    // 이미 선택되었는지 확인
    const exists = selectedList.querySelector(
      `.selected-attendee-item[data-user-id="${userId}"]`
    );
    if (exists) return;

    const li = document.createElement("li");
    li.className = "selected-attendee-item";
    li.dataset.userId = userId;

    const nameSpan = document.createElement("span");
    nameSpan.textContent = userName;

    const removeBtn = document.createElement("button");
    removeBtn.type = "button";
    removeBtn.className = "attendee-remove-btn";
    removeBtn.innerHTML = "×";

    removeBtn.addEventListener("click", function () {
      li.remove();
      refreshCount();
    });

    li.appendChild(nameSpan);
    li.appendChild(removeBtn);

    // ★ 추가 참석자에만 hidden input 생성 (주최자는 생성하지 않음)
    li.appendChild(createHiddenInput(userId));

    selectedList.appendChild(li);
    refreshCount();
  }

  // 부서 트리 참석자 클릭 이벤트 바인딩
  treeUsers.forEach((userEl) => {
    if (userEl.dataset.userId === loginUserId) {
      userEl.classList.add("tree-user-disabled"); // 시각적 비활성화
    } else {
      userEl.addEventListener("click", handleUserClick);
    }
  });

  /* ===== 로그인 유저를 기본 참석자로 자동 추가 (UI 용도만, hidden X) ===== */
  if (loginUserId && loginUserName && selectedList) {
    const exists = selectedList.querySelector(
      `.selected-attendee-item[data-user-id="${loginUserId}"]`
    );
    if (!exists) {
      const li = document.createElement("li");
      li.className = "selected-attendee-item";
      li.dataset.userId = loginUserId;

      const nameSpan = document.createElement("span");
      nameSpan.textContent = loginUserName + " (주최자)";

      const removeBtn = document.createElement("button");
      removeBtn.type = "button";
      removeBtn.className = "attendee-remove-btn";
      removeBtn.innerHTML = "×";

      // 주최자는 제거 불가
      removeBtn.addEventListener("click", function () {
        alert("회의 생성자는 참석자 목록에서 제거할 수 없습니다.");
      });

      li.appendChild(nameSpan);
      li.appendChild(removeBtn);

      // ※ 여기서는 createHiddenInput(loginUserId)를 붙이지 않는다
      selectedList.appendChild(li);

      refreshCount();
    }
  }

  // "모두 지우기" 버튼: 주최자는 남기고, 나머지만 삭제
  if (clearBtn && selectedList) {
    clearBtn.addEventListener("click", function () {
      const items = selectedList.querySelectorAll(".selected-attendee-item");
      items.forEach((item) => {
        if (item.dataset.userId !== loginUserId) {
          item.remove();
        }
      });
      refreshCount();
    });
  }

  refreshCount();

  // === 폼 제출 시: 주최자를 제외한 '추가 참석자'가 최소 1명인지 검사 ===
  const form = document.querySelector(".meeting-form");
  const attendeeError = document.getElementById("attendee-error");
  const attendeeSelectedBox = document.querySelector(".attendee-selected");

  if (form && selectedList) {
    form.addEventListener("submit", function (e) {
      const items = selectedList.querySelectorAll(".selected-attendee-item");

      // 주최자를 제외한 참석자 수
      const extraCount = Array.from(items).filter(
        (item) => item.dataset.userId !== loginUserId
      ).length;

      if (extraCount === 0) {
        // 추가 참석자 0명이면 제출 막고 에러 표시
        e.preventDefault();

        if (attendeeError) {
          attendeeError.style.display = "block";
        }
        if (attendeeSelectedBox) {
          attendeeSelectedBox.classList.add("attendee-error-border");
          attendeeSelectedBox.scrollIntoView({
            behavior: "smooth",
            block: "center",
          });
        }
      } else {
        // 1명 이상이면 에러 숨기고 그대로 제출
        if (attendeeError) {
          attendeeError.style.display = "none";
        }
        if (attendeeSelectedBox) {
          attendeeSelectedBox.classList.remove("attendee-error-border");
        }
      }
    });
  }
  const date_time_input = document.getElementById("id_meet_date");
  date_time_input.max = new Date().toLocaleDateString('sv').replace(/ /g,'');
});
