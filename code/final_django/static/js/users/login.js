document.addEventListener("DOMContentLoaded", function () {
  // ===== 공통 요소/함수 =====
  const form = document.querySelector(".login-form");
  const csrftoken = document.querySelector("input[name='csrfmiddlewaretoken']")?.value;

  const initModal = document.getElementById("init-pw-modal");
  const initForm = document.getElementById("init-pw-form");
  const initPw1 = document.getElementById("init_new_pw");
  const initPw2 = document.getElementById("init_new_pw2");
  const initPwError = document.getElementById("init-pw-error");
  const initPwMessage = document.getElementById("init-pw-message");
  const initPwCancel = document.getElementById("init-pw-cancel");
  const initPwOk = document.getElementById("init_new_pw_ok");
  const initPwErr = document.getElementById("init_new_pw_error");
  const initPw2Ok = document.getElementById("init_new_pw2_ok");
  const initPw2Err = document.getElementById("init_new_pw2_error");
  const pwPattern = /^(?=.*[A-Za-z])(?=.*\d)[A-Za-z\d]{8,15}$/;

  function openInitModal(message) {
    if (!initModal) return;
    [initPwErr, initPw2Err, initPwOk, initPw2Ok].forEach((el) => {
      if (el) el.classList.remove("visible");
    });
    if (initPwError) {
      initPwError.textContent = "";
      initPwError.classList.remove("visible");
    }
    if (initPw1) initPw1.value = "";
    if (initPw2) initPw2.value = "";
    if (initPwMessage && message) initPwMessage.textContent = message;
    initModal.classList.remove("hidden");
  }

  function closeInitModal() {
    if (initModal) initModal.classList.add("hidden");
  }

  // ===== 1. 로그인 폼 =====
  if (form) {
    form.addEventListener("submit", async function (e) {
      e.preventDefault();
      const idInput = document.getElementById("id_user_id");
      const pwInput = document.getElementById("id_password");

      const idVal = idInput ? idInput.value.trim() : "";
      const pwVal = pwInput ? pwInput.value.trim() : "";

      if (!idVal || !pwVal) {
        alert("ID와 Password를 모두 입력해 주세요.");
        if (!idVal && idInput) idInput.focus();
        else if (!pwVal && pwInput) pwInput.focus();
        return;
      }

      // 기존 에러 메세지 제거
      document.querySelectorAll(".error").forEach((el) => el.remove());
      const msgArea = document.getElementById("messages-area");
      if (msgArea) {
        while (msgArea.firstChild) {
          msgArea.removeChild(msgArea.firstChild);
        }
      }

      const formData = new FormData(form);

      try {
        const res = await fetch("/users/login-api/", {
          method: "POST",
          headers: {
            "X-CSRFToken": csrftoken,
          },
          body: formData,
        });

        const data = await res.json();
        if (!res.ok) {
          const errors = data.errors || {};

          // messages-area에 에러 모아서 찍기
          if (msgArea) {
            const allMessages = [];

            if (Array.isArray(errors.__all__)) {
              allMessages.push(...errors.__all__);
            }
            if (Array.isArray(errors.user_id)) {
              allMessages.push(...errors.user_id);
            }
            if (Array.isArray(errors.password)) {
              allMessages.push(...errors.password);
            }

            allMessages.forEach((text) => {
              const p = document.createElement("p");
              p.className = "error";
              p.style.color = "red";
              p.textContent = text;
              msgArea.appendChild(p);
            });
          }

          // 비밀번호 정책 미충족 → 초기 변경 모달 띄우기
          if (data.force_change) {
            openInitModal(data.message || "초기 비밀번호를 변경해 주세요.");
          }
          return;
        }

        if (data.ok) {
          window.location.href = data.redirect_url || "/";
        }
      } catch (error) {
        console.error("로그인 중 오류 발생:", error);
        alert("로그인 처리 중 문제가 발생했습니다. 다시 시도하세요.");
      }
    });
  }

  // ===== 2. 비밀번호 찾기 버튼 =====
  const findPwBtn = document.getElementById("open-findpw-modal");
  if (findPwBtn) {
    findPwBtn.addEventListener("click", function(event) {
      event.preventDefault();
      alert("비밀번호 변경을 원하시면 관리자에게 문의하세요.");
    });
  }

  // ===== 3. 초기 비밀번호 변경 모달 =====
  if (initPwCancel) {
    initPwCancel.addEventListener("click", function () {
      closeInitModal();
    });
  }

  // 실시간 형식 체크
  if (initPw1 && initPwOk && initPwErr) {
    initPw1.addEventListener("input", function () {
      const val = initPw1.value.trim();
      initPwOk.classList.remove("visible");
      initPwErr.classList.remove("visible");

      // 첫 번째 비밀번호 변경 시 두 번째 입력란 메시지도 초기화
      if (initPw2Ok) initPw2Ok.classList.remove("visible");
      if (initPw2Err) initPw2Err.classList.remove("visible");

      if (!val) return;

      if (pwPattern.test(val)) {
        initPwOk.textContent = "올바른 형식입니다.";
        initPwOk.classList.add("visible");
      } else {
        initPwErr.textContent = "비밀번호 형식을 맞추어주세요.";
        initPwErr.classList.add("visible");
      }
    });
  }

  // 실시간 일치 여부 체크
  if (initPw1 && initPw2 && initPw2Ok && initPw2Err) {
    initPw2.addEventListener("input", function () {
      const v1 = initPw1.value.trim();
      const v2 = initPw2.value.trim();

      initPw2Ok.classList.remove("visible");
      initPw2Err.classList.remove("visible");

      if (!v2) return;

      // 첫 번째 비밀번호가 형식에 맞지 않으면 아무것도 표시하지 않음
      if (!pwPattern.test(v1)) {
        return;
      }

      // 형식이 맞을 때만 일치 여부 확인
      if (v1 === v2) {
        initPw2Ok.textContent = "비밀번호가 일치합니다.";
        initPw2Ok.classList.add("visible");
      } else {
        initPw2Err.textContent = "비밀번호가 일치하지 않습니다.";
        initPw2Err.classList.add("visible");
      }
    });
  }

  if (initForm) {
    initForm.addEventListener("submit", async function (e) {
      e.preventDefault();

      const n1 = initPw1 ? initPw1.value.trim() : "";
      const n2 = initPw2 ? initPw2.value.trim() : "";

      // 모든 에러 메시지 초기화
      [initPwOk, initPwErr, initPw2Ok, initPw2Err].forEach((el) => {
        if (el) el.classList.remove("visible");
      });

      if (initPwError) {
        initPwError.textContent = "";
        initPwError.classList.remove("visible");
      }

      if (!n1 || !n2) {
        if (initPwError) {
          initPwError.textContent = "모든 항목을 입력해 주세요.";
          initPwError.classList.add("visible");
        }
        return;
      }

      if (!pwPattern.test(n1)) {
        if (initPwErr) {
          initPwErr.textContent = "비밀번호 형식을 맞추어주세요.";
          initPwErr.classList.add("visible");
        }
        if (initPwError) {
          initPwError.textContent = "비밀번호 형식을 맞추어주세요.";
          initPwError.classList.add("visible");
        }
        return;
      }

      if (n1 !== n2) {
        if (initPw2Err) {
          initPw2Err.textContent = "비밀번호가 일치하지 않습니다.";
          initPw2Err.classList.add("visible");
        }
        return;
      }

      // 유효성 검사 통과 - 성공 메시지 표시
      if (initPwOk) {
        initPwOk.textContent = "올바른 형식입니다.";
        initPwOk.classList.add("visible");
      }
      if (initPw2Ok) {
        initPw2Ok.textContent = "비밀번호가 일치합니다.";
        initPw2Ok.classList.add("visible");
      }

      const formData = new FormData(initForm);

      try {
        const res = await fetch(initForm.action, {
          method: "POST",
          headers: {
            "X-CSRFToken": csrftoken,
            "X-Requested-With": "XMLHttpRequest",
          },
          body: formData,
        });

        const data = await res.json();

        if (!data.ok) {
          if (data.field === "new_password" && initPwErr) {
            initPwErr.textContent = data.message || "비밀번호를 확인해주세요.";
            initPwErr.classList.add("visible");
          } else if (data.field === "new_password2" && initPw2Err) {
            initPw2Err.textContent = data.message || "비밀번호를 확인해주세요.";
            initPw2Err.classList.add("visible");
          } else if (initPwError) {
            initPwError.textContent = data.message || "비밀번호를 확인해주세요.";
            initPwError.classList.add("visible");
          }
          return;
        }

        window.location.href = data.redirect_url || "/";
      } catch (err) {
        console.error("초기 비밀번호 변경 오류:", err);
        if (initPwError) {
          initPwError.textContent = "오류가 발생했습니다. 다시 시도해 주세요.";
          initPwError.classList.add("visible");
        }
      }
    });
  }
});
