document.addEventListener("DOMContentLoaded", () => {
  // 여러 위치의 링크/버튼을 한 번에 처리
  const loginEls = Array.from(
    document.querySelectorAll("#btn-google-login, #link-google-login")
  );
  const logoutEls = Array.from(
    document.querySelectorAll("#btn-google-logout, #link-google-logout")
  );
  const hintEl = document.querySelector("[data-google-authed]");

  if (!loginEls.length || !logoutEls.length) return;

  const setState = (authed) => {
    loginEls.forEach((el) => {
      el.style.display = authed ? "none" : "inline-flex";
    });
    logoutEls.forEach((el) => {
      el.style.display = authed ? "inline-flex" : "none";
    });
  };

  // 서버에서 내려준 힌트(세션에 토큰 있으면 true)
  const initialAuthed =
    (hintEl?.dataset.googleAuthed || "").toLowerCase() === "true";
  setState(initialAuthed);

  function getCSRFToken() {
    const name = "csrftoken";
    const match = document.cookie
      .split(";")
      .map((c) => c.trim())
      .find((c) => c.startsWith(`${name}=`));
    return match ? decodeURIComponent(match.split("=").slice(1).join("=")) : "";
  }

  async function checkStatus() {
    try {
      const res = await fetch("/api/google-auth-status/");
      if (!res.ok) throw new Error("status_failed");
      const data = await res.json();
      setState(!!data.authenticated);
    } catch (err) {
      console.warn("google auth status check failed:", err);
      // API 실패 시 서버 힌트값을 유지
    }
  }

  loginEls.forEach((btn) =>
    btn.addEventListener("click", () => {
      window.location.href = "/google/login/";
    })
  );

  logoutEls.forEach((btn) =>
    btn.addEventListener("click", async () => {
      logoutEls.forEach((b) => (b.disabled = true));
      try {
        const res = await fetch("/api/google-auth/revoke/", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "X-CSRFToken": getCSRFToken(),
          },
          body: "{}",
        });
        if (!res.ok) {
          throw new Error("구글 인증 해제에 실패했습니다.");
        }
        setState(false);
        alert("구글 인증이 해제되었습니다.");
        // 새로고침하여 캘린더/Tasks 연동 상태를 초기화
        window.location.reload();
      } catch (err) {
        console.error("google revoke error:", err);
        alert(err.message || "구글 인증 해제 중 오류가 발생했습니다.");
      } finally {
        logoutEls.forEach((b) => (b.disabled = false));
      }
    })
  );

  checkStatus();
});

