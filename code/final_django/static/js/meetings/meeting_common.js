// static/js/meetings/meeting_common.js
document.addEventListener("DOMContentLoaded", function () {
  const copyBtn = document.getElementById("btn-copy-full-transcript");
  if (!copyBtn) return;

  copyBtn.addEventListener("click", function () {
    const scrollEl = document.getElementById("assign-transcript-scroll");
    if (!scrollEl) {
      alert("복사할 전문 영역을 찾을 수 없습니다.");
      return;
    }

    const text = (scrollEl.innerText || scrollEl.textContent || "").trim();
    if (!text) {
      alert("복사할 내용이 존재하지 않습니다.");
      return;
    }

    // 최신 브라우저: Clipboard API
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard
        .writeText(text)
        .then(() => alert("전문 전체가 클립보드에 복사되었습니다."))
        .catch(() => fallbackCopy(text));
    } else {
      fallbackCopy(text);
    }
  });

  // 구형 브라우저용 복사 방식
  function fallbackCopy(text) {
    const textarea = document.createElement("textarea");
    textarea.value = text;
    textarea.style.position = "fixed";
    textarea.style.top = "-9999px";
    textarea.style.left = "-9999px";
    document.body.appendChild(textarea);

    textarea.focus();
    textarea.select();

    try {
      const successful = document.execCommand("copy");
      if (successful) {
        alert("전문 전체가 클립보드에 복사되었습니다.");
      } else {
        alert("복사 실패. 다른 브라우저에서 시도해주세요.");
      }
    } catch (err) {
      alert("복사 중 오류가 발생했습니다.");
    }

    document.body.removeChild(textarea);
  }
});

document.addEventListener("DOMContentLoaded", function () {
  const saveBtn = document.getElementById("btn-transcript-save");
  if (!saveBtn) return;

  const summaryUrl = saveBtn.dataset.summaryUrl;
  if (!summaryUrl) {
    // 요약 URL이 없는 경우에는 아무 동작도 하지 않고 종료
    return;
  }

  saveBtn.addEventListener("click", function (event) {
    event.preventDefault();
    window.location.href = summaryUrl;
  });
});
