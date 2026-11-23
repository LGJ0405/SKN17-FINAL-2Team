(function (window) {
  "use strict";

  function startJobPolling({
    storageKey,
    prepareUrl,
    successUrl,
    failUrl,
    defaultErrorMessage,
    networkErrorMessage,
    pollIntervalMs = 3000,
    maxRetryErrors = 3,
  }) {
    let consecutiveErrors = 0;

    async function pollStatus() {
      const savedJobId = sessionStorage.getItem(storageKey);
      const url = savedJobId
        ? `${prepareUrl}?job_id=${encodeURIComponent(savedJobId)}`
        : prepareUrl;

      try {
        const res = await fetch(url);
        let data = {};
        try {
          data = await res.json();
        } catch (_parseErr) {
          data = {};
        }

        if (res.status >= 502 && consecutiveErrors < maxRetryErrors) {
          consecutiveErrors += 1;
          setTimeout(pollStatus, pollIntervalMs);
          return;
        }

        consecutiveErrors = 0;

        if (data.status === "done") {
          sessionStorage.removeItem(storageKey);
          window.location.href = successUrl;
          return;
        }

        if (data.status === "processing") {
          if (data.job_id) {
            sessionStorage.setItem(storageKey, data.job_id);
          }
          setTimeout(pollStatus, pollIntervalMs);
          return;
        }

        sessionStorage.removeItem(storageKey);
        alert(data.message || defaultErrorMessage);
        window.location.href = failUrl;
      } catch (e) {
        console.error(e);
        if (consecutiveErrors < maxRetryErrors) {
          consecutiveErrors += 1;
          setTimeout(pollStatus, pollIntervalMs);
          return;
        }
        sessionStorage.removeItem(storageKey);
        alert(networkErrorMessage);
        window.location.href = failUrl;
      }
    }

    pollStatus();
  }

  window.startJobPolling = startJobPolling;
})(window);
