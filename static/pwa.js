(function () {
  if (typeof window === "undefined") return;

  const STORAGE_KEY = "smart_home_pwa_hint_dismissed";
  let deferredPrompt = null;
  let installButton = null;

  function ensureInstallButton() {
    if (installButton) return installButton;

    installButton = document.createElement("button");
    installButton.type = "button";
    installButton.textContent = "Cai app";
    installButton.setAttribute("aria-label", "Cai dat app Smart Home");
    installButton.style.position = "fixed";
    installButton.style.right = "16px";
    installButton.style.bottom = "16px";
    installButton.style.zIndex = "99999";
    installButton.style.padding = "12px 16px";
    installButton.style.borderRadius = "999px";
    installButton.style.border = "1px solid rgba(0, 212, 255, 0.45)";
    installButton.style.background = "linear-gradient(135deg, #00d4ff, #00ffb8)";
    installButton.style.color = "#04111f";
    installButton.style.fontWeight = "700";
    installButton.style.boxShadow = "0 0 18px rgba(0, 212, 255, 0.45)";
    installButton.style.cursor = "pointer";
    installButton.style.display = "none";

    installButton.addEventListener("click", async function () {
      if (!deferredPrompt) return;
      deferredPrompt.prompt();
      try {
        await deferredPrompt.userChoice;
      } catch (err) {
        console.warn("Install prompt error:", err);
      }
      deferredPrompt = null;
      installButton.style.display = "none";
    });

    document.body.appendChild(installButton);
    return installButton;
  }

  function showInstallButton() {
    const btn = ensureInstallButton();
    btn.style.display = "inline-flex";
    btn.style.alignItems = "center";
    btn.style.justifyContent = "center";
    btn.style.gap = "8px";
  }

  function isIosSafari() {
    const ua = window.navigator.userAgent || "";
    const isiOS = /iPad|iPhone|iPod/.test(ua);
    const isSafari = /Safari/.test(ua) && !/CriOS|FxiOS|EdgiOS/.test(ua);
    return isiOS && isSafari;
  }

  function isStandalone() {
    return window.matchMedia("(display-mode: standalone)").matches || window.navigator.standalone === true;
  }

  function maybeShowIosHint() {
    if (!isIosSafari() || isStandalone()) return;
    try {
      if (window.localStorage.getItem(STORAGE_KEY) === "1") return;
    } catch (err) {}

    const hint = document.createElement("div");
    hint.style.position = "fixed";
    hint.style.left = "16px";
    hint.style.right = "16px";
    hint.style.bottom = "16px";
    hint.style.zIndex = "99998";
    hint.style.padding = "14px 16px";
    hint.style.borderRadius = "16px";
    hint.style.background = "rgba(6, 14, 27, 0.94)";
    hint.style.color = "#d6f7ff";
    hint.style.border = "1px solid rgba(0, 212, 255, 0.25)";
    hint.style.boxShadow = "0 8px 30px rgba(0,0,0,0.3)";
    hint.innerHTML =
      "<strong style='display:block; margin-bottom:6px;'>Cai tren iPhone</strong>" +
      "<span>Trong Safari, bam Share va chon Add to Home Screen de dung nhu app.</span>";

    const close = document.createElement("button");
    close.type = "button";
    close.textContent = "Dong";
    close.style.marginTop = "10px";
    close.style.padding = "8px 12px";
    close.style.borderRadius = "999px";
    close.style.border = "1px solid rgba(0, 212, 255, 0.25)";
    close.style.background = "transparent";
    close.style.color = "#d6f7ff";
    close.style.cursor = "pointer";
    close.addEventListener("click", function () {
      try {
        window.localStorage.setItem(STORAGE_KEY, "1");
      } catch (err) {}
      hint.remove();
    });

    hint.appendChild(close);
    document.body.appendChild(hint);
  }

  window.addEventListener("beforeinstallprompt", function (event) {
    event.preventDefault();
    deferredPrompt = event;
    showInstallButton();
  });

  window.addEventListener("appinstalled", function () {
    deferredPrompt = null;
    if (installButton) installButton.style.display = "none";
  });

  window.addEventListener("load", function () {
    if ("serviceWorker" in navigator) {
      navigator.serviceWorker
        .register("/sw.js")
        .catch(function (err) {
          console.warn("Service worker register failed:", err);
        });
    }
    maybeShowIosHint();
  });
})();
