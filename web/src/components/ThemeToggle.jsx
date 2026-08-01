import React from "react";

export default function ThemeToggle({ theme }) {
  return (
    <button className="icon-btn" onClick={theme.toggle} title="Toggle light / dark" aria-label="Toggle theme">
      {theme.isDark() ? "☀︎ light" : "☾ dark"}
    </button>
  );
}
