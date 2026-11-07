"use strict";

const statusLine = document.getElementById("status");
const chapters = document.getElementById("chapters");
const controls = ["prev", "next", "smaller", "larger"];

function showError(error) {
  console.error(error);
  statusLine.textContent = "Не удалось открыть этот фрагмент. Книгу можно скачать по ссылке выше.";
}

async function start() {
  const book = ePub("../../resources/plan-b.epub", { openAs: "epub" });
  const reader = book.renderTo("reader", {
    width: "100%",
    height: "100%",
    spread: "none",
    allowScriptedContent: false
  });
  const navigation = await book.loaded.navigation;
  function addEntries(entries, depth = 0) {
    for (const item of entries) {
      chapters.add(new Option("\u00a0\u00a0".repeat(depth) + item.label.trim(), item.href));
      addEntries(item.subitems || [], depth + 1);
    }
  }
  addEntries(navigation.toc);
  await reader.display();
  statusLine.textContent = "";
  for (const id of controls) document.getElementById(id).disabled = false;
  chapters.disabled = chapters.options.length < 2;

  function turn(action) {
    statusLine.textContent = "";
    Promise.resolve().then(action).catch(showError);
  }
  document.getElementById("prev").onclick = () => turn(() => reader.prev());
  document.getElementById("next").onclick = () => turn(() => reader.next());
  chapters.onchange = () => {
    if (chapters.value) turn(() => reader.display(chapters.value));
  };
  let size = 100;
  function resize(delta) {
    size = Math.max(75, Math.min(200, size + delta));
    reader.themes.fontSize(size + "%");
  }
  document.getElementById("smaller").onclick = () => resize(-10);
  document.getElementById("larger").onclick = () => resize(10);
}

start().catch(showError);
