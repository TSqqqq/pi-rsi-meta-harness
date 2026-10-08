// Loads the real dashboard page in jsdom against a live test server and clicks every button.
// A button passes only if it (1) shows the busy spinner, (2) ends with a success or error effect,
// and (3) produces the expected server-side change or a visible error toast.
const { JSDOM } = require("jsdom"); // resolved via NODE_PATH set by run.sh
const { execFileSync } = require("child_process");
const port = process.argv[2], dbPath = process.argv[3];
const base = `http://127.0.0.1:${port}`;
const sql = (q) => execFileSync("sqlite3", [dbPath, q]).toString().trim();
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const results = [];

(async () => {
  // jsdom has no fetch; inject Node's, resolving the page's relative URLs against the server.
  const dom = await JSDOM.fromURL(base + "/", { runScripts: "dangerously", resources: "usable", pretendToBeVisual: true,
    beforeParse(win) { win.fetch = (u, o) => fetch(new URL(u, base), o); } });
  const w = dom.window, d = w.document, $ = (id) => d.getElementById(id);
  w.confirm = () => true;
  w.addEventListener("error", (e) => console.log("PAGE_ERROR", e.message));
  await sleep(2500);
  console.log("state pill:", $("stateText").textContent, "| runs listed:", d.querySelectorAll("[data-run]").length);

  async function click(name, el, check) {
    if (!el) { results.push({ name, pass: false, check: "button not rendered" }); return; }
    let sawBusy = false;
    const obs = new w.MutationObserver(() => { if (el.classList.contains("busy")) sawBusy = true; });
    obs.observe(el, { attributes: true, attributeFilter: ["class"] });
    el.click();
    let effect = "";
    for (let i = 0; i < 60 && !effect; i++) {
      await sleep(250);
      if (el.classList.contains("flash-ok")) effect = "ok";
      else if (el.classList.contains("flash-err")) effect = "err";
    }
    obs.disconnect();
    const toast = $("toast");
    const res = { name, busy: sawBusy, effect, toast: toast.className.includes("show") ? toast.textContent : "" };
    res.check = check ? await check(res) : true;
    res.pass = res.busy && !!res.effect && !!res.toast && res.check === true;
    results.push(res);
    await sleep(300);
  }
  const state = () => sql("select value from meta where key='state'").replace(/"/g, "");

  await click("暂停", $("pauseBtn"), (r) => r.effect === "ok" && state() === "PAUSED" || `state=${state()}`);
  await click("继续", $("resumeBtn"), (r) => r.effect === "ok" && state() === "RUNNING" || `state=${state()}`);

  // Validation errors must be visible, not silent.
  $("ideaInput").value = "";
  await click("加入草稿(空)", $("addIdeaBtn"), (r) => r.effect === "err" && r.toast.includes("请先输入") || r.toast);
  $("ideaInput").value = "try later layers";
  await click("加入草稿", $("addIdeaBtn"), () => sql("select count(*) from idea_queue where status='draft'") === "1" || "no draft row");
  await sleep(1800); // let the idea list re-render
  const sendOne = d.querySelector("[data-idea-send]");
  await click("草稿·发给下一轮", sendOne, () => sql("select status from idea_queue order by id limit 1") === "sent" || "not sent");
  $("ideaInput").value = "second idea";
  await click("加入草稿#2", $("addIdeaBtn"));
  await click("全部草稿发给下一轮", $("sendAllIdeasBtn"), () => sql("select count(*) from idea_queue where status='draft'") === "0" || "drafts left");
  await sleep(1800);
  const del = d.querySelector("[data-idea-delete]");
  await click("草稿·删除", del, () => sql("select count(*) from idea_queue") === "1" || "not deleted");

  $("notes").value = "my note";
  await click("笔记→想法队列", $("noteToIdeaBtn"), () => sql("select count(*) from idea_queue where text='my note'") === "1" || "missing");

  // Steering: pick the active agent, wait through two refreshes, the choice must survive.
  $("agentSelect").value = "worker-exp_000002";
  await sleep(3500);
  const kept = $("agentSelect").value;
  $("steerInput").value = "only edit model/attention.py";
  await click("实时纠偏", $("steerBtn"), () => (kept === "worker-exp_000002" && sql("select count(*) from human_guidance where kind='steer'") === "1") || `kept=${kept}`);

  // Node actions need a selection; the page auto-selects the best node only when one is validated, so select explicitly.
  await w.selectNode("exp_000001");
  await sleep(800);
  await click("置顶", $("pinBtn"), () => sql("select pinned from experiments where id='exp_000001'") === "1" || "not pinned");
  $("branchInsight").value = "reopen with a bug fix";
  await click("从这里继续", $("branchBtn"), () => sql("select count(*) from human_guidance where kind='branch'") === "1" || "no guidance");
  await click("完整详情", $("fullDetailBtn"), () => $("modal").classList.contains("open") || "modal closed");
  $("closeModal").click();

  const dep = d.querySelector("[data-dep^='approve']");
  await click("依赖·批准", dep, () => sql("select status from dependency_requests") === "approved" || "not approved");

  await click("停止", $("stopBtn"), (r) => r.effect === "ok" && state() === "STOPPED_BY_USER" || `state=${state()}`);

  // Tabs switch views (no server call).
  d.querySelector(".tab[data-view='trend']").click();
  results.push({ name: "标签·指标趋势", pass: $("trendView").style.display === "block" });

  for (const r of results) console.log(`${r.pass ? "PASS" : "FAIL"}  ${r.name}${r.effect ? "  [" + r.effect + "]" : ""}${r.toast ? "  toast=" + r.toast : ""}${r.check !== true && r.check !== undefined ? "  check=" + r.check : ""}`);
  console.log(results.every((r) => r.pass) ? "ALL_PASS" : "SOME_FAIL");
  w.close();
  process.exit(0);
})();
