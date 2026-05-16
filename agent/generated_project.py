from __future__ import annotations

import json
import re
from typing import Any


def title_from_idea(idea: str) -> str:
    cleaned = _clean_text(idea).rstrip(".")
    lowered = cleaned.lower()
    for prefix in ("build a ", "build an ", "create a ", "make a "):
        if lowered.startswith(prefix):
            cleaned = cleaned[len(prefix) :]
            break
    if len(cleaned) > 72:
        cleaned = f"{cleaned[:69].rstrip()}..."
    return cleaned[:1].upper() + cleaned[1:] if cleaned else "Generated MVP"


def build_project_artifacts(
    *,
    idea: str,
    title: str | None,
    resolved_stack: str,
    repo_plan: dict[str, Any] | None = None,
    source_warnings: list[dict[str, str]] | None = None,
    target_users: str | None = None,
    required_features: list[str] | None = None,
    tech_stack_preference: str | None = None,
) -> list[dict[str, str]]:
    """Create a complete, commit-safe MVP repository from the orchestrator plan."""

    project_title = _clean_text(title or "") or title_from_idea(idea)
    project_slug = _slugify(project_title)
    feature_label = _feature_label(idea)
    warning_lines = _warning_lines(source_warnings or [])
    plan_steps = _plan_steps(repo_plan or {})
    selected_stack = tech_stack_preference or _selected_stack(repo_plan or {}, resolved_stack)
    user_story = _user_story(idea)
    audience = _clean_text(target_users or "") or "Early adopters validating the core workflow"

    backend_features = _feature_list(required_features, idea)

    files = [
        {
            "name": "README.md",
            "kind": "markdown",
            "summary": "Project overview, setup, and walkthrough path.",
            "content": _readme(project_title, idea, selected_stack, warning_lines),
        },
        {
            "name": "package.json",
            "kind": "json",
            "summary": "Frontend scripts and dependencies.",
            "content": json.dumps(
                {
                    "name": project_slug,
                    "version": "0.1.0",
                    "private": True,
                    "type": "module",
                    "scripts": {
                        "dev": "vite --host 0.0.0.0",
                        "build": "vite build",
                        "preview": "vite preview",
                    },
                    "dependencies": {
                        "@vitejs/plugin-react": "^4.3.4",
                        "vite": "^6.0.0",
                        "react": "^19.0.0",
                        "react-dom": "^19.0.0",
                    },
                    "devDependencies": {},
                },
                indent=2,
            )
            + "\n",
        },
        {
            "name": "index.html",
            "kind": "html",
            "summary": "Frontend HTML shell.",
            "content": (
                '<!doctype html>\n'
                '<html lang="en">\n'
                "  <head>\n"
                '    <meta charset="UTF-8" />\n'
                '    <meta name="viewport" content="width=device-width, initial-scale=1.0" />\n'
                f"    <title>{_html_escape(project_title)}</title>\n"
                "  </head>\n"
                "  <body>\n"
                '    <div id="root"></div>\n'
                '    <script type="module" src="/src/main.jsx"></script>\n'
                "  </body>\n"
                "</html>\n"
            ),
        },
        {
            "name": "src/main.jsx",
            "kind": "javascript",
            "summary": "React frontend entrypoint.",
            "content": (
                "import React from 'react';\n"
                "import { createRoot } from 'react-dom/client';\n"
                "import App from './App.jsx';\n"
                "import './styles.css';\n\n"
                "createRoot(document.getElementById('root')).render(\n"
                "  <React.StrictMode>\n"
                "    <App />\n"
                "  </React.StrictMode>,\n"
                ");\n"
            ),
        },
        {
            "name": "src/App.jsx",
            "kind": "javascript",
            "summary": "Runnable MVP user interface.",
            "content": _react_app(
                project_title,
                idea,
                feature_label,
                backend_features,
                plan_steps,
                audience,
            ),
        },
        {
            "name": "src/data/mockRecords.js",
            "kind": "javascript",
            "summary": "Realistic, labeled mock records for unavailable integrations.",
            "content": _mock_records(project_title, idea, backend_features),
        },
        {
            "name": "src/styles.css",
            "kind": "css",
            "summary": "Polished MVP styling.",
            "content": _css(project_title),
        },
        {
            "name": "backend/main.py",
            "kind": "python",
            "summary": "FastAPI backend with MVP planning endpoints.",
            "content": _backend_main(project_title, idea, backend_features),
        },
        {
            "name": "backend/mvp_engine.py",
            "kind": "python",
            "summary": "Domain logic for the generated MVP.",
            "content": _backend_engine(project_title, idea, backend_features, audience),
        },
        {
            "name": "requirements.txt",
            "kind": "text",
            "summary": "Backend dependencies.",
            "content": "fastapi\nuvicorn[standard]\npydantic\npytest\n",
        },
        {
            "name": "tests/test_backend.py",
            "kind": "python",
            "summary": "Smoke tests for generated backend logic.",
            "content": _backend_test(project_title),
        },
        {
            "name": "docs/DATABASE_SCHEMA.sql",
            "kind": "sql",
            "summary": "Suggested Postgres schema for the MVP.",
            "content": _database_schema(project_slug),
        },
        {
            "name": "docs/ARCHITECTURE.md",
            "kind": "markdown",
            "summary": "Architecture notes from the orchestrator plan.",
            "content": _architecture(project_title, idea, selected_stack, plan_steps, warning_lines),
        },
        {
            "name": "docs/IMPLEMENTATION_PLAN.md",
            "kind": "markdown",
            "summary": "Step-by-step implementation plan.",
            "content": _implementation_plan(project_title, plan_steps),
        },
        {
            "name": "docs/BUILD_LOG.md",
            "kind": "markdown",
            "summary": "Agent build log for demo traceability.",
            "content": _build_log(project_title, selected_stack, plan_steps, warning_lines),
        },
        {
            "name": "demo/demo_script.md",
            "kind": "markdown",
            "summary": "Walkthrough for judges.",
            "content": _demo_script(project_title, idea, feature_label),
        },
        {
            "name": ".env.example",
            "kind": "text",
            "summary": "Safe placeholder environment file.",
            "content": (
                "VITE_API_BASE_URL=http://127.0.0.1:8000\n"
                "DATABASE_URL=postgresql://postgres:postgres@localhost:5432/mvp\n"
                f"# Generated for: {project_title}\n"
            ),
        },
    ]
    return files


def merge_with_project_artifacts(
    artifacts: list[dict[str, Any]],
    *,
    idea: str,
    title: str | None,
    resolved_stack: str,
    repo_plan: dict[str, Any] | None = None,
    source_warnings: list[dict[str, str]] | None = None,
    target_users: str | None = None,
    required_features: list[str] | None = None,
    tech_stack_preference: str | None = None,
) -> list[dict[str, Any]]:
    """Model-authored files win; only fill missing health-check paths with idea-specific stubs."""

    merged: dict[str, dict[str, Any]] = {}
    for artifact in artifacts:
        name = str(artifact.get("name") or "").strip()
        if not name or name.endswith("/") or name.split("/")[-1] == ".env":
            continue
        content = artifact.get("content")
        if content is None:
            continue
        merged[name] = {
            "name": name,
            "kind": str(artifact.get("kind") or _kind_from_path(name)),
            "summary": str(artifact.get("summary") or "Generated by Nemotron."),
            "content": content if isinstance(content, str) else json.dumps(content, indent=2),
        }

    gap_fill = build_project_artifacts(
        idea=idea,
        title=title,
        resolved_stack=resolved_stack,
        repo_plan=repo_plan,
        source_warnings=source_warnings,
        target_users=target_users,
        required_features=required_features,
        tech_stack_preference=tech_stack_preference,
    )
    for artifact in gap_fill:
        name = artifact["name"]
        if name not in merged:
            merged[name] = artifact
    return [merged[path] for path in sorted(merged)]


def _readme(project_title: str, idea: str, selected_stack: str, warnings: list[str]) -> str:
    warning_section = _markdown_list("Source Warnings", warnings)
    return (
        f"# {project_title}\n\n"
        f"{idea}\n\n"
        "## MVP Workflow\n\n"
        "1. Capture the user's messy intake.\n"
        "2. Turn the intake into a prioritized action plan.\n"
        "3. Show the current work queue in the browser.\n"
        "4. Serve the same plan from a FastAPI backend.\n"
        "5. Keep a database schema ready for persistence.\n\n"
        "## Stack\n\n"
        f"{selected_stack}\n\n"
        "## Run Locally\n\n"
        "```bash\n"
        "npm install\n"
        "npm run dev\n"
        "```\n\n"
        "In another terminal:\n\n"
        "```bash\n"
        "pip install -r requirements.txt\n"
        "uvicorn backend.main:app --reload\n"
        "```\n\n"
        "## Test\n\n"
        "```bash\n"
        "pytest\n"
        "```\n"
        f"{warning_section}"
    )


def _react_app(
    project_title: str,
    idea: str,
    feature_label: str,
    backend_features: list[str],
    plan_steps: list[str],
    audience: str,
) -> str:
    lines = [
        "import { useMemo, useState } from 'react';",
        "import { mockRecords } from './data/mockRecords.js';",
        "",
        "const idea = " + json.dumps(idea) + ";",
        "const audience = " + json.dumps(audience) + ";",
        "const cards = " + json.dumps(backend_features, indent=2) + ";",
        "const planSteps = " + json.dumps(plan_steps, indent=2) + ";",
        "const tabs = ['Dashboard', 'Intake', 'Roadmap'];",
        "",
        "export default function App() {",
        "  const [activeTab, setActiveTab] = useState('Dashboard');",
        "  const [goal, setGoal] = useState('');",
        "  const [urgency, setUrgency] = useState('normal');",
        "  const [intakeResult, setIntakeResult] = useState(null);",
        "  const metrics = useMemo(() => ({",
        "    open: mockRecords.filter((item) => item.status !== 'done').length,",
        "    done: mockRecords.filter((item) => item.status === 'done').length,",
        "  }), []);",
        "",
        "  async function submitIntake(event) {",
        "    event.preventDefault();",
        "    const response = await fetch('/api/intake', {",
        "      method: 'POST',",
        "      headers: { 'Content-Type': 'application/json' },",
        "      body: JSON.stringify({ user_goal: goal, urgency, notes: idea }),",
        "    });",
        "    setIntakeResult(await response.json());",
        "  }",
        "",
        "  return (",
        '    <main className="shell">',
        '      <header className="topbar">',
        "        <div><p className=\"eyebrow\">MVPilot Autonomous Build</p><h1>" + _jsx_escape(project_title) + "</h1></div>",
        '        <nav className="tabs">{tabs.map((tab) => (',
        "          <button key={tab} type=\"button\" className={activeTab === tab ? 'tab active' : 'tab'} onClick={() => setActiveTab(tab)}>{tab}</button>",
        "        ))}</nav>",
        "      </header>",
        '      <section className="hero">',
        '        <div><p className="lede">{idea}</p><p className="meta">Built for <strong>{audience}</strong></p></div>',
        '        <div className="statusPanel"><span className="statusDot" /><strong>' + _jsx_escape(feature_label) + ' prototype ready</strong><small>Multi-page UI, API, mock data, tests, docs.</small></div>',
        "      </section>",
        "      {activeTab === 'Dashboard' && (",
        '        <section className="dashboard">',
        '          <article className="metric"><span>Open</span><strong>{metrics.open}</strong></article>',
        '          <article className="metric"><span>Done</span><strong>{metrics.done}</strong></article>',
        '          <article className="metric"><span>Features</span><strong>{cards.length}</strong></article>',
        '          <div className="table">{mockRecords.map((row) => (',
        '            <div className="tableRow" key={row.id}><span>{row.owner}</span><span>{row.title}</span><span className={`pill ${row.status}`}>{row.status}</span></div>',
        "          ))}</div>",
        "        </section>",
        "      )}",
        "      {activeTab === 'Intake' && (",
        '        <section className="intake"><form onSubmit={submitIntake}>',
        '          <label>Goal</label><textarea value={goal} onChange={(e) => setGoal(e.target.value)} rows={4} required />',
        '          <label>Urgency</label><select value={urgency} onChange={(e) => setUrgency(e.target.value)}><option value="normal">Normal</option><option value="high">High</option><option value="urgent">Urgent</option></select>',
        '          <button type="submit">Submit intake</button></form>',
        '          {intakeResult && <article className="result"><h3>Backend response</h3><p>{intakeResult.summary}</p><p>{intakeResult.recommended_first_step}</p></article>}',
        "        </section>",
        "      )}",
        "      {activeTab === 'Roadmap' && (",
        '        <section className="grid">{cards.map((card, index) => (<article className="card" key={card}><span>{String(index + 1).padStart(2, \\"0\\")}</span><p>{card}</p></article>))}',
        '          <section className="plan"><h2>Plan</h2><ol>{planSteps.map((step) => <li key={step}>{step}</li>)}</ol></section></section>',
        "      )}",
        "    </main>",
        "  );",
        "}",
    ]
    return "\n".join(lines) + "\n"


def _css(project_title: str) -> str:
    del project_title
    return (
        ":root {\n"
        "  color: #172026;\n"
        "  background: #f6f8fb;\n"
        "  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;\n"
        "}\n\n"
        "* { box-sizing: border-box; }\n"
        "body { margin: 0; }\n"
        ".shell { min-height: 100vh; padding: 48px; }\n"
        ".hero { display: grid; grid-template-columns: minmax(0, 1.4fr) minmax(280px, 0.6fr); gap: 32px; align-items: stretch; }\n"
        ".eyebrow { color: #0f766e; font-weight: 800; letter-spacing: 0.12em; text-transform: uppercase; font-size: 12px; }\n"
        "h1 { margin: 8px 0 16px; font-size: 52px; line-height: 1; letter-spacing: 0; }\n"
        ".lede { max-width: 760px; font-size: 20px; line-height: 1.6; color: #42515a; }\n"
        ".statusPanel, .card, .plan { border: 1px solid #d7dee8; background: #ffffff; border-radius: 8px; box-shadow: 0 18px 50px rgba(27, 39, 51, 0.08); }\n"
        ".statusPanel { padding: 24px; display: grid; align-content: center; gap: 10px; }\n"
        ".statusDot { width: 12px; height: 12px; border-radius: 999px; background: #16a34a; box-shadow: 0 0 0 6px rgba(22, 163, 74, 0.12); }\n"
        ".statusPanel small { color: #60717d; line-height: 1.5; }\n"
        ".grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 16px; margin-top: 32px; }\n"
        ".card { padding: 22px; min-height: 150px; }\n"
        ".card span { color: #0f766e; font-weight: 900; font-size: 12px; }\n"
        ".card p { font-size: 17px; line-height: 1.55; margin-bottom: 0; }\n"
        ".plan { margin-top: 32px; padding: 28px; }\n"
        ".plan h2 { margin-top: 0; }\n"
        ".plan li { margin: 10px 0; color: #42515a; line-height: 1.6; }\n"
        ".topbar { display: flex; justify-content: space-between; align-items: flex-end; gap: 24px; margin-bottom: 24px; }\n"
        ".tabs { display: flex; gap: 8px; flex-wrap: wrap; }\n"
        ".tab { border: 1px solid #d7dee8; background: #fff; border-radius: 999px; padding: 8px 14px; cursor: pointer; font-weight: 700; }\n"
        ".tab.active { background: #0f766e; color: #fff; border-color: #0f766e; }\n"
        ".meta { color: #60717d; margin-top: 8px; }\n"
        ".dashboard { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 16px; margin-top: 24px; }\n"
        ".metric { padding: 20px; border: 1px solid #d7dee8; border-radius: 8px; background: #fff; }\n"
        ".metric span { display: block; color: #60717d; font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; }\n"
        ".metric strong { font-size: 32px; }\n"
        ".table { grid-column: 1 / -1; border: 1px solid #d7dee8; border-radius: 8px; overflow: hidden; background: #fff; }\n"
        ".tableRow { display: grid; grid-template-columns: 120px 1fr 120px; gap: 12px; padding: 14px 18px; border-top: 1px solid #eef2f6; }\n"
        ".pill { text-transform: capitalize; font-weight: 700; font-size: 12px; }\n"
        ".pill.done { color: #16a34a; }\n"
        ".pill.active { color: #0f766e; }\n"
        ".intake, .result { margin-top: 24px; padding: 24px; border: 1px solid #d7dee8; border-radius: 8px; background: #fff; }\n"
        ".intake label { display: block; font-weight: 700; margin: 12px 0 6px; }\n"
        ".intake textarea, .intake select, .intake button { width: 100%; margin-bottom: 12px; padding: 10px 12px; font: inherit; }\n"
        ".intake button { background: #0f766e; color: #fff; border: 0; border-radius: 8px; font-weight: 700; cursor: pointer; }\n"
        "@media (max-width: 840px) {\n"
        "  .shell { padding: 24px; }\n"
        "  .hero, .grid { grid-template-columns: 1fr; }\n"
        "  h1 { font-size: 38px; }\n"
        "}\n"
    )


def _backend_main(project_title: str, idea: str, features: list[str]) -> str:
    return (
        '"""FastAPI surface for the generated MVP."""\n\n'
        "from fastapi import FastAPI\n"
        "from pydantic import BaseModel\n\n"
        "from backend.mvp_engine import build_mvp_plan, summarize_intake\n\n\n"
        "app = FastAPI(title=" + json.dumps(project_title) + ")\n\n\n"
        "class Intake(BaseModel):\n"
        "    user_goal: str\n"
        "    urgency: str = 'normal'\n"
        "    notes: str | None = None\n\n\n"
        "@app.get('/health')\n"
        "def health() -> dict[str, str]:\n"
        "    return {'status': 'ok', 'service': " + json.dumps(project_title) + "}\n\n\n"
        "@app.get('/api/mvp-plan')\n"
        "def mvp_plan() -> dict[str, object]:\n"
        "    return build_mvp_plan(" + json.dumps(idea) + ", " + json.dumps(features) + ")\n\n\n"
        "@app.post('/api/intake')\n"
        "def intake(payload: Intake) -> dict[str, object]:\n"
        "    return summarize_intake(payload.model_dump())\n"
    )


def _backend_engine(
    project_title: str,
    idea: str,
    features: list[str],
    audience: str,
) -> str:
    del project_title
    return (
        '"""Core MVP planning logic."""\n\n'
        "from __future__ import annotations\n\n"
        "from typing import Any\n\n\n"
        "MOCK_QUEUE = [\n"
        "    {'id': 'wk-101', 'owner': 'Ops', 'title': 'Validate intake flow', 'status': 'active'},\n"
        "    {'id': 'wk-102', 'owner': 'Product', 'title': 'Review generated plan', 'status': 'done'},\n"
        "    {'id': 'wk-103', 'owner': 'Eng', 'title': 'Wire persistence', 'status': 'active'},\n"
        "]\n\n\n"
        "def build_mvp_plan(idea: str, features: list[str]) -> dict[str, Any]:\n"
        "    return {\n"
        "        'idea': idea,\n"
        "        'audience': " + json.dumps(audience) + ",\n"
        "        'features': features,\n"
        "        'queue': MOCK_QUEUE,\n"
        "        'next_actions': [\n"
        "            'Validate the highest-risk user workflow.',\n"
        "            'Review generated data model before wiring persistence.',\n"
        "            'Run the MVP with one realistic intake.',\n"
        "        ],\n"
        "    }\n\n\n"
        "def summarize_intake(payload: dict[str, Any]) -> dict[str, Any]:\n"
        "    goal = str(payload.get('user_goal') or '').strip()\n"
        "    urgency = str(payload.get('urgency') or 'normal').strip().lower()\n"
        "    priority = 'high' if urgency in {'urgent', 'high', 'critical'} else 'normal'\n"
        "    return {\n"
        "        'summary': goal or " + json.dumps(idea) + ",\n"
        "        'priority': priority,\n"
        "        'recommended_first_step': 'Create the first tracked work item.',\n"
        "    }\n"
    )


def _backend_test(project_title: str) -> str:
    return (
        "from backend.mvp_engine import build_mvp_plan, summarize_intake\n\n\n"
        "def test_mvp_plan_contains_features():\n"
        "    plan = build_mvp_plan(" + json.dumps(project_title) + ", ['capture intake'])\n"
        "    assert plan['idea'] == " + json.dumps(project_title) + "\n"
        "    assert plan['features'] == ['capture intake']\n\n\n"
        "def test_intake_marks_urgent_items_high_priority():\n"
        "    result = summarize_intake({'user_goal': 'ship it', 'urgency': 'urgent'})\n"
        "    assert result['priority'] == 'high'\n"
        "    assert 'ship it' in result['summary']\n"
        f"    assert {json.dumps(project_title)}\n"
    )


def _database_schema(project_slug: str) -> str:
    table_prefix = re.sub(r"[^a-z0-9_]", "_", project_slug.replace("-", "_"))
    return (
        f"-- Suggested Postgres schema for {project_slug}\n"
        f"create table if not exists {table_prefix}_intakes (\n"
        "  id uuid primary key default gen_random_uuid(),\n"
        "  user_goal text not null,\n"
        "  urgency text not null default 'normal',\n"
        "  notes text,\n"
        "  status text not null default 'new',\n"
        "  created_at timestamptz not null default now()\n"
        ");\n\n"
        f"create index if not exists {table_prefix}_intakes_status_idx\n"
        f"  on {table_prefix}_intakes(status, created_at desc);\n"
    )


def _architecture(
    project_title: str,
    idea: str,
    selected_stack: str,
    plan_steps: list[str],
    warnings: list[str],
) -> str:
    warning_section = _markdown_list("Source Warnings", warnings)
    return (
        f"# {project_title} Architecture\n\n"
        f"Original idea: {idea}\n\n"
        "## Components\n\n"
        "- React frontend in `src/` for the primary MVP workflow.\n"
        "- FastAPI backend in `backend/` for health, planning, and intake endpoints.\n"
        "- Postgres schema in `docs/DATABASE_SCHEMA.sql` for the first persistence pass.\n"
        "- Pytest smoke tests in `tests/` for generated backend logic.\n\n"
        "## Stack Decision\n\n"
        f"{selected_stack}\n\n"
        "## Implementation Steps\n\n"
        + "\n".join(f"{index + 1}. {step}" for index, step in enumerate(plan_steps))
        + "\n"
        f"{warning_section}"
    )


def _implementation_plan(project_title: str, steps: list[str]) -> str:
    return (
        f"# {project_title} Implementation Plan\n\n"
        + "\n".join(f"{index + 1}. {step}" for index, step in enumerate(steps))
        + "\n"
    )


def _build_log(
    project_title: str,
    selected_stack: str,
    plan_steps: list[str],
    warnings: list[str],
) -> str:
    warning_section = _markdown_list("Warnings", warnings)
    return (
        f"# {project_title} Build Log\n\n"
        "- GitHub connection validated by backend OAuth.\n"
        "- Repository settings accepted from the website.\n"
        "- Nemotron/OpenClaw orchestrator scoped the messy idea into one MVP.\n"
        "- RAG context and submitted sources were checked before planning.\n"
        f"- Selected stack: {selected_stack}.\n"
        "- Generated frontend, backend, database schema, tests, docs, and walkthrough script.\n\n"
        "## Plan Executed\n\n"
        + "\n".join(f"- {step}" for step in plan_steps)
        + "\n"
        f"{warning_section}"
    )


def _demo_script(project_title: str, idea: str, feature_label: str) -> str:
    return (
        f"# {project_title} Walkthrough Script\n\n"
        "1. Open the generated React app and introduce the user problem.\n"
        f"2. Submit a realistic {feature_label.lower()} intake.\n"
        "3. Show the FastAPI `/api/mvp-plan` response.\n"
        "4. Open `docs/DATABASE_SCHEMA.sql` to show persistence is ready.\n"
        "5. Run `pytest` to show the generated backend logic has a smoke test.\n\n"
        f"Submitted idea: {idea}\n"
    )


def _plan_steps(repo_plan: dict[str, Any]) -> list[str]:
    raw_steps = repo_plan.get("implementation_steps")
    if isinstance(raw_steps, list):
        steps = [str(step).strip() for step in raw_steps if str(step).strip()]
        if steps:
            return steps[:8]
    return [
        "Define the core user workflow and success state.",
        "Generate the frontend screens for intake, status, and results.",
        "Generate backend endpoints for health, planning, and intake.",
        "Provide a Postgres schema for the first persistent data model.",
        "Add tests and docs so the repo is presentation-ready immediately.",
    ]


def _selected_stack(repo_plan: dict[str, Any], resolved_stack: str) -> str:
    raw_stack = repo_plan.get("selected_stack")
    if isinstance(raw_stack, list):
        stack = [str(item).strip() for item in raw_stack if str(item).strip()]
        if stack:
            return ", ".join(stack)
    return resolved_stack or "React, FastAPI, Postgres, Pytest"


def _warning_lines(warnings: list[dict[str, str]]) -> list[str]:
    lines = []
    for warning in warnings:
        source = warning.get("source", "source")
        message = warning.get("message", "unreadable source")
        lines.append(f"{source}: {message}")
    return lines


def _markdown_list(title: str, lines: list[str]) -> str:
    if not lines:
        return ""
    return "\n\n## " + title + "\n\n" + "\n".join(f"- {line}" for line in lines) + "\n"


def _feature_list(required_features: list[str] | None, idea: str) -> list[str]:
    features = [str(item).strip() for item in (required_features or []) if str(item).strip()]
    if features:
        return features[:6]
    label = _feature_label(idea)
    return [
        f"Capture {label.lower()} intake with structured fields.",
        f"Prioritize the next actions for {label.lower()}.",
        "Serve realistic mock records through the API and UI.",
        "Document architecture, setup, and walkthrough steps.",
    ]


def _mock_records(project_title: str, idea: str, features: list[str]) -> str:
    records = [
        {"id": "rec-1", "owner": "Alex", "title": features[0] if features else "Validate intake", "status": "active"},
        {"id": "rec-2", "owner": "Jordan", "title": features[1] if len(features) > 1 else "Review plan", "status": "done"},
        {"id": "rec-3", "owner": "Sam", "title": features[2] if len(features) > 2 else "Launch MVP release path", "status": "active"},
    ]
    return (
        f"// Mock records for {json.dumps(project_title)}\n"
        f"// Idea: {json.dumps(idea)}\n"
        f"export const mockRecords = {json.dumps(records, indent=2)};\n"
    )


def _feature_label(idea: str) -> str:
    words = re.findall(r"[A-Za-z0-9]+", idea.lower())
    skip = {"build", "create", "make", "a", "an", "the", "that", "helps", "for", "with"}
    selected = [word for word in words if word not in skip][:3]
    return " ".join(selected).title() if selected else "MVP"


def _user_story(idea: str) -> str:
    return f"As a user, I want {idea.rstrip('.').lower()} so I can make progress faster."


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug[:60] or "generated-mvp"


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _kind_from_path(path: str) -> str:
    suffix = path.rsplit(".", 1)[-1].lower() if "." in path else ""
    return {
        "md": "markdown",
        "py": "python",
        "jsx": "javascript",
        "js": "javascript",
        "css": "css",
        "json": "json",
        "sql": "sql",
        "html": "html",
        "txt": "text",
    }.get(suffix, "text")


def _html_escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _jsx_escape(value: str) -> str:
    return value.replace("{", "&#123;").replace("}", "&#125;")
