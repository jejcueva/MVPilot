from __future__ import annotations

import json
import re
from typing import Any

from agent.project_depth import (
    depth_profile,
    enrich_project_requirements,
    project_collection_route,
    project_tabs,
    title_from_idea,
)


def build_project_artifacts(
    *,
    idea: str,
    title: str | None,
    resolved_stack: str,
    architecture_plan: dict[str, Any] | None = None,
    repo_plan: dict[str, Any] | None = None,
    source_warnings: list[dict[str, str]] | None = None,
    target_users: str | None = None,
    required_features: list[str] | None = None,
    tech_stack_preference: str | None = None,
    project_requirements: dict[str, Any] | None = None,
    mvp_scope: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Create a complete generated project package from the orchestration plan."""

    plan = architecture_plan or repo_plan or {}
    requirements = enrich_project_requirements(
        project_requirements or mvp_scope or {},
        idea=idea,
        intake={
            "requiredFeatures": required_features or [],
            "targetUsers": target_users,
            "techStackPreference": tech_stack_preference,
        },
    )
    project_title = _clean_text(title or "") or title_from_idea(idea)
    slug = _slugify(project_title)
    stack = tech_stack_preference or _selected_stack(plan, resolved_stack)
    warnings = _warning_lines(source_warnings or [])
    features = _feature_list(requirements, required_features)
    advanced_features = _string_list(requirements.get("advanced_features"))
    personas = _string_list(requirements.get("user_personas")) or ["Primary user"]
    user_flows = requirements.get("user_flows") if isinstance(requirements.get("user_flows"), list) else []
    archetype = str(requirements.get("project_archetype") or "workflow")
    depth = str(requirements.get("project_depth") or "Advanced Project")
    target_platform = str(requirements.get("target_platform") or "web app")
    route = project_collection_route(archetype)
    tabs = project_tabs(archetype, idea)
    is_study = _looks_like_study_project(idea, features)
    api_routes = _string_list(requirements.get("api_routes"))
    if not api_routes:
        api_routes = [
            "POST /api/auth/login",
            "POST /api/uploads",
            "POST /api/summaries",
            "POST /api/quizzes",
            "GET /api/flashcards/review",
            "GET /api/dashboard",
        ]

    files = [
        {
            "name": "README.md",
            "kind": "markdown",
            "summary": "Full project overview, setup, architecture, env guide, testing, and deployment notes.",
            "content": _readme(
                project_title=project_title,
                idea=idea,
                stack=stack,
                depth=depth,
                target_platform=target_platform,
                personas=personas,
                features=features,
                advanced_features=advanced_features,
                api_routes=api_routes,
                warnings=warnings,
            ),
        },
        {
            "name": "package.json",
            "kind": "json",
            "summary": "Frontend scripts, build command, and test dependencies.",
            "content": _package_json(slug),
        },
        {
            "name": "index.html",
            "kind": "html",
            "summary": "Frontend HTML shell.",
            "content": _index_html(project_title),
        },
        {
            "name": "src/main.jsx",
            "kind": "javascript",
            "summary": "React entrypoint.",
            "content": _main_jsx(),
        },
        {
            "name": "src/App.jsx",
            "kind": "javascript",
            "summary": "Full project studio UI with auth, upload, generation, review, and dashboard flows.",
            "content": _react_app(
                project_title=project_title,
                idea=idea,
                features=features,
                advanced_features=advanced_features,
                personas=personas,
                user_flows=user_flows,
                api_routes=api_routes,
                tabs=tabs,
                route=route,
                is_study=is_study,
            ),
        },
        {
            "name": "src/lib/api.js",
            "kind": "javascript",
            "summary": "Typed API client helpers for the generated backend routes.",
            "content": _api_client(api_routes),
        },
        {
            "name": "src/state/projectState.js",
            "kind": "javascript",
            "summary": "Domain seed state used by the generated UI and tests.",
            "content": _frontend_state(project_title, features, user_flows, is_study),
        },
        {
            "name": "src/styles.css",
            "kind": "css",
            "summary": "Responsive application styling for a production-style tool surface.",
            "content": _css(),
        },
        {
            "name": "backend/main.py",
            "kind": "python",
            "summary": "FastAPI app with auth, upload, summary, quiz, flashcard, dashboard, and health routes.",
            "content": _backend_main(project_title, idea, features, api_routes, is_study),
        },
        {
            "name": "backend/models.py",
            "kind": "python",
            "summary": "Pydantic request and response models for generated API flows.",
            "content": _backend_models(project_title, features),
        },
        {
            "name": "backend/services.py",
            "kind": "python",
            "summary": "Business logic for parsing inputs, generating study or project outputs, and dashboard metrics.",
            "content": _backend_services(project_title, idea, features, is_study),
        },
        {
            "name": "backend/db.py",
            "kind": "python",
            "summary": "SQLite-backed persistence adapter with Postgres-ready boundaries.",
            "content": _backend_db(slug),
        },
        {
            "name": "requirements.txt",
            "kind": "text",
            "summary": "Backend runtime and test dependencies.",
            "content": "fastapi\nuvicorn[standard]\npydantic\npython-multipart\npytest\nhttpx\n",
        },
        {
            "name": "tests/test_backend.py",
            "kind": "python",
            "summary": "Generated API and service tests for the project workflows.",
            "content": _backend_tests(project_title, is_study),
        },
        {
            "name": "docs/PROJECT_PLAN.md",
            "kind": "markdown",
            "summary": "Expanded product plan, features, milestones, and success criteria.",
            "content": _project_plan(project_title, idea, requirements, features, advanced_features),
        },
        {
            "name": "docs/ARCHITECTURE.md",
            "kind": "markdown",
            "summary": "Frontend, backend, data, auth, integration, testing, and deployment architecture.",
            "content": _architecture(project_title, idea, stack, requirements, plan, warnings),
        },
        {
            "name": "docs/API_SPEC.md",
            "kind": "markdown",
            "summary": "API route plan with responsibilities and data flow.",
            "content": _api_spec(project_title, api_routes),
        },
        {
            "name": "docs/DATABASE_SCHEMA.sql",
            "kind": "sql",
            "summary": "Postgres/Supabase schema for users, uploads, generated assets, reviews, and logs.",
            "content": _database_schema(slug, is_study),
        },
        {
            "name": "docs/TESTING_STRATEGY.md",
            "kind": "markdown",
            "summary": "Unit, API, frontend, validation, and build verification strategy.",
            "content": _testing_strategy(project_title, depth),
        },
        {
            "name": "docs/DEPLOY.md",
            "kind": "markdown",
            "summary": "Deployment guide for frontend, backend, database, and environment variables.",
            "content": _deploy_doc(project_title),
        },
        {
            "name": "docs/AGENT_LOG.md",
            "kind": "markdown",
            "summary": "Visible orchestration log with agent responsibilities and generation decisions.",
            "content": _agent_log(project_title, requirements, plan, warnings),
        },
        {
            "name": "docs/BUILD_LOG.md",
            "kind": "markdown",
            "summary": "Build and validation log for the generated codebase.",
            "content": _build_log(project_title, stack, depth, features, warnings),
        },
        {
            "name": "docs/KNOWN_LIMITATIONS.md",
            "kind": "markdown",
            "summary": "Known limitations and future improvements.",
            "content": _limitations(project_title),
        },
        {
            "name": "docs/WALKTHROUGH.md",
            "kind": "markdown",
            "summary": "End-to-end product walkthrough and demo flow.",
            "content": _walkthrough(project_title, user_flows, api_routes),
        },
        {
            "name": "demo/demo_script.md",
            "kind": "markdown",
            "summary": "Judge-facing demo script for the generated product.",
            "content": _walkthrough(project_title, user_flows, api_routes),
        },
        {
            "name": ".env.example",
            "kind": "text",
            "summary": "Safe placeholder environment file.",
            "content": _env_example(project_title),
        },
    ]
    return files


def merge_with_project_artifacts(
    artifacts: list[dict[str, Any]],
    *,
    idea: str,
    title: str | None,
    resolved_stack: str,
    architecture_plan: dict[str, Any] | None = None,
    repo_plan: dict[str, Any] | None = None,
    source_warnings: list[dict[str, str]] | None = None,
    target_users: str | None = None,
    required_features: list[str] | None = None,
    tech_stack_preference: str | None = None,
    project_requirements: dict[str, Any] | None = None,
    mvp_scope: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
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
        architecture_plan=architecture_plan,
        repo_plan=repo_plan,
        source_warnings=source_warnings,
        target_users=target_users,
        required_features=required_features,
        tech_stack_preference=tech_stack_preference,
        project_requirements=project_requirements,
        mvp_scope=mvp_scope,
    )
    for artifact in gap_fill:
        merged.setdefault(artifact["name"], artifact)
    return [merged[path] for path in sorted(merged)]


def _readme(
    *,
    project_title: str,
    idea: str,
    stack: str,
    depth: str,
    target_platform: str,
    personas: list[str],
    features: list[str],
    advanced_features: list[str],
    api_routes: list[str],
    warnings: list[str],
) -> str:
    return (
        f"# {project_title}\n\n"
        f"{idea}\n\n"
        f"Generated as a **{depth}** for a **{target_platform}**. The repository is structured as a complete full-stack project with authentication, data flows, API contracts, tests, documentation, and deployment notes.\n\n"
        "## Target Users\n\n"
        + _markdown_items(personas)
        + "\n## Core Features\n\n"
        + _markdown_items(features)
        + "\n## Advanced Features\n\n"
        + _markdown_items(advanced_features or ["Production-ready extension points", "Operational logs", "Deployment readiness"])
        + "\n## Tech Stack\n\n"
        f"{stack}\n\n"
        "## API Plan\n\n"
        + _markdown_items(api_routes)
        + "\n## Run Locally\n\n"
        "```bash\nnpm install\nnpm run dev\n```\n\n"
        "In another terminal:\n\n"
        "```bash\npip install -r requirements.txt\nuvicorn backend.main:app --reload\n```\n\n"
        "## Test\n\n"
        "```bash\npytest\nnpm run build\n```\n\n"
        "## Environment\n\n"
        "Copy `.env.example` and set backend-only secrets in the backend deployment environment. Never commit real `.env` files.\n"
        + _warning_section(warnings)
    )


def _package_json(slug: str) -> str:
    return json.dumps(
        {
            "name": slug,
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
    ) + "\n"


def _index_html(project_title: str) -> str:
    return (
        "<!doctype html>\n"
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
    )


def _main_jsx() -> str:
    return (
        "import React from 'react';\n"
        "import { createRoot } from 'react-dom/client';\n"
        "import App from './App.jsx';\n"
        "import './styles.css';\n\n"
        "createRoot(document.getElementById('root')).render(\n"
        "  <React.StrictMode>\n"
        "    <App />\n"
        "  </React.StrictMode>,\n"
        ");\n"
    )


def _react_app(
    *,
    project_title: str,
    idea: str,
    features: list[str],
    advanced_features: list[str],
    personas: list[str],
    user_flows: list[dict[str, Any]],
    api_routes: list[str],
    tabs: list[str],
    route: str,
    is_study: bool,
) -> str:
    return (
        "import { useMemo, useState } from 'react';\n"
        "import { seedAssets, reviewQueue, dashboardMetrics } from './state/projectState.js';\n"
        "import { createUpload, generateQuiz, getDashboard } from './lib/api.js';\n\n"
        f"const projectTitle = {json.dumps(project_title)};\n"
        f"const idea = {json.dumps(idea)};\n"
        f"const features = {json.dumps(features, indent=2)};\n"
        f"const advancedFeatures = {json.dumps(advanced_features, indent=2)};\n"
        f"const personas = {json.dumps(personas, indent=2)};\n"
        f"const userFlows = {json.dumps(user_flows, indent=2)};\n"
        f"const apiRoutes = {json.dumps(api_routes, indent=2)};\n"
        f"const tabs = {json.dumps(tabs)};\n"
        f"const collectionRoute = {json.dumps(route)};\n"
        f"const isStudyProject = {json.dumps(is_study)};\n\n"
        "export default function App() {\n"
        "  const [activeTab, setActiveTab] = useState(tabs[0]);\n"
        "  const [session, setSession] = useState({ name: 'Avery Student', role: 'owner' });\n"
        "  const [noteText, setNoteText] = useState('Photosynthesis lecture: spaced repetition works best when reviews are scheduled right before forgetting.');\n"
        "  const [generated, setGenerated] = useState(null);\n"
        "  const [status, setStatus] = useState('Ready');\n"
        "  const metrics = useMemo(() => dashboardMetrics, []);\n\n"
        "  async function handleGenerate() {\n"
        "    setStatus('Generating');\n"
        "    const upload = await createUpload({ title: 'Lecture notes', content: noteText, route: collectionRoute });\n"
        "    const quiz = await generateQuiz({ source_id: upload.id, content: noteText });\n"
        "    setGenerated({ upload, quiz });\n"
        "    setStatus('Generated');\n"
        "  }\n\n"
        "  async function refreshDashboard() {\n"
        "    setStatus('Refreshing dashboard');\n"
        "    const dashboard = await getDashboard();\n"
        "    setGenerated((current) => ({ ...(current || {}), dashboard }));\n"
        "    setStatus('Dashboard refreshed');\n"
        "  }\n\n"
        "  return (\n"
        "    <main className=\"appShell\">\n"
        "      <header className=\"topbar\">\n"
        "        <div>\n"
        "          <p className=\"eyebrow\">Full-Stack Generated Project</p>\n"
        "          <h1>{projectTitle}</h1>\n"
        "          <p className=\"lede\">{idea}</p>\n"
        "        </div>\n"
        "        <div className=\"sessionCard\">\n"
        "          <span>Signed in</span>\n"
        "          <strong>{session.name}</strong>\n"
        "          <button type=\"button\" onClick={() => setSession({ name: 'Morgan Admin', role: 'admin' })}>Switch role</button>\n"
        "        </div>\n"
        "      </header>\n"
        "      <nav className=\"tabs\">{tabs.map((tab) => <button key={tab} type=\"button\" className={activeTab === tab ? 'active' : ''} onClick={() => setActiveTab(tab)}>{tab}</button>)}</nav>\n"
        "      {activeTab === tabs[0] && <section className=\"grid\"><ProjectBrief personas={personas} features={features} apiRoutes={apiRoutes} /><Metrics metrics={metrics} /></section>}\n"
        "      {activeTab === tabs[1] && <section className=\"workspace\"><textarea value={noteText} onChange={(event) => setNoteText(event.target.value)} rows={8} /><div className=\"actions\"><button type=\"button\" onClick={handleGenerate}>{isStudyProject ? 'Generate study assets' : 'Generate workflow assets'}</button><button type=\"button\" onClick={refreshDashboard}>Refresh dashboard</button></div><p className=\"status\">{status}</p></section>}\n"
        "      {activeTab === tabs[2] && <section className=\"grid\"><ReviewQueue generated={generated} /><Advanced features={advancedFeatures} /></section>}\n"
        "      {activeTab === tabs[3] && <section className=\"dashboard\"><FlowList flows={userFlows} /><AssetTable /></section>}\n"
        "    </main>\n"
        "  );\n"
        "}\n\n"
        "function ProjectBrief({ personas, features, apiRoutes }) {\n"
        "  return <article className=\"panel\"><h2>Product System</h2><p>Built for {personas.join(', ')}.</p><ul>{features.slice(0, 8).map((feature) => <li key={feature}>{feature}</li>)}</ul><h3>API routes</h3><ul>{apiRoutes.map((route) => <li key={route}>{route}</li>)}</ul></article>;\n"
        "}\n\n"
        "function Metrics({ metrics }) {\n"
        "  return <article className=\"panel metrics\"><h2>Readiness Dashboard</h2>{metrics.map((metric) => <div key={metric.label} className=\"metric\"><span>{metric.label}</span><strong>{metric.value}</strong><small>{metric.detail}</small></div>)}</article>;\n"
        "}\n\n"
        "function ReviewQueue({ generated }) {\n"
        "  const quizItems = generated?.quiz?.questions || reviewQueue;\n"
        "  return <article className=\"panel\"><h2>Generated Review</h2><ul>{quizItems.map((item) => <li key={item.id || item.question}>{item.question || item.prompt}<span>{item.answer || item.due}</span></li>)}</ul></article>;\n"
        "}\n\n"
        "function Advanced({ features }) {\n"
        "  return <article className=\"panel\"><h2>Advanced System Features</h2><ul>{features.map((feature) => <li key={feature}>{feature}</li>)}</ul></article>;\n"
        "}\n\n"
        "function FlowList({ flows }) {\n"
        "  return <article className=\"panel\"><h2>User Flow</h2>{flows.map((flow) => <div key={flow.step} className=\"flow\"><strong>{flow.screen}</strong><span>{flow.action}</span><code>{flow.api || 'UI'}</code></div>)}</article>;\n"
        "}\n\n"
        "function AssetTable() {\n"
        "  return <article className=\"panel\"><h2>Generated Assets</h2>{seedAssets.map((asset) => <div key={asset.id} className=\"asset\"><span>{asset.type}</span><strong>{asset.title}</strong><em>{asset.status}</em></div>)}</article>;\n"
        "}\n"
    )


def _api_client(api_routes: list[str]) -> str:
    del api_routes
    return (
        "const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';\n\n"
        "async function request(path, options = {}) {\n"
        "  const response = await fetch(`${API_BASE}${path}`, {\n"
        "    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },\n"
        "    ...options,\n"
        "  });\n"
        "  if (!response.ok) throw new Error(`API ${response.status}: ${path}`);\n"
        "  return response.json();\n"
        "}\n\n"
        "export function createUpload(payload) {\n"
        "  return request('/api/uploads', { method: 'POST', body: JSON.stringify(payload) });\n"
        "}\n\n"
        "export function generateQuiz(payload) {\n"
        "  return request('/api/quizzes', { method: 'POST', body: JSON.stringify(payload) });\n"
        "}\n\n"
        "export function getDashboard() {\n"
        "  return request('/api/dashboard');\n"
        "}\n"
    )


def _frontend_state(project_title: str, features: list[str], user_flows: list[dict[str, Any]], is_study: bool) -> str:
    assets = [
        {"id": "asset-1", "type": "summary", "title": features[0] if features else project_title, "status": "ready"},
        {"id": "asset-2", "type": "quiz", "title": "Adaptive quiz set" if is_study else "Validation checklist", "status": "draft"},
        {"id": "asset-3", "type": "dashboard", "title": "Personalized dashboard", "status": "live"},
    ]
    review = [
        {"id": "q1", "question": "What concept needs review next?", "answer": "Spaced repetition queue" if is_study else "Highest-risk workflow"},
        {"id": "q2", "question": "What is the next recommended action?", "answer": "Review weak topics" if is_study else "Complete validation"},
    ]
    metrics = [
        {"label": "Features", "value": str(len(features)), "detail": "Generated feature modules"},
        {"label": "Flows", "value": str(max(1, len(user_flows))), "detail": "End-to-end user journeys"},
        {"label": "Tests", "value": "API", "detail": "Backend smoke coverage included"},
    ]
    return (
        f"export const seedAssets = {json.dumps(assets, indent=2)};\n\n"
        f"export const reviewQueue = {json.dumps(review, indent=2)};\n\n"
        f"export const dashboardMetrics = {json.dumps(metrics, indent=2)};\n"
    )


def _css() -> str:
    return (
        ":root { color: #172026; background: #f6f8fb; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }\n"
        "* { box-sizing: border-box; }\n"
        "body { margin: 0; }\n"
        ".appShell { min-height: 100vh; padding: 32px; }\n"
        ".topbar { display: grid; grid-template-columns: minmax(0, 1fr) 280px; gap: 24px; align-items: start; }\n"
        ".eyebrow { margin: 0 0 8px; color: #0f766e; font-size: 12px; font-weight: 800; text-transform: uppercase; }\n"
        "h1 { margin: 0; font-size: 44px; line-height: 1.05; letter-spacing: 0; }\n"
        ".lede { max-width: 820px; color: #4b5c66; font-size: 18px; line-height: 1.6; }\n"
        ".sessionCard, .panel, .workspace { border: 1px solid #d7dee8; background: #fff; border-radius: 8px; box-shadow: 0 16px 40px rgba(27, 39, 51, 0.08); }\n"
        ".sessionCard { padding: 18px; display: grid; gap: 8px; }\n"
        ".sessionCard span, .metric span, .asset span { color: #60717d; font-size: 12px; text-transform: uppercase; font-weight: 800; }\n"
        "button { border: 0; border-radius: 8px; background: #0f766e; color: white; padding: 10px 14px; font-weight: 800; cursor: pointer; }\n"
        ".tabs { display: flex; gap: 8px; flex-wrap: wrap; margin: 28px 0; }\n"
        ".tabs button { background: #fff; color: #172026; border: 1px solid #d7dee8; }\n"
        ".tabs button.active { background: #0f766e; color: #fff; border-color: #0f766e; }\n"
        ".grid, .dashboard { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; }\n"
        ".panel, .workspace { padding: 24px; }\n"
        ".panel h2 { margin-top: 0; }\n"
        "li { margin: 8px 0; line-height: 1.5; }\n"
        ".workspace textarea { width: 100%; border: 1px solid #d7dee8; border-radius: 8px; padding: 14px; font: inherit; line-height: 1.5; }\n"
        ".actions { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 14px; }\n"
        ".status { color: #0f766e; font-weight: 800; }\n"
        ".metric, .flow, .asset { border-top: 1px solid #eef2f6; padding: 14px 0; display: grid; gap: 4px; }\n"
        ".flow code { color: #0f766e; }\n"
        ".asset { grid-template-columns: 90px 1fr 90px; align-items: center; }\n"
        "@media (max-width: 840px) { .appShell { padding: 20px; } .topbar, .grid, .dashboard { grid-template-columns: 1fr; } h1 { font-size: 34px; } }\n"
    )


def _backend_main(project_title: str, idea: str, features: list[str], api_routes: list[str], is_study: bool) -> str:
    return (
        '"""FastAPI backend for the generated full-stack project."""\n\n'
        "from fastapi import FastAPI, UploadFile, File\n\n"
        "from backend.db import list_activity, save_activity\n"
        "from backend.models import LoginRequest, QuizRequest, TextAssetRequest\n"
        "from backend.services import build_dashboard, generate_quiz, parse_uploaded_notes, summarize_text\n\n\n"
        f"app = FastAPI(title={json.dumps(project_title)})\n"
        f"PROJECT_IDEA = {json.dumps(idea)}\n"
        f"PROJECT_FEATURES = {json.dumps(features, indent=2)}\n"
        f"API_ROUTES = {json.dumps(api_routes, indent=2)}\n"
        f"IS_STUDY_PROJECT = {json.dumps(is_study)}\n\n\n"
        "@app.get('/health')\n"
        "def health() -> dict[str, object]:\n"
        "    return {'status': 'ok', 'service': app.title, 'routes': len(API_ROUTES)}\n\n\n"
        "@app.post('/api/auth/login')\n"
        "def login(payload: LoginRequest) -> dict[str, object]:\n"
        "    return {'access_token': f'dev-token-{payload.email}', 'user': {'email': payload.email, 'role': payload.role}}\n\n\n"
        "@app.post('/api/uploads')\n"
        "def create_upload(payload: TextAssetRequest) -> dict[str, object]:\n"
        "    asset = parse_uploaded_notes(payload.title, payload.content)\n"
        "    save_activity({'type': 'upload', 'title': payload.title, 'status': 'parsed'})\n"
        "    return asset\n\n\n"
        "@app.post('/api/files/upload')\n"
        "async def upload_file(file: UploadFile = File(...)) -> dict[str, object]:\n"
        "    content = (await file.read()).decode('utf-8', errors='replace')\n"
        "    asset = parse_uploaded_notes(file.filename or 'upload.txt', content)\n"
        "    save_activity({'type': 'file_upload', 'title': file.filename, 'status': 'parsed'})\n"
        "    return asset\n\n\n"
        "@app.post('/api/summaries')\n"
        "def create_summary(payload: TextAssetRequest) -> dict[str, object]:\n"
        "    return summarize_text(payload.title, payload.content)\n\n\n"
        "@app.post('/api/quizzes')\n"
        "def create_quiz(payload: QuizRequest) -> dict[str, object]:\n"
        "    quiz = generate_quiz(payload.content, study_mode=IS_STUDY_PROJECT)\n"
        "    save_activity({'type': 'quiz', 'title': 'Generated quiz', 'status': 'ready'})\n"
        "    return quiz\n\n\n"
        "@app.get('/api/flashcards/review')\n"
        "def flashcard_review() -> dict[str, object]:\n"
        "    return {'queue': generate_quiz(PROJECT_IDEA, study_mode=True)['flashcards']}\n\n\n"
        "@app.get('/api/dashboard')\n"
        "def dashboard() -> dict[str, object]:\n"
        "    return build_dashboard(PROJECT_IDEA, PROJECT_FEATURES, list_activity())\n\n\n"
        "@app.get('/api/project-plan')\n"
        "def project_plan() -> dict[str, object]:\n"
        "    return {'idea': PROJECT_IDEA, 'features': PROJECT_FEATURES, 'api_routes': API_ROUTES}\n"
    )


def _backend_models(project_title: str, features: list[str]) -> str:
    return (
        '"""Request and response schemas for the generated project."""\n\n'
        "from pydantic import BaseModel, Field\n\n\n"
        "class LoginRequest(BaseModel):\n"
        "    email: str = 'user@example.com'\n"
        "    role: str = 'owner'\n\n\n"
        "class TextAssetRequest(BaseModel):\n"
        "    title: str = Field(min_length=1)\n"
        "    content: str = Field(min_length=1)\n"
        "    route: str | None = None\n\n\n"
        "class QuizRequest(BaseModel):\n"
        "    source_id: str | None = None\n"
        "    content: str = Field(min_length=1)\n"
        "    difficulty: str = 'adaptive'\n\n\n"
        f"PROJECT_NAME = {json.dumps(project_title)}\n"
        f"FEATURES = {json.dumps(features, indent=2)}\n"
    )


def _backend_services(project_title: str, idea: str, features: list[str], is_study: bool) -> str:
    return (
        '"""Business logic for the generated full-stack project."""\n\n'
        "from __future__ import annotations\n\n"
        "from hashlib import sha1\n"
        "from typing import Any\n\n\n"
        f"PROJECT_TITLE = {json.dumps(project_title)}\n"
        f"PROJECT_IDEA = {json.dumps(idea)}\n"
        f"FEATURES = {json.dumps(features, indent=2)}\n"
        f"IS_STUDY_PROJECT = {json.dumps(is_study)}\n\n\n"
        "def _asset_id(value: str) -> str:\n"
        "    return sha1(value.encode('utf-8')).hexdigest()[:12]\n\n\n"
        "def parse_uploaded_notes(title: str, content: str) -> dict[str, Any]:\n"
        "    words = [word.strip('.,:;!?').lower() for word in content.split() if len(word.strip('.,:;!?')) > 4]\n"
        "    key_terms = sorted(dict.fromkeys(words))[:8]\n"
        "    return {\n"
        "        'id': _asset_id(title + content),\n"
        "        'title': title,\n"
        "        'kind': 'lecture_notes' if IS_STUDY_PROJECT else 'source_document',\n"
        "        'summary': summarize_text(title, content)['summary'],\n"
        "        'key_terms': key_terms,\n"
        "        'next_actions': [\n"
        "            'Generate summary',\n"
        "            'Create quiz questions',\n"
        "            'Schedule review tasks',\n"
        "        ],\n"
        "    }\n\n\n"
        "def summarize_text(title: str, content: str) -> dict[str, Any]:\n"
        "    sentences = [part.strip() for part in content.replace('\\n', ' ').split('.') if part.strip()]\n"
        "    summary = '. '.join(sentences[:2]) or content[:240]\n"
        "    return {'title': title, 'summary': summary, 'source_length': len(content)}\n\n\n"
        "def generate_quiz(content: str, *, study_mode: bool) -> dict[str, Any]:\n"
        "    theme = 'spaced repetition' if study_mode else 'project workflow'\n"
        "    questions = [\n"
        "        {'id': 'q1', 'question': f'What is the key idea behind {theme}?', 'answer': 'Review important material at increasing intervals.'},\n"
        "        {'id': 'q2', 'question': 'Which area should the user review next?', 'answer': (content[:80] or PROJECT_TITLE)},\n"
        "    ]\n"
        "    flashcards = [\n"
        "        {'front': 'Core concept', 'back': questions[0]['answer'], 'due': 'today'},\n"
        "        {'front': 'Next action', 'back': questions[1]['answer'], 'due': 'tomorrow'},\n"
        "    ]\n"
        "    return {'questions': questions, 'flashcards': flashcards, 'study_mode': study_mode}\n\n\n"
        "def build_dashboard(idea: str, features: list[str], activity: list[dict[str, Any]]) -> dict[str, Any]:\n"
        "    return {\n"
        "        'idea': idea,\n"
        "        'feature_count': len(features),\n"
        "        'activity_count': len(activity),\n"
        "        'readiness': 86 if activity else 72,\n"
        "        'weak_topics': ['retrieval practice', 'review cadence'] if IS_STUDY_PROJECT else ['validation', 'handoff'],\n"
        "        'next_actions': features[:4],\n"
        "    }\n"
    )


def _backend_db(slug: str) -> str:
    return (
        '"""SQLite adapter with a Postgres-ready boundary for generated project state."""\n\n'
        "from __future__ import annotations\n\n"
        "import json\n"
        "import sqlite3\n"
        "from pathlib import Path\n"
        "from typing import Any\n\n\n"
        "DB_PATH = Path(__file__).resolve().parent / 'data' / 'project.db'\n\n\n"
        "def _connection() -> sqlite3.Connection:\n"
        "    DB_PATH.parent.mkdir(parents=True, exist_ok=True)\n"
        "    conn = sqlite3.connect(DB_PATH)\n"
        "    conn.row_factory = sqlite3.Row\n"
        "    conn.execute(\n"
        "        'create table if not exists activity ('\n"
        "        'id integer primary key autoincrement, type text not null, title text, status text, payload text)'\n"
        "    )\n"
        "    return conn\n\n\n"
        "def save_activity(payload: dict[str, Any]) -> dict[str, Any]:\n"
        "    with _connection() as conn:\n"
        "        cursor = conn.execute(\n"
        "            'insert into activity (type, title, status, payload) values (?, ?, ?, ?)',\n"
        "            (payload.get('type', 'event'), payload.get('title'), payload.get('status', 'new'), json.dumps(payload)),\n"
        "        )\n"
        "        conn.commit()\n"
        "    return {'id': cursor.lastrowid, **payload}\n\n\n"
        "def list_activity() -> list[dict[str, Any]]:\n"
        "    with _connection() as conn:\n"
        "        rows = conn.execute('select id, type, title, status, payload from activity order by id desc').fetchall()\n"
        "    return [{**json.loads(row['payload']), 'id': row['id']} for row in rows]\n\n\n"
        f"PROJECT_SCHEMA_PREFIX = {json.dumps(slug.replace('-', '_'))}\n"
    )


def _backend_tests(project_title: str, is_study: bool) -> str:
    return (
        "from fastapi.testclient import TestClient\n\n"
        "from backend.main import app\n"
        "from backend.services import generate_quiz, parse_uploaded_notes\n\n\n"
        "client = TestClient(app)\n\n\n"
        "def test_health_reports_service():\n"
        "    response = client.get('/health')\n"
        "    assert response.status_code == 200\n"
        f"    assert response.json()['service'] == {json.dumps(project_title)}\n\n\n"
        "def test_upload_and_quiz_flow():\n"
        "    upload = client.post('/api/uploads', json={'title': 'Lecture', 'content': 'Spaced repetition improves exam recall.'})\n"
        "    assert upload.status_code == 200\n"
        "    quiz = client.post('/api/quizzes', json={'source_id': upload.json()['id'], 'content': 'Spaced repetition improves exam recall.'})\n"
        "    assert quiz.status_code == 200\n"
        "    assert quiz.json()['questions']\n\n\n"
        "def test_services_generate_domain_assets():\n"
        "    asset = parse_uploaded_notes('Notes', 'retrieval practice and review cadence')\n"
        "    assert asset['key_terms']\n"
        f"    assert generate_quiz('review cadence', study_mode={json.dumps(is_study)})['flashcards']\n"
    )


def _project_plan(project_title: str, idea: str, requirements: dict[str, Any], features: list[str], advanced_features: list[str]) -> str:
    profile = depth_profile(str(requirements.get("project_depth") or "Advanced Project"))
    return (
        f"# {project_title} Project Plan\n\n"
        f"## Product Description\n\n{idea}\n\n"
        f"## Depth\n\n{profile['name']} - minimum features: {profile['minimum_features']}.\n\n"
        "## Personas\n\n"
        + _markdown_items(_string_list(requirements.get("user_personas")) or ["Primary user"])
        + "\n## Core Features\n\n"
        + _markdown_items(features)
        + "\n## Advanced Features\n\n"
        + _markdown_items(advanced_features)
        + "\n## Success Criteria\n\n"
        + _markdown_items(_string_list(requirements.get("success_criteria")))
    )


def _architecture(project_title: str, idea: str, stack: str, requirements: dict[str, Any], plan: dict[str, Any], warnings: list[str]) -> str:
    return (
        f"# {project_title} Architecture\n\n"
        f"Original idea: {idea}\n\n"
        "## Stack\n\n"
        f"{stack}\n\n"
        "## Frontend Architecture\n\n"
        "- React application with authenticated workspace, generation flow, review queue, and dashboard views.\n"
        "- `src/lib/api.js` isolates backend calls from the UI.\n"
        "- `src/state/projectState.js` seeds domain-specific review, dashboard, and asset state.\n\n"
        "## Backend Architecture\n\n"
        "- FastAPI app in `backend/main.py` exposes auth, upload, generation, review, dashboard, and health routes.\n"
        "- `backend/services.py` owns domain logic and can be replaced with live model/provider calls.\n"
        "- `backend/db.py` stores activity locally and marks the persistence boundary for Postgres/Supabase.\n\n"
        "## Data Model\n\n"
        + _markdown_items(_string_list(requirements.get("data_entities")))
        + "\n## API Design\n\n"
        + _markdown_items(_string_list(requirements.get("api_routes")))
        + "\n## Auth And Authorization\n\n"
        "- Development token route is included for local flow testing.\n"
        "- Production deployment should replace the dev token with Supabase Auth, Clerk, Auth.js, or the selected auth provider.\n"
        "- Service-role keys and provider secrets must remain backend-only.\n\n"
        "## Implementation Notes\n\n"
        + _markdown_items(_string_list(plan.get("implementation_steps")) or ["Generate frontend, backend, database, docs, tests, and deployment files."])
        + _warning_section(warnings)
    )


def _api_spec(project_title: str, api_routes: list[str]) -> str:
    return (
        f"# {project_title} API Spec\n\n"
        + "\n".join(f"- `{route}` - generated route contract for the project workflow." for route in api_routes)
        + "\n\nAll request handlers return JSON and avoid exposing secrets to the browser.\n"
    )


def _database_schema(slug: str, is_study: bool) -> str:
    prefix = re.sub(r"[^a-z0-9_]", "_", slug.replace("-", "_"))
    asset_name = "study_assets" if is_study else "project_assets"
    return (
        f"-- Postgres/Supabase schema for {slug}\n"
        "create extension if not exists pgcrypto;\n\n"
        f"create table if not exists {prefix}_users (\n"
        "  id uuid primary key default gen_random_uuid(),\n"
        "  email text unique not null,\n"
        "  role text not null default 'owner',\n"
        "  created_at timestamptz not null default now()\n"
        ");\n\n"
        f"create table if not exists {prefix}_{asset_name} (\n"
        "  id uuid primary key default gen_random_uuid(),\n"
        f"  user_id uuid references {prefix}_users(id) on delete cascade,\n"
        "  title text not null,\n"
        "  source_text text,\n"
        "  parsed_summary text,\n"
        "  metadata jsonb not null default '{}'::jsonb,\n"
        "  status text not null default 'ready',\n"
        "  created_at timestamptz not null default now()\n"
        ");\n\n"
        f"create table if not exists {prefix}_generated_items (\n"
        "  id uuid primary key default gen_random_uuid(),\n"
        f"  asset_id uuid references {prefix}_{asset_name}(id) on delete cascade,\n"
        "  item_type text not null,\n"
        "  prompt text,\n"
        "  answer text,\n"
        "  due_at timestamptz,\n"
        "  confidence integer default 0,\n"
        "  created_at timestamptz not null default now()\n"
        ");\n\n"
        f"create table if not exists {prefix}_activity_logs (\n"
        "  id uuid primary key default gen_random_uuid(),\n"
        "  user_id uuid,\n"
        "  event_type text not null,\n"
        "  payload jsonb not null default '{}'::jsonb,\n"
        "  created_at timestamptz not null default now()\n"
        ");\n\n"
        f"create index if not exists {prefix}_{asset_name}_user_idx on {prefix}_{asset_name}(user_id, created_at desc);\n"
        f"create index if not exists {prefix}_generated_items_due_idx on {prefix}_generated_items(due_at, confidence);\n"
    )


def _testing_strategy(project_title: str, depth: str) -> str:
    return (
        f"# {project_title} Testing Strategy\n\n"
        f"Depth target: {depth}.\n\n"
        "- Unit test service functions for parsing, generation, and dashboard metrics.\n"
        "- API integration test health, upload, quiz, and dashboard routes.\n"
        "- Frontend build check with `npm run build`.\n"
        "- Add journey tests for sign-in, upload, generation, review, and dashboard once browser automation is configured.\n"
        "- Keep provider integrations behind adapters so missing credentials fail with actionable errors.\n"
    )


def _deploy_doc(project_title: str) -> str:
    from agent.project_depth import deploy_readme_section

    return deploy_readme_section(project_title=project_title, repo_url=None)


def _agent_log(project_title: str, requirements: dict[str, Any], plan: dict[str, Any], warnings: list[str]) -> str:
    agent_assignments = _string_list(plan.get("agent_assignments")) or [
        "Product Strategist Agent expanded requirements.",
        "Research/RAG Agent collected source context.",
        "System Architect Agent planned architecture.",
        "Data/API Agent designed API and data model.",
        "Frontend Agent generated UI flows.",
        "Backend Agent generated services and routes.",
        "QA Agent generated validation strategy.",
        "Documentation Agent generated docs.",
        "GitHub Agent exports repository files.",
    ]
    return (
        f"# {project_title} Agent Log\n\n"
        f"Project depth: {requirements.get('project_depth', 'Advanced Project')}\n\n"
        + _markdown_items(agent_assignments)
        + _warning_section(warnings)
    )


def _build_log(project_title: str, stack: str, depth: str, features: list[str], warnings: list[str]) -> str:
    return (
        f"# {project_title} Build Log\n\n"
        "- Nemotron/OpenClaw orchestration attempted full project planning before file generation.\n"
        "- Product, research, architecture, data/API, frontend, backend, QA, docs, GitHub, and logger agents contributed to the plan.\n"
        f"- Project depth: {depth}.\n"
        f"- Selected stack: {stack}.\n"
        f"- Generated {len(features)} feature-oriented implementation targets.\n"
        "- Generated source files, tests, architecture docs, database schema, setup, and deployment instructions.\n"
        + _warning_section(warnings)
    )


def _limitations(project_title: str) -> str:
    return (
        f"# {project_title} Known Limitations\n\n"
        "- Local development auth uses a development token route and should be replaced before production.\n"
        "- AI provider calls are isolated behind service functions and need real credentials for live generation.\n"
        "- SQLite is included for local persistence; run the Postgres schema for production.\n"
        "- Browser journey tests should be added once the deployment target is selected.\n\n"
        "## Future Improvements\n\n"
        "- Add background jobs for long-running generation tasks.\n"
        "- Add role-specific authorization policies.\n"
        "- Add richer analytics and user notifications.\n"
    )


def _walkthrough(project_title: str, user_flows: list[dict[str, Any]], api_routes: list[str]) -> str:
    flows = user_flows or [
        {"step": "1", "screen": "Workspace", "action": "Sign in and open the workspace", "api": "POST /api/auth/login"},
        {"step": "2", "screen": "Generate", "action": "Create project assets", "api": "POST /api/uploads"},
        {"step": "3", "screen": "Dashboard", "action": "Review generated outputs", "api": "GET /api/dashboard"},
    ]
    return (
        f"# {project_title} Walkthrough\n\n"
        + "\n".join(
            f"{index + 1}. **{step.get('screen', 'Screen')}** - {step.get('action', 'Action')} (`{step.get('api', 'UI')}`)"
            for index, step in enumerate(flows)
            if isinstance(step, dict)
        )
        + "\n\n## API Evidence\n\n"
        + _markdown_items(api_routes)
    )


def _env_example(project_title: str) -> str:
    return (
        "VITE_API_BASE_URL=http://127.0.0.1:8000\n"
        "DATABASE_URL=sqlite:///backend/data/project.db\n"
        "AUTH_SECRET=replace-me\n"
        "AI_PROVIDER_API_KEY=\n"
        "SUPABASE_URL=\n"
        "SUPABASE_ANON_KEY=\n"
        f"# Generated for: {project_title}\n"
    )


def _feature_list(requirements: dict[str, Any], required_features: list[str] | None) -> list[str]:
    features = _string_list(required_features)
    for feature in _string_list(requirements.get("core_features")):
        if feature not in features:
            features.append(feature)
    return features[:14]


def _selected_stack(plan: dict[str, Any], resolved_stack: str) -> str:
    raw_stack = plan.get("selected_stack")
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


def _warning_section(warnings: list[str]) -> str:
    if not warnings:
        return ""
    return "\n## Source Warnings\n\n" + _markdown_items(warnings)


def _markdown_items(items: list[str]) -> str:
    values = [str(item).strip() for item in items if str(item).strip()]
    if not values:
        return "- Not specified yet.\n"
    return "\n".join(f"- {item}" for item in values) + "\n"


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _looks_like_study_project(idea: str, features: list[str]) -> bool:
    corpus = " ".join([idea, *features]).lower()
    return any(word in corpus for word in ("study", "lecture", "quiz", "flashcard", "spaced repetition", "exam"))


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug[:60] or "generated-project"


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
