# MVPilot

MVPilot turns a vague startup idea into a deployable GitHub MVP through an **OpenClaw-aware orchestration pipeline**, Nemotron planning, RAG context, and visible autonomous build telemetry in Mission Control.

## What MVPilot Does Now

1. **Intake** — project idea, target users, tech stack preference, required features, reference URL, GitHub repo settings.
2. **OpenClaw orchestration** — phased pipeline (understand idea → extract requirements → plan architecture → decompose tasks → repo → frontend → backend/API → labeled mocks → docs → validation → finalize).
3. **MVP generation** — idea-specific React UI, FastAPI backend, realistic labeled mock integrations where credentials are unavailable, tests, architecture docs, `.env.example`, and build log.
4. **GitHub delivery** — OAuth-backed repo create/update, commit, and health verification.
5. **Mission Control UI** — flight path, **OpenClaw Build Timeline**, **MVP Plan** panel, agent activity log, and landing links.

When `OPENCLAW_API_KEY` is set, tool calls are wrapped with OpenClaw trace metadata (`runtime: openclaw`). LangGraph still executes the workflow graph; OpenClaw is the tool boundary and demo-facing orchestration layer. If live Nemotron is unavailable and `ALLOW_IDEA_AWARE_PARTIAL=true`, MVPilot generates a partial, project-specific implementation and labels pending/mock areas instead of switching to a generic starter app.

## Local Setup

1. Copy environment variables:

```bash
cp .env.example .env
```

2. Create a GitHub OAuth App at <https://github.com/settings/developers>.

- Homepage URL: `http://localhost:3000`
- Authorization callback URL: `http://127.0.0.1:3001/api/auth/github/callback`
- Required scope is requested by the backend: `repo read:user user:email`

3. Fill these backend values in `.env`:

```bash
GITHUB_OAUTH_CLIENT_ID=...
GITHUB_OAUTH_CLIENT_SECRET=...
GITHUB_OAUTH_REDIRECT_URI=http://127.0.0.1:3001/api/auth/github/callback
GITHUB_TOKEN_ENCRYPTION_KEY=...
FRONTEND_BASE_URL=http://localhost:3000
CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
NEXT_PUBLIC_AGENT_API_URL=http://127.0.0.1:3001
```

Generate the encryption key with:

```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

For live model/repo execution, also set:

```bash
ADAPTER_MODE=live
MOCK_MODE=false
MVPILOT_MOCK_TOOLS=false
NVIDIA_API_KEY=...
OPENCLAW_API_KEY=...          # optional; enables OpenClaw runtime label + tool traces
OPENCLAW_ENV=development
ALLOW_IDEA_AWARE_PARTIAL=true # graceful idea-specific partial output if Nemotron is unavailable
SUPABASE_URL=...
SUPABASE_SERVICE_ROLE_KEY=...
```

Apply Supabase migrations before live GitHub connections:

```bash
supabase db push
```

## Run Locally

Backend:

```bash
.venv/bin/uvicorn agent.main:app --host 127.0.0.1 --port 3001 --reload
```

Frontend:

```bash
cd frontend
npm install
NEXT_PUBLIC_AGENT_API_URL=http://127.0.0.1:3001 npm run dev
```

Open <http://localhost:3000>, connect GitHub, fill the launch brief (idea, users, stack, features), and launch the build.

## Demo Script (Hackathon)

1. Show **Launch Parameters** — idea + target users + required features + repo name.
2. Connect GitHub and click **Launch MVPilot**.
3. Point to **OpenClaw Build Timeline** as phases complete (requirements → architecture → code → validation → commit).
4. Open **MVP Plan** — features, stack, implementation steps from Nemotron.
5. Expand **Agent Activity Log** for node-level decisions.
6. Land on GitHub repo — open generated React app (`npm run dev`) and FastAPI (`uvicorn backend.main:app --reload`).
7. Highlight `docs/BUILD_LOG.md`, `docs/ARCHITECTURE.md`, and `demo/demo_script.md`.

## Architecture (Service Layers)

| Module | Role |
|--------|------|
| `agent/openclaw_orchestrator.py` | Pipeline phases, MVP plan snapshot, build timeline |
| `agent/workflow.py` | LangGraph nodes (scope, plan, generate, commit, verify) |
| `agent/project_generation.py` | Idea-specific artifact generation + model artifact merge |
| `agent/generated_project.py` | Rich MVP package (frontend tabs, API, labeled mocks, docs) |
| `agent/mvp_validation.py` | Generic-output detection, validation report, and repair gate |
| `agent/openclaw_runtime.py` | OpenClaw tool adapter + traces |
| `tools/github_tool.py` | GitHub repo + commit tools |
| `frontend/app/page.tsx` | Mission Control UI |

## Tests

```bash
pytest
cd frontend && npm run lint
```

Main project docs live in `docs/`.
