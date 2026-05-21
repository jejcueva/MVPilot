# StudyPilot — Product Reference (MVP)

## Problem
College students juggle multiple courses, deadlines, and unstructured notes. They need a lightweight planner that turns vague goals into actionable weekly plans.

## Target users
- Undergraduate students with 3–6 active courses
- Students preparing for exams who need focus sessions and progress visibility

## Core MVP features
1. **Study goal intake** — capture course, deadline, and priority in one form.
2. **Weekly plan generator** — break goals into 5–7 concrete tasks with time estimates.
3. **Focus session tracker** — start/stop sessions and log what was completed.
4. **Progress dashboard** — open tasks, completed tasks, and streak-style metrics.
5. **API health** — `/health` and `/api/demo-plan` for demo reliability.

## Non-goals (demo boundary)
- No full LMS integration in v1
- No mobile native app — responsive web only
- No paid subscriptions or auth beyond demo mock users

## Suggested stack
- React + Vite frontend
- FastAPI backend
- Postgres-ready schema (SQL file in repo)

## Demo success criteria
- User submits a study goal and sees a prioritized plan.
- Dashboard shows at least three realistic mock work items.
- README explains local setup in under five minutes.
