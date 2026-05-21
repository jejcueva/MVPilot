-- Persist Stack Selector output on project sessions.

alter table public.project_sessions
    add column if not exists recommended_stack jsonb not null default '{}'::jsonb;

create index if not exists project_sessions_status_idx on public.project_sessions(status);
