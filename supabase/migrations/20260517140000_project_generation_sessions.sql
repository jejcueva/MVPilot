-- Extended tables for complex project generation sessions (non-breaking additions).

create table if not exists public.project_sessions (
    id uuid primary key default gen_random_uuid(),
    task_id text unique not null,
    idea text not null,
    project_depth text not null default 'Advanced Project',
    target_platform text not null default 'web app',
    orchestration_mode text not null default 'langgraph',
    status text not null default 'started',
    project_plan jsonb not null default '{}'::jsonb,
    build_timeline jsonb not null default '[]'::jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create table if not exists public.agent_logs (
    id uuid primary key default gen_random_uuid(),
    task_id text not null references public.project_sessions(task_id) on delete cascade,
    agent_key text not null,
    agent_name text not null,
    stage_id text not null,
    status text not null default 'completed',
    message text not null,
    detail text,
    logged_at timestamptz not null default now()
);

create index if not exists agent_logs_task_id_idx on public.agent_logs(task_id);

create table if not exists public.project_architectures (
    id uuid primary key default gen_random_uuid(),
    task_id text not null references public.project_sessions(task_id) on delete cascade,
    architecture jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now()
);

create table if not exists public.project_requirements (
    id uuid primary key default gen_random_uuid(),
    task_id text not null references public.project_sessions(task_id) on delete cascade,
    requirements jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now()
);

create table if not exists public.validation_results (
    id uuid primary key default gen_random_uuid(),
    task_id text not null references public.project_sessions(task_id) on delete cascade,
    passed boolean not null default false,
    report jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now()
);
