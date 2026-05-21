import os
import re


def test_memories_migration_exists_and_contains_requirements():
    migrations_dir = os.path.join(
        os.path.dirname(__file__),
        "..",
        "supabase",
        "migrations"
    )
    
    # Find the memories migration file
    memories_file = None
    for filename in os.listdir(migrations_dir):
        if "memories" in filename.lower() and filename.endswith(".sql"):
            memories_file = os.path.join(migrations_dir, filename)
            break
            
    assert memories_file is not None, "Memories migration file not found."
    
    with open(memories_file, "r") as f:
        content = f.read().lower()
        
    # Assert public.memories table
    assert "create table public.memories" in content or "create table if not exists public.memories" in content
    
    # Assert RLS
    assert "enable row level security" in content
    
    # Assert indexes
    assert "create index" in content
    assert "task_id" in content
    assert "created_at" in content
    # Note: Embedding index was removed because pgvector does not support indexing vectors over 2000 dimensions
    
    # Assert match_memories RPC
    assert "create or replace function public.match_memories" in content or "create function public.match_memories" in content


def test_github_connections_migration_exists_and_contains_security_requirements():
    migrations_dir = os.path.join(
        os.path.dirname(__file__),
        "..",
        "supabase",
        "migrations",
    )

    migration_file = None
    for filename in sorted(os.listdir(migrations_dir)):
        if "github_connections" in filename.lower() and filename.endswith(".sql"):
            migration_file = os.path.join(migrations_dir, filename)
            break

    assert migration_file is not None, "github_connections migration file not found."

    with open(migration_file, "r") as f:
        content = f.read().lower()

    assert "create table public.github_connections" in content or "create table if not exists public.github_connections" in content
    assert "state_hash" in content
    assert "encrypted_pending_code" in content
    assert "encrypted_access_token" in content
    assert "github_login" in content
    assert "github_user_id" in content
    assert "error_summary" in content
    assert "enable row level security" in content
    assert "create index" in content
    assert "task_id" in content
    assert "status" in content
