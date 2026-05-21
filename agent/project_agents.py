from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ProjectAgent:
    key: str
    name: str
    responsibility: str


PROJECT_AGENTS: tuple[ProjectAgent, ...] = (
    ProjectAgent(
        "product_strategist",
        "Product Strategist Agent",
        "Expands the idea into requirements, personas, features, and success criteria.",
    ),
    ProjectAgent(
        "research_rag",
        "Research/RAG Agent",
        "Retrieves reference URLs, uploaded files, repository context, public docs, and project memory.",
    ),
    ProjectAgent(
        "stack_selector",
        "Stack Selector Agent",
        "Recommends a project-specific tech stack from idea, depth, platform, features, and hackathon rules.",
    ),
    ProjectAgent(
        "system_architect",
        "System Architect Agent",
        "Designs architecture, folder structure, integrations, and module boundaries using the recommended stack.",
    ),
    ProjectAgent(
        "data_api",
        "Data/API Agent",
        "Designs data models, API routes, validation, storage, auth, and service boundaries.",
    ),
    ProjectAgent(
        "frontend",
        "Frontend Agent",
        "Designs pages, components, UI flows, state handling, and user experience.",
    ),
    ProjectAgent(
        "backend",
        "Backend Agent",
        "Generates server logic, business logic, integrations, database access, and APIs.",
    ),
    ProjectAgent(
        "qa",
        "QA Agent",
        "Creates validation checks, missing-feature audits, test strategy, and build verification.",
    ),
    ProjectAgent(
        "documentation",
        "Documentation Agent",
        "Generates README, setup, env guide, architecture notes, and usage examples.",
    ),
    ProjectAgent(
        "github",
        "GitHub Agent",
        "Creates or updates a repository, commits generated files, and reports export status.",
    ),
    ProjectAgent(
        "logger",
        "Logger Agent",
        "Streams progress to the frontend and stores orchestration logs.",
    ),
)


def project_agent_manifest() -> list[dict[str, Any]]:
    return [
        {
            "key": agent.key,
            "name": agent.name,
            "responsibility": agent.responsibility,
        }
        for agent in PROJECT_AGENTS
    ]


def agent_name(agent_key: str) -> str:
    for agent in PROJECT_AGENTS:
        if agent.key == agent_key:
            return agent.name
    return agent_key.replace("_", " ").title()
