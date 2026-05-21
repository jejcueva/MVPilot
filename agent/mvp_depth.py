"""Backward-compatible imports for older tests and callers.

The active product language is project depth and full project generation.
"""

from __future__ import annotations

from agent.project_depth import (
    PROJECT_ARCHETYPES as VERTICAL_PACKS,
    build_deploy_artifact,
    classify_project_archetype,
    default_api_routes,
    default_user_flows,
    deploy_readme_section,
    enrich_project_requirements,
    primary_entity_for_archetype,
    project_collection_route,
    project_tabs,
    user_flow_checklist,
)


classify_vertical = classify_project_archetype
default_demo_path = default_user_flows
enrich_mvp_scope = enrich_project_requirements
primary_entity_for_vertical = primary_entity_for_archetype
pack_tabs = project_tabs
pack_collection_route = project_collection_route
demo_path_checklist = user_flow_checklist
