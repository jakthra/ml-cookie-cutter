from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union

import typer
from rich import print
from rich.panel import Panel

if TYPE_CHECKING:
    from ml_cookie_cutter.projects import Project


app = typer.Typer()


def get_project_by_name(project: str):
    from ml_cookie_cutter.projects import projects

    _project: Optional[Project] = next((p for p in projects if p == project), None)

    if _project is None:
        typer.echo(f"Project: {_project} not found!")
        raise typer.Exit(1)

    return _project


def print_datasets_for_project(project: Union[str, Project]) -> Optional[list[str]]:
    """Get datasets for a project"""
    if isinstance(project, str):
        _project = get_project_by_name(project)
    else:
        _project = project
    datasets = _project.get_datasets()

    if not datasets:
        typer.echo(
            f"No datasets found for project: {project}, materialize data first. Use `ml materialize-project {project}`"
        )
        raise typer.Exit(1)

    typer.echo(
        f"""Available datasets:
        {(', ').join([str(path) for path in datasets])}"""
    )


@app.command(help="Materialize data for a project")
def materialize_project(project: str, materialization: str = "parquet"):
    """Materialize a project"""
    from ml_cookie_cutter.orchestration import materialize_timeseries_data_assets

    _project = get_project_by_name(project)

    typer.echo(f"Materializing data for project: {_project} ({materialization})")
    materialize_timeseries_data_assets(io_manager="local_polars_parquet_io_manager")

    typer.echo("Datasets materialized successfully!")
    print_datasets_for_project(_project)


@app.command("datasets", help="List datasets for a project")
def list_datasets(project: str):
    """List datasets for a project"""
    print_datasets_for_project(project)


@app.command("list", help="List projects")
def list_projects():
    """List projects"""
    from ml_cookie_cutter.projects import projects
    projects = [project for project in projects]
    strings = []
    for index, project in enumerate(projects):
        description = f"({project.description})" if project.description else ""
        strings.append(f"[{index}] {project.name} {description}")
    strings_formatted = '\n'.join(strings)
    print(Panel.fit(
        f"""{strings_formatted}"""
    , title="Projects"))


if __name__ == "__main__":
    list_datasets("timeseries")
