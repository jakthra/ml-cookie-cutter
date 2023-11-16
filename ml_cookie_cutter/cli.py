from typing import Optional

import typer

from ml_cookie_cutter.data.constants import DATALAKE_DIRECTORY
from ml_cookie_cutter.orchestration import materialize_timeseries_data_assets, projects

app = typer.Typer()


@app.command(help="Materialize data for a project")
def materialize_project(project: str):
    """Materialize a project"""
    if project not in projects:
        typer.echo(f"Project: {project} not found!")
        raise typer.Exit(1)

    typer.echo(f"Materializing data for project: {project}")
    materialize_timeseries_data_assets()

    typer.echo("Datasets materialized successfully!")
    typer.echo(
        f"""Available datasets:
        {(', ').join([str(path) for path in (DATALAKE_DIRECTORY / project / 'dataset').rglob('*.parquet')])}"""
    )


@app.command(help="Status of projects")
def status(project: Optional[str] = None):
    """Status of projects"""
    # TODO: Imple,ent status for project
    if project is None:
        typer.echo("Status of all projects: ")
        raise typer.Exit(0)
    typer.echo(f"Status of project: {project}")


if __name__ == "__main__":
    materialize_project("timeseries")
