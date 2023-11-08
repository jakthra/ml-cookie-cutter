from typing import Optional
import typer


app = typer.Typer()


@app.command(help="Materialize data for a project")
def materialize(project: str):
    """Materialize a project"""
    # TODO: Implement materialization for project
    typer.echo(f"Materializing data for project: {project}")


@app.command(help="Status of projects")
def status(project: Optional[str] = None):
    """Status of projects"""
    # TODO: Imple,ent status for project
    if project is None:
        typer.echo("Status of all projects: ")
        raise typer.Exit(0)
    typer.echo(f"Status of project: {project}")
