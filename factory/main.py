import typer
from dotenv import load_dotenv

load_dotenv()  # loads .env before any command runs

from factory.commands.status import cmd_status
from factory.commands.run import cmd_run
from factory.commands.review import cmd_review
from factory.commands.review_mr import cmd_review_mr
from factory.commands.refine import cmd_refine
from factory.commands.continue_mr import cmd_continue_mr
from factory.core.monitor import cmd_monitor
from factory.commands.plan import cmd_plan
from factory.commands.update_wiki import cmd_update_wiki
from factory.commands.fix_mr_pipeline import cmd_fix_mr_pipeline
from factory.commands.chat import cmd_po

app = typer.Typer(
    name="factory",
    help="SoftwareTeamFabrik — Autonomous AI-driven software development factory.",
    add_completion=False,
    no_args_is_help=True,
)

app.command("status", help="Show GitLab connection, active sprint, and agent roster.")(cmd_status)
app.command("run", help="Spawn an AI agent to analyse an issue and open a draft MR.")(cmd_run)
app.command("review", help="Generate a Sprint Review wiki page; --approve to advance to next sprint.")(cmd_review)
app.command("review-mr", help="Run the Code Reviewer agent against a merge request.")(cmd_review_mr)
app.command("refine", help="Run Developer agent on an MR to address review feedback, push changes to MR branch.")(cmd_refine)
app.command("continue-mr", help="Continue an incomplete implementation on an existing MR branch.")(cmd_continue_mr)
app.command("monitor", help="Start monitoring service for auto-review and refinement loops.")(cmd_monitor)
app.command("plan", help="Create or display a sprint plan with token-aware scheduling.")(cmd_plan)
app.command("update-wiki", help="Update wiki pages based on merge request changes.")(cmd_update_wiki)
app.command("fix-mr-pipeline", help="Act on failed pipelines for a merge request.")(cmd_fix_mr_pipeline)
app.command("po", help="Initiate a Product Owner (PO) agent discussion flow for an open issue.")(cmd_po)

if __name__ == "__main__":
    app()
