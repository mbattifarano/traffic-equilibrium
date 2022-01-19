import click
import os

from traffic_equilibrium.solver import Result


@click.command()
@click.argument('result_dir', type=click.Path(exists=True))
@click.option('--iterations', type=int, default=None)
@click.option('--line-search-tolerance', type=float, default=None)
def main(result_dir, iterations, line_search_tolerance):
    click.echo(f"Loading result from {result_dir}...")
    parent_dir, _ = os.path.split(result_dir.rstrip(os.sep))
    name = os.path.basename(parent_dir)
    result = Result.load(result_dir)
    click.echo(f"Loaded result for {name}. Improving...")
    result.improve(max_iterations=iterations, line_search_tolerance=line_search_tolerance)
    click.echo(f"Saving...")
    result.save(parent_dir)
    return 0


if __name__ == '__main__':
    main()
