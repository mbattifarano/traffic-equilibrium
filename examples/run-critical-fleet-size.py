import click
import cvxpy as cp
from traffic_equilibrium.solver import Result, Problem
from traffic_equilibrium.projected_gradient import ProjectedGradientResult
from traffic_equilibrium.link_cost import LinkCostBPR, LinkCostLinear
from traffic_equilibrium.vector import Vector
from traffic_equilibrium.critical_fleet_size import (
    prepare_critical_fleet_size,
    critical_fleet_size_mip,
    critical_fleet_size_lp,
    check_fleet_paths,
)
import numpy as np
import scipy.sparse

import pickle

from scipy.sparse import hstack, diags
import time


# one hour in seconds
HOUR = 60 * 60


@click.command()
@click.argument('result_path', type=click.Path(exists=True))
@click.option('-o', '--outfile')
@click.option('--result-kind', default='fw',
              type=click.Choice(['fw', 'pg'], case_sensitive=False))
@click.option('--epsilon-user', type=float, default=0.01)
@click.option('--epsilon-fleet', type=float, default=0.01)
@click.option('--epsilon-fleet-marginal-cost', type=float, default=None)
@click.option('--max-paths-per-trip', type=int, default=100)
@click.option('--mip/--no-mip', default=False)
@click.option('--mcr/--no-mcr', default=True)
@click.option('--cfs-lp/--no-cfs-lp', default=True)
@click.option('--abandon/--no-abandon', default=True)
@click.option('--col-gen/--no-col-gen', default=True)
@click.option('--cfs-so/--no-cfs-so', default=True)
@click.option('--use-all-paths', is_flag=True)
@click.option('--beta', type=float, default=1.0, help="link flow regularization weight")
@click.option('--verbose', is_flag=True)
@click.option('--solver', type=click.Choice(cp.settings.SOLVERS, case_sensitive=False))
@click.option('--obj-lb', type=float, help="lower cutoff for MIP objective")
@click.option('--obj-ub', type=float, help="upper cutoff for MIP objective")
@click.option('--time-limit', type=float, help="time limit in hours for the MIP")
@click.option('--mip-solutions', type=int, help="terminate MIP when this many solutions are found")
@click.option('--link-error-as-constraint', is_flag=True, help="treat link error as a box constraint")
def critical_fleet_size(result_path, result_kind, outfile, epsilon_user, epsilon_fleet,
                        epsilon_fleet_marginal_cost,
                        max_paths_per_trip,
                        mip, mcr, cfs_lp, abandon, col_gen, cfs_so, use_all_paths, beta, verbose, solver,
                        obj_lb, obj_ub, time_limit, mip_solutions, link_error_as_constraint):
    click.echo(f"Running critical fleet size for {result_path}...")
    if epsilon_fleet_marginal_cost is None:
        epsilon_fleet_marginal_cost = epsilon_fleet
    if time_limit is not None and mip_solutions is not None:
        click.echo(
            "Cannot specify both `--time-limit` and `--mip-solutions`; setting time limit to None.")
        time_limit = None
    if cfs_so:
        print("SOLVING CFS-SO")
    else:
        print("SOLVING CFS-UE")
    if result_kind == 'fw':
        result = Result.load(result_path)
    else:
        result = ProjectedGradientResult.load(result_path)
    mlpf = result.problem.cost_fn
    # TODO clean this up
    if hasattr(mlpf, "alpha"):
        lpf = LinkCostBPR(mlpf.alpha, mlpf.beta,
                          mlpf.capacity, mlpf.free_flow_travel_time)
    elif hasattr(mlpf, "coefficients"):
        lpf = LinkCostLinear(mlpf.coefficients, mlpf.constants)
    else:
        raise Exception(f"Unrecognized marginal link cost function: {mlpf}")
    (lpm, tpm, fp, up, flow, flow_est, flow_eps, path_flow_est, grad, lmc, lc, tc, tmc) = prepare_critical_fleet_size(
        result.problem, result.paths, result.prev_flow, result.flow, lpf, mlpf,
        epsilon_user, epsilon_fleet, use_all_paths, max_paths_per_trip
    )
    demand = result.problem.demand.volumes.to_array()
    total_demand = demand.sum()
    print(f"USING SOLVER: {solver}")
    solver_opts = {
        cp.GUROBI: dict(
            Presolve=2,   # 2 = aggressive presolve
            #Aggregate=0,  # Don't aggregate during presolve
            NumericFocus=3,
            DualReductions=0,
            ScaleFlag=3,  # 2 = aggressive coefficient scaling
            # use the homogeneous barrier algorithm (suggeseted by guobi logs)
            BarHomogeneous=1,
            # Method=1,  # 1: dual simplex
            Quad=1,       # use quad precision
            # BarOrder=0,   # Speed up ordering
            BarQCPConvTol=1e-7,  # more accurate solutions
        )
    }
    if mcr:
        t0 = time.time()
        mcr_lp = critical_fleet_size_lp(
            flow,
            demand,
            epsilon_fleet,
            lpm, tpm,
            fp, up,
            lc.to_array(), grad.to_array(), lmc.to_array(),
            flow_eps,
            flow_est,
            min_control_ratio=True,
            beta=beta,
            grad_scale=1,
            flow_scale=1,
            grad_cutoff=0.0,
            link_error_as_constraint=link_error_as_constraint,
        )
        violated_constraints = mcr_lp.is_feasible(path_flow_est)
        if violated_constraints:
            raise Exception(
                f"SO path flow violates {len(violated_constraints)}: {violated_constraints}")
        else:
            print("MCR is feasible.")
        #solve_with_scaling(mcr_lp.problem, None, verbose=True)
        mcr_lp.solve(solver=solver, verbose=verbose,
                     **solver_opts.get(solver, {}))
        click.echo(
            f"Minimum Control Ratio solved in {time.time() - t0} seconds (beta={beta}): {100 * mcr_lp.fleet_fraction():0.4f}%.")
        click.echo(
            f"    ||link flow error||: {mcr_lp.link_flow_error() * mcr_lp.total_volume()} (gap={mcr_lp.link_flow_error()}) (total volume = {mcr_lp.total_volume()})")
        click.echo(
            f"    objective value: {mcr_lp.problem.value} ~= {mcr_lp.fleet_fraction() + beta * mcr_lp.link_flow_error()} ")
        mcr_fleet_volume = mcr_lp.fleet_path_flow.value.sum()
        if obj_lb is None or mcr_fleet_volume >= obj_lb:
            obj_lb = mcr_fleet_volume
            print(f"MIP lower bound = {obj_lb}")
        if outfile:
            with open(f"{outfile}.mcr.pkl", 'wb') as outfile_p:
                pickle.dump({
                    'result': str(result_path),
                    'fleet_flow': mcr_lp.fleet_path_flow.value,
                    'user_flow': mcr_lp.user_path_flow.value,
                    'fleet_paths': fp,
                    'user_paths': up,
                    'link_path': lpm,
                    'trip_path': tpm,
                    'flow': flow,
                    'flow_projected': flow_est,
                    'demand': demand,
                    'link_cost': lc.to_array(),
                    'link_marginal_cost': lmc.to_array(),
                    'link_cost_gradient': grad.to_array(),
                    'trip_cost': tc.to_array(),
                    'trip_marginal_cost': tmc.to_array(),
                    'fleet_marginal_path_cost': mcr_lp.fleet_marginal_path_cost.value,
                }, outfile_p)
    if cfs_lp:
        click.echo("Building critical fleet size LP...")
        done = False
        n_runs = 0
        check_duplicates = False
        t0 = time.time()
        initial_solve = False
        while not done:
            if check_duplicates:
                _lpm_n = diags(1 / lpm.sum(0).A.ravel())
                _lpm_check = lpm.T.dot(lpm).dot(_lpm_n)
                duplicates = _lpm_check.multiply(_lpm_check.T) == 1
                duplicates.setdiag(0, 0)
                duplicates.eliminate_zeros()
                if duplicates.nnz:
                    print(
                        f"Found {duplicates.nnz / 2} duplicate paths before run {n_runs}")
                    for i, j in zip(*duplicates.nonzero()):
                        print(
                            f"paths {i} and {j}: {lpm.getcol(i).indices}, {lpm.getcol(j).indices}")
                assert duplicates.nnz == 0
            lp = critical_fleet_size_lp(
                flow,
                demand,
                epsilon_fleet_marginal_cost,
                lpm, tpm,
                fp, up,
                lc.to_array(), grad.to_array(), lmc.to_array(),
                flow_eps,
                flow_est,
                beta=beta,
                grad_scale=1.0,
                flow_scale=1.0,
                grad_cutoff=0.0,
                link_error_as_constraint=link_error_as_constraint,
                trip_cost=tc.to_array(),
            )
            if initial_solve:
                initial_solve = False
                violated_constraints = lp.is_feasible(path_flow_est)
                if violated_constraints:
                    raise Exception(
                        f"SO path flow violates {len(violated_constraints)}: {violated_constraints}")
                else:
                    print("CFS LP is feasible.")
            click.echo("Solving problem...")
            #solve_with_scaling(lp.problem, None, verbose=True)
            lp.solve(solver=solver, verbose=verbose,
                     **solver_opts.get(solver, {}))
            status = lp.problem.status
            print(f"Solver returned status {status}")
            n_runs += 1
            click.echo(
                f"Critical fleet size: {100 * lp.fleet_fraction():0.4f}%.")
            if col_gen:
                new_paths, link_path_extra, trip_path_extra = check_fleet_paths(
                    result.problem, result.paths, Vector.copy_of(flow),
                    Vector.copy_of(lp.fleet_link_flow.value),
                    lpf, lpm, lp.fleet_marginal_trip_cost.value
                )
                print(
                    f"Found {new_paths} new paths ({lpm.shape[1] + new_paths} total paths).")
                if new_paths:
                    # add new un-usable paths
                    lpm = hstack([lpm, link_path_extra])
                    _, n_paths = lpm.shape
                    tpm = hstack([tpm, trip_path_extra])
                    fp.resize(n_paths, refcheck=False)
                    up.resize(n_paths, refcheck=False)
                else:
                    #done = True
                    # if there are paths that are unused but could be, mark them as un-usable
                    unused_paths = ~ (lp.fleet_path_flow.value > 0.0)
                    n_eligible = (unused_paths & fp).sum()
                    if n_eligible and abandon:
                        print(f"Abandoning paths ({n_eligible} eligible)")
                        fp[unused_paths] = False
                    else:
                        done = True
            else:
                done = True
        click.echo(
            f"\nCritical fleet size lp upper bound solved in {time.time()-t0} seconds: {100 * lp.fleet_fraction():0.4f}%.\n")
        cfs_fleet_volume = lp.fleet_path_flow.value.sum()
        if obj_ub is None or cfs_fleet_volume <= obj_ub:
            obj_ub = cfs_fleet_volume
            print(f"mip upper bound = {obj_ub}")
        if outfile:
            with open(f"{outfile}.lp.pkl", 'wb') as outfile_p:
                pickle.dump({
                    'result': str(result_path),
                    'fleet_flow': lp.fleet_path_flow.value,
                    'user_flow': lp.user_path_flow.value,
                    'fleet_paths': fp,
                    'user_paths': up,
                    'link_path': lpm,
                    'trip_path': tpm,
                    'flow': flow,
                    'flow_projected': flow_est,
                    'demand': demand,
                    'link_cost': lc.to_array(),
                    'link_marginal_cost': lmc.to_array(),
                    'link_cost_gradient': grad.to_array(),
                    'trip_cost': tc.to_array(),
                    'trip_marginal_cost': tmc.to_array(),
                    'fleet_marginal_path_cost': lp.fleet_marginal_path_cost.value,
                }, outfile_p)
    # TODO: fix this
    if not cfs_so:
        obj_lb = cfs_ue(result, lpf, mlpf, epsilon_user, epsilon_fleet, use_all_paths, max_paths_per_trip,
               link_error_as_constraint, solver_opts, outfile, beta, solver, verbose, result_path)

    if mip:
        mip_solver_opts = {
            cp.GUROBI: {
                "MIPGapAbs": 0.01,
                # "MIPFocus": 1  # 1: feasible solutions
                # "Cutoff": (total_demand if not cfs_lp else lp.problem.value) * (1 + 1e-3),
            },
            cp.CPLEX: dict(cplex_params={
                "mip.display": 3,  # node log verbosity
                "mip.tolerances.absmipgap": 0.01 * total_demand,
                "mip.tolerances.uppercutoff": total_demand if not cfs_lp else lp.problem.value,
                # "emphasis.mip": 2, # 0: balanced, 1: feasibility, 2: optimality, 3: best bound
                # "mip.tolerances.mipgap": 0.01,
                # "mip.strategy.lbheur": True,
                # "mip.strategy.miqcpstrat": 1, # 0: default, 1: QCP relaxation, 2: LP relaxation
                # "mip.strategy.heuristiceffort": 3, # 1.0: default
                # "mip.strategy.rinsheur": 30, # run RINS every n nodes
                # "mip.strategy.probe": 3, # 3: most aggressive
                # Cuts
                # "mip.cuts.flowcovers": 2,
                # "mip.cuts.pathcut": 2,
                # "mip.cuts.mcfcut": 2,
                # "mip.cuts.liftproj": 3,
                # "mip.cuts.implied": 2,
                # other cuts
                # "mip.cuts.covers": 3,
                # "mip.cuts.cliques": 3,
                # "mip.cuts.disjunctive": 3, # most aggressive
                # "mip.cuts.localimplied": 3,
                # "mip.cuts.bqp": 3,
                # "mip.strategy.branch": 1,  # up branch
            })
        }
        if time_limit is not None:
            mip_solver_opts[cp.GUROBI]["TimeLimit"] = time_limit * HOUR
            mip_solver_opts[cp.CPLEX]["cplex_params"]["timelimit"] = time_limit * HOUR
        if mip_solutions is not None:
            mip_solver_opts[cp.GUROBI]["SolutionLimit"] = mip_solutions
            mip_solver_opts[cp.CPLEX]["mip.limits.solutions"] = mip_solutions
        (lpm, tpm, fp, up, flow, flow_est, flow_eps, path_flow_est, grad, lmc, lc, tc, tmc) = prepare_critical_fleet_size(
            result.problem, result.paths, result.prev_flow, result.flow, lpf, mlpf,
            epsilon_user, epsilon_fleet, use_all_paths, max_paths_per_trip, cfs_so
        )
        done = False
        t0 = time.time()
        while not done:
            problem_mip = critical_fleet_size_mip(
                flow,
                result.problem.demand.volumes.to_array(),
                epsilon_fleet,
                lpm, tpm,
                fp, up,
                lc.to_array(), grad.to_array(), lmc.to_array(),
                flow_eps,
                flow_est,
                beta,
                ub=obj_ub,
                lb=obj_lb,
                cfs_so=cfs_so,
            )
            #violated_constraints = problem_mip.is_feasible(path_flow_est)
            # if not violated_constraints:
            #    raise Exception(
            #        f"SO path flow violates {len(violated_constraints)}: {violated_constraints}")
            # else:
            #    print(f"CFS MIP is feasible (obj = {problem_mip.problem.objective.value})")
            click.echo("Solving MIP...")
            problem_mip.solve(solver=solver, verbose=verbose,
                              **mip_solver_opts.get(solver, {})
                              )
            new_paths, link_path_extra, trip_path_extra = check_fleet_paths(
                result.problem, result.paths, Vector.copy_of(flow),
                Vector.copy_of(problem_mip.fleet_link_flow.value),
                lpf, lpm, problem_mip.fleet_marginal_trip_cost.value,
                cfs_so=cfs_so
            )
            print(
                f"Found {new_paths} new paths ({lpm.shape[1] + new_paths} total paths).")
            click.echo(
                f"Critical fleet size (mip): {100 * problem_mip.fleet_fraction():0.4f}%.")
            if new_paths:
                # add new un-usable paths
                lpm = hstack([lpm, link_path_extra])
                _, n_paths = lpm.shape
                tpm = hstack([tpm, trip_path_extra])
                fp.resize(n_paths, refcheck=False)
                up.resize(n_paths, refcheck=False)
            else:
                done = True
        click.echo(
            f"\nCritical fleet size exact solved in {time.time() - t0} seconds: {100 * problem_mip.fleet_fraction():0.4f}%.\n")
        click.echo(f"fleet volume = {problem_mip.fleet_path_flow.value.sum()}")
        if outfile:
            if cfs_so:
                fname = f"{outfile}.mip.pkl"
            else:
                fname = f"{outfile}.cfs-ue.mip.pkl"
            with open(fname, 'wb') as outfile_p:
                pickle.dump({
                    'result': str(result_path),
                    'fleet_flow': problem_mip.fleet_path_flow.value,
                    'user_flow': problem_mip.user_path_flow.value,
                    'fleet_paths': fp,
                    'user_paths': up,
                    'link_path': lpm,
                    'trip_path': tpm,
                    'flow': flow,
                    'flow_projected': flow_est,
                    'demand': demand,
                    'link_cost': lc.to_array(),
                    'link_marginal_cost': lmc.to_array(),
                    'link_cost_gradient': grad.to_array(),
                    'trip_cost': tc.to_array(),
                    'trip_marginal_cost': tmc.to_array(),
                    'fleet_marginal_path_cost': problem_mip.fleet_marginal_path_cost.value,
                }, outfile_p)


def cfs_ue(result, lpf, mlpf, epsilon_user, epsilon_fleet, use_all_paths, max_paths_per_trip,
           link_error_as_constraint, solver_opts, outfile, beta, solver, verbose, result_path, col_gen=True):
    print("="*80)
    print("Building critical fleet size UE LP...")
    cfs_so = False
    (lpm, tpm, fp, up, flow, flow_est, flow_eps, path_flow_est, grad, lmc, lc, tc, tmc) = prepare_critical_fleet_size(
        result.problem, result.paths, result.prev_flow, result.flow, lpf, mlpf,
        epsilon_user, epsilon_fleet, use_all_paths, max_paths_per_trip, cfs_so)
    demand = result.problem.demand.volumes.to_array()
    print(f"n paths not shared: {(fp != up).sum()}")
    assert (fp == up).all()
    done = False
    n_runs = 0
    t0 = time.time()
    while not done:
        lp = critical_fleet_size_lp(
            flow,
            demand,
            epsilon_fleet,
            lpm, tpm,
            fp, up,
            lc.to_array(), grad.to_array(), lmc.to_array(),
            flow_eps,
            flow_est,
            beta=beta,
            min_control_ratio=False,
            grad_scale=1.0,
            flow_scale=1.0,
            grad_cutoff=0.0,
            link_error_as_constraint=link_error_as_constraint,
            cfs_so=cfs_so,
            trip_cost=tc.to_array(),
        )
        #violated_constraints = lp.is_feasible_ue(path_flow_est)
        #print(violated_constraints)
        #for i, c in enumerate(lp.problem.constraints):
        #    print(i, c.value(), c.violation().mean())
        #if violated_constraints:
        #    raise Exception(
        #        f"SO path flow violates {len(violated_constraints)} constraints: {violated_constraints}")
        #else:
        #    print(
        #        f"CFS UE LP is feasible objective = {lp.problem.objective.value}")
        click.echo("Solving problem...")
        #solve_with_scaling(lp.problem, None, verbose=True)
        lp.solve(solver=solver, verbose=verbose,
                 **solver_opts.get(solver, {}))
        status = lp.problem.status
        print(f"Solver returned status {status}")
        n_runs += 1
        click.echo(
            f"Critical fleet size UE: {100 * lp.fleet_fraction():0.4f}%.")
        if col_gen:
            new_paths, link_path_extra, trip_path_extra = check_fleet_paths(
                result.problem, result.paths, Vector.copy_of(flow),
                Vector.copy_of(lp.fleet_link_flow.value),
                lpf, lpm, lp.fleet_marginal_trip_cost.value,
                cfs_so=False,
            )
            print(
                f"Found {new_paths} new paths ({lpm.shape[1] + new_paths} total paths).")
            if new_paths:
                # add new un-usable paths
                lpm = hstack([lpm, link_path_extra])
                _, n_paths = lpm.shape
                tpm = hstack([tpm, trip_path_extra])
                fp.resize(n_paths, refcheck=False)
                up.resize(n_paths, refcheck=False)
            else:
                #done = True
                # if there are paths that are unused but could be, mark them as un-usable
                unused_paths = ~ (lp.fleet_path_flow.value > 0.0)
                n_eligible = (unused_paths & fp).sum()
                if n_eligible:
                    print(f"Abandoning paths ({n_eligible} eligible)")
                    fp[unused_paths] = False
                else:
                    done = True
        else:
            done = True
    click.echo(
        f"\nCFS-UE lower bound solved in {time.time()-t0} seconds: {100 * lp.fleet_fraction():0.4f}%.\n")
    cfs_fleet_volume = lp.fleet_path_flow.value.sum()
    if outfile:
        with open(f"{outfile}.cfs-ue.lp.pkl", 'wb') as outfile_p:
            pickle.dump({
                'result': str(result_path),
                'fleet_flow': lp.fleet_path_flow.value,
                'user_flow': lp.user_path_flow.value,
                'fleet_paths': fp,
                'user_paths': up,
                'link_path': lpm,
                'trip_path': tpm,
                'flow': flow,
                'flow_projected': flow_est,
                'demand': demand,
                'link_cost': lc.to_array(),
                'link_marginal_cost': lmc.to_array(),
                'link_cost_gradient': grad.to_array(),
                'trip_cost': tc.to_array(),
                'trip_marginal_cost': tmc.to_array(),
                'fleet_marginal_path_cost': lp.fleet_marginal_path_cost.value,
            }, outfile_p)
    return cfs_fleet_volume


def solve_with_scaling(problem, magnitude_bounds=(-4, 6), solver=cp.GUROBI, verbose=False,
                       **kwargs):
    data, chain, inverse_data = problem.get_problem_data(solver,
                                                         verbose=verbose)
    print("data:")
    print(data)
    if magnitude_bounds is None:
        scale_matrix = scipy.sparse.identity(len(data["c"]))
    else:
        scale_factors = get_scale_factors(data["A"], *magnitude_bounds)
        scale_matrix = scipy.sparse.diags(scale_factors)
    print(
        f"Scaling problem with factors in [{scale_matrix.data.min()}, {scale_matrix.data.max()}]")
    data["A"] = data["A"] @ scale_matrix
    data["c"] = data["c"] @ scale_matrix
    solution = chain.solve_via_data(problem, data, verbose=verbose,
                                    solver_opts=kwargs)
    print(scale_matrix.shape, solution["primal"].shape)
    #solution["primal"] = scale_matrix.dot(solution["primal"])
    problem.unpack_results(solution, chain, inverse_data)
    return problem.value


def get_scale_factors(matrix, min_magnitude=-4, max_magnitude=6):
    _, n_cols = matrix.shape
    factors = np.ones(n_cols, dtype=np.float)
    for i in range(n_cols):
        col = matrix.getcol(i)
        magnitudes = np.log10(abs(col[col != 0.0]))
        _min = magnitudes.min()
        _max = magnitudes.max()
        if _min < min_magnitude:
            factors[i] = 10**(min_magnitude - _min)
        elif _max > max_magnitude:
            factors[i] = 10**(max_magnitude - _max)
    return factors


if __name__ == '__main__':
    critical_fleet_size()
