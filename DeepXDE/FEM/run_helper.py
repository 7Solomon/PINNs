from FEM.output import evaluate_solution_at_points_on_rank_0, initialize_point_evaluation
from dolfinx import geometry,   fem
from dolfinx.fem import petsc
from petsc4py import PETSc
import numpy as np

def _evaluate_at_points_wrapper(comm, rank, solution_func, points, bb_tree, domain):
        """
        Handles evaluation for both mixed and simple (scalar/vector) elements.
        If the element is mixed, it evaluates each subspace and stacks the results.
        """
        is_mixed = solution_func.function_space.element.num_sub_elements > 0
        if not is_mixed:
            # Standard case for scalar or vector functions
            return evaluate_solution_at_points_on_rank_0(solution_func, points, bb_tree, domain, comm)

        # Handle mixed elements by evaluating each component
        all_sub_evals = []
        V = solution_func.function_space
        for i in range(V.element.num_sub_elements):
            # --- ROBUST WORKAROUND for collapse() bug ---
            # Manually create the sub-function.
            # 1. Get the collapsed subspace definition and the degree-of-freedom map.
            V_sub, dof_map = V.sub(i).collapse()
            # 2. Create a new, separate function on that subspace.
            sub_func = fem.Function(V_sub)
            # 3. Copy the relevant data from the main mixed function into the new sub-function.
            sub_func.x.array[:] = solution_func.x.array[dof_map]
            
            sub_eval_data = evaluate_solution_at_points_on_rank_0(sub_func, points, bb_tree, domain, comm)
            
            if rank == 0 and sub_eval_data is not None:
                # Ensure data is 2D for consistent horizontal stacking
                if sub_eval_data.ndim == 1:
                    sub_eval_data = sub_eval_data[:, np.newaxis]
                all_sub_evals.append(sub_eval_data)
        
        if rank == 0 and all_sub_evals:
            return np.hstack(all_sub_evals)
        return None

def execute_transient_simulation(
    domain,
    t_start: float,
    t_end: float,
    dt_initial: float,
    solver_function,
    problem_type: str,
    fem_states: dict,
    fem_constants: dict,
    evaluation_times: np.ndarray = None,
    evaluation_spatial_points_xy: np.ndarray = None,
    max_fem_steps_between_eval: int = 10000
):
    comm = domain.comm
    rank = comm.rank

    perform_eval, eval_points_3d, bb_tree = initialize_point_evaluation(
        domain, evaluation_spatial_points_xy, comm
    )

    # --- Time-stepping loop setup ---
    t_fem_current = t_start
    dt_const = fem_constants.get("dt")
    if dt_const is None:
        raise KeyError("fem_constants must contain a 'dt' dolfinx.fem.Constant.")
    dt_const.value = dt_initial

    # Generalize getting the solution variables
    uh = next((v for k, v in fem_states.items() if k.startswith("uh")), None)
    un = next((v for k, v in fem_states.items() if k.startswith("un")), None)
    if uh is None or un is None:
        raise KeyError("Could not find 'uh' and 'un' variables in fem_states.")

    all_evaluated_data_rank0 = []
    evaluation_times_sorted = np.sort(np.unique(evaluation_times)) if evaluation_times is not None else []
    eval_idx = 0

    # --- Initial evaluation at t_start ---
    if perform_eval and eval_idx < len(evaluation_times_sorted) and np.isclose(evaluation_times_sorted[eval_idx], t_start):
        if rank == 0: print(f"Evaluating at initial time t={t_start}")
        eval_data = _evaluate_at_points_wrapper(comm, rank, uh, eval_points_3d, bb_tree, domain)
        if rank == 0 and eval_data is not None: all_evaluated_data_rank0.append(eval_data)
        eval_idx += 1

    # --- Adaptive stepping parameters ---
    max_retries = 5
    min_dt = 1e-6

    while t_fem_current < t_end:
        target_time = evaluation_times_sorted[eval_idx] if eval_idx < len(evaluation_times_sorted) else t_end
        
        if t_fem_current + dt_const.value > target_time:
            dt_const.value = target_time - t_fem_current
        
        t_fem_current += dt_const.value

        if problem_type == "nonlinear":
            retries = 0
            converged = False
            while retries < max_retries and not converged:
                try:
                    # Attempt to solve with the current dt
                    num_its = solver_function(uh)
                    converged = True
                    if rank == 0 and num_its[0] > 10: # Optional: print if many iterations were needed
                        print(f"Newton solver took {num_its[0]} iterations.")
                except RuntimeError as e:
                    if "Newton solver did not converge" in str(e):
                        # Backtrack time and solution
                        t_fem_current -= dt_const.value
                        dt_const.value /= 2.0 # Reduce dt
                        uh.x.array[:] = un.x.array # Reset guess to previous state
                        t_fem_current += dt_const.value # Recalculate new time
                        retries += 1
                        if rank == 0:
                            print(f"WARNING: Convergence failed. Retrying with dt={dt_const.value:.2e}")
                        if dt_const.value < min_dt:
                            raise RuntimeError(f"Simulation failed: dt reduced below minimum ({min_dt}).") from e
                    else:
                        raise e # Re-raise other runtime errors
            
            if not converged:
                raise RuntimeError(f"Solver failed to converge after {max_retries} retries. Aborting.")
        else:
            solver_function() # For linear problems

        un.x.array[:] = uh.x.array

        if perform_eval and eval_idx < len(evaluation_times_sorted) and np.isclose(t_fem_current, evaluation_times_sorted[eval_idx]):
            if rank == 0: print(f"Evaluating at t={t_fem_current:.2f}")
            eval_data = _evaluate_at_points_wrapper(comm, rank, uh, eval_points_3d, bb_tree, domain)
            if rank == 0 and eval_data is not None: all_evaluated_data_rank0.append(eval_data)
            eval_idx += 1
        
        # Optional: Increase dt again if convergence was easy
        if problem_type == "nonlinear":
            dt_const.value = min(dt_const.value * 1.1, dt_initial)

    if rank == 0:
        print("Transient simulation finished.")
        if perform_eval:
            return uh, np.array(all_evaluated_data_rank0)
    
    return uh, None

