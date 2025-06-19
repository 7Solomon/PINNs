from FEM.output import evaluate_solution_at_points_on_rank_0, initialize_point_evaluation
from dolfinx import geometry
from dolfinx.fem import petsc
from petsc4py import PETSc
import numpy as np

def perform_fem_step(dt_const_fem, current_dt_value,
                      b_vec, compiled_L_form, compiled_a_form_matrix_part, bcs_list,
                      ksp_solver, uh_solution, un_previous_solution):
    pass
    

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
    uh = next((v for k, v in fem_states.items() if k.startswith("uh_")), None)
    un = next((v for k, v in fem_states.items() if k.startswith("un_")), None)
    if uh is None or un is None:
        raise KeyError("Could not find 'uh_' and 'un_' variables in fem_states.")

    all_evaluated_data_rank0 = []
    evaluation_times_sorted = np.sort(np.unique(evaluation_times)) if evaluation_times is not None else []
    eval_idx = 0

    # --- Initial evaluation at t_start ---
    if perform_eval and eval_idx < len(evaluation_times_sorted) and np.isclose(evaluation_times_sorted[eval_idx], t_start):
        if rank == 0: print(f"Evaluating at initial time t={t_start}")
        eval_data = evaluate_solution_at_points_on_rank_0(uh, eval_points_3d, bb_tree, domain, comm)
        if rank == 0: all_evaluated_data_rank0.append(eval_data)
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
                    solver_function(uh)
                    converged = True
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
            eval_data = evaluate_solution_at_points_on_rank_0(uh, eval_points_3d, bb_tree, domain, comm)
            if rank == 0: all_evaluated_data_rank0.append(eval_data)
            eval_idx += 1
        
        # Optional: Increase dt again if convergence was easy
        if problem_type == "nonlinear":
            dt_const.value = min(dt_const.value * 1.1, dt_initial)

    if rank == 0:
        print("Transient simulation finished.")
        if perform_eval:
            return uh, np.array(all_evaluated_data_rank0)
    
    return uh, None

