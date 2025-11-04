import random
from scipy.optimize import minimize

class SkipParObjectiveMinimizer:
    def __init__(self, num_layers, t_comm_g_c, t_upd_c, t_comm_c_g, t_fwd, t_bwd, stale_array = None, do_filter = True):
        self.N = num_layers
        self.t_comm_g_c = t_comm_g_c
        self.t_upd_c = t_upd_c
        self.t_comm_c_g = t_comm_c_g
        self.t_fwd = t_fwd
        self.t_bwd = t_bwd
        self.stale_array = stale_array
        self.do_filter = do_filter
        self.constraints = [
            {'type': 'ineq', 'fun': lambda x: x[1] - x[0]-1},
            {'type': 'ineq', 'fun': lambda x: self.N - x[0]},
            {'type': 'ineq', 'fun': lambda x: self.N - x[1]},
            {'type': 'ineq', 'fun': lambda x: x[0]},
            {'type': 'ineq', 'fun': lambda x: x[1]}
        ]
        
    def pipe(self, n):
        return self.t_comm_g_c + max((n-1)*self.t_comm_g_c, n*self.t_upd_c, (n-1)*self.t_comm_c_g) + self.t_comm_c_g
    
    def objective_function(self, x):
        disp_eq = max(0, self.pipe(x[1] - x[0]) - x[0]*self.t_fwd)
        gap_eq = self.pipe(x[0]) + max((self.N - x[0])*self.t_bwd, self.pipe(self.N - x[1])) - self.N*self.t_bwd

        #return self.pipe(x[0]) + max((self.N - x[0])*self.t_bwd, self.pipe(self.N - x[1])) - self.N*self.t_bwd + max(0, self.pipe(x[1] - x[0]) - x[0]*self.t_fwd)
        # return abs(disp_eq - gap_eq)
        return disp_eq + gap_eq

    def objective_function_with_filtering(self, x):
        assert self.stale_array is not None, "Stale array can not be None using filtering."
        start = int(x[0])
        end = int(x[1])
        disp_eq = max(0, self.pipe(x[1] - x[0] - sum(self.stale_array[start - 1: end - 1])) - ((x[0] - sum(self.stale_array[:start - 1]))*self.t_fwd))
        gap_eq = self.pipe(x[0]- sum(self.stale_array[:start - 1])) + max((self.N - x[0] - sum(self.stale_array[:start - 1]))*self.t_bwd, self.pipe(self.N - x[1] - sum(self.stale_array[(end - 1) - 1:]))) - self.N*self.t_bwd

        return disp_eq + gap_eq
    
    

    def optimize_objective(self):
        x0 = [8, 16]
        result = minimize(fun = self.objective_function_with_filtering if self.do_filter else self.objective_function, x0 = x0, constraints = self.constraints)
        return result

# lp_op = SkipParObjectiveMinimizer(
#     num_layers= 35, 
#     t_comm_g_c= 0.91 * 160.6/1000, 
#     t_upd_c= 0.91 * 450/1000,
#     t_comm_c_g= 0.91 * 248/1000,
#     t_fwd= 0.91 * 200/1000,
#     t_bwd= 0.91 * 400/1000
# )
if __name__ == "__main__":
    lp_op = SkipParObjectiveMinimizer(
        num_layers= 35, 
        t_comm_g_c= 2.77/35, 
        t_upd_c= 8.993/35,
        t_comm_c_g= 2.49/35,
        t_fwd= 1.344/35,
        t_bwd= 3.873/35,
        stale_array= [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        do_filter = True
    )
    
    print("done")
    res = lp_op.optimize_objective()
    print("res: ",res)

    # 
    #  gap: pipe(n - y) -> (max(N - X) tbwd, pipe * (N - y)) - N * tbwd
    # x, y = 29, 33
    # 
    # x = 6
    # y = 17
    # N = 35

    # x = 29
    # y = 33

    x = 17
    y = 25

    disp_value = max(0, lp_op.pipe(y - x) - x*lp_op.t_fwd)
    gap_value= lp_op.pipe(x) + max((lp_op.N - x)*lp_op.t_bwd, lp_op.pipe(lp_op.N - y)) - lp_op.N*lp_op.t_bwd

    #print("dispertion value:", disp_value)
    #print("gap value:", gap_value)
    #print("Found Minimum: ", disp_value + gap_value)



    # def lsp_transition_layer(
    #           self,
    #           N,
    #           t_bwd,
    #           t_offload,
    #           t_update,
    #           t_upload,
    #     ):
    #         return N - (-t_bwd + (t_offload + t_update + t_upload))/ max(t_offload, t_update, t_upload)

    # print(lsp_transition_layer(
        
    # ))
    # min_value = lp_op.pipe(x) + max((N - x)*lp_op.t_bwd, lp_op.pipe(N-y)) - lp_op.N*lp_op.t_bwd + max(0, lp_op.pipe(y-x) - x*lp_op.t_fwd)
    # print("The manual calculated minimum is: ", min_value)
    stale_arrays_test = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    ]

    for stale_array in stale_arrays_test:
        lp_op.do_filter = True
        lp_op.stale_array = stale_array
        print("With optimal solution:")
        res = lp_op.optimize_objective()
        print(res)
        x_start = res.x[0]
        y_start = res.x[1]  
        disp_eq = max(0, lp_op.pipe(y_start - x_start - sum(lp_op.stale_array[int(x_start) - 1: int(y_start) - 1])) - ((x_start - sum(lp_op.stale_array[:int(x_start) - 1]))*lp_op.t_fwd))
        gap_eq = lp_op.pipe(x_start- sum(lp_op.stale_array[:int(x_start) - 1])) + max((lp_op.N - x_start - sum(lp_op.stale_array[:int(x_start) - 1]))*lp_op.t_bwd, lp_op.pipe(lp_op.N - y_start - sum(lp_op.stale_array[(int(y_start) - 1) - 1:]))) - lp_op.N*lp_op.t_bwd
        print("Gap:", gap_eq)
        print("Disp:", disp_eq)
        print("Gap + Disp", disp_eq + gap_eq)
        print("----------\n")

        times_found = 0
        while times_found < 40:
            x_start = random.randint(1, 35)
            y_start = random.randint(1, 35)

            if x_start >= y_start:
                continue


            # Equation without filtering.
            # disp_value = max(0, lp_op.pipe(y_start - x_start) - x_start*lp_op.t_fwd)
            # gap_value= lp_op.pipe(x_start) + max((lp_op.N - x_start)*lp_op.t_bwd, lp_op.pipe(lp_op.N - y_start)) - lp_op.N*lp_op.t_bwd
            
            # Equation with filtering.
            disp_eq = max(0, lp_op.pipe(y_start - x_start - sum(lp_op.stale_array[x_start - 1: y_start - 1])) - ((x_start - sum(lp_op.stale_array[:x_start - 1]))*lp_op.t_fwd))
            gap_eq = lp_op.pipe(x_start- sum(lp_op.stale_array[:x_start - 1])) + max((lp_op.N - x_start - sum(lp_op.stale_array[:x_start - 1]))*lp_op.t_bwd, lp_op.pipe(lp_op.N - y_start - sum(lp_op.stale_array[(y_start - 1) - 1:]))) - lp_op.N*lp_op.t_bwd
            print(f"X: {x_start}, Y: {y_start}; Disp: {disp_eq}; Gap: {gap_eq} Minimized Value: {disp_eq + gap_eq}")
            times_found += 1

        # x_start = 24
        # y_start = 33  
    