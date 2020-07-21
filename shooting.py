"""
Project          : Shooting Algorithm for partially-affine optimal control problems

Author           : João Miguel Machado, joint work with Maria Soledad Aronna

Date of creation : 11/06/2020

Purpose          : This is an implementation demo of the shooting algorithm for optimal control
                   problems that present controls appearing both linearly and non linearly.
                   We aim to solve Goddard's problem, retrieving Oberle results (in his 1990 paper)

 Comments        :  Is this the real live? Is this just fantasy? 
                    Caught in the landside, no escape from reality...
                    Open your eyes... look up to the skies and seeeeeeee
                    I'am just a poor boy, I need no sympathy
"""
import numpy as np 

# Make NumPy warnings errors. Without this, we can't catch overflow
# errors that can occur in the step() function, which might indicate a
# problem with the ODE.
#np.seterr(all='raise')

import symengine as sp
from symengine_function import gradient, lie_bracket, xp_bracket
from sympy.parsing.sympy_parser import parse_expr
from pygsl import odeiv
import sympy

class shooting:

    def __init__(self, states_list, arcs_list = None, lin_cont_list = None, nonlin_cont_list = None, running_cost = None, time_horizon = ['free', 'free']):
        valid_arcs = ['bang_plus', 'sing', 'bang_minus']
        
        if arcs_list != None: 
            self.arcs              = arcs_list
            self.N                 = len(arcs_list)
            self.singular_indexes  = [i for i,arc in enumerate(arcs_list) if arc == 'sing']
            for arc in arcs_list:
                if arc not in valid_arcs:
                    print('Invalid arc type given in arcs_list. Use arc_types in:\n ', valid_arcs)
                    return

        costates_list            = ''
        states_list_splited      = states_list.split(' ')         # String to parce the states given by user
        if lin_cont_list != None:
            lin_cont_list_splited    = lin_cont_list.split(' ')   # String to parce the linear controls given by user

        self.n                 = len(states_list.split(' '))
        self.n_p               = self.n
        self.time_horizon      = time_horizon
        
        
        # Defining the symbolic states, costates, controls and dynamics
        for state_str in states_list_splited:    
            costates_list  += 'p_'+state_str+' '

        if running_cost != None:
            states_list_splited.append('x'+str(len(states_list_splited)+1) )
            states_list = ' '.join(states_list_splited)
            self.n += 1
            self.running_cost  = parse_expr(running_cost)
        pass

        # Symbols for states and costates
        #       x,p are used in the computations of functions of interest
        #       ,e.g. Hamiltonian and singular controls
        self.x  = sp.Matrix([sp.symbols(states_list)]).T
        self.p  = sp.Matrix([sp.symbols(costates_list)]).T

        # Assemble symbolic variables for initial-final constraints
        initial_states_list      = ''                              
        initial_costates_list    = ''                              
        final_states_list        = ''                              
        final_costates_list      = ''
        
        for x in states_list_splited:
            if x != '':
                initial_states_list   += x+'_i'+' '
                final_states_list     += x+'_f'+' '
            pass
        pass 

        for p in costates_list.split(' '):
            if p != '':
                initial_costates_list += p+'_i'+' '
                final_costates_list   += p+'_f'+' '
            pass 
        pass

        # x_i,p_i for i=0,1 are the variables used in the IF constraints and lagrangian 
        self.x_0 = sp.Matrix([sp.var(initial_states_list)]).T
        self.x_1 = sp.Matrix([sp.var(final_states_list)]).T
        self.p_0 = sp.Matrix([sp.var(initial_costates_list)]).T
        self.p_1 = sp.Matrix([sp.var(final_costates_list)]).T


        # Assimple variables for problem TP, if we arcs_list != None
        if arcs_list != None:    
            switching_times_list     = ['T0']                          # Used to generate switching times
            co_switching_times_list  = ['p_T0']
            

            states_i_list            = ''                              # Used the generate TP states
            costates_i_list          = ''                              # Used to generate TP costates
            lin_cont_i_list          = ''                              # Used to generate TP lin controls                              

            switching_times_list_i    = []
            switching_times_list_f    = []
            co_switching_times_list_i = []
            co_switching_times_list_f = []

            initial_states_list_tp   = ''                           
            initial_costates_list_tp = ''                           
            final_states_list_tp     = ''                           
            final_costates_list_tp   = ''                           

            # Symbolic variables for (TP)
            for i in range(len(self.arcs)):
                # Assemble symbolic variables for states and costates
                for state_str in states_list_splited:
                    if state_str != '':
                        states_i_list += state_str+str(i)+' '
                        initial_states_list_tp += state_str+str(i)+'_i '
                        final_states_list_tp += state_str+str(i)+'_f '

                for costate_str in costates_list.split(' '):
                    if costate_str != '': 
                        costates_i_list  += costate_str+str(i)+' '
                        initial_costates_list_tp += costate_str+str(i)+'_i '
                        final_costates_list_tp += costate_str+str(i)+'_f '

                # Assemble symbolic variables for linear controls
                if arcs_list[i] == 'sing':
                    for lin_cont_str in lin_cont_list_splited:
                        lin_cont_i_list += lin_cont_str+str(i)+' '
                
                # Assemble symbolic variables for switching times
                switching_times_list.append('T'+str(i+1))
                co_switching_times_list.append('p_T'+str(i+1))
    
            # Checks if the initial and final times are free in the string time_horizon
            # If they are, we leave the optimization variables T0 and TN
            # Else, we remove these variables from the switching_times_list

            lim_0, lim_N = 0, self.N+1
            if self.time_horizon[0] != 'free': lim_0 = 1
            if self.time_horizon[1] != 'free': lim_N = self.N
            
            switching_times_list    = switching_times_list[lim_0:lim_N]
            co_switching_times_list = co_switching_times_list[lim_0:lim_N]

            for T, p in zip(switching_times_list, co_switching_times_list):
                switching_times_list_i.append(T)
                switching_times_list_f.append(T)
                co_switching_times_list_i.append(p+'_i')
                co_switching_times_list_f.append(p+'_f')


            self.n_T = len(switching_times_list)
            switching_times_list      = ' '.join(switching_times_list)
            co_switching_times_list   = ' '.join(co_switching_times_list)
            switching_times_list_i    = ' '.join(switching_times_list_i)
            switching_times_list_f    = ' '.join(switching_times_list_f)


            xp_states_list = states_i_list+' '+costates_i_list+' '+switching_times_list
            xp_i_list      = initial_states_list_tp+' '+initial_costates_list_tp+' '+switching_times_list_i 
            xp_f_list      = final_states_list_tp+' '+final_costates_list_tp+' '+switching_times_list_f
            
            # Symbols for controls
            self.lin_cont  = sp.Matrix([sp.symbols(lin_cont_list)])
            if nonlin_cont_list != None:
                self.nlin_cont = sp.Matrix([sp.symbols(nonlin_cont_list)])

            # xp and TP_lin_cont are the actual variables of the Transformed problem (TP)
            self.xp = sp.Matrix([sp.symbols(xp_states_list)]).T
            if len(self.singular_indexes) > 0: 
                self.TP_lin_cont = sp.Matrix([sp.var(lin_cont_i_list)]).T
    
        else:
            xp_states_list = states_list+' '+costates_list
            xp_i_list      = initial_states_list+' '+initial_costates_list 
            xp_f_list      = final_states_list+' '+final_costates_list
        pass # end if arcs_list != None

        # Generate symbolic variables for the Shooting Function
        #                                     and states/costates dynamics, for TP and unconstrained

        self.xp   = sp.Matrix([sp.symbols(xp_states_list)]).T  # state variables for the TPBVP
        self.xp_i = sp.Matrix([sp.var(xp_i_list)]).T           # initial variables for shooting function
        self.xp_f = sp.Matrix([sp.var(xp_f_list)]).T           # final variables for shooting function
        
        # Creates a vector of symbols and time_horizons
        if hasattr(self, 'N'):
            N, n, n_p = self.N, self.n, self.n_p
            if N > 1 :
                self.T   = sp.zeros(N+1,1)
                if self.time_horizon[0] != 'free':
                    self.T[0,0]     = self.time_horizon[0]
                    self.T[1:N,0]   = self.xp[N*(n+n_p):N*(n+n_p)+N,0]
                else:
                    self.T[0:N,0]   = self.xp[N*(n+n_p):N*(n+n_p) + N + 1,0]
                        
                if self.time_horizon[1] != 'free':
                    self.T[N,0]   = self.time_horizon[1]
                else:
                    ###############33
                    # BUG 
                    ###############
                    self.T[N,0]   = self.xp[N*(n+n_p)+N-1, 0]
    
    pass # end __init__

    def add_drift_dynamics(self, dynamics_str):
        k = 0
        self.f0 = sp.zeros(self.n, 1)
        for dyn_str in dynamics_str:
            self.f0[k] = parse_expr(dyn_str)
            k += 1

        if hasattr(self, 'running_cost') == True:
            expr = self.running_cost
            if hasattr(self, 'lin_cont') == True:
                self.f0[self.n - 1] = expr - sp.diff(expr, self.lin_cont[0])*self.lin_cont[0]
            else:
                self.f0[self.n - 1] = expr 
        pass
    
    def add_lin_dynamics(self, dynamics_str, bounds = [0.0, 1.0]):
        k = 0
        self.lin_bounds = bounds
        self.f1 = sp.zeros(self.n, 1)

        for dyn_str in dynamics_str:
            self.f1[k] = parse_expr(dyn_str)
            k += 1
        if hasattr(self, 'running_cost') == True:
            expr = self.running_cost
            self.f1[self.n - 1] = sp.diff(expr, self.lin_cont[0])
        pass
    
    def add_constrainst_IF(self, cost_function_str = None, constraints_IF_list = None):
        '''
        This function parses the IF constraints and defines the initial-final lagrangian
        '''
        # Assert if the user has given a cost function, either criterion on final states, or running cost
        assert(hasattr(self, 'running_cost') == True or cost_function_str != None)
        
        # And initialize the initial-final lagrangian
        self.lagrangian_IF    = sp.Matrix([0.0])

        if hasattr(self, 'running_cost') == True:                 # Check for running cost
            self.lagrangian_IF[0] += self.x_1[len(self.x_1) - 1]     # Final value of introduced state
        if cost_function_str != None:                             # Check for Maryer cost 
            self.lagrangian_IF[0] += parse_expr(cost_function_str)   # Parse cost function

        # Check for final constraints    
        if constraints_IF_list != None:
            self.dim_IF_constraints = len(constraints_IF_list)
            self.constraints_IF     = sp.zeros(self.dim_IF_constraints, 1)
        
            # Parse initial-final constraints
            for i in range(self.dim_IF_constraints):
                self.constraints_IF[i] = parse_expr(constraints_IF_list[i])

            # Generate symbolic variable for lagrange multipliers
            lagrange_mult_list = 'beta_0'
            for i in range(1, self.dim_IF_constraints):
                lagrange_mult_list += ' '+'beta_'+str(i)
            self.lagrange_mult = sp.Matrix([sp.var(lagrange_mult_list)])

            # Increment the lagrangian with the initial final constraints terms
            #print( (self.lagrange_mult).dot(self.constraints_IF) )
            self.lagrangian_IF += sp.Matrix([(self.lagrange_mult).dot(self.constraints_IF)])

        if hasattr(self, 'arcs') == True:
            # Substitute the initial-final states with the values in the lagrangian with the appropriate TP variables
            ini_final_subs_dic = dict(zip(self.x_0, self.xp_i[0:self.n, 0]))
            ini_final_subs_dic.update(dict(zip(self.x_1, self.xp_f[(self.N-1)*self.n:self.N*self.n, 0]) ))

            self.lagrangian_IF = self.lagrangian_IF.subs(ini_final_subs_dic)
        pass

    def parse_dict(self, dict_str):
        """
        Parse a dictionary of strings into sympy variables
        """
        dict_sympy = {}
        for key in dict_str:
            key_sympy  = parse_expr(key)
            hash_sympy = parse_expr(dict_str[key])
            dict_sympy.update({key_sympy: hash_sympy})

        return dict_sympy

    def singular_control_symbolic(self, nonlin_cont_subs_dict_str = None):
        if nonlin_cont_subs_dict_str != None:
            # Case the substitution dictionary is given by the user in a string format
            nonlin_cont_subs_dict = self.parse_dict(nonlin_cont_subs_dict_str)
        else:
            # Routine to compute solve the stationarity of Hamiltonian for nonlin controls and 
            # obtain the substitution dictionary
            pass 
        
        if hasattr(self, 'running_cost') == True:
            p = sp.zeros(self.n, 1)
            p[0:self.n_p, 0] = self.p
            p[self.n_p, 0] = 1.0
        else:
            p = self.p

        # Quantities that need to be computed before control substitution
        lief0f1          = lie_bracket(self.f0, self.f1, self.x)
        Dlief0f1         = lief0f1.jacobian(self.nlin_cont)
        lie_f0_f_0f_1    = lie_bracket(self.f0, lief0f1, self.x)
        lie_f1_f_0f_1    = lie_bracket(self.f1, lief0f1, self.x)
        self.Hamiltonian = p.dot(self.f0 + self.f1*self.lin_cont)
        self.dp          = -sp.Matrix([self.Hamiltonian]).jacobian(self.x[0:self.n_p,0])

        # Control substitution
        self.f0          = self.f0.subs(nonlin_cont_subs_dict)
        self.f1          = self.f1.subs(nonlin_cont_subs_dict)
        self.dp          = self.dp.subs(nonlin_cont_subs_dict)
        self.Hamiltonian = self.Hamiltonian.subs(nonlin_cont_subs_dict)
        
        lief0f1  = lief0f1.subs(nonlin_cont_subs_dict)
        Dlief0f1 = Dlief0f1.subs(nonlin_cont_subs_dict)
        lie_f0_f_0f_1 = lie_f0_f_0f_1.subs(nonlin_cont_subs_dict) 
        lie_f1_f_0f_1 = lie_f1_f_0f_1.subs(nonlin_cont_subs_dict)

        U_feedback = self.nlin_cont.subs(nonlin_cont_subs_dict)
        bra_U_f0   = xp_bracket(U_feedback, p.dot(self.f0), self.x, p)
        bra_U_f1   = xp_bracket(U_feedback, p.dot(self.f1), self.x, p)

        gamma10 = p.dot(lie_f0_f_0f_1 + Dlief0f1*bra_U_f0)
        gamma10 = sympy.simplify(gamma10)
        gamma11 = p.dot(lie_f1_f_0f_1 + Dlief0f1*bra_U_f1)
        gamma11 = sympy.simplify(gamma11)

        #print('gamma10            : ', gamma10)
        #print('gamma10 symplified : ', sympy.simplify(gamma10))

        #print('gamma11: ', gamma11)
        #print('gamma11 symplified : ', sympy.simplify(gamma11))
        
        self.delta_sing            = -(gamma10/gamma11)     # The symbolic expression for the linear control
        self.switching_function    = p.dot(self.f1)  # Symbolic expression for the switching function
        self.switching_function_dt = p.dot(lief0f1)  # Symbolic expression for the switching function's first time derivative
        pass
        
    def sing_cont_lin_only(self):
        """
        Function to compute symbolic singular controls for the totally affine case.
        """
        if hasattr(self, 'running_cost') == True:
            p = sp.zeros(self.n, 1)
            p[0:self.n_p, 0] = self.p
            p[self.n_p, 0]   = 1.0
        else:
            p = self.p

        # Quantities that need to be computed before control substitution
        lief0f1  = lie_bracket(self.f0, self.f1, self.x)
        self.Hamiltonian = p.dot(self.f0 + self.f1*self.lin_cont)
        self.dp  = -sp.Matrix([self.Hamiltonian]).jacobian(self.x[0:self.n_p, 0])
    
        lie_f0_f_0f_1 = lie_bracket(self.f0, lief0f1, self.x)
        lie_f1_f_0f_1 = lie_bracket(self.f1, lief0f1, self.x)

        gamma10 = p.dot(lie_f0_f_0f_1)
        gamma11 = p.dot(lie_f1_f_0f_1)
        
        self.delta_sing            = -gamma10/gamma11     # The symbolic expression for the linear control
        self.switching_function    = p.dot(self.f1)  # Symbolic expression for the switching function
        self.switching_function_dt = p.dot(lief0f1)  # Symbolic expression for the switching function's first time derivative
        pass

    def symbolic_shooting_function_unconstrained(self):
        assert(hasattr(self, 'N')==False)
        n, n_p   = self.n, self.n_p

        # Check if initial-final constraints were added to the Optimal Control Problem
        if hasattr(self, 'dim_IF_constraints') == True:
            dim_IF_constraints = self.dim_IF_constraints
        else:
            dim_IF_constraints = 0

        # Dimension of the Shooting function accounts for, respectively
        #   dim_IF_constraints : Number of initial-final constraints (if there are any)
        #   n_p                : Transversality conditions, final conditions for costates
        #   2*len(self.singular_indexes) : switching functions and switching function derivative at singular arcs
        
        dim_shooting = dim_IF_constraints + n_p 
        if hasattr(self, 'delta_sing'):
            dim_shooting += 2*len(self.singular_indexes)
        if hasattr(self, 'dim_IF_constraints'):
            # Add also the dimension for transversality condition in initial values of costates, if applicable
            dim_shooting += n
        
        Shooting_function = sympy.zeros(dim_shooting, 1)
        index             = 0 # Index we start to fill the Shooting function

        # Check if initial-final constraints were added to the Optimal Control Problem
        if hasattr(self, 'dim_IF_constraints') == True:
            Shooting_function[0:dim_IF_constraints,0] = self.constraints_IF
            index = dim_IF_constraints
        else:
            print('No initial-final constraints added to the Shooting Function')

        # When there are no initial constraints, the initial Transversality conditions are redundant with the lagrange multipliers, 
        # We get trivially p(0) = -beta, or p(0) = 0
        if hasattr(self, 'dim_IF_constraints') == True:
            # Transversality conditions
            #      initial conditions on costates
            Shooting_function[index:index+n,0] = self.xp_i[n:2*n, 0] + self.lagrangian_IF.jacobian(self.xp_i[0:n, 0]).T
            index += n

        # Transversality conditions
        #      final conditions on costates
        Shooting_function[index:index+n_p,0] = self.xp_f[n:n+n_p, 0] - self.lagrangian_IF.jacobian(self.xp_f[0:n_p, 0]).T
        index += n_p

        # Hamiltonian derivatives of order 0 and 1
        # Note that it is possible we include conditions on the shooting function that are redundant to the previous entries,
        # This will lead to linearly dependent rows in the derivative of the shooting function. 
        # We will check if any new entry is not a multiple of the previous.
        
        if hasattr(self, 'singular_indexes'):
            for i in self.singular_indexes:
                subs_dic_initial = dict(zip(self.x, self.xp_f[0:n, 0] ))
                subs_dic_initial.update(dict(zip(self.p, self.xp_f[n:n+n_p, 0]) ))

                expr = self.switching_function.subs(subs_dic_initial)

                # First we check if the expression is a multiple of any other previous entry
                bool_is_constant_list = [(Shooting_function[k]/expr).is_constant() for k in range(index)]
                if not any(bool_is_constant_list):
                    Shooting_function[index, 0] = expr
                    index += 1
                    
                expr = self.switching_function_dt.subs(subs_dic_initial)
                bool_is_constant_list = [(Shooting_function[k]/expr).is_constant() for k in range(index)]
                if not any(bool_is_constant_list):
                    Shooting_function[index, 0] = expr
                    index += 1
            pass # end loop over singular indexes
        pass # end check for singular indexes
            
        Shooting_function = sp.Matrix(Shooting_function[0:index])
        return Shooting_function

    def symbolic_shooting_function(self):
        """
        Assemble shooting function with symbolic variables.
            OBS: use self.xp_i and self.xp_f as defined outside the function
        """
        n, n_p, n_T, N = self.n, self.n_p, self.n_T, self.N
        
        # Check if initial-final constraints were added to the Optimal Control Problem
        if hasattr(self, 'dim_IF_constraints') == True:
            dim_IF_constraints = self.dim_IF_constraints
        else:
            dim_IF_constraints = 0

        # Dictionary to evaluate the states and costates in initial and final conditions
        subs_dic_final = dict(zip(self.x, self.xp_f[N*n:(N+1)*n, 0] ))
        subs_dic_final.update(dict(zip(self.p, self.xp_f[N*n+(N-1)*n_p:N*(n+n_p), 0]) ))

    
        # Dictionary for Hamiltonian in each arc 
        Hamiltonian_dict  = {'bang_plus': self.Hamiltonian.subs(self.lin_cont[0], self.lin_bounds[1]),
                             'bang_minus': self.Hamiltonian.subs(self.lin_cont[0], self.lin_bounds[0]),
                             'sing':       self.Hamiltonian.subs(self.lin_cont[0], self.delta_sing)}
        

        # Dimension of the Shooting function accounts for, respectively
        #   dim_IF_constraints : Number of initial-final constraints (if there are any)
        #   (N-1)*(n+n_p)      : Continuity of the states and costates in TP formulation
        #   n                  : Transversality conditions, final conditions for costates
        #   n_T                : Transversality conditions of costates of switching times, equivalent to continuity of the Hamiltonian in TP formulation
        #   2*len(self.singular_indexes) : switching functions and switching function derivative at singular arcs
        
        dim_shooting      = dim_IF_constraints + (N-1)*(n+n_p) + n + n_T + 2*len(self.singular_indexes)
        if hasattr(self, 'dim_IF_constraints'):
            ########################################################
            # BUG: need to check if the IF_constraints depend on p_0
            ########################################################
            # Add also the dimension for transversality condition in initial for, if applicable
            dim_shooting += n_p
        
        Shooting_function = sympy.zeros(dim_shooting, 1)
        index             = 0 # Index we start to fill the Shooting function

        # Check if initial-final constraints were added to the Optimal Control Problem
        if hasattr(self, 'dim_IF_constraints') == True:
            Shooting_function[0:dim_IF_constraints,0] = self.constraints_IF
            index = dim_IF_constraints
        else:
            print('No initial-final constraints added to the Shooting Function')

        # continuity of the states condition
        Shooting_function[index:index+(N-1)*n,0] = self.xp_f[0:(N-1)*n,0] - self.xp_i[n:N*n,0]
        index += (N-1)*n

        # continuity of the costates condition
        Shooting_function[index:index+(N-1)*n_p,0] = self.xp_f[N*n: N*n+(N-1)*n_p,0] - self.xp_i[N*n + n_p:N*(n+n_p),0]
        index += (N-1)*n_p 

        # When there are no initial constraints, the initial Transversality conditions are redundant with the lagrange multipliers, 
        # We get trivially p(0) = -beta, or p(0) = 0
        if hasattr(self, 'dim_IF_constraints') == True:
            # Transversality conditions
            #      initial conditions on costates
            Shooting_function[index:index+n_p,0] = self.xp_i[N*n:N*n+n_p, 0] + self.lagrangian_IF.jacobian(self.xp_i[0:n_p, 0]).T
            index += n_p

        # Transversality conditions
        #      final conditions on costates
        Shooting_function[index:index+n_p,0] = self.xp_f[N*n+(N-1)*n_p:N*(n+n_p), 0] - self.lagrangian_IF.jacobian(self.xp_f[(N-1)*n:(N-1)*n+n_p, 0]).T
        index += n_p

        # Continuity of the Hamiltonian condition
        for i in range(len(self.arcs)-1):
            subs_dic_final = dict(zip(self.x, self.xp_f[i*n:(i+1)*n, 0] ))
            subs_dic_final.update(dict(zip(self.p, self.xp_f[N*n+i*n_p:N*n+(i+1)*n_p, 0]) ))
            # Gets Hamiltonian corresponding to the right arc and substitutes the appropriate final values
            H_i_final = Hamiltonian_dict[self.arcs[i]].subs(subs_dic_final)

            subs_dic_initial = dict(zip(self.x, self.xp_i[(i+1)*n:(i+2)*n, 0] ))
            subs_dic_initial.update(dict(zip(self.p, self.xp_i[N*n+(i+1)*n_p:N*n+(i+2)*n_p, 0]) ))
            # Gets Hamiltonian corresponding to the right arc and substitutes the appropriate initial values
            H_ipp_initial = Hamiltonian_dict[self.arcs[i+1]].subs(subs_dic_initial)

            Shooting_function[index, 0] = H_ipp_initial - H_i_final
            index += 1
   
        # Hamiltonian derivatives of order 0 and 1
        # Note that it is possible we include conditions on the shooting function that are redundant to the previous entries,
        # This will lead to linearly dependent rows in the derivative of the shooting function. 
        # We will check if any new entry is not a multiple of the previous.
        for i in self.singular_indexes:

            subs_dic_final = dict(zip(self.x, self.xp_i[i*n:(i+1)*n, 0] ))
            subs_dic_final.update(dict(zip(self.p, self.xp_i[N*n+(i)*n_p:N*n+(i+1)*n_p, 0]) ))

            subs_dic_final = dict(zip(self.x, self.xp_f[i*n:(i+1)*n, 0] ))
            subs_dic_final.update(dict(zip(self.p, self.xp_f[N*n+(i)*n_p:N*n+(i+1)*n_p, 0]) ))

            expr = (self.T[i+1] - self.T[i])*self.switching_function.subs(subs_dic_final)
            #expr = self.switching_function.subs(subs_dic_final)
            # First we check if the expression is a multiple of any other previous entry
            bool_is_constant_list = [(Shooting_function[k]/expr).is_constant() for k in range(index)]
            if not any(bool_is_constant_list):
                Shooting_function[index, 0] = expr
                index += 1
                
            expr = (self.T[i+1] - self.T[i])*self.switching_function_dt.subs(subs_dic_final)
            #expr = self.switching_function_dt.subs(subs_dic_final)
            bool_is_constant_list = [(Shooting_function[k]/expr).is_constant() for k in range(index)]
            if not any(bool_is_constant_list):
                Shooting_function[index, 0] = expr
                index += 1
        
        Shooting_function = sp.Matrix(Shooting_function[0:index])
        return Shooting_function

    def symbolic_shooting_function_fast(self):
        """
        Assemble shooting function with symbolic variables.
            OBS: use self.xp_i and self.xp_f as defined outside the function
        """
        n, n_p, n_T, N = self.n, self.n_p, self.n_T, self.N
        
        # Check if initial-final constraints were added to the Optimal Control Problem
        if hasattr(self, 'dim_IF_constraints') == True:
            dim_IF_constraints = self.dim_IF_constraints
        else:
            dim_IF_constraints = 0

        # Dictionary to evaluate the states and costates in initial and final conditions
        subs_dic_final = dict(zip(self.x, self.xp_f[N*n:(N+1)*n, 0] ))
        subs_dic_final.update(dict(zip(self.p, self.xp_f[N*n+(N-1)*n_p:N*(n+n_p), 0]) ))

    
        # Dictionary for Hamiltonian in each arc 
        Hamiltonian_dict  = {'bang_plus': self.Hamiltonian.subs(self.lin_cont[0], self.lin_bounds[1]),
                             'bang_minus': self.Hamiltonian.subs(self.lin_cont[0], self.lin_bounds[0]),
                             'sing':       self.Hamiltonian.subs(self.lin_cont[0], self.delta_sing)}
        

        # Dimension of the Shooting function accounts for, respectively
        #   dim_IF_constraints : Number of initial-final constraints (if there are any)
        #   (N-1)*(n+n_p)      : Continuity of the states and costates in TP formulation
        #   n                  : Transversality conditions, final conditions for costates
        #   n_T                : Transversality conditions of costates of switching times, equivalent to continuity of the Hamiltonian in TP formulation
        #   2*len(self.singular_indexes) : switching functions and switching function derivative at singular arcs
        
        dim_shooting = dim_IF_constraints + (N-1)*(n+n_p) + n + n_T + 2*len(self.singular_indexes)
        if hasattr(self, 'dim_IF_constraints'):
            # Add also the dimension for transversality condition in initial for, if applicable
            ####################
            # BUG 
            # Applicable if there are initial conditions of the states
            # in the constraints function
            ####################
            dim_shooting += n_p
        
        Shooting_function = sympy.zeros(dim_shooting, 1)
        index             = 0 # Index we start to fill the Shooting function

        # Check if initial-final constraints were added to the Optimal Control Problem
        if hasattr(self, 'dim_IF_constraints') == True:
            ini_final_subs_dic = dict(zip(self.x_0, self.xp_i[0:self.n, 0]))
            ini_final_subs_dic.update(dict(zip(self.x_1, self.xp_f[(self.N-1)*self.n:self.N*self.n, 0]) ))
            Shooting_function[0:dim_IF_constraints,0] = self.constraints_IF.subs(ini_final_subs_dic)
            index = dim_IF_constraints
        else:
            print('No initial-final constraints added to the Shooting Function')

        # continuity of the states condition
        Shooting_function[index:index+(N-1)*n,0] = self.xp_f[0:(N-1)*n,0] - self.xp_i[n:N*n,0]
        index += (N-1)*n

        # continuity of the costates condition
        Shooting_function[index:index+(N-1)*n_p,0] = self.xp_f[N*n: N*n+(N-1)*n_p,0] - self.xp_i[N*n + n_p:N*(n+n_p),0]
        index += (N-1)*n_p 

        # When there are no initial constraints, the initial Transversality conditions are redundant with the lagrange multipliers, 
        # We get trivially p(0) = -beta, or p(0) = 0
        if hasattr(self, 'dim_IF_constraints') == True:
            # Transversality conditions
            #      initial conditions on costates
            Shooting_function[index:index+n_p,0] = self.xp_i[N*n:N*n+n_p, 0] + self.lagrangian_IF.jacobian(self.xp_i[0:n_p, 0]).T
            #index += n_p
            pass

        # Transversality conditions
        #      final conditions on costates
        Shooting_function[index:index+n_p,0] = self.xp_f[N*n+(N-1)*n_p:N*(n+n_p), 0] - self.lagrangian_IF.jacobian(self.xp_f[(N-1)*n:(N-1)*n+n_p, 0]).T
        index += n_p

        # Continuity of the Hamiltonian condition
        for i in range(len(self.arcs)-1):
            subs_dic_final = dict(zip(self.x, self.xp_f[i*n:(i+1)*n, 0] ))
            subs_dic_final.update(dict(zip(self.p, self.xp_f[N*n+i*n_p:N*n+(i+1)*n_p, 0]) ))
            # Gets Hamiltonian corresponding to the right arc and substitutes the appropriate final values
            H_i_final = Hamiltonian_dict[self.arcs[i]].subs(subs_dic_final)

            subs_dic_initial = dict(zip(self.x, self.xp_i[(i+1)*n:(i+2)*n, 0] ))
            subs_dic_initial.update(dict(zip(self.p, self.xp_i[N*n+(i+1)*n_p:N*n+(i+2)*n_p, 0]) ))
            # Gets Hamiltonian corresponding to the right arc and substitutes the appropriate initial values
            H_ipp_initial = Hamiltonian_dict[self.arcs[i+1]].subs(subs_dic_initial)

            Shooting_function[index, 0] = H_ipp_initial - H_i_final
            index += 1
   
        # Hamiltonian derivatives of order 0 and 1
        # Note that it is possible we include conditions on the shooting function that are redundant to the previous entries,
        # This will lead to linearly dependent rows in the derivative of the shooting function. 
        # We will check if any new entry is not a multiple of the previous.
        for i in self.singular_indexes:

            subs_dic_initial = dict(zip(self.x, self.xp_i[i*n:(i+1)*n, 0] ))
            subs_dic_initial.update(dict(zip(self.p, self.xp_i[N*n+(i)*n_p:N*n+(i+1)*n_p, 0]) ))

            subs_dic_final = dict(zip(self.x, self.xp_f[i*n:(i+1)*n, 0] ))
            subs_dic_final.update(dict(zip(self.p, self.xp_f[N*n+(i)*n_p:N*n+(i+1)*n_p, 0]) ))

            expr = (self.T[i+1] - self.T[i])*self.switching_function.subs(subs_dic_final)
            Shooting_function[index, 0] = expr
            index += 1
                
            expr = (self.T[i+1] - self.T[i])*self.switching_function_dt.subs(subs_dic_initial)
            Shooting_function[index, 0] = expr
            index += 1
        
        Shooting_function = sp.Matrix(Shooting_function[0:index])
        return Shooting_function

    def dynamics_symbolic(self):
        n, n_p, n_T, N = self.n, self.n_p, self.n_T, self.N
        
        # Defining the dynamics for each arc
        dp = self.dp
        dx = self.f0 + self.f1*self.lin_cont

        states_dynamics_plus    =  dx.subs(self.lin_cont[0], self.lin_bounds[1])
        states_dynamics_minus   =  dx.subs(self.lin_cont[0], self.lin_bounds[0])
        states_dynamics_sing    =  dx.subs(self.lin_cont[0], self.delta_sing)
        costates_dynamics_plus  = (dp.subs(self.lin_cont[0], self.lin_bounds[1])).T
        costates_dynamics_minus = (dp.subs(self.lin_cont[0], self.lin_bounds[0])).T
        costates_dynamics_sing  = (dp.subs(self.lin_cont[0], self.delta_sing)).T

        # Dicstionary for state dynamics in each arc
        states_dynamics_dic    = {'bang_plus' : states_dynamics_plus,
                                  'bang_minus': states_dynamics_minus,
                                  'sing'      : states_dynamics_sing }

        # Dictionary for costate dynamics in each arc
        costates_dynamics_dic  = {'bang_plus' : costates_dynamics_plus,
                                  'bang_minus': costates_dynamics_minus,
                                  'sing'      : costates_dynamics_sing }

        F = sp.zeros(len(self.xp), 1)
        i = 0 # index for xp positioning

        for arc in self.arcs:   
            # assemble dictionary to substitute x with x_k, p with p_k
            substitution_dict = dict(zip(self.x, self.xp[i*n:(i+1)*n, 0]))
            substitution_dict.update(dict(zip(self.p, self.xp[N*n+i*n_p:N*n+(i+1)*n_p,0]))) 

            substitution_dict_pp = dict(zip(self.x, self.xp[(i+1)*n:(i+2)*n, 0]))
            substitution_dict_pp.update(dict(zip(self.p, self.xp[N*n+(i+1)*n_p:N*n+(i+2)*n_p,0]))) 

            # obtain state dynamics appropriated to each arc from dictionary
            F[i*n:(i+1)*n,0] = (self.T[i+1] - self.T[i])*states_dynamics_dic[arc]
            F[i*n:(i+1)*n,0] = F[i*n:(i+1)*n,0].subs(substitution_dict)

            # obtain costate dynamics appropriated to each arc from dictionary
            F[N*n+i*n_p:N*n+(i+1)*n_p,0] = (self.T[i+1] - self.T[i])*costates_dynamics_dic[arc]
            F[N*n+i*n_p:N*n+(i+1)*n_p,0] = F[N*n+i*n_p:N*n+(i+1)*n_p,0].subs(substitution_dict)

            i += 1
            pass 
        return F

    def dynamics_symbolic_unconstrained(self):
        n, n_p = self.n, self.n_p
        F = sp.zeros(len(self.xp), 1)
        if hasattr(self, 'delta_sing'):
            # State dynamics
            F[0:n] = self.f0 + self.f1*self.delta_sing
            # Costate dynamics
            F[n:n+n_p] = (self.dp.subs(self.lin_cont[0], self.delta_sing)).T
        else:
            # State dynamics
            F[0:n] = self.f0 
            # Costate dynamics
            self.Hamiltonian = self.p.dot(self.f0)
            if hasattr(self, 'running_cost') == True:
                self.Hamiltonian += self.running_cost
            F[n:n+n_p]       = -sp.Matrix([self.Hamiltonian]).jacobian(self.x[0:self.n_p, 0]).T
        pass 

        substitution_dict = dict(zip(self.x, self.xp[0:n, 0]))
        substitution_dict.update(dict(zip(self.p, self.xp[n:n+n_p,0])))
        
        return F.subs(substitution_dict)
    pass 
        
    def sym_to_numpy(self):
        """
        Generate symbolic functions for
            - singular controls
            - switching function
            - switching function time derivative
            - TP dynamics
            - variational TP dynamics
            - Shooting Function 
            - Derivatives of Shooting Function, w.r.t. initial values, final values, lagrange multipliers
        Transforms these symbolic expressions into numpy functions with symengine's Lambidify method.
        The obtained lambda functions are stored as class attributes.
        """
        # TP dynamics
        # First we check if the problem is constrained or unconstrained
        arg_sing_control = sp.DenseMatrix(len(self.x) + len(self.p), 1, [self.x, self.p])
        if hasattr(self, 'delta_sing') == True:
            self.singular_controls_np = sp.Lambdify(arg_sing_control, self.delta_sing)
        pass  
        if hasattr(self, 'switching_function') == True:
            self.switching_function_np    = sp.Lambdify(arg_sing_control, self.switching_function)
            self.switching_function_dt_np = sp.Lambdify(arg_sing_control, self.switching_function_dt)
        pass

        # Check if problem is unconstrained or not and generate corresponding symbolic functions
        if hasattr(self, 'arcs') == True:
            dyn_TP_sym = self.dynamics_symbolic()                  # TP dynamics
            S_F = self.symbolic_shooting_function_fast()           # TP Shooting function
        else:
            dyn_TP_sym = self.dynamics_symbolic_unconstrained()    # unconstrained dynamics
            S_F = self.symbolic_shooting_function_unconstrained()  # unconstrained Shooting function
        
        # generate symbolic variational dynamics
        var_dyn_sym = dyn_TP_sym.jacobian(self.xp)

        # Generate lambda functions
        self.dynamics_np     = sp.Lambdify(self.xp, dyn_TP_sym)  # lambda function to numerical TP dynamics 
        self.var_dynamics_np = sp.Lambdify(self.xp, var_dyn_sym) # lambda function to numerical Variational TP dynamics

        # Wrapper for arguments of the Shooting Function and its derivatives
        #   If there are no initial final constraints, the lagrange multipliers are not included as Shooting parameters
        if hasattr(self, 'lagrange_mult') == True:
            args = sp.DenseMatrix(len(self.xp_i) + len(self.xp_f) + len(self.lagrange_mult), 1, [self.xp_i, self.xp_f, self.lagrange_mult])
            
            D_bSF = S_F.jacobian(self.lagrange_mult)     # Derivative w.r.t. lagrange multipliers
            self.D_bSF_num_np = sp.Lambdify(args, D_bSF) # 
        else:
            args = sp.DenseMatrix(len(self.xp_i) + len(self.xp_f), 1, [self.xp_i, self.xp_f] )
        pass # end check for lagrange multipliers

        # Symbolic Shooting Function Derivatives
        D_iSF = S_F.jacobian(self.xp_i) # Derivative w.r.t. initial conditions
        D_fSF = S_F.jacobian(self.xp_f) # Derivative w.r.t. final conditions

        # Make Lambda functions
        self.shooting_np  = sp.Lambdify(args, S_F)   # lambda function to numerical Shooting Funtion
        self.D_iSF_num_np = sp.Lambdify(args, D_iSF) # lambda function to numerical SF w.r.t. initial conditions
        self.D_fSF_num_np = sp.Lambdify(args, D_fSF) # lambda function to numerical SF w.r.t. final conditions
    pass # end function  sym_to_numpy

    def implicit_Euler(self, xp_initial, step_size):
        step_size = float(step_size)
        N         = int(np.floor(1.0/step_size))

        I      = np.eye(len(self.xp))
        xp_k   = np.copy(xp_initial)
        flow_k = I
        for k in range(N):
            xp_k  += step_size*self.dynamics_np(xp_k.T)[0]
            flow_k = np.linalg.solve(I - step_size*self.var_dynamics_np(xp_k.T)[0], flow_k)
            
        return xp_k, flow_k, N

    def heun_trapezoidal(self, xp_initial, step_size):
        step_size = float(step_size)
        half_step = step_size/2.0
        N         = int(np.floor(1.0/step_size))

        I      = np.eye(len(self.xp))
        M_k    = np.eye(len(self.xp))
        M_kpp  = np.eye(len(self.xp))

        xp_k   = np.copy(xp_initial)
        xp_aux = np.copy(xp_initial)
        flow_k = np.eye(len(self.xp))
        for k in range(N):
            M_k = self.var_dynamics_np(xp_k.T)[0]
            # Heun integration of the states
            xp_aux = xp_k + step_size*self.dynamics_np(xp_k.T)[0]
            xp_k  += half_step*(self.dynamics_np(xp_k.T)[0] + self.dynamics_np(xp_aux.T)[0])

            # Trapezoidal integration of the variational dynamics
            M_kpp  = self.var_dynamics_np(xp_k.T)[0]
            flow_k = np.linalg.solve((I - half_step*M_kpp),(I + half_step*M_k)@flow_k)
            
        return xp_k, flow_k, N

    def dynamics4gsl(self, t, xp, args = None):
        return self.dynamics_np(xp).reshape(len(xp)) 
        
    def jacobian4gsl(self, t, xp, args = None):
        return self.var_dynamics_np(xp) 

    def gsl_matrix_dyn(self, t, X_array, args = None):
        dim_TP = len(self.xp)
        mat_shape = (dim_TP, dim_TP)

        xp_t  = X_array[0:dim_TP]
        var_t = X_array[dim_TP:].reshape(mat_shape)
        
        matrix_dynamics = np.zeros((dim_TP*(dim_TP+1), ))
        # Dynamics components
        matrix_dynamics[0:dim_TP] = self.dynamics_np(xp_t).reshape((dim_TP,))

        # Variational Dynamics components
        var_dyn = self.var_dynamics_np(xp_t)@var_t
        matrix_dynamics[dim_TP:] = var_dyn.reshape((dim_TP**2,))

        return matrix_dynamics
    
    def gsl_matrix_ode(self, xp_initial, step_size):
        dim_TP = len(self.xp)
        dimension = dim_TP*(dim_TP+1)

        X           = np.zeros((dimension, ))
        X[0:dim_TP] = xp_initial.reshape((dim_TP,))
        X[dim_TP:]  = np.eye(dim_TP).reshape((dim_TP**2,))
    
        # Setting the ODE solver from gsl 
        step      = odeiv.step_rk4imp(dimension, self.gsl_matrix_dyn, jac = None, args = None)
        control   = odeiv.control_yp_new(step, 1e-6, 1e-6)
        evolve    = odeiv.evolve(step, control, dimension)

        t = 0.0
        h = step_size
        n_steps = 0
        while t < 1.0:
            if 1.0 - h < t: h = 1.0 - t
            t, h, X = evolve.apply(t, 1.0, h, X)
            n_steps += 1
        
        xp_final = X[0:dim_TP].reshape([dim_TP, 1])
        var_final = X[dim_TP:].reshape([dim_TP, dim_TP])
        
        return xp_final, var_final, n_steps
        
    def gsl_ode_GaussII(self, xp_initial, step_size):
        #
        # Matrices for IRK-2 at gaussian points 
        A = np.array([[0.25, 0.25 - np.sqrt(3.0)/6.0], [0.25 + np.sqrt(3.0)/6.0, 0.25]])
        b = np.array([0.5, 0.5])
        c = np.array([0.5 - np.sqrt(3.0)/6.0, 0.5 + np.sqrt(3.0)/6.0])
        d = np.linalg.solve(A.T, b)
        n_stages = 2

        #I  = np.eye(len(self.xp))
        #M1 = np.eye(len(self.xp))
        #M2 = np.eye(len(self.xp))

        xp_k    = xp_initial.reshape(len(xp_initial))
        xp_mid1 = xp_initial.reshape(len(xp_initial))
        xp_mid2 = xp_initial.reshape(len(xp_initial))
        flow_k  = np.eye(len(self.xp))

        # Setting the ODE solver from gsl to solve original problem
        dimension = len(xp_initial)
        step      = odeiv.step_rk4imp(dimension, self.dynamics4gsl, jac = self.jacobian4gsl, args = None)
        control   = odeiv.control_yp_new(step, 1e-9, 1e-9)
        evolve    = odeiv.evolve(step, control, dimension)

        # Auxiliary matrices for LS in variational update equation
        I_sn = np.eye(dimension*n_stages)
        A_kron_I = np.kron(A, np.eye(dimension))

        diag_Ms  = np.copy(I_sn)
        B = np.zeros([n_stages*dimension, dimension])
        Z = np.copy(B)

        if hasattr(self, 'arcs') == True: 
            t_k, t_kpp = 0.0, 0.0   # If we solve TP, the integration period is (0,1)
            T_f = 1.0
        else: 
            t_k, t_kpp = self.time_horizon[0], self.time_horizon[0] # If we don't create problem TP, the integration period is the time horizon prescribed
            T_f = self.time_horizon[1]
            
        h = step_size
        n_steps = 0
        while t_k < T_f:
            if T_f - h < t_k: h = T_f - t_k
            
            # Updates the state and obtains the value of step size at current step
            t_kpp, h, xp_k = evolve.apply(t_k, T_f, h, xp_k)
            #evolve.reset()
            #print(xp_k, t_k, t_kpp, h, T_f)
            
            # Estimates the values of the states at midpoints for Gaussian quadrature
            #  and computes jacobian at such points for IRK-2 
            t_trash, h_trash, xp_mid1 = evolve.apply(t_k, t_k + c[0]*h, c[0]*h, xp_mid1)
            #evolve.reset()

            t_trash, h_trash, xp_mid2 = evolve.apply(t_k, t_k + c[1]*h, c[1]*h, xp_mid2)
            #evolve.reset()

            diag_Ms[0:dimension, 0:dimension] = self.var_dynamics_np(xp_mid1)
            diag_Ms[dimension:, dimension:]   = self.var_dynamics_np(xp_mid2)
            
            A_kron_I_diag_Ms = h*A_kron_I@diag_Ms
            
            B[0:dimension,:] = flow_k
            B[dimension:,:]  = flow_k
            B = A_kron_I_diag_Ms@B

            Z = np.linalg.solve(I_sn - A_kron_I_diag_Ms, B)
            flow_k += d[0]*Z[0:dimension,:] + d[1]*Z[dimension:,:]        

            xp_mid1, xp_mid2 = np.copy(xp_k), np.copy(xp_k)
            t_k = t_kpp
            n_steps += 1

        return xp_k.reshape([len(xp_k), 1]), flow_k, n_steps

    def line_search(self, rule, current_point, q_at_0, q_derivative_at_0, direction, initial_conditions):
        valid_rules = ['Wolfe', 'Armijo', 'Goldstein_Price']
        if rule not in valid_rules:
            print('Invalid linea search rule type given. Use linear search rule in:\n ', valid_rules)
            return
        step = 1.0
        if rule == 'Wolfe':
            if q_derivative_at_0 >=0:
                print('Direction not optimal. q\'(0) = {} >=0 '.format(q_derivative_at_0))
            # Set constants for Wolfe line search
            m1, m2, tL, tR = 0.1, 0.9, 0.0, 100.0

            # Compute quantities related to the objective function
            shot_t, D_shot_t, N_steps = self.shooting_function(current_point + step*direction, initial_conditions)
            shot_t            = shot_t.reshape(len(shot_t),)
            q_at_t            = 0.5*np.linalg.norm(shot_t, ord= 2)**2
            q_derivative_at_t = np.dot(shot_t, D_shot_t@direction)
            
            # While not the condition to terminate the Wolfe line search
            while q_at_t > q_at_0 + m1*step*q_derivative_at_0 or q_derivative_at_t < m2*q_derivative_at_0:
                if (q_at_t <= q_at_0 + m1*step*q_derivative_at_0) and (q_derivative_at_t < m2*q_derivative_at_0): 
                    # step too small
                    tL = step
                    if tR == 100.0:
                        step = 2*step
                    else:
                        step = (tL + tR)/2.0
                else:
                    # step too large
                    tR = step
                    if tR == 100.0:
                        step = 2*step
                    else:
                        step = (tL + tR)/2.0
            
                shot_t, D_shot_t, N_steps = self.shooting_function(current_point + step*direction, initial_conditions)
                shot_t = shot_t.reshape(len(shot_t),)
                q_at_t            = 0.5*np.linalg.norm(shot_t, ord= 2)**2
                q_derivative_at_t = np.dot(shot_t, D_shot_t@direction)
            pass # end while
        pass # end Wolfe line search
        
        if rule == 'Armijo':
            print('Error: Armijo Line Search not yet implemented. Using constant step.')
            pass

        if rule == 'Goldstein_Price':
            print('Error: Goldstein Price Line Search not yet implemented. Using constant step.')
            pass
        return step 

    def shooting_function(self, shooting_args, initial_conditions):
        """
        This function returns the numerical values of the Shooting Function and the Derivative with respect to the Shooting Parameters

            Inputs:
                all_args           - numpy array with all parameters necessary to evaluate the shooting function and integrate TP and the variational system
                                     [x^0(0), x^1(0), ..., x^N-1(0), p^0(0), ..., p^0(0), switching_times, beta]
                integration_method - method used by scipy to integrate the ODEs
                initial_conditions - string value specifying if the initial conditions are fixed or optimization variables

            Outputs:
                shooting   - shooting function evaluated with prescribed parameters
                D_shooting - derivative of the shooting function with respect to the shooting parameters
        """
        d_nu = len(shooting_args)
        if initial_conditions != 'free':
            all_args    = np.append(initial_conditions, shooting_args, axis = 0)
            start_index = self.n
        else:
            all_args    = shooting_args
            start_index = 0

        # all_args consists in [x^0(0), x^1(0), ..., x^N-1(0), p^0(0), ..., p^0(0), switching_times, beta]
        # If there are no initial-final constraints, we ignore any other argument given by mistake
        dim_TP     = len(self.xp)
        xp_initial = np.copy(all_args[0:dim_TP])

        # Integration with our naive implementation
        step = 0.01
        xp_final, var_final, N_steps = self.gsl_ode_GaussII(xp_initial, step_size = step)
        #xp_final, var_final, N_steps = self.gsl_matrix_ode(xp_initial, step_size = step)
        
        # Compute the Shooting Function
        # The values of the switching times and respective costates are only necessary at final time
        arg = np.append(xp_initial, xp_final, axis = 0)
        
        if hasattr(self,'lagrange_mult') == True:
            lagrange_mult = all_args[dim_TP:,0]
            lagrange_mult = lagrange_mult.reshape(len(all_args) - dim_TP,1)
            arg = np.append(arg, lagrange_mult, axis = 0)
        
        shot = self.shooting_np(arg.T)[0]
        
        # Compute the Shooting Derivative 
        # Derivatives of the shooting function

        d_iSF = self.D_iSF_num_np(arg.T)[0]  # w.r.t. initial conditions
        d_fSF = self.D_fSF_num_np(arg.T)[0]  # w.r.t. final conditions

        if hasattr(self, 'lagrange_mult') == True:
            d_beta = len(self.lagrange_mult)
            d_bSF  = self.D_bSF_num_np(arg.T)[0]  # Shooting derivative w.r.t. lagrange multipliers (beta)
            D_shot = np.zeros([len(shot), d_nu])
            D_shot[:,0:d_nu-d_beta] = d_iSF[:,start_index:] + d_fSF@var_final[:,start_index:]
            D_shot[:,d_nu-d_beta:]  = d_bSF
        else:
            D_shot = d_iSF[:,start_index:] + d_fSF@var_final[:,start_index:]
            
        return shot, D_shot, N_steps

    def solve_shooting(self, shooting_args, initial_conditions = 'free', tol = 1e-6, integration_method = 'Radau', step_size = 1.0):
        
        shot   = np.ones([10,1])
        max_it = 0

        while np.linalg.norm(shot, ord = np.inf) > tol and max_it <= 100:
            shot, D_shot, N_steps = self.shooting_function(shooting_args, initial_conditions)

            A      = D_shot.T@D_shot
            b      = D_shot.T@shot
            inc    = np.copy(np.linalg.solve(A,b))
            shooting_args -= step_size*inc
            max_it += 1

            print('iteration:   {}, |S|:   {}, integration steps:   {};'.format(max_it, np.linalg.norm(shot), N_steps))
            print('rank A:  ', np.linalg.matrix_rank(A),'rank D_shot:  ', np.linalg.matrix_rank(D_shot),'\n\n')

        if initial_conditions != 'free':
            return np.append(initial_conditions, shooting_args, axis = 0), shot
        else:
            return shooting_args, shot
    
    def solve_shooting_test(self, shooting_args, initial_conditions = 'free', tol = 1e-6, integration_method = 'Radau', step_size = 1.0):
        
        shot   = np.ones([10,1])
        max_it = 0

        while np.linalg.norm(shot, ord = np.inf) > tol and max_it <= 100:
            shot, D_shot, N_steps = self.shooting_function(shooting_args, initial_conditions) 

            # Compute next direction 
            A      = D_shot.T@D_shot
            b      = D_shot.T@shot
            inc    = np.copy(-np.linalg.solve(A,b))

            # Quantities to initialize the line search
            shot              = shot.reshape(len(shot),)
            q_at_0            = 0.5*np.linalg.norm(shot, ord= 2)**2
            q_derivative_at_0 = np.dot(shot, D_shot@inc)

            # Compute step_size 
            step_size = self.line_search(rule = 'Wolfe', current_point = shooting_args, 
                                         q_at_0 = q_at_0, q_derivative_at_0 = q_derivative_at_0, 
                                         direction = inc, initial_conditions = initial_conditions)
            
            # Update 
            shooting_args += step_size*inc
            max_it += 1

            print('iteration: {}, |S|: {}, integration steps: {}, step size: {};'.format(max_it, np.linalg.norm(shot), N_steps, step_size))
            print('rank A:  ', np.linalg.matrix_rank(A),'rank D_shot:  ', np.linalg.matrix_rank(D_shot),'\n\n')

        if initial_conditions != 'free':
            return np.append(initial_conditions, shooting_args, axis = 0), shot
        else:
            return shooting_args, shot


    def optimal_trajectory(self, opt_initial_cond, control_bounds):

        if hasattr(self, 'arcs') == True:
            if hasattr(self, 'lagrange_mult') == True:
                opt_initial_cond = opt_initial_cond[0:len(opt_initial_cond) - len(self.lagrange_mult)]
            # Integrate the system with optimal initial conditions
            n, n_p, N = self.n, self.n_p, self.N

            T = opt_initial_cond[N*(n+n_p):]
            if self.time_horizon[0] != 'free':
                T = np.append(self.time_horizon[0], T)
            if self.time_horizon[1] != 'free':
                T = np.append(T, self.time_horizon[1])

            xp_init = opt_initial_cond.reshape((len(opt_initial_cond,)))
            
            # Setting up gsl solver
            dimension = len(xp_init)
            step      = odeiv.step_rk4imp(dimension, self.dynamics4gsl, jac = self.jacobian4gsl, args = None)
            control   = odeiv.control_yp_new(step, 1e-9, 1e-9)
            evolve    = odeiv.evolve(step, control, dimension)

            # First we integrate the transformed system in the normalized time interval (0,1)
            h, t_k, xp_k     = 0.001, 0.0, np.copy(xp_init)
            times_normalized = [0.0]
            xp_splitted      = [opt_initial_cond.reshape((len(opt_initial_cond,)))]

            while t_k < 1.0:
                if 1.0 - h < t_k: h = 1.0 - t_k
                # Updates the state and obtains the value of step size at current step
                # h_ is the suggested next step, not the last step size taken. 
                # Since we want resolution, we integrate with constant step size
                t_k, h, xp_k = evolve.apply(t_k, 1.0, h, xp_k)
                times_normalized.append(t_k)
                xp_splitted.append(xp_k)
        
            N_t = len(times_normalized)
            xp_splitted      = np.array(xp_splitted)
            times_normalized = np.array(times_normalized)

            states   = xp_splitted[:, 0:n]
            costates = xp_splitted[:, N*n: N*n+n_p]
            times    = (T[1] - T[0])*times_normalized + T[0]


            for k in range(1, len(self.arcs)):
                times    = np.append(times, (T[k+1] - T[k])*times_normalized + T[k])
                states   = np.append(states, xp_splitted[:, k*n:(k+1)*n], axis = 0)
                costates = np.append(costates, xp_splitted[:, N*n+k*n_p:N*n+(k+1)*n_p], axis = 0)
                
            xp_all = np.append(states, costates, axis = 1)

            controls                  = self.singular_controls_np(xp_all)
            switching_function_all    = self.switching_function_np(xp_all)
            switching_function_dt_all = self.switching_function_dt_np(xp_all)

            for k in range(len(self.arcs)):
                if self.arcs[k] == 'bang_minus':
                    controls[N_t*k:N_t*(k+1)] = control_bounds[0]
                if self.arcs[k] == 'bang_plus':
                    controls[N_t*k:N_t*(k+1)] = control_bounds[1]
                    
            return times, controls, states, costates, switching_function_all, switching_function_dt_all
        else:
            n, n_p    = self.n, self.n_p 
            T_0, T_f = self.time_horizon[0], self.time_horizon[1]

            xp_init = opt_initial_cond.reshape((len(opt_initial_cond,)))
            
            # Setting up gsl solver
            dimension = len(xp_init)
            step      = odeiv.step_rk4imp(dimension, self.dynamics4gsl, jac = self.jacobian4gsl, args = None)
            control   = odeiv.control_yp_new(step, 1e-9, 1e-9)
            evolve    = odeiv.evolve(step, control, dimension)

            # First we integrate the transformed system in the normalized time interval (0,1)
            h, t_k, xp_k = 0.001, T_0, np.copy(xp_init)
            times        = [T_0]
            xp           = [opt_initial_cond.reshape((len(opt_initial_cond,)))]
           
            while t_k < T_f:
                if T_f - h < t_k: h = T_f - t_k
                # Updates the state and obtains the value of step size at current step
                # h_ is the suggested next step, not the last step size taken. 
                # Since we want resolution, we integrate with constant step size
                t_k, h, xp_k = evolve.apply(t_k, T_f, h, xp_k)
                times.append(t_k)
                xp.append(xp_k)
            pass # end while
            xp = np.array(xp)

            states   = xp[:,0:n]
            costates = xp[:,n:n+n_p]
            return states, costates, times
        pass # end check for arcs
    pass ## end function 


