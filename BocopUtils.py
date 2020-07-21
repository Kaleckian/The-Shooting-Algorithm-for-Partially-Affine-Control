# Bocop python utils for bocop
# Pierre Martinon
# Inria
# 2018-2019

# General
# - buildProblem: build bocop executable
# - launchProblem: run bocop executable to solve OCP

# Getters
# - readResultFile: get optimization status from file result.out
# - readSolFile: read .sol file
# - getState: get a single state variable

# Setters
# - setConstant: set a constant in file problem.constants

import os
import shutil
import subprocess
import numpy as np

#-----------------------------------------
# (re)build executable for bocop
#-----------------------------------------
def buildProblem(clean,debug,verbose):

	status = 0
	if debug == 1:
		debug_option = ' -d '
	else:
		debug_option = ''
		
	if clean == 1:
		if os.path.exists('build'):
			shutil.rmtree('build')
	if os.path.exists('bocop'):
		os.remove('bocop')

	fnull = open(os.devnull, 'w')
	if verbose == 0:
		status = subprocess.call('sh build.sh'+debug_option,shell=True,stdout=fnull,stderr=fnull)
	else:
		status = subprocess.call('sh build.sh'+debug_option,shell=True)

	if status != 0:
		print("Error: build step failed with return "+str(status))

	return status


#-----------------------------------------
# launch optimization  +++ windows
#-----------------------------------------
def launchProblem(clean,maxtime,verbose=0):

	fnull = open(os.devnull, 'w')
	if clean == 1:
		subprocess.call('rm result.out',shell=True,stdout=fnull,stderr=fnull)

	try:

		if verbose == 0:
			status = subprocess.call('./bocop',shell=True,stdout=fnull,stderr=fnull,timeout=maxtime)
		else:
			status = subprocess.call('./bocop',shell=True,timeout=maxtime)

	except subprocess.TimeoutExpired:
		if verbose > 0:
			print(' Optimization aborted after timeout...')
		status = 666

	return status


#-----------------------------------------
# read result.out file
#-----------------------------------------
def readResultFile(filename):

  status = -1
  objective = 0
  constraint = 0
  iterations = 0
  cpuipopt = 0
  cpunl = 0
  
  with open(filename,'r') as resultfile:
    for line in resultfile:
			# STATUS 0,1,3 ARE OK
      if 'EXIT: Optimal Solution Found' in line:
        status = 0
      elif 'EXIT: Solved To Acceptable Level' in line:
        status = 1
      elif 'EXIT: Search Direction is becoming Too Small' in line:
        status = 3
      # STATUS BELOW ARE NOT OK
      elif 'EXIT: Maximum Number Of Iterations' in line:
        status = -1
      elif 'EXIT: Restoration Failed!' in line:
        status = -2
      elif 'Number of Iterations....' in line:
        iterations = int(line.split()[-1])
      elif 'Objective...............' in line:
        objective = float(line.split()[-1])
      elif 'Constraint violation....' in line:
        constraint = float(line.split()[-1])
      elif 'Total CPU secs in IPOPT' in line:
        cpuipopt = float(line.split()[-1])
      elif 'Total CPU secs in NLP' in line:
        cpunl = float(line.split()[-1])
        
  cpu = cpuipopt + cpunl

  return status, objective, constraint, iterations, cpuipopt, cpunl, cpu

# ipopt return codes  
#static final int 	SOLVE_SUCCEEDED = 0
#static final int 	ACCEPTABLE_LEVEL = 1
#static final int 	INFEASIBLE_PROBLEM = 2
#static final int 	SEARCH_DIRECTION_TOO_SMALL = 3
#static final int 	DIVERGING_ITERATES = 4
#static final int 	USER_REQUESTED_STOP = 5
#static final int 	ITERATION_EXCEEDED = -1
#static final int 	RESTORATION_FAILED = -2
#static final int 	ERROR_IN_STEP_COMPUTATION = -3
#static final int 	CPUTIME_EXCEEDED = -4
#static final int 	NOT_ENOUGH_DEGREES_OF_FRE = -10
#static final int 	INVALID_PROBLEM_DEFINITION = -11
#static final int 	INVALID_OPTION = -12
#static final int 	INVALID_NUMBER_DETECTED = -13
#static final int 	UNRECOVERABLE_EXCEPTION = -100
#static final int 	NON_IPOPT_EXCEPTION = -101
#static final int 	INSUFFICIENT_MEMORY = -102
#static final int 	INTERNAL_ERROR = -199



#-----------------------------------------
# read sol file and retrieve trajectory
#-----------------------------------------
def readSolFile(filename):
	
	#+++ probably better to read whole file then extract parts with subfunctions
	with open(filename,'r') as solfile:
		
		# read dimensions
		line = solfile.readline()
		while not '# *****     SOLUTION     *****' in line:
			if '# state.dimension' in line:
				dim_state = int(line.split()[-1])
			elif '# control.dimension' in line:
				dim_control = int(line.split()[-1]) 
			elif '# discretization.steps' in line:
				n_steps = int(line.split()[-1])
			elif '# boundarycond.dimension' in line:
				dim_boundarycond =  int(line.split()[-1])
			elif 'parameter.dimension' in line:
				dim_parameter =  int(line.split()[-1])				
			line = solfile.readline()

		# read objective
		objective = getValue(solfile,'# Objective value :')
		
		# read time steps and stages
		n_stage = int(getValue(solfile,'# Number of stages of discretization method :'))
		line = solfile.readline()
		time_steps = np.empty(n_steps+1)
		time_stages = np.empty(n_steps*n_stage)
		index_s = 0
		for k in range(0,n_steps):
			line = solfile.readline()
			time_steps[k] = float(line)
			for s in range(0,n_stage):
				line = solfile.readline()
				time_stages[index_s] = float(line)
				index_s = index_s + 1
		line = solfile.readline()
		time_steps[-1] = float(line)

		# read state, control, parameters
		state = getBlock2D(solfile,'# State',dim_state,n_steps+1)
		control = getBlock2D(solfile,'# Control',dim_control,n_stage*n_steps)
		parameters = getBlock1D(solfile,'# Parameters',dim_parameter)
		
		# read boundary conditions multipliers, costate
		boundary_mult = getBlock1D(solfile,'# Multipliers associated to the boundary conditions',dim_boundarycond)
		costate = getBlock2D(solfile,'# Adjoint',dim_state,n_steps) 
		
		# read ipopt status
		status = getValue(solfile,'# Ipopt status :') 

	return objective, time_steps, time_stages, state, control, parameters, costate, boundary_mult, status


def getValue(solfile,label):
	
	line = solfile.readline()
	while not label in line:
		line = solfile.readline()
	value = float(solfile.readline())
	
	return value


def getBlock1D(solfile,label,dim1):
	
	block = np.empty([dim1])
	line = solfile.readline()
	while not label in line:
		line = solfile.readline()
	for k in range(0,dim1):
		line = solfile.readline()
		block[k] = float(line)

	return block


def getBlock2D(solfile,label,dim1,dim2):

	block = np.empty([dim1,dim2])
	line = solfile.readline()
	for i in range(0,dim1):
		while not label in line:
			line = solfile.readline()
		for k in range(0,dim2):
			line = solfile.readline()
			block[i][k] = float(line)

	return block

#-----------------------------------------
# get a single state variable
#-----------------------------------------
def getState(state_index):

	state = readSolFile('problem.sol') 
	return state[state_index]


#-----------------------------------------
# get sets of bounds
#-----------------------------------------
def getBounds(filename):

	bounds = [ ]

	with open(filename,'r') as boundsfile:

		# warning: bounds file are sometimes mis-formatted between labels and value
		line = boundsfile.readline()
		while not 'Dimensions' in	line:
			line = boundsfile.readline()		
		dims = boundsfile.readline().split()
		dims = [int(i) for i in dims]

		for line in boundsfile:
			line = line.strip()
			if len(line) > 0 and not '#' in line:
				bounds.append(line.split())

	return dims, bounds


#-----------------------------------------
# set one constant in file problem.constants
#-----------------------------------------
def setConstant(index, value):

	constants = []
	lines = open('problem.constants','r').readlines()
	lines[index+5] = str(value)+'\n'
	outfile = open('problem.constants','w')
	outfile.writelines(lines)
	outfile.close()


#-----------------------------------------
# set one bound in file problem.bounds
#-----------------------------------------
def setBound(group, index, lower, upper, kind):
	
	dims, bounds = getBounds('problem.bounds')
	
	if group == 'boundary':
		index_group = 0		
	elif group == 'state':
		index_group = dims[0]
	elif group == 'control':
		index_group = sum(dims[0:1+1])
	elif group == 'algvar':
		index_group = sum(dims[0:2+1])
	elif group == 'param':
		index_group = sum(dims[0:3+1])
	elif group == 'path':
		index_group = sum(dims[0:4+1])
	else:
		print('ERROR: setBound >>> kind should be among \'boundary\',  \'state\',  \'control\', \'algvar\',  \'param\', \'path\' ')
		return
	
	bounds[index_group+index] = [str(lower),str(upper),kind]
	
	writeBounds('problem.bounds',dims,bounds)
	
	
def writeBounds(filename,dims,bounds):
		
	outfile = open(filename,'w')
	outfile.writelines("""# This file contains all the bounds of your problem.
# Bounds are stored in standard format :
# [lower bound]  [upper bound] [type of bound]

# Dimensions (i&f conditions, y, u, z, p, path constraints) :\n""")
	outfile.writelines([str(d)+' ' for d in dims])
	outfile.writelines('\n')

	index = 0
	outfile.writelines("""\n# Bounds for the initial and final conditions :\n""")
	while index < dims[0]:
		outfile.writelines([str(value)+' ' for value in bounds[index] ])
		outfile.writelines('\n')
		index = index + 1

	outfile.writelines("""\n# Bounds for the state variables :\n""")
	while index < sum(dims[0:1+1]):
		outfile.writelines([str(value)+' ' for value in bounds[index] ])
		outfile.writelines('\n')
		index = index + 1

	outfile.writelines("""\n# Bounds for the control variables :\n""")
	while index < sum(dims[0:2+1]):
		outfile.writelines([str(value)+' ' for value in bounds[index] ])
		outfile.writelines('\n')
		index = index + 1
		
	outfile.writelines("""\n# Bounds for the algebraic variables :\n""")
	while index < sum(dims[0:3+1]):
		outfile.writelines([str(value)+' ' for value in bounds[index] ])
		outfile.writelines('\n')
		index = index + 1
		
	outfile.writelines("""\n# Bounds for the optimization parameters :\n""")
	while index < sum(dims[0:4+1]):
		outfile.writelines([str(value)+' ' for value in bounds[index] ])
		outfile.writelines('\n')
		index = index + 1
		
	outfile.writelines("""\n# Bounds for the path constraints :\n""")
	while index < sum(dims[0:5+1]):
		outfile.writelines([str(value)+' ' for value in bounds[index] ])
		outfile.writelines('\n')
		index = index + 1

	outfile.close()


#-----------------------------------------
# set one parameter in .def file
#-----------------------------------------
def	setInDef(label,kind,value):

	lines = open('problem.def','r').readlines()
	index = 0
	for line in lines:
		if label in line:
			lines[index] = label + ' ' + kind + ' ' +value+ '\n'
		index = index + 1 
	outfile = open('problem.def','w')
	outfile.writelines(lines)
	outfile.close()
	
	
