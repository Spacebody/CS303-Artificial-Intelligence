import numpy as np
import os, sys
import copy
try:
		from tkinter import *
except ImportError:  #Python 2.x
		PythonVersion = 2
		from Tkinter import *
		from tkFont import Font
		from ttk import *
		from tkMessageBox import *
		import tkFileDialog
else:  #Python 3.x
		PythonVersion = 3
		from tkinter.font import Font
		from tkinter.ttk import *
		from tkinter.messagebox import *

# tags for file
file_tag='train' #train/test

# The board size of go game
BOARD_SIZE = 9
COLOR_BLACK=-1
COLOR_WHITE=1
COLOR_NONE=0
POINT_STATE_CHECKED=100
POINT_STATE_UNCHECKED=101
POINT_STATE_NOT_ALIVE=102
POINT_STATE_ALIVE=103
POINT_STATE_EMPYT=104


def read_go(file_name):
	# read from txt file and save as a matrix
	go_arr = np.zeros((BOARD_SIZE, BOARD_SIZE))
	for line in open(file_name):
		line = line.strip()
		lst = line.split()
		row = int(lst[0])
		col = int(lst[1])
		val = int(lst[2])
		go_arr[row, col] = val
	return go_arr


def plot_go(go_arr, txt='Default'):
	# Visualization of a go matrix
	# First draw a canvas with 9*9 grid
	root = Tk()
	cv = Canvas(root, width=50*(BOARD_SIZE+1), height=50*(BOARD_SIZE+1), bg='#F7DCB4')
	cv.create_text(250,10,text=txt,fill='blue')
	cv.pack(side=LEFT)
	size = 50
	for x in range(BOARD_SIZE):
		cv.create_line(size+x*size, size, size+x*size, size+(BOARD_SIZE-1)*size)
	for y in range(BOARD_SIZE):
		cv.create_line(size, size+y*size, size+(BOARD_SIZE-1)*size, size+size*y)
	# Second draw white and black circles on cross points
	offset = 20
	idx_black = np.argwhere(go_arr == COLOR_BLACK)
	idx_white = np.argwhere(go_arr == COLOR_WHITE)
	len_black = idx_black.shape[0]
	len_white = idx_white.shape[0]
	for i in range(len_black):
		if idx_black[i,0] >= BOARD_SIZE or idx_black[i,1] >= BOARD_SIZE:
			print ('IndexError: index out of range')
			sys.exit(0)
		else:
			new_x = 50*(idx_black[i,1]+1)
			new_y = 50*(idx_black[i,0]+1)
			cv.create_oval(new_x-offset, new_y-offset, new_x+offset, new_y+offset, width=1, fill='black', outline='black')
	for i in range(len_white):
		if idx_white[i,0] >= BOARD_SIZE or idx_white[i,1] >= BOARD_SIZE:
			print ('IndexError: index out of range')
			sys.exit(0)
		else:
			new_x = 50*(idx_white[i,1]+1)
			new_y = 50*(idx_white[i,0]+1)
			cv.create_oval(new_x-offset, new_y-offset, new_x+offset, new_y+offset, width=1, fill='white', outline='white')
	root.mainloop()

#-------------------------------------------------------
# Rule judgement  --Finished
#-------------------------------------------------------
def is_alive(check_state, go_arr, i, j, color_type):
	'''
	This function checks whether the point (i,j) and its connected points with the same color are alive, it can only be used for white/black chess only
	Depth-first searching.
	:param check_state: The guard array to verify whether a point is checked
	:param go_arr: chess board
	:param i: x-index of the start point of searching
	:param j: y-index of the start point of searching
	:return: POINT_STATE_CHECKED/POINT_STATE_ALIVE/POINT_STATE_NOT_ALIVE, POINT_STATE_CHECKED=> the start point (i,j) is checked, POINT_STATE_ALIVE=> the point and its linked points with the same color are alive, POINT_STATE_NOT_ALIVE=>the point and its linked points with the same color are dead
	'''
	points = []
	points = find_chess_block(check_state, go_arr, i, j, points, color_type)
	for (k,p) in points:
		for (r,t) in ((k-1,p), (k+1,p), (k,p-1), (k,p+1)):
			if r in range(go_arr.shape[0]) and t in range(go_arr.shape[1]):
				if go_arr[r,t] == COLOR_NONE:
					for (s,z) in points:
						check_state[s,z] = POINT_STATE_CHECKED
					return POINT_STATE_ALIVE
			else:
				continue
		else:
			continue
	return POINT_STATE_NOT_ALIVE

def find_chess_block(check_state, go_arr, i, j, points, color_type):
	'''
	:param check_state: The guard array to verify whether a point is checked
	:param go_arr: chess board
	:param i: x-index of the start point of searching
	:param j: y-index of the start point of searching
	:return: points=>The array of chess block
	'''
	points.append((i, j))
	for (k,p) in ((i-1,j), (i+1,j), (i,j-1), (i,j+1)):
		if k in range(go_arr.shape[0]) and p in range(go_arr.shape[1]):
			if go_arr[k,p] == color_type and (k,p) not in points:
				points = find_chess_block(check_state, go_arr, k, p, points, color_type)
			else:
				continue
		else:
			continue
	return points

def go_judege(go_arr):
	'''
	:param go_arr: the numpy array contains the chess board
	:return: whether this chess board fit the go rules in the document
					 False => unfit rule
					 True => ok
	'''
	is_fit_go_rule = True
	check_state = np.zeros(go_arr.shape)
	check_state[:] = POINT_STATE_EMPYT
	tmp_indx = np.where(go_arr != 0)
	check_state[tmp_indx] = POINT_STATE_UNCHECKED
	for i in range(go_arr.shape[0]):
		for j in range(go_arr.shape[1]):
			if check_state[i, j] == POINT_STATE_UNCHECKED:
				tmp_alive = is_alive(check_state, go_arr,i,j, go_arr[i,j])
				if tmp_alive == POINT_STATE_NOT_ALIVE: # once the go rule is broken, stop the searching and return the state
					is_fit_go_rule = False
					return is_fit_go_rule
			else:
				pass # pass if the point and its lined points are checked
	return is_fit_go_rule

#-------------------------------------------------------
# User strategy   --Finished
#-------------------------------------------------------
def user_step_eat(go_arr):
	'''
	:param go_arr: chessboard
	:return: ans=>where to put one step forward for white chess pieces so that some black chess pieces will be killed; user_arr=> the result chessboard after the step
	'''
	ans = []
	user_arr = go_arr
	killed_arr = []
	vitality = []
	check_state = np.zeros(user_arr.shape)
	check_state[:] = POINT_STATE_EMPYT
	tmp_indx = np.where(user_arr != 0)
	check_state[tmp_indx] = POINT_STATE_UNCHECKED
	for i in range(user_arr.shape[0]):
		for j in range(user_arr.shape[1]):
			if user_arr[i,j] == COLOR_BLACK and (len(killed_arr) == 0 or (i,j) not in killed_arr):
				killed_arr = []
				killed_arr = find_chess_block(check_state, user_arr, i, j, killed_arr, user_arr[i,j])
				for (r,t) in killed_arr:
					for (s,z) in ((r-1,t),(r+1,t),(r,t-1),(r,t+1)):
						if s in range(user_arr.shape[0]) and z in range(user_arr.shape[1]):
							if user_arr[s,z] == COLOR_NONE and (s,z) not in vitality:
								vitality.append((s,z))
							else:
								pass
						else:
							continue
				if len(vitality) == 1:
					ans.append(vitality[0])
					for (i,j) in killed_arr:
						user_arr[i,j] = COLOR_NONE
					vitality = []
				else:
					vitality = []
			else:
				continue
	if len(ans) > 0:
		for (i,j) in ans:
			user_arr[i,j] = COLOR_WHITE
	return ans, user_arr	

def user_step_possible(go_arr):
	'''
	:param go_arr: chessboard
	:return: ans=> all the possible locations to put one step forward for white chess pieces
	'''
	ans = []
	user_arr1 = copy.copy(go_arr)
	for i in range(user_arr1.shape[0]):
		for j in range(user_arr1.shape[1]):
			if user_arr1[i,j] == COLOR_NONE and (i,j) not in ans:
				user_arr1[i,j] = COLOR_WHITE
				is_fit_go_rule = go_judege(user_arr1)
				if is_fit_go_rule == True:
					ans.append((i,j))
					user_arr1[i,j] = COLOR_NONE
				else:
					user_arr1[i,j] = COLOR_NONE
			else:
				continue
	eat_ans = []
	eat_ans, user_arr1 = user_step_eat(user_arr1)
	ans = sorted(ans + eat_ans)
	return ans

def plot_possible_step(go_arr, ans,txt='The possible step: marked by red cross'):
	# Visualization of a go matrix
	# First draw a canvas with 9*9 grid
	root = Tk()
	cv = Canvas(root, width=50*(BOARD_SIZE+1), height=50*(BOARD_SIZE+1), bg='#F7DCB4')
	cv.create_text(250,10,text=txt,fill='blue')
	cv.pack(side=LEFT)
	size = 50
	for x in range(BOARD_SIZE):
		cv.create_line(size+x*size, size, size+x*size, size+(BOARD_SIZE-1)*size)
	for y in range(BOARD_SIZE):
		cv.create_line(size, size+y*size, size+(BOARD_SIZE-1)*size, size+size*y)
	# Second draw white and black circles on cross points
	offset = 20
	idx_black = np.argwhere(go_arr == COLOR_BLACK)
	idx_white = np.argwhere(go_arr == COLOR_WHITE)
	idx_ans = np.argwhere(ans)
	len_black = idx_black.shape[0]
	len_white = idx_white.shape[0]
	len_ans = idx_ans.shape[0]
	for i in range(len_black):
		if idx_black[i,0] >= BOARD_SIZE or idx_black[i,1] >= BOARD_SIZE:
			print ('IndexError: index out of range')
			sys.exit(0)
		else:
			new_x = 50*(idx_black[i,1]+1)
			# print("new_x:{}".format(new_x))
			new_y = 50*(idx_black[i,0]+1)
			cv.create_oval(new_x-offset, new_y-offset, new_x+offset, new_y+offset, width=1, fill='black', outline='black')
	for i in range(len_white):
		if idx_white[i,0] >= BOARD_SIZE or idx_white[i,1] >= BOARD_SIZE:
			print ('IndexError: index out of range')
			sys.exit(0)
		else:
			new_x = 50*(idx_white[i,1]+1)
			new_y = 50*(idx_white[i,0]+1)
			cv.create_oval(new_x-offset, new_y-offset, new_x+offset, new_y+offset, width=1, fill='white', outline='white')
	for (k,p) in ans:
		new_x = 50*(p+1)
		new_y = 50*(k+1)
		cv.create_oval(new_x-offset, new_y-offset, new_x+offset, new_y+offset, width=1, fill='red', outline='red')		
	root.mainloop()

#-------------------------------------------------------
# Write results into txt
#-------------------------------------------------------
def write_result(file_name, file_tag, results):
	with open(file_name, 'a+') as f:
		f.writelines(file_tag+'\n')
		if type(results) == bool:
			f.writelines(str(results)+'\n')
		else:
			for (i,j) in results: 
				f.writelines("{} {}".format(i,j)+'\n')
		f.writelines('\n')


if __name__ == "__main__":
	chess_rule_monitor = True
	problem_tag="Default"
	ans=[]
	user_arr=np.zeros([0,0])


	# The first problem: rule checking
	problem_tag = "Problem 0: rule checking"
	go_arr = read_go('{}_0.txt'.format(file_tag))
	plot_go(go_arr, problem_tag)
	chess_rule_monitor=go_judege(go_arr)
	print ("{}:{}".format(problem_tag, chess_rule_monitor))
	plot_go(go_arr, '{}=>{}'.format(problem_tag, chess_rule_monitor))
	write_result("answer_for_train.txt", file_tag+"_0",chess_rule_monitor)

	problem_tag = "Problem 00: rule checking"
	go_arr = read_go('{}_00.txt'.format(file_tag))
	plot_go(go_arr, problem_tag)
	chess_rule_monitor = go_judege(go_arr)
	print ("{}:{}".format(problem_tag, chess_rule_monitor))
	plot_go(go_arr, '{}=>{}'.format(problem_tag, chess_rule_monitor))
	write_result("answer_for_train.txt", file_tag+"_00",chess_rule_monitor)

	# The second~fifth prolbem: forward one step and eat the adverse points on the chessboard
	for i in range(1,5):
	  problem_tag = "Problem {}: forward on step".format(i)
	  go_arr = read_go('{}_{}.txt'.format(file_tag, i))
	  plot_go(go_arr, problem_tag)
	  chess_rule_monitor = go_judege(go_arr)
	  ans, user_arr = user_step_eat(go_arr) # need finish
	  print ("{}:{}".format(problem_tag, ans))
	  plot_go(user_arr, '{}=>{}'.format(problem_tag, ans))
	  write_result("answer_for_train.txt", file_tag+"_"+str(i), ans)


	# The sixth problem: find all the postion which can place a white chess pieces
	problem_tag = "Problem {}: all possible position".format(5)
	go_arr = read_go('{}_{}.txt'.format(file_tag,5))
	plot_go(go_arr, problem_tag)
	chess_rule_monitor = go_judege(go_arr)
	ans = user_step_possible(go_arr) # need finish
	print ("{}:{}".format(problem_tag, ans))
	# plot_go(go_arr, '{}=>{}'.format(problem_tag, chess_rule_monitor))
	plot_possible_step(go_arr, ans, '{}=>{}'.format(problem_tag, "Marked by red ovals"))
	write_result("answer_for_train.txt", file_tag+"_5", ans)
