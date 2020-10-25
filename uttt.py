import sys
import math
import random
import time
import copy
import mcts as MonteCarlo

p1 = 0
p2 = 1
mapSize = 3

winPositions = set()

mapSize = 3

Nmast = {}
Wmast = {}

Nmast[p1] = {}
Nmast[p2] = {}
Wmast[p1] = {}
Wmast[p2] = {}


def UpdateNmast(N, moves, player):
	for i in range(N):
		it = len(moves) - N + i
		if(tuple(moves[max(it,0) : len(moves)]) not in Nmast[player]):
			Nmast[player][tuple(moves[max(it,0) : len(moves)])] = 1
		else:
			Nmast[player][tuple(moves[max(it,0) : len(moves)])] += 1

def UpdateWmast(N, moves, player):
	for x in range(0,len(moves), 2):
		for i in range(N):
			it = len(moves) - N + i
			if(tuple(moves[x+i : x + N]) not in Wmast[player]):
				Wmast[player][tuple(moves[x+i : x + N])] = 1
			else:
				Wmast[player][tuple(moves[x+i : x + N])] += 1


def SetMast():
	for i in range(mapSize * mapSize):
		for j in range(mapSize * mapSize):
			Nmast[p1][(i,j)] = 0
			Nmast[p2][(i,j)] = 0
			Wmast[p1][(i,j)] = 0
			Wmast[p2][(i,j)] = 0

def Qmast(player, actions):
	if(actions not in Nmast[player] or actions not in Wmast[player]):
		return 0.0

	if(Nmast[player][actions] == 0):
		return 0.0

	return Wmast[player][actions]/Nmast[player][actions]


boardCoords = []
for i in range(3):
	for j in range(3):
		boardCoords.append((j,i))

SetMast()


for y in range(mapSize):
	lineX = []
	lineY = []
	for x in range(mapSize):
		lineX.append((x, y))
		lineY.append((y, x))
	winPositions.add(tuple(lineX))
	winPositions.add(tuple(lineY))

winPositions.add(tuple([(0, 0), (1, 1), (2, 2)]))
winPositions.add(tuple([(2, 0), (1, 1), (0, 2)]))

def checkBoard(Os, Xs, BigOs, BigXs, Draws):
	
	newBigOs = set()
	newBigXs = set()
	newDraws = set()

	for b in range(mapSize): #OOO
		for a in range(mapSize):
			if((a, b) not in BigOs and (a, b) not in BigXs and (a, b) not in Draws):
				for line in winPositions:
					win = True
					for pos in line:
						if(pos not in Os):
							win = False
							break
					if(win):
						newBigOs.add((a, b))
						break

	newBigOs = newBigOs.union(BigOs)

	for b in range(mapSize): #XXX
		for a in range(mapSize):
			if((a, b) not in BigOs and (a, b) not in BigXs and (a, b) not in Draws):
				for line in winPositions:
					win = True
					for pos in line:
						if(pos not in Xs):
							win = False
							break
					if(win):
						newBigXs.add((a, b))
						break

	newBigXs = newBigXs.union(BigXs)

	for b in range(mapSize): #DRAWS
		for a in range(mapSize):
			if((a, b) not in BigOs and (a, b) not in BigXs and (a, b) not in Draws):
				tileCounter = 0
				for y in range(mapSize):
					for x in range(mapSize):
						if((x, y) in Os or (x, y) in Xs):
							tileCounter += 1
				if(tileCounter >= mapSize*mapSize):
					newDraws.add((a, b))

	newDraws = newDraws.union(Draws)

	return newBigOs, newBigXs, newDraws


class State:
	def __init__(self, Os, Xs, BigOs, BigXs, Draws, player, curBoard):
		self.player = player

		self.Os = Os
		self.Xs = Xs

		self.curBoard = curBoard

		self.BigOs = BigOs
		self.BigXs = BigXs
		self.Draws = Draws

	def actions(self):
		ret = set()
		if(self.curBoard is None):
			for b in range(mapSize):
				for a in range(mapSize):
					if((a, b) not in self.BigOs and (a, b) not in self.BigXs and (a, b) not in self.Draws):
						for y in range(mapSize):
							for x in range(mapSize):
									if( (x + a*3, y + b*3) not in self.Os and (x + a*3, y + b*3) not in self.Xs):
										ret.add((x, y))
		else:
			for y in range(mapSize):
				for x in range(mapSize):
					pos = (x + self.curBoard[0] * 3,  y + self.curBoard[1] * 3)
					if(pos not in self.Os and pos not in self.Xs):
						ret.add((x + self.curBoard[0] * 3,  y + self.curBoard[1] * 3))
		return ret

	def MakeMove(self, a):
		if(self.player == p1):
			newOs = copy.deepcopy(self.Os)
			newOs.add(a)
			newBigOs, newBigXs, newDraws = checkBoard(newOs, self.Xs, self.BigOs, self.BigXs, self.Draws)
			newCurBoard = (a[0] % mapSize, a[1] % mapSize)
			if(newCurBoard in newBigOs or newCurBoard in newBigXs or newCurBoard in newDraws):
				newCurBoard = None
			return State(newOs, self.Xs, newBigOs, newBigXs, newDraws, 1-self.player, newCurBoard)
		else:
			newXs = copy.deepcopy(self.Xs)
			newXs.add(a)
			newBigOs, newBigXs, newDraws = checkBoard(self.Os, newXs, self.BigOs, self.BigXs, self.Draws)
			newCurBoard = (a[0] % mapSize, a[1] % mapSize)
			if(newCurBoard in newBigOs or newCurBoard in newBigXs or newCurBoard in newDraws):
				newCurBoard = None
			return State(self.Os, newXs, newBigOs, newBigXs, newDraws, 1-self.player, newCurBoard)

	def Terminal(self):
		if(self.Win(p1) or self.Win(p2)):
			return True

		if(len(self.BigOs) + len(self.BigXs) + len(self.Draws) >= 9):
			return True
		else:
			return False

	def Win(self, player):
		if(player == p1):
			for line in winPositions:
				win = True
				for pos in line:
					if(pos not in self.BigOs):
						win = False
						break
				if(win):
					return True
		else:
			for line in winPositions:
				win = True
				for pos in line:
					if(pos not in self.BigXs):
						win = False
						break
				if(win):
					return True

	def PrettyPrint(self):
		for j in range(mapSize):
			for y in range(mapSize):
				for i in range(mapSize):
					print('|', end = '')
					for x in range(mapSize):
						#print((x + i*3, y + j*3), end = '')
						
						if((x + i*3, y + j*3) in self.Os):
							print('O', end = '')
						elif((x + i*3, y + j*3) in self.Xs):
							print('X', end = '')
						else:
							print(' ', end = '')
						
						#print((x + i*3, y + j*3), end = '')
				print('|')
			print('---------------')
		print("\n\n")
		
	def __eq__(self, o):
		return self.Os == o.Os and self.Xs == o.Xs

	def __hash__(self):
		return hash(frozenset(self.Os)) ^ hash(frozenset(self.Xs))

class Node:
	def __init__(self, state, parent):
		self.visits = 1
		self.wins = 0
		self.state = state
		self.parent = parent
		self.children = set()

	def FullyExpanded(self):
		'''
		for a in self.state.actions():
			if(Node(self.state.MakeMove(a), self) not in self.children):
				return False
		return True
		'''
		#print(len(self.state.actions()), len(self.children))
		#if(len(self.state.actions()) <= len(self.children)):
		#	return True

		for a in self.state.actions():
			if(Node(self.state.MakeMove(a), self) not in self.children):
				return False

		return True

	def GetChild(self, a):
		for child in self.children:
			if(self.state.MakeMove(a) == child.state):
				return child

	def NewChild(self):
		for a in self.state.actions():
			newState = self.state.MakeMove(a)
			if(Node(newState, self) not in self.children):
				addNode = Node(newState, self)
				self.children.add(addNode)
				return addNode


	def Q(self):
		return self.wins/self.visits

	def UCB(self, t):
		c = 0.6
		return float(self.wins)/float(t) + c*math.sqrt(math.log(t)/float(self.visits))

	def GetChildUCB(self, t):
		bestVal = -10000.0
		bestChild = self
		for child in self.children:
			childUCB = child.UCB(t)
			if(childUCB > bestVal):
				bestVal = childUCB
				bestChild = child
		return bestChild

	def GetChildBestReward(self):
		bestVal = -1
		bestChild = self
		for child in self.children:
			if(child.wins/child.visits > bestVal):
				bestVal = child.wins/child.visits
				bestChild = child
		return bestChild

	def __eq__(self, o):
		return self.state == o.state

	def __hash__(self):
		return hash(self.state) ^ hash(self.parent)

def GetMoveBetweenStates(fromState, toState):
	for a in fromState.actions():
		newState = fromState.MakeMove(a)
		if(newState == toState):
			return a

class MCTS:
	def __init__(self, root):
		self.root = root
		self.time = 1
		self.MCdict = {}

	def traverse(self):
		node = self.root

		if(node.state.Terminal()):
			return node

		while(node.FullyExpanded()):
			if(node.state.Terminal()):
				return node

			node = node.GetChildUCB(self.time)

		if(node.state.Terminal()):
			return node

		return node.NewChild()

	def PlayRandomGame(self, node):
		state = copy.deepcopy(node.state)
		while(not state.Terminal()):
			act = random.choice(tuple(state.actions()))
			state = state.MakeMove(act)
			
		return state.Win(self.root.state.player)

	def BackPropagate(self, node, result):
		if(node == self.root):
			return
		node.visits += 1
		if(result):
			node.wins += 1
		self.BackPropagate(node.parent, result)

	def Search(self):
		
		leaf = self.traverse()
		
		#leaf.state.PrettyPrint()
		result = self.PlayRandomGame(leaf)
		self.BackPropagate(leaf, result)

	def GetGOTONode(self, timeLimit):#in seconds
		start = time.time()
		self.time = 1
		sim_counter = 0
		while(time.time() - start < timeLimit):
			self.Search()
			self.time += 1
			sim_counter += 1
		
		return self.root.GetChildBestReward(), sim_counter

	def ChangeRoot(self, newRootState):
		found = False
		for node in self.root.children:
			if(node.state == newRootState):
				self.root = node
				found = True
				break
		return found

class MAST:
	def __init__(self, root, eps):
		self.root = root
		self.time = 1
		self.MCdict = {}
		self.eps = eps

	def traverse(self):
		node = self.root
		if(node.state.Terminal()):
			return node
		while(node.FullyExpanded()):
			if(len(node.state.actions()) == 0):
				return node

			if(node.state.Terminal()):
				return node

			a = self.ChooseActionMAST(node.state)
			
			newNode = node.GetChild(a)
			node = newNode

		return node.NewChild()

	def ChooseActionMAST(self, state):
		#state.PrettyPrint()
		#print(len(state.Os) + len(state.Xs), state.Terminal())
		if(random.uniform(0,1) > self.eps):
			return random.choice(tuple(state.actions()))
		else:
			bestVal = -1
			bestAct = None
			for a in state.actions():
				if(Qmast(state.player, a) > bestVal):
					bestVal = Qmast(state.player, a)
					bestAct = a
			if(bestAct is None):
				return random.choice(tuple(state.actions()))

			return bestAct

	def PlayRandomGame(self, node):
		state = copy.deepcopy(node.state)
		thisGame_actions = {}
		thisGame_actions[self.root.state.player] = set()
		thisGame_actions[1-self.root.state.player] = set()
		moves = []
		while(not state.Terminal()):
			act = self.ChooseActionMAST(state)	
			Nmast[state.player][act] += 1
			thisGame_actions[state.player].add(act)
			state = state.MakeMove(act)

		if(state.Win(self.root.state.player)):
			for a in thisGame_actions[self.root.state.player]:
				Wmast[self.root.state.player][a] += 1
		else:
			for a in thisGame_actions[1-self.root.state.player]:
				Wmast[1-self.root.state.player][a] += 1

		#print('dupa3')
		return state.Win(self.root.state.player)

	def BackPropagate(self, node, result):
		if(node is None or node == self.root):
			return
		node.visits += 1
		a = GetMoveBetweenStates(node.parent.state, node.state)

		if(result):
			node.wins += 1
			Wmast[self.root.state.player][a] += 1
		else:
			Wmast[1-self.root.state.player][a] += 1

		self.BackPropagate(node.parent, result)

	def Search(self):
		leaf = self.traverse()
		if(leaf is None):
			return
		#leaf.state.PrettyPrint()
		result = self.PlayRandomGame(leaf)

		self.BackPropagate(leaf, result)

	def GetGOTONode(self, timeLimit):#in seconds
		start = time.time()
		sim_counter = 0
		while(time.time() - start < timeLimit):
			self.Search()
			self.time += 1
			sim_counter += 1

		return self.ChooseActionMAST(self.root.state), sim_counter

	def ChangeRoot(self, newRootState):
		found = False
		for node in self.root.children:
			if(node.state == newRootState):
				self.root = node
				found = True
				break
		return found


class FLAT:
	def __init__(self, root):
		self.root = root
		self.time = 1
		self.MCdict = {}

	def PlayRandomGame(self, node):
		state = copy.deepcopy(node.state)
		while(not state.Terminal()):
			act = random.choice(tuple(state.actions()))
			state = state.MakeMove(act)
			
		return state.Win(self.root.state.player)

	def MCGames(self):
		for a in self.root.state.actions():
			newState = self.root.state.MakeMove(a)
			result = self.PlayRandomGame(Node(newState, newState))
			if(result):
				win = 1
			else:
				win = 0
			
			if(newState in self.MCdict):
				self.MCdict[newState] = (self.MCdict[newState][0] + 1, self.MCdict[newState][1] + win)
			else:
				self.MCdict[newState] = (1, win)

	def GetBestActionMC(self):
		bestVal = -1000.0
		retState = None
		for state, Q in self.MCdict.items():
			if(Q[1]/Q[0] > bestVal):
				bestVal = Q[1]/Q[0]
				retState = state
		return retState

	def FlatMC(self, timeLimit):
		start = time.time()
		self.MCdict = {}
		simcount = 0
		while(time.time() - start < timeLimit):
			self.MCGames()
			simcount += len(self.root.state.actions())
			
		return self.GetBestActionMC(), simcount

	def ChangeRoot(self, newRootState):
		found = False
		for node in self.root.children:
			if(node.state == newRootState):
				self.root = node
				found = True
				break
		return found


class NST:
	def __init__(self, root, eps, N):
		self.root = root
		self.time = 1
		self.MCdict = {}
		self.eps = eps
		self.N = N

	def traverse(self):
		node = copy.deepcopy(self.root)
		if(node.state.Terminal()):
			return node

		moves = []
		while(node.FullyExpanded()):
			if(len(node.state.actions()) == 0):
				return node

			if(node.state.Terminal()):
				return node

			a = self.ChooseActionMAST(node.state, moves)
			moves.append(a)

			UpdateNmast(self.N, moves, node.state.player)

			newNode = node.GetChild(a)
			node = newNode

		return node.NewChild(), moves

	def ChooseActionMAST(self, state, hist):
		#state.PrettyPrint()
		#print(self.N)
		it = max(len(hist)-self.N + 1, 0)
		last = hist[ it : len(hist)]
		if(random.uniform(0,1) > self.eps):
			return random.choice(tuple(state.actions()))
		else:
			bestVal = -1
			bestAct = random.choice(tuple(state.actions()))
			for a in state.actions():
				thisLast = tuple(last + [a])
				if(Qmast(state.player, thisLast) > bestVal):
					bestVal = Qmast(state.player, thisLast)
					bestAct = a

			return bestAct

	def PlayRandomGame(self, node, moves):
		state = copy.deepcopy(node.state)
		thisGame_actions = {}
		thisGame_actions[self.root.state.player] = []
		thisGame_actions[1-self.root.state.player] = []
		#print('dupa2')
		firstPlayer = state.player

		while(not state.Terminal()):
			act = self.ChooseActionMAST(state, moves)
			moves.append(act)
			UpdateNmast(self.N, moves, state.player)
			#thisGame_actions[state.player].append(act)
			state = state.MakeMove(act)
		#print('dupa3')
		return state.Win(self.root.state.player), moves

	def Backup(self, moves, result):
		if(result):
			UpdateWmast(self.N, moves, self.root.state.player)
		else:
			UpdateWmast(self.N, moves[1:], 1-self.root.state.player)

	def BackPropagate(self, node, result):
		if(node is None or node == self.root):
			return
		node.visits += 1
		self.BackPropagate(node.parent, result)


	def Search(self):
		leaf, moves = self.traverse()
		if(leaf is None):
			return
		#leaf.state.PrettyPrint()
		result, moves1 = self.PlayRandomGame(leaf, moves)
		self.Backup(moves1, result)
		self.BackPropagate(leaf, result)

	def GetGOTONode(self, timeLimit, history):#in seconds
		start = time.time()
		self.time = 0
		sim_counter = 0
		while(time.time() - start < timeLimit):
			self.Search()
			self.time += 1
			sim_counter += 1
		return self.ChooseActionMAST(self.root.state, history), sim_counter

	def ChangeRoot(self, newRootState):
		for node in self.root.children:
			if(node.state == newRootState):
				self.root = node
				return True
		return False



'''
for e in range(5, 100, 5):
	eps = e/100
	thisRes = []
	
	'''

#F6WE-STXU-5E58-SXYH-N4GC

def PlayGame(state, agent1, agent2, me):
	curState = copy.deepcopy(state)

	history = []
	changeCount = 0
	firstMove = True
	firstMoveSims = 0
	simCounts = []


	while(True):
		#curState.PrintBoard()
		if(curState.Terminal()):
			break

		a = (0,0)
		if(curState.player == p2):#bot
			a, sim_counter = agent2.GetGOTONode(1.0)
			#a = GetMoveBetweenStates(curState, GOTOstate)
			#a = random.choice(tuple(curState.actions()))
		else:#me
			a, sim_counter = agent1.GetGOTONode(1.0, history)
			if(firstMove):
				firstMove = False
				firstMoveSims = sim_counter
			simCounts.append(sim_counter)
			#a = GetMoveBetweenStates(curState, GOTONode.state)


		history.append(a)
		curState = curState.MakeMove(a)
	
		#print(a)
		#curState.PrettyPrint()
		if(not agent1.ChangeRoot(curState)):
			agent1 = NST(Node(curState, None), agent1.eps, agent1.N)

		if(not agent2.ChangeRoot(curState)):
			agent2 = MAST(Node(curState, None), agent2.eps)

	#print(changeCount, "/", len(history))
	if(curState.Win(me)):
		return True, firstMoveSims, sum(simCounts)/len(simCounts)
	else:
		return False, firstMoveSims, sum(simCounts)/len(simCounts)


num_games = 100
results = {}

Qamaf = {}
Namaf = {}

eps = 0.6

myWins = 0

for it in range(num_games):
	bigState = State(set(), set(), set(), set(), set(), p1, None)
	curState = bigState

	SetMast()
	
	Qamaf = {}
	Namaf = {}

	mcts = MCTS(Node(curState, None))
	mast = MAST(Node(curState, None), eps)
	#rave = RAVE(Node(curState, curState), K)
	nst = NST(Node(curState, None), eps, 2)
	flat = FLAT(Node(curState, None))

	res, simcount_start, simcount_all = PlayGame(curState, nst, mast, p1)
	if(res):
		myWins += 1
	print(myWins,"/",it+1)
print(myWins/num_games * 100, '% wins')




'''
f = open("MASTvsMCTS.txt", "a")
for e, res in results.items():
	f.write(str(e) + ": " + str(res) + "\n")
f.close()
'''