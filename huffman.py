import numpy as np
from scipy.misc import imread,imresize
import matplotlib.pyplot as plt
from operator import itemgetter, attrgetter
import queue

class Node:
	def __init__(self):
		self.prob = None
		self.code = None
		self.data = None
		self.left = None
		self.right = None 	# the color (the bin value) is only required in the leaves
	def __lt__(self, other):
		if (self.prob < other.prob):		# define rich comparison methods for sorting in the priority queue
			return 1
		else:
			return 0
	def __ge__(self, other):
		if (self.prob > other.prob):
			return 1
		else:
			return 0
def rgb2gray(img):
	gray_img = np.rint(img[:,:,0]*0.2989 + img[:,:,1]*0.5870 + img[:,:,2]*0.1140)
	gray_img = gray_img.astype(int)
	return gray_img

def get2smallest(data):			# can be used instead of inbuilt function get(). was not used in  implementation
    first = second = 1;
    fid=sid=0
    for idx,element in enumerate(data):
        if (element < first):
            second = first
            sid = fid
            first = element
            fid = idx
        elif (element < second and element != first):
            second = element
    return fid,first,sid,second
    
def tree(probabilities):
	prq = queue.PriorityQueue()
	for color,probability in enumerate(probabilities):
		leaf = Node()
		leaf.data = color
		leaf.prob = probability
		prq.put(leaf)

	while (prq.qsize()>1):
		newnode = Node()		# create new node
		l = prq.get()
		r = prq.get()			# get the smalles probs in the leaves
						# remove the smallest two leaves
		newnode.left = l 		# left is smaller
		newnode.right = r
		newprob = l.prob+r.prob	# the new prob in the new node must be the sum of the other two
		newnode.prob = newprob
		prq.put(newnode)	# new node is inserted as a leaf, replacing the other two 
	return prq.get()		# return the root node - tree is complete

def huffman_traversal(root_node,tmp_array,f):		# traversal of the tree to generate codes
	if (root_node.left is not None):
		tmp_array[huffman_traversal.count] = 1
		huffman_traversal.count+=1
		huffman_traversal(root_node.left,tmp_array,f)
		huffman_traversal.count-=1
	if (root_node.right is not None):
		tmp_array[huffman_traversal.count] = 0
		huffman_traversal.count+=1
		huffman_traversal(root_node.right,tmp_array,f)
		huffman_traversal.count-=1
	else:
		huffman_traversal.output_bits[root_node.data] = huffman_traversal.count		#count the number of bits for each color
		bitstream = ''.join(str(cell) for cell in tmp_array[1:huffman_traversal.count]) 
		color = str(root_node.data)
		wr_str = color+' '+ bitstream+'\n'
		f.write(wr_str)		# write the color and the code to a file
	return

# Read an bmp image into a numpy array
img = imread('tiger.bmp')
img = imresize(img,10)		# resize to 10% (not strictly necessary - done for faster computation)

# convert to grayscale
gray_img = rgb2gray(img)

# compute histogram of pixels
hist = np.bincount(gray_img.ravel(),minlength=256)

probabilities = hist/np.sum(hist)		# a priori probabilities from frequencies

root_node = tree(probabilities)			# create the tree using the probs.
tmp_array = np.ones([64],dtype=int)
huffman_traversal.output_bits = np.empty(256,dtype=int) 
huffman_traversal.count = 0
f = open('codes.txt','w')
huffman_traversal(root_node,tmp_array,f)		# traverse the tree and write the codes

input_bits = img.shape[0]*img.shape[1]*8	# calculate number of bits in grayscale 
compression = (1-np.sum(huffman_traversal.output_bits*hist)/input_bits)*100	# compression rate
print('Compression is ',compression,' percent')
