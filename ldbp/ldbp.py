#========================================================#
# Learned Digital Backpropagation (LDBP)
# Author: Christian Haeger (christian.haeger@chalmers.se)
# Last modified: December, 2018
#========================================================#
# imports and constants {{{
#========================================================#
import tensorflow as tf
import sys # sys.exit()
import warnings
import os # os.path.exists(), os.environ['v'], os.makedirs()
import numpy as np
import scipy as sp # sp.fft(), sp.linalg.solve(), sp.optimize.fsolve(), sp.special.erf()
import time # time.gmtime(), time.strftime()
import math # math.isnan()
import random # random.shuffle()
import threading # threading.Thread()
import multiprocessing
import argparse as ap
import configparser
import shutil # shutil.copyfile(src, dst)
from lib import fir # fir.cd_fir_filter()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # avoid TF logging output for GPU devices etc. 
sp.set_printoptions(precision = 4, suppress = True)
np.set_printoptions(precision = 4)
#np.seterr(divide='ignore', invalid='ignore') # ignore "divide by 0" warning

# constants
co_h = 6.6260657e-34
co_c0 = 299792458
co_lambda = 1550.0e-9
co_dB = 10.0*np.log10(np.exp(1.0))
nu = co_c0/co_lambda
dB_conv = 4.342944819032518

# }}}
#========================================================#
# functions {{{
#========================================================#
def rrcosine(rolloff, delay, OS):
	""" Root-raised cosine filter for pulse shaping
	Args:
		rolloff: between 0 and 1
		delay: in symbols 
		OS: oversampling factor (samples per symbol)
	
	Returns:
		A vector of length 2*(OS*delay)+1
	"""
	rrcos = np.zeros(2*delay*OS+1)
	rrcos[delay*OS] = 1 + rolloff*(4/np.pi-1)
	for i in range(1,delay*OS+1):
		t = i/OS
		if(t == 1/4/rolloff):
			val = rolloff/np.sqrt(2)*((1+2/np.pi)*np.sin(np.pi/(4*rolloff)) + (1-2/np.pi)*np.cos(np.pi/(4*rolloff)))
		else:
			val = (np.sin(np.pi*t*(1-rolloff)) + 4*rolloff*t*np.cos(np.pi*t*(1+rolloff))) / (np.pi*t*(1-(4*rolloff*t)**2))
		rrcos[delay*OS+i] = val
		rrcos[delay*OS-i] = val
	return rrcos / np.sqrt(np.sum(rrcos**2))

def get_fvec(N,fs):
	return np.concatenate((np.linspace(0,N//2-1,N//2), np.linspace(-N//2,-1,N//2))) * fs/N

def tf_print(tmp_var):
	init = tf.global_variables_initializer()
	sess = tf.Session()
	sess.run(init)
	print(sess.run(tmp_var))

def get_optimizer():
	if optimizer == "adam":
		opt = tf.train.AdamOptimizer(learning_rate, float(conf_train['adam_A']), float(conf_train['adam_B']))
	elif optimizer == "rmsprop":
		opt = tf.train.RMSPropOptimizer(learning_rate, float(conf_train['rmsprop_A']), float(conf_train['rmsprop_B']))
	elif optimizer == "adadelta":
		opt = tf.train.AdadeltaOptimizer(learning_rate, float(conf_train['adadelta_A']))
	elif optimizer == "adagrad":
		opt = tf.train.AdagradOptimizer(learning_rate, float(conf_train['adagrad_A']))
	else:
		raise ValueError("wrong optimizer string: optimizer = '"+optimizer+"'")
	return opt

def complex_multiply(x,y):
	"""
	Args:
		x: Tensor of shape = [batch_size, N, 2]
		y: Tensor of shape = [batch_size, N, 2]
	
	Returns:
		A Tensor of shape = [batch_size, N, 2]
	"""
	xr = x[:,:,0]
	xi = x[:,:,1]
	yr = y[:,:,0]
	yi = y[:,:,1]
	return tf.stack([xr*yr-xi*yi, xr*yi+xi*yr], axis=2)

def tf_real_filter(init_coeffs, opt=False):
	""" Create arbitrary FIR filter with real coefficients
	
	Args:
		init_coeffs: real numpy array of shape [filter_length,] with initial filter coefficients
		opt: (Optional) True for variable, False for constant, default=False
	
	Returns: 
		A Tensor with shape = [filter_length]
	"""
	
	if opt == True:
		h = tf.Variable(init_coeffs, dtype=tf.float32)
	else:
		h = tf.constant(init_coeffs, tf.float32)
	
	return h

def tf_real_symmetric_filter(init_coeffs, opt=False, mask=1):
	""" Create odd-length symmetric FIR filter with real coefficients
	
	For symmetric filters of length 2*L+1, there are (L+1) tunable parameters:
		coeffs: h_L, ..., h_1, [h_0, h_1, ..., h_L]
	
	Args:
		init_coeffs: real numpy array of shape [filter_length,] with initial filter coefficients, 
			filter_length should be odd
			coefficients should be symmetric
		opt: (Optional) True for variable, False for constant, default=False
		mask: (Optional) binary mask for pruning, default=1
	
	Returns: 
		A Tensor with shape = [filter_length]
	"""
	
	filter_length = len(init_coeffs)
	if filter_length%2 is 0:
		raise ValueError("filter length has to be odd: filter_length = {}".format(filter_length))
	filter_delay = (filter_length-1)//2
	right_half = init_coeffs[filter_delay::]
	
	for i in range(filter_delay): # check if symmetric
		absdiff = abs(init_coeffs[i] - right_half[filter_delay-i])
		if absdiff > 1e-2:
			warnings.warn("inital filter coefficients are not symmetric: absolute difference = {}".format(absdiff))
	
	if opt == True:
		h_vars = tf.Variable(right_half, dtype=tf.float32)
	else:
		h_vars = tf.constant(right_half, tf.float32)
	
	hmasked = h_vars*mask # apply binary mask for pruning
	return tf.concat([tf.reverse(hmasked[1:], axis=[0]), hmasked], axis=0)

def tf_complex_symmetric_filter(init_coeffs, opt=False, mask=1):
	""" Create odd-length symmetric FIR filter with complex coefficients
	
	For complex-valued symmetric filters of length 2*L+1, there are 2*(L+1) tunable parameters:
		real coeffs: h_L, ..., h_1, [h_0, h_1, ..., h_L]
		imag coeffs: g_L, ..., g_1, [g_0, g_1, ..., g_L]
	
	Args:
		init_coeffs: complex numpy array of shape [filter_length,] with initial filter coefficients, 
			filter_length should be odd
			coefficients should be symmetric
		opt: (Optional) True for variable, False for constant, default=False
		mask: (Optional) binary mask for pruning, default=1
	
	Returns:
		A Tensor with shape = [filter_length, 2]
			column 1: real coefficients
			column 2: imaginary coefficients
	"""
	
	h_real = tf_real_symmetric_filter(np.real(init_coeffs), opt, mask)
	h_imag = tf_real_symmetric_filter(np.imag(init_coeffs), opt, mask)
	return tf.stack([h_real, h_imag], axis=1)

def cconv(x, h):
	""" y = cconv(x, h) uses tf.nn.conv1d to perform circular convolution of signal x with filter h 
	
	Notes: 
	- This function also circularly shifts y to remove the filter delay caused by h
	- The delay is (filter_length-1)/2 for odd-length filters and filter_length/2-1 for even-length filters
	- TensorFlow does not do convolution but correlation, so the filter "flipping" 
		is performed manually in this function.
	
	Args:
		x: signal Tensor
			shape(x) = [batch_size, N]     real signal
			shape(x) = [batch_size, N, 2]  complex signal
		h: filter Tensor
			shape(h) = [filter_length]     real filter
			shape(h) = [filter_length, 2]  complex filter
	
	Returns:
		A Tensor with shape
			shape(y) = [batch_size, N]     if both x and h are real
			shape(y) = [batch_size, N, 2]  all other cases
	"""
	
	filter_length = int(h.shape[0])
	batch_size = int(x.shape[0])
	N = int(x.shape[1])
	
	# expand dimensions in case of real signal and/or filter
	if len(x.shape) == 2:
		x = tf.expand_dims(x, axis=2)
	
	if len(h.shape) == 1:
		h = tf.expand_dims(h, axis=1)
	
	# extend x to achieve circular convolution and remove the filter delay
	if(filter_length%2==0): # even-length filters
		filter_delay = filter_length//2-1
		x = tf.concat([x[:, N-filter_delay-1:, :], x, x[:, :filter_delay, :]], axis=1) # [x_end, x, x_begin]
	else: # odd-length filters
		filter_delay = (filter_length-1)//2
		x = tf.concat([x[:, N-filter_delay:, :], x, x[:, :filter_delay, :]], axis=1) # [x_end, x, x_begin]
	
	# reshape filter 
	if(x.shape[2] == 1 and h.shape[1] == 1): # real signal, real filter
		h = tf.reshape(h, [filter_length, 1, 1])
		conv1d_filter = tf.reverse(h, axis=[0]) # flip
	elif(x.shape[2] == 1 and h.shape[1] == 2): # real signal, complex filter
		hr = tf.reshape(h[:,0], [filter_length, 1, 1])
		hr = tf.reverse(hr, axis=[0]) # flip
		hi = tf.reshape(h[:,1], [filter_length, 1, 1])
		hi = tf.reverse(hi, axis=[0]) # flip
		conv1d_filter = tf.concat([hr, hi], axis=2)
	elif(x.shape[2] == 2 and h.shape[1] == 1): # complex signal, real filter
		hr = tf.reshape(h[:,0], [filter_length, 1, 1])
		hr = tf.reverse(hr, axis=[0]) # flip
		z = tf.zeros(shape=[filter_length, 1, 1]) # dummy 
		filter_1 = tf.concat([hr, z], axis=1)
		filter_2 = tf.concat([z , hr], axis=1)
		conv1d_filter = tf.concat([filter_1, filter_2], axis=2)
	elif(x.shape[2] == 2 and h.shape[1] == 2): # complex signal, complex filter
		hr = tf.reshape(h[:,0], [filter_length, 1, 1])
		hr = tf.reverse(hr, axis=[0]) # flip
		hi = tf.reshape(h[:,1], [filter_length, 1, 1])
		hi = tf.reverse(hi, axis=[0]) # flip
		filter_1 = tf.concat([hr, -hi], axis=1)
		filter_2 = tf.concat([hi,  hr], axis=1)
		conv1d_filter = tf.concat([filter_1, filter_2], axis=2)
	else:
		raise ValueError("signal or filter has wrong shape")
	
	# call conv1d
	#   input  has shape = [batch_size, N, in_channels]
	#   filter has shape = [filter_length, in_channels, out_channels]
	#   output has shape = [batch_size, N, out_channels]
	y = tf.nn.conv1d(x, conv1d_filter, stride=1, padding="VALID", name='conv1d')
	
	if y.shape[2] == 1: # real signal and filter
		return y[:,:,0]
	else:
		return y

def periodically_extend(x, M):
	""" Extends a numpy vector of length N to length M>N by periodically copying the elements """
	N = x.shape[0]
	y = np.zeros(M, dtype=x.dtype);
	for i in range(M):
		y[i] = x[i%N]
	return y

def line2array(line):
	''' converts a string of comma-separated numbers to numpy array '''
	return np.array([float(v) for v in line.strip().split(",")])

def effective_length(length, alpha_lin):
	if alpha_lin == 0:
		return length
	else:
		return (1-np.exp(-alpha_lin*length))/alpha_lin

class ssfm_parameters:
	""" handles parameters related to the split-step Fourier method (SSFM)
	
	Initialization is performed with a dictionary that should have the following keys:
		step_size_method
			logarithmic
			linear 
			step_size 
			predefined 
		StPS: steps per span (only for logarithmic and linear)
		adjusting_factor: recommended is 0.4 (only for logarithmic)
		ssfm_method
			symmetric: linear->nonlinear->linear 
			asymmetric: linear->nonlinear
		combine_half_steps: wether to combine half-steps of adjacent spans (only for symmetric)
		alpha: attenuation parameter; should be 0 for less steps than spans
		beta2: dispersion parameter
		gamma: nonlinear parameter
		Nsp: number of spans
		Lsp: span length [m]
		fsamp: sampling frequency
		Nsamp: length of the assumed FFT
		direction: +1 for forward, -1 for backpropagation
	
	computed attributes:
		model_steps
		cd_length
		nl_param
		nl_length (not used)
	 
	Usage example: 
	
	bw = ssfm_parameters(parameter_dict)
	for NN in range(bw.model_steps):
		u = sp.ifft(bw.get_cd_filter_freq(NN)*sp.fft(u))
		u = u*np.exp(1J*bw.nl_param[NN]*np.abs(u)**2)
	"""
	
	def __init__(self, opts):
		self.__dict__.update(opts) # converts all dictionary entries to attributes 
		
		alpha_lin = self.alpha/(10*np.log10(np.exp(1)))
		Nsp = self.Nsp
		Lsp = self.Lsp
		direction = self.direction
		
		if direction == +1 and self.Nsp > 1:
			raise ValueError("forward propagation valid only for 1 span")
		
		if self.step_size_method == 'logarithmic':
			if 'adjusting_factor' not in opts:
				self.adjusting_factor = 0.4 # 0: linear, 1: very logarithmic
		
		if 'combine_half_steps' not in opts:
			self.combine_half_steps = True
		
		if self.step_size_method == 'step_size': # used only for subband processing
			step_size = self.step_size
			Ltot = Lsp*Nsp
			model_steps = int(np.floor(Ltot/step_size)+1)
			last_step_size = Ltot - (model_steps-1)*step_size
			
			cd_length = step_size*np.ones(model_steps)
			cd_length[model_steps-1] = last_step_size
			
			tmp = np.mod(np.cumsum(cd_length), Lsp)
			len_before = np.zeros(model_steps)
			len_after = np.zeros(model_steps)
			amplifier_location = np.zeros(model_steps)
			for NN in range(1, model_steps):
				if(tmp[NN-1] > tmp[NN]):
					amplifier_location[NN] = 1;
					len_after[NN] = tmp[NN]
					len_before[NN] = cd_length[NN] - len_after[NN]
			amplifier_location[0] = 1
			amplifier_location[-1] = 0
			
			nl_length = np.zeros(model_steps)
			eff_len_before = np.zeros(model_steps)
			for NN in range(model_steps):
				if (amplifier_location[NN] == 1) and (NN != 0):
					h = len_after[NN]
					eff_len_before[NN] = effective_length(len_before[NN],np.abs(alpha_lin))
				else:
					h = cd_length[NN]
				nl_length[NN] = effective_length(h,np.abs(alpha_lin))
		else:
			StPS = self.StPS
			# ====================================================== #
			# compute step sizes for one span
			# ====================================================== #
			if self.step_size_method == 'logarithmic':
				alpha_adj = self.adjusting_factor*alpha_lin
				delta = (1-np.exp(-alpha_adj*Lsp))/StPS
				if(direction == -1):
					nn = np.arange(StPS)+1    # 1,2,...,StPS
				else:
					nn = StPS-np.arange(StPS) # StPS,...,2,1
				step_size = -1/(alpha_adj) * np.log((1-(StPS-nn+1)*delta)/(1-(StPS-nn)*delta))
			elif self.step_size_method == "linear":
				step_size = Lsp/StPS*np.ones(StPS)
			else:
				raise ValueError("wrong step_size_method given (should be 'linear' or 'logarithmic'): "+self.step_size_method)
			# ====================================================== #
			# compute cd_length, nl_length, amplifier_location
			# ====================================================== #
			if self.ssfm_method == "symmetric":
				if self.combine_half_steps == True:
					model_steps = Nsp*StPS+1
					cd_length = np.zeros(model_steps)
					nl_length = np.zeros(model_steps)
					for NN in range(Nsp):
						for MM in range(StPS):
							cd_length[NN*StPS+MM] = step_size[MM]/2 + step_size[(MM+StPS-1)%StPS]/2
							nl_length[NN*StPS+MM] = step_size[MM]
					cd_length[0] = step_size[0]/2
					cd_length[model_steps-1] = step_size[StPS-1]/2
					
					amplifier_location = np.zeros(model_steps)
					amplifier_location[:-1:StPS] = 1
				else:
					model_steps = Nsp*(StPS+1)
					cd_length = np.concatenate([[step_size[0]/2], (step_size[0:-1]+step_size[1:])/2, [step_size[-1]/2]])
					cd_length = np.tile(cd_length, Nsp)
					nl_length = np.concatenate([step_size, [0]])
					nl_length = np.tile(nl_length, Nsp)
					
					amplifier_location = np.zeros(model_steps)
					amplifier_location[::StPS+1] = 1
			elif self.ssfm_method == "asymmetric":
				model_steps = Nsp*StPS
				cd_length = np.zeros(model_steps)
				nl_length = np.zeros(model_steps)
				for NN in range(Nsp):
					for MM in range(StPS):
						cd_length[NN*StPS+MM] = step_size[MM]
						nl_length[NN*StPS+MM] = effective_length(step_size[MM], np.abs(alpha_lin))
				
				amplifier_location = np.zeros(model_steps)
				amplifier_location[::StPS] = 1
			else:
				raise ValueError("wrong split step method given (should be 'symmetric' or 'asymmetric'): "+self.ssfm_method)
		# ====================================================== #
		# compute attenuation and nl_param
		# ====================================================== #
		nl_param = direction*self.gamma*nl_length
		
		attenuation = np.exp(-direction*alpha_lin*cd_length/2)
		for NN in range(model_steps):
			if direction == -1 and amplifier_location[NN] == 1:
				attenuation[NN] = attenuation[NN] * np.exp(direction*alpha_lin*Lsp/2)
		
		# re-normalize nl_param
		for NN in range(model_steps):
			nl_param[NN] = nl_param[NN]*np.prod(attenuation[0:NN+1:])**2
		
		if self.step_size_method == "step_size":
			for NN in range(model_steps):
				if amplifier_location[NN] == 1:
					nl_param[NN] = nl_param[NN] + direction*self.gamma*eff_len_before[NN]
		
		self.model_steps = model_steps
		self.cd_length = cd_length
		self.nl_length = nl_length
		self.nl_param = nl_param
		
		N = self.Nsamp
		self.fvec = np.concatenate((np.linspace(0,N//2-1,N//2), np.linspace(-N//2,-1,N//2))) * self.fsamp/N
	
	def get_cd_filter_freq(self, NN):
		return np.exp(1j*(self.beta2/2)*(2*np.pi*self.fvec)**2*(self.direction*self.cd_length[NN]))

def ordered_direct_product(A,B):
	p = A.shape[0]
	q = B.shape[0]
	n = A.shape[1]
	m = B.shape[1]
	
	C = np.zeros([p*q,n+m])
	for i in range(q):
		C[i*q:(i+1)*q:, :n:] = A[i,:]
	for i in range(p):
		C[i*q:(i+1)*q:, n::] = B
	return C

def QAM(M):
    Msqrt = (np.sqrt(M)).astype(np.int)
    if Msqrt**2 != M:
        raise ValueError("M has to be of the form M=4^m where m>0")
    x_pam = np.expand_dims(-(Msqrt-2*np.arange(start=1, stop=Msqrt+1)+1), axis=1)
    x_qam = ordered_direct_product(x_pam, x_pam)
    const = x_qam[:,0] + 1j * x_qam[:,1]
    return const/np.sqrt(np.mean(np.abs(const)**2))

# }}}
#========================================================#
# parse function arguments {{{
#========================================================#
parser = ap.ArgumentParser("python3 ldbp.py")
parser.description = "Learned Digital Backpropagation (LDBP)"
parser.add_argument("P", help="set of training powers in dB, e.g., [5] or [5,6,7]")
parser.add_argument("Lr", help="learning rate, e.g., 0.01")
parser.add_argument("iter", help="gradient descent iterations, e.g., 1000")
parser.add_argument("-c", "--config_path", help="path to configuration file (default is ldbp_config.ini)", default="ldbp_config.ini")
parser.add_argument("-l", "--logdir", help="directory for log files (default is log)", default="log")
parser.add_argument("-t", "--timing", help="time the forward propagation", action="store_true")

args = parser.parse_args()
args_dict = vars(args) # converts to a dictionary

opt_list="P,Lr,iter".split(",")
arg_str = ""
for i in range(len(opt_list)):
	arg_str += opt_list[i]
	arg_str += args_dict[opt_list[i]]
	if(i != len(opt_list)-1):
		arg_str += "_"

config_path = args.config_path
P_dB_r = np.asarray(eval(args.P))
P_W_r = pow(10, P_dB_r/10)*1e-3
iterations = int(args.iter)
learning_rate = float(args.Lr)

# }}}
#========================================================#
# read config file {{{
#========================================================#
defaults = {
	# system
	"sigma scaling"             : "1",
	"modulation"                : "16-QAM",
	# LDBP
	"combine half-steps"        : "yes",
	"load cd filter"            : "no",
	"load cd filter filename"   : "parameters.csv",
	"optimize cd filters"       : "yes",
	"optimize Kerr parameters"  : "no",
	"complex Kerr parameters"   : "no",
	"tied Kerr parameters"      : "no",
	"pruning"                   : "no",
	"less steps than spans"     : "no",
	"cd alpha"                  : "1",
	"cd filter length margin"   : "2.0",
	"cd filter length minimum"  : "13",
	"nl alpha"                  : "1",
	"nl filter length"          : "1",
	# training
	"adam_A"                    : "0.9", # decay for running average of the gradient
	"adam_B"                    : "0.999", # decay for running average of the square of the gradient
	"rmsprop_A"                 : "0.9",
	"rmsprop_B"                 : "0.1",
	"adadelta_A"                : "0.1",
	"adagrad_A"                 : "0.1",
	# data
	'forward step size method'  : 'logarithmic',
	'forward split step method' : 'symmetric'
}

config = configparser.ConfigParser(defaults)

config_folder, config_file = os.path.split(config_path)
print("configuration file name: '"+config_file+"'")

if not os.path.exists(config_path):
	raise RuntimeError("config file in '"+config_file+"' does not exist")
config.read(config_path)

# system parameters
conf_sys      = config['system parameters']
Lsp           = conf_sys.getfloat('span length [km]')*1.0e3
alpha         = conf_sys.getfloat('alpha [dB/km]')*1.0e-3
gamma         = conf_sys.getfloat('gamma [1/W/km]')*1.0e-3
noise_figure  = conf_sys.getfloat('amplifier noise figure [dB]')
sigma_scaling = conf_sys.getfloat('sigma scaling')
Nsp           = conf_sys.getint('number of spans')
fsym          = conf_sys.getfloat('symbol rate [Gbaud]')*1.0e9
modulation    = conf_sys['modulation']
rolloff       = conf_sys.getfloat('RRC roll-off')
delay         = conf_sys.getint('RRC delay')
lp_bandwidth  = conf_sys.getfloat('low-pass filter bandwidth [GHz]')*1.0e9
Nsym          = conf_sys.getint('data symbols per block')
OS_a          = conf_sys.getint('analog oversampling')
OS_d          = conf_sys.getint('digital oversampling')
if config.has_option('system parameters', 'D [ps/nm/km]'):
	D     = conf_sys.getfloat('D [ps/nm/km]')*1.0e-6
	beta2 = -D*co_lambda**2/(2*np.pi*co_c0)
else:
	beta2 = conf_sys.getfloat('beta2 [ps^2/km]')*1.0e-27

# LDBP parameters
conf_ldbp              = config['LDBP parameters']
step_size_method_bw    = conf_ldbp['step size method']
ssfm_method_bw         = conf_ldbp['split step method']
combine_half_steps     = conf_ldbp['combine half-steps']
cd_opt                 = conf_ldbp.getboolean('optimize cd filters')
cd_alpha               = conf_ldbp.getfloat('cd alpha')
load_cd_filter         = conf_ldbp.getboolean('load cd filter')
if load_cd_filter == True:
	cd_filter_filename  = conf_ldbp['load cd filter filename']
else:
	cd_filter_method    = conf_ldbp['cd filter method']
	cd_filter_bandwidth = conf_ldbp.getfloat('cd filter bandwidth')
	cd_filter_oob_gain  = conf_ldbp.getfloat('cd filter max out-of-band gain') # 0.58 for 17-taps 20 Gbaud
nl_opt                 = conf_ldbp.getboolean('optimize Kerr parameters')
if nl_opt == True:
	tied_Kerr           = conf_ldbp.getboolean('tied Kerr parameters')
nl_alpha               = conf_ldbp.getfloat('nl alpha')
nl_filter_length       = conf_ldbp.getint('nl filter length')
less_steps_than_spans  = conf_ldbp.getboolean('less steps than spans')

# training parameters
conf_train       = config['training']
minibatch_size   = conf_train.getint('minibatch size')
optimizer        = conf_train['optimizer']
summary_interval = conf_train.getint('summary writing interval')
SAVE_FILE        = conf_train.getboolean('save results to file')

# data generation
conf_data           = config['data generation']
StPS_fw             = conf_data.getint('forward steps per span')
step_size_method_fw = conf_data['forward step size method']
ssfm_method_fw      = conf_data['forward split step method']
QMAX                = conf_data.getint('number of queue elements') # number of queue elements
QBSIZE              = conf_data.getint('generation batch size') # batch size to pupulate the queue
NPROC               = conf_data.getint('number of parallel processors') # number of processors used to populate the queue
REPF                = conf_data.getint('data replication factor') # replication factor for data

if OS_a%OS_d != 0:
	raise ValueError('oversampling factors have to be divisible: OS_a={}, OS_d={}'.format(OS_a, OS_d))

if nl_filter_length%2 == 0:
	raise ValueError('nl_filter_length has to be odd: nl_filter_length = {}'.format(nl_filter_length))
nl_filter_delay = (nl_filter_length-1)//2

# derived parameters
L = Lsp*Nsp
Gain = 10.0**(alpha*Lsp/10.0)
sef = 10.0**(noise_figure/10.0)/2.0/(1.0-1.0/Gain)
alpha_lin = alpha / dB_conv
N0 = sigma_scaling*Nsp*(np.exp(alpha_lin*Lsp)-1.0)*co_h*nu*sef
sigma2 = N0 * fsym * OS_a
Nsamp_a = Nsym*OS_a
Nsamp_d = Nsym*OS_d
fsamp_a = fsym*OS_a
fsamp_d = fsym*OS_d
f_a = get_fvec(Nsamp_a, fsamp_a)
f_d = get_fvec(Nsamp_d, fsamp_d)

if "QAM" in modulation:
    splitstr = modulation.split("-")
    modulation_order = int(splitstr[0])
    modulation = "QAM"

print("total memory of the data queue: {} MB".format(QMAX*64*(Nsamp_d+Nsym)/8/1e6))

# }}}
#========================================================#
# forward propagation generative model {{{
#========================================================#
ps_filter_tx_coeffs = rrcosine(rolloff, delay, OS_a) # pulse shaping filter
ps_filter_tx_length = 2*(OS_a*delay)+1
ps_filter_tx_delay = OS_a*delay # delay in samples

# pre-compute frequency responses
ps_tmp = np.concatenate((ps_filter_tx_coeffs, np.zeros(Nsamp_a-ps_filter_tx_length)))
ps_tmp = np.roll(ps_tmp, -ps_filter_tx_delay)
ps_filter_tx_freq = sp.fft(ps_tmp, n=Nsamp_a)
lp_filter_freq = (abs(f_a) <= lp_bandwidth/2).astype(float)

if modulation == "QAM":
    const = QAM(modulation_order)

ssfm_opts = {
    "alpha": alpha,
    "beta2": beta2,
    "gamma": gamma,
    "Nsp": 1,
    "Lsp": Lsp,
    "fsamp": fsamp_a,
    "Nsamp": Nsamp_a,
    "step_size_method": step_size_method_fw,
    "ssfm_method": ssfm_method_fw,
    "StPS": StPS_fw,
    "direction": 1
}

fw = ssfm_parameters(ssfm_opts)

def forward_propagation():
    """
    Returns:
        y: received signal (shape = [Nsamp_d, 2], separate real and imaginary part)
        x: symbol vector (shape = [Nsym], complex)
        P: launch power (in W)
    """
    np.random.seed() # new seed is necessary for multiprocessor 
    P = P_W_r[np.random.randint(P_W_r.shape[0])] # get random launch power
    # [SOURCE] random points from the signal constellation
    if modulation == "QAM":
        x = const[np.random.randint(const.shape[0], size=[1, Nsym])]
    elif modulation == "Gaussian":
        x = (np.random.normal(0,1,size=[1, Nsym]) + 1j*np.random.normal(0,1,size=[1, Nsym]))/np.sqrt(2)
    else:
        raise ValueError("wrong modulation format: " + modulation)
    # [MODULATION] upsample + pulse shaping
    x_up = np.zeros([1, Nsamp_a], dtype=np.complex64)
    x_up[:, ::OS_a] = x*np.sqrt(OS_a)
    u = sp.ifft(sp.fft(x_up)*ps_filter_tx_freq)*np.sqrt(P)
    # [CHANNEL] simulate forward propagation
    for NN in range(Nsp): # enter a span
        for MM in range(fw.model_steps): # enter a segment
            u = sp.ifft(fw.get_cd_filter_freq(MM)*sp.fft(u))
            u = u*np.exp(1j*fw.nl_param[MM]*np.abs(u)**2)
            #u = u*np.exp(1j*(8/9)*fw.nl_param[MM]*(np.abs(u[0,:])**2+np.abs(u[1,:])**2))
        # add noise, NOTE: amplifier gain (u = u*np.exp(alpha_lin*Lsp/2.0)) is absorbed in nl_param
        u = u + np.sqrt(sigma2/2/Nsp) * (np.random.randn(1,Nsamp_a) + 1j*np.random.randn(1,Nsamp_a))
    # [RECEIVER] low-pass filter + downsample
    u = sp.ifft(sp.fft(u)*lp_filter_freq)
    y = u[0, ::OS_a//OS_d]
    y = np.stack([np.real(y), np.imag(y)], axis=1)
    return y, x[0,:], P

if args.timing == True:
	print("")
	print("timing the forward propation ...")
	t = time.time()
	_,_,_ = forward_propagation()
	elapsed = time.time()-t
	print("{0:.2f} seconds to generate 1 input/output data pair".format(elapsed))
	print("Generating approx. {0:.0f} input/output data pairs per seconds".format(NPROC*REPF/elapsed))
	sys.exit("")

# }}}
#========================================================#
# compute step sizes for DBP {{{
#========================================================#
ssfm_opts = {}
ssfm_opts['beta2'] = beta2
ssfm_opts['gamma'] = gamma
ssfm_opts['fsamp'] = fsamp_d
ssfm_opts['Nsamp'] = Nsamp_d
ssfm_opts['step_size_method'] = step_size_method_bw
ssfm_opts['ssfm_method'] = ssfm_method_bw
ssfm_opts['combine_half_steps'] = combine_half_steps
ssfm_opts['direction'] = -1

if less_steps_than_spans == False:
	ssfm_opts['alpha'] = alpha
	ssfm_opts['Nsp'] = Nsp
	ssfm_opts['Lsp'] = Lsp
	ssfm_opts['StPS'] = int(conf_ldbp['steps per span'])
else:
	ssfm_opts['alpha'] = 0
	ssfm_opts['Nsp'] = 1
	ssfm_opts['Lsp'] = Lsp*Nsp
	ssfm_opts['StPS'] = int(conf_ldbp['total steps'])

bw = ssfm_parameters(ssfm_opts)

#}}}
#========================================================#
# compute or load initial cd filter coefficients {{{
#========================================================#
cd_filter_coeffs = {}

if load_cd_filter == False:
	# create object for cd filter design
	fopt = {}
	fopt['beta2'] = beta2
	fopt['fsamp'] = fsamp_d
	fopt['Nsamp'] = Nsamp_d
	fopt['method'] = cd_filter_method
	fopt['bandwidth'] = cd_filter_bandwidth
	fopt['max_out_of_band_gain'] = cd_filter_oob_gain # 0.58 for 17-taps 20 Gbaud
	fir_obj = fir.cd_fir_filter(fopt)
	
	# determine length of cd filters 
	if config.has_option('LDBP parameters', 'cd filter length'):
		tmp = (line2array(conf_ldbp['cd filter length'])).astype(np.int32)
		cd_filter_length = periodically_extend(tmp, bw.model_steps) # tile
	else:
		cd_filter_length_margin = float(conf_ldbp['cd filter length margin'])
		cd_filter_length_min = int(conf_ldbp['cd filter length minimum'])
		print("No cd filter length provided. Computing automatically with margin {} and minimum {}:".format(cd_filter_length_margin,cd_filter_length_min))
		
		req_len = fir_obj.get_required_filter_length(bw.cd_length)
		cd_filter_length = (2*np.ceil(req_len/2*(1+cd_filter_length_margin))+1).astype(np.int32)
		for NN in range(bw.model_steps):
			if cd_filter_length[NN] < cd_filter_length_min:
				cd_filter_length[NN] = cd_filter_length_min
		print(cd_filter_length)
	
	# compute filter taps
	for NN in range(bw.model_steps):
		cd_filter_coeffs[NN] = fir_obj.get_filter(bw.cd_length[NN], cd_filter_length[NN])
else: # or load from file
	f = open(cd_filter_filename)
	lines = f.readlines()
	f.close()
	
	if len(lines) != 3*bw.model_steps:
		raise RuntimeError("File '"+cd_filter_filename+"' should have {} lines but has {}".format(3*bw.model_steps, len(lines)))
	
	cd_filter_length = np.zeros(bw.model_steps, dtype=np.int64)
	for NN in range(bw.model_steps):
		h_r = line2array(lines[NN*3+0])
		h_i = line2array(lines[NN*3+1])
		if np.size(h_r) != np.size(h_i):
			raise ValueError('real and imaginary part of loaded cd filters should be the same')
		cd_filter_length[NN] = np.size(h_r)
		if cd_filter_length[NN]%2 == 0:
			raise ValueError('loaded cd filter should have odd length: cd_filter_length[{}]={}'.format(NN,cd_filter_length[NN]))
		cd_filter_coeffs[NN] = h_r+1j*h_i
	print("loaded cd filters have lengths:")
	print(cd_filter_length)

cd_filter_delay = (cd_filter_length-1)//2

#}}}
#========================================================#
# define pruning parameters {{{
#========================================================#
pruning = config.getboolean('LDBP parameters', 'pruning')

def get_prune_op(mask, mask_len):
	mask_len_descreased = tf.assign(mask_len, mask_len-1)
	pos = tf.constant(np.arange(0, int(mask.get_shape()[0]), 1, np.int32), tf.int32)
	new_mask = tf.cast(tf.less(pos, mask_len_descreased), tf.float32)
	return tf.assign(mask, new_mask)

cd_mask = {}

if pruning == True:
	# get target lengths and memory
	tmp = (line2array(conf_ldbp['target cd filter length'])).astype(np.int32)
	target_length = periodically_extend(tmp, bw.model_steps) # tile
	for NN in range(bw.model_steps):
		if target_length[NN] < 0:
			target_length[NN] = cd_filter_length[NN] + target_length[NN]
		if target_length[NN]%2 == 0:
			raise ValueError("target filter lengths have to be odd")
	print("pruned filters will have lengths:")
	print(target_length)
	target_delay = (target_length-1)//2
	# determine the pruning order
	prune_order = []
	max_len = np.max(cd_filter_delay)
	min_len = np.min(target_delay)
	for i in range(max_len-min_len+1):
		for NN in range(bw.model_steps):
			if cd_filter_delay[NN] >= max_len-i and target_delay[NN] < max_len-i:
				prune_order.append(NN)
	# shuffle the order
	random.shuffle(prune_order)
	# determine pruning schedule: train-prune-train-prune-train
	pruning_steps = len(prune_order)
	#pruning_interval = np.ceil(iterations/(pruning_steps+1))
	print("total pruning steps: {}".format(pruning_steps))
	pruning_schedule = (np.ceil(np.ceil(2.0**(-np.arange(pruning_steps,0,-1))*iterations)))
	pruning_schedule = np.ceil(pruning_schedule + np.arange(pruning_steps)*iterations/8/pruning_steps)
	
	cd_mask_len = {}
	prune_op = {} # each mask has a pruning op associated with it
	
	for NN in range(bw.model_steps):
		cd_mask[NN] = tf.Variable(np.ones([cd_filter_delay[NN]+1]), dtype=tf.float32, trainable=False)
		cd_mask_len[NN] = tf.Variable(cd_filter_delay[NN]+1, dtype=tf.int32, trainable=False)
		prune_op[NN] = get_prune_op(cd_mask[NN], cd_mask_len[NN])
else:
	print("no pruning")
	for NN in range(bw.model_steps):
		cd_mask[NN] = 1

# }}}
#========================================================#
# define tunable parameters {{{
#========================================================#
no_filter = np.zeros(nl_filter_length, dtype=np.float32)
no_filter[nl_filter_delay] = 1.0

if nl_opt == True and tied_Kerr == True:
	nl_filter_all = tf_real_symmetric_filter(no_filter*nl_alpha, nl_opt)

cd_filter = {}
nl_filter = {}

for NN in range(bw.model_steps):
	# linear parameters
	cd_filter[NN] = tf_complex_symmetric_filter(cd_filter_coeffs[NN]*cd_alpha, cd_opt, mask=cd_mask[NN])
	# nonlinear parameters
	if nl_opt == True:
		if tied_Kerr == True:
			nl_filter[NN] = nl_filter_all
		else:
			nl_filter[NN] = tf_real_symmetric_filter(no_filter*nl_alpha, nl_opt)
	else:
		nl_filter[NN] = tf_real_symmetric_filter(no_filter*nl_alpha, nl_opt)

# matched filter 
ps_filter = tf_real_symmetric_filter(rrcosine(rolloff, delay, OS_d))

# }}}
#========================================================#
# build the computation graph in TensorFlow {{{
#========================================================#
print("building the TensorFlow graph ", end='', flush=True)

y_enq = tf.placeholder(tf.float32, shape=[None, Nsamp_d, 2])
x_enq = tf.placeholder(tf.complex64, shape=[None, Nsym])
P_enq = tf.placeholder(tf.float32, shape=[None, 1])

min_after_dequeue = int(conf_data["minimum elements after dequeue"]) # at least this many elements must remain after dequeue

myq = tf.RandomShuffleQueue(QMAX, min_after_dequeue, dtypes=[tf.float32, tf.complex64, tf.float32], shapes=[[Nsamp_d, 2], [Nsym], [1]])
enqueue_op = myq.enqueue_many([y_enq, x_enq, P_enq])
dummy_dequeue = myq.dequeue_many(QBSIZE*NPROC*REPF)
y,x,P_W = myq.dequeue_many(minibatch_size)

# [LDBP], signals have shape = [batch_size, N, 2] (if complex) or [batch_size, N] (if real)
for NN in range(bw.model_steps):
	print('.', end='', flush=True)
	# linear step
	y = cconv(y, cd_filter[NN]/cd_alpha) # complex(y) = complex(x) * complex(h)
	# nonlinear step, includes possible filtering of {|y_i|^2}
	#ysq = bw.nl_param[NN]*tf.reduce_sum(tf.square(y), axis=2)
	ysq = bw.nl_param[NN]*tf.reduce_sum(tf.square(y), axis=2)
	ysq_filtered = cconv(ysq, nl_filter[NN]/nl_alpha) # real(y) = real(x) * real(h)
	y = complex_multiply(y, tf.stack([tf.cos(ysq_filtered), tf.sin(ysq_filtered)], axis=2))

# matched filter
y = cconv(y, ps_filter) # complex(y) = complex(x) * real(h)
y = tf.complex(y[:,:,0],y[:,:,1])
# downsample
y = y[:,::OS_d] / tf.complex(tf.sqrt(P_W), 0.0) / np.sqrt(OS_d)
# constant phase-offset rotation
tmp = tf.reduce_sum(tf.conj(x)*y, 1, keepdims=True)
phi_cpe = -tf.atan2(tf.imag(tmp),tf.real(tmp))
x_hat = y * tf.exp(tf.complex(0.0, phi_cpe))

mean_squared_error = tf.reduce_mean(tf.square(tf.abs(x-x_hat)))
effective_snr = -10.0*tf.log(mean_squared_error+1e-12)/tf.log(10.0)

print("")
print("calling optimizer ... ", end="", flush=True)
optimizer = get_optimizer()
train = optimizer.minimize(mean_squared_error)

# compute total number of tunable parameters
total_parameters = 0
for variable in tf.trainable_variables():
	shape = variable.get_shape() # shape is an array of tf.Dimension
	variable_parameters = 1
	for dim in shape:
		variable_parameters *= dim.value
	total_parameters += variable_parameters

print("done, total tunable parameters: {}".format(total_parameters))

# }}}
#========================================================#
# start session {{{
#========================================================#
tf.summary.scalar("effective_snr", effective_snr)
tf.summary.scalar("data_queue_size", myq.size())
summary = tf.summary.merge_all()

init_op = tf.global_variables_initializer()
sess = tf.Session()

# create log dir
logdir = args.logdir+"/"+arg_str
logdir += "/" + time.strftime("%Y-%m-%d_%H.%M.%S", time.gmtime())

if not os.path.exists(logdir):
	os.makedirs(logdir)
else:
	raise RuntimeError("log directory \'" + logdir + "\' already exists")

print("name of the log directory: " + logdir)

# copy the .ini file to log folder
shutil.copyfile(config_path, logdir+"/"+config_file)
summary_writer = tf.summary.FileWriter(logdir, sess.graph)
sess.run(init_op) # run the OP that initializes global variables

# }}}
#========================================================#
# populate the data queue {{{
#========================================================#
def forward_propagation_batch(ignore_arg):
	y_read = np.zeros([QBSIZE, Nsamp_d, 2], np.float32)
	x_read = np.zeros([QBSIZE, Nsym], np.complex64)
	P_read = np.zeros([QBSIZE, 1], np.float32)
	for i in range(QBSIZE):
		y_read[i,:,:], x_read[i,:], P_read[i,:] = forward_propagation()
	return y_read, x_read, P_read

def populate_queue(sess, enqueue_op, coord):
	m = multiprocessing.cpu_count()
	pool = multiprocessing.Pool(m)
	while not coord.should_stop():
		results = pool.map(forward_propagation_batch, [0]*NPROC)
		y_batch = np.zeros([QBSIZE*NPROC*REPF, Nsamp_d, 2], np.float32)
		x_batch = np.zeros([QBSIZE*NPROC*REPF, Nsym], np.complex64)
		P_batch = np.zeros([QBSIZE*NPROC*REPF, 1], np.float32)
		for j in range(REPF):
			for i in range(NPROC):
				off = j*QBSIZE*NPROC
				y_batch[i*QBSIZE+off:(i+1)*QBSIZE+off,:,:] = results[i][0]
				x_batch[i*QBSIZE+off:(i+1)*QBSIZE+off,:]   = results[i][1]
				P_batch[i*QBSIZE+off:(i+1)*QBSIZE+off,:]   = results[i][2]
		sess.run(enqueue_op, feed_dict={y_enq: y_batch, x_enq: x_batch, P_enq: P_batch})

coord = tf.train.Coordinator()
t = threading.Thread(target=populate_queue, args=(sess, enqueue_op, coord))
t.start()

# }}}
#========================================================#
# optimization routine {{{
#========================================================#
# inital values 
mse_tmp, snr_tmp = sess.run([mean_squared_error, effective_snr])
print("---------------------------------------------")
print("initial: MSE = {0:.6f}, effective SNR = {1:.3f} dB".format(mse_tmp, snr_tmp))
print("elements in the data queue: {}".format(sess.run(myq.size())))

# write initial summary
sstr = sess.run(summary)
summary_writer.add_summary(sstr, 0)
summary_writer.flush()

# gradient descent
start = time.time()

pruned = 0
for i in range(1,iterations+1):
	_, mse_tmp, snr_tmp, sstr = sess.run([train, mean_squared_error, effective_snr, summary]) # 1 step in gradient descent
	if(math.isnan(mse_tmp)):
		print("nan detected, exiting optimization loop")
		break
	# summary
	if i%summary_interval == 0 or i==iterations:
		summary_writer.add_summary(sstr, i)
		summary_writer.flush()
		print("iter {0}: MSE = {1:.6f}, effective SNR = {2:.3f} dB, summary written".format(i, mse_tmp, snr_tmp))
	# pruning
	if pruning == True:
		pr_cnt = (pruning_schedule == i).sum()
		for II in range(pr_cnt):
			print("pruning (iter: {}, filter: {}, progress: {}/{})".format(i,prune_order[pruned],pruned+1,pruning_steps))
			sess.run(prune_op[prune_order[pruned]])
			pruned = pruned + 1

end = time.time()

print("requesting stop")
coord.request_stop()

queue_size = sess.run(myq.size())
if(QMAX - queue_size < QBSIZE*NPROC*REPF):
	print("dummy dequeue")
	sess.run(dummy_dequeue) # otherwise the threads hang at enqueue_op 

print("joining threads")
coord.join([t]) # wait for threads to terminate

opt_time = end-start
print("total optimization time: {0:.1f}s".format(opt_time))
print("processing approx. {0:.0f} input/output data pairs per second".format(iterations*minibatch_size/opt_time))

# }}}
#========================================================#
# save results to csv file {{{
#========================================================#
if SAVE_FILE == True:
	print("saving optimized parameters ... ", end="", flush=True)
	
	f=open(logdir+'/parameters.csv', 'ab') # a: append, b: binary mode
	f.truncate(0)
	
	cd_filter_print = sess.run(cd_filter)
	nl_filter_print = sess.run(nl_filter)
	
	for NN in range(bw.model_steps):
		tmp = np.transpose(cd_filter_print[NN] / cd_alpha)
		if pruning == True: # only store the pruned filter
			delay_diff = cd_filter_delay[NN] - target_delay[NN]
			tmp = tmp[:,delay_diff:delay_diff+target_length[NN]:]
		np.savetxt(f, tmp, delimiter=',')
		np.savetxt(f, np.transpose(nl_filter_print[NN]*bw.nl_param[NN] / nl_alpha), delimiter=',')
	f.close()
	print("done")
else:
	print("nothing is saved ...")

# }}}
#========================================================#
