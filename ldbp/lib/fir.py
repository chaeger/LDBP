import numpy as np
from numpy import pi, cos, sin, exp, sqrt
from scipy import special, linalg, optimize, ifft
import warnings
#import sympy as sym

class cd_fir_filter:
	""" finite impulse response (FIR) filters for compensating chromatic dispersion (CD)
	
	Second-order dispersion is modeled in the (forward) split-step Fourier method as
		
		u = -j beta2/2 d^2u/dt^2
		
	which has the frequency-domain solution
		
		U(w,z) = exp(j beta/2 z w^2) U(w,0)
		
	In order to compensate, the ideal filter response is given by
		
		EDC(w) = exp(-j beta/2 z w^2)
	
	Args (provided as a dictionary):
		beta2: dispersion parameter
		fsamp: sampling frequency in Hz
		Nsamp: number of samples (for FFT)
		method: employed filter-design method
			'truncate IDFT'
			'direct sampling'
			'least squares'
			'LS-CO': least squares with constrained out-of-band gain
		bandwidth: percentage of compensated bandwidth, between 0 and 1, defaults to 1
			only used for 'least squares' and 'LS-CO'
		eps: regularization parameter for 'least squares', defaults to 1e-12
		max_out_of_band_gain: for 'LS-CO'
	"""
	
	def __init__(self, opts):
		self.__dict__.update(opts) # converts all dictionary entries to attributes 
		
		if 'bandwidth' not in opts:
			self.bandwidth = 1.0
		if opts['method'] == 'least squares' and 'eps' not in opts:
			self.eps = 1e-12
		if self.bandwidth < 0.0 or self.bandwidth > 1.0:
			raise ValueError("bandwidth parameter has to be between 0 and 1: bandwidth = {}".format(self.bandwidth))
	
	def get_required_filter_length(self, L):
		L = np.array(L)
		Df = self.bandwidth*self.fsamp
		return (2*np.floor(2*pi*np.abs(self.beta2)*L*Df*self.fsamp/2)+1).astype(np.int32)
	
	def get_filter(self, L, cd_filter_length):
		""" approximates exp(j*KK*w^2), where w=-pi..pi 
		
		Args:
			L: fiber length [m]
			cd_filter_length: length of the FIR filter, has to be odd
		
		Returns:
			Filter coefficients as numpy array
		"""
		 
		if cd_filter_length%2 == 0:
			raise ValueError("cd_filter_length has to be odd: cd_filter_length={}".format(cd_filter_length))
		
		KK = -(self.beta2)/2*L*(self.fsamp**2)
		delay = (cd_filter_length-1)//2
		N = self.Nsamp
		xi = self.bandwidth
		Omega1 = -pi*xi
		Omega2 =  pi*xi
		K = int(np.floor(N/2*self.bandwidth))
		
		if xi < 1:
			i = np.reshape(np.arange(K+1,(N-K-1)+1), [N-2*K-1, 1])
			k = np.reshape(np.arange(-delay, delay+1), [1, cd_filter_length])
			B = exp(-1j*i*k*2*pi/N) * sqrt((2*pi/N)/(2*pi+Omega1-Omega2))
		else:
			B = np.zeros([1, cd_filter_length])
		
		out_of_band_gain = lambda h: np.sum(np.abs(np.matmul(B,h))**2)
		
		i = np.reshape(np.arange(-K,K+1), [2*K+1, 1])
		k = np.reshape(np.arange(-delay, delay+1), [1, cd_filter_length])
		A = exp(-1j*i*k*2*pi/N) #* sqrt((2*pi/N)/(Omega2-Omega1))
		des = exp(1j*KK*(2*pi*i/N)**2) #* sqrt((2*pi/N)/(Omega2-Omega1))
		
		inband_error = lambda h: np.sum(np.abs(np.matmul(A,h)-des)**2)#/(2*xi*pi)/N
		 
		if self.method == 'truncate IDFT':
			fvec = np.fft.fftshift(np.arange(-N//2, N//2)*self.fsamp/N)
			htmp = ifft(exp(-1j*self.beta2/2*L*(2*pi*fvec)**2)) 
			cd_filter_coeffs = np.concatenate([np.flipud(htmp[1:delay+1:]), htmp[0:delay+1:]])
		elif self.method == 'direct sampling': # Savory (2008)
			cd_filter_coeffs = np.zeros([cd_filter_length], dtype=np.complex64)
			for n in range(-delay, delay+1):
				cd_filter_coeffs[n+delay] = sqrt(1j/(4*KK*pi))*exp(-1j*n**2/(4*KK))
		elif self.method == 'least squares2':
			Q = xi*np.ones([delay+1, delay+1]) # Q[0,0] = xi
			for i in range(1,delay+1):
				Q[0,i] = Q[i,0] = 2*sin(i*pi*xi)/(i*pi)
			for i in range(1,delay+1):
				for j in range(1,delay+1):
					if i != j:
						AA = i*cos(j*pi*xi)*sin(i*pi*xi)
						BB = j*cos(i*pi*xi)*sin(j*pi*xi)
						Q[i,j] = 4*(AA-BB)/(i**2-j**2)/pi
					else:
						Q[i,i] = (2*pi*xi+sin(2*i*pi*xi)/i)/pi
			
			nn = np.arange(delay+1)
			D = exp(-1j*(nn**2/(4*KK) + 3*pi/4))/(2*sqrt(pi*KK+0j)) \
					*(special.erf(exp(1j*3*pi/4)*(2*Omega2*KK-nn)/(2*sqrt(KK+0j))) \
					+ special.erf(exp(1j*3*pi/4)*(2*Omega2*KK+nn)/(2*sqrt(KK+0j)))) 
			D[0] = D[0]/2
			
			I = 2*np.eye(delay+1)
			I[0,0] = 1
			
			tmp = linalg.solve(Q+self.eps*I, D)
			cd_filter_coeffs = np.concatenate([np.flipud(tmp[1:]),tmp])
		elif self.method == 'least squares': # Eghbali et al. (2014)
			Q = np.ones([cd_filter_length, cd_filter_length])
			for i in range(cd_filter_length):
				for j in range(cd_filter_length):
					if i != j: # diagonal entries are 1
						Q[i,j] = sin(pi*(i-j)*xi)/(pi*(i-j)*xi)
			Q = xi*Q
			
			nn = np.arange(-delay, delay+1) 
			v = xi*self.int_aux(nn, KK, xi)
			
			cd_filter_coeffs = linalg.solve(Q+self.eps*np.eye(cd_filter_length), v)
		elif self.method == 'LS-CO': # Sheikh et al. (2016)
			Q1 = np.ones([cd_filter_length, cd_filter_length])
			Q2 = np.ones([cd_filter_length, cd_filter_length])
			for i in range(cd_filter_length):
				for j in range(cd_filter_length):
					if i != j: # diagonal entries are 1
						Q1[i,j] = sin(pi*(i-j)*xi)/(pi*(i-j)*xi)
						if xi < 1:
							Q2[i,j] = sin(pi*(i-j)*xi)/(pi*(i-j)*(xi-1))
						else:
							Q2[i,j] = 0.0
			
			nn = np.arange(-delay, delay+1)[np.newaxis].T
			v = self.int_aux(nn, KK, xi)
			
			hopt = lambda l: linalg.solve(Q1+l*Q2, v)
			fun2 = lambda l: out_of_band_gain(hopt(np.abs(l)))-self.max_out_of_band_gain
			 
			lambda_opt = np.abs(optimize.fsolve(fun2, 1e-10))
			#if fun2(0.0) > 0:
			#	lambda_opt = np.abs(optimize.fsolve(fun2, 0.0))
			#else:
			#	lambda_opt = 0.0
			#print(lambda_opt)
			
			cd_filter_coeffs = hopt(lambda_opt)
		elif self.method == 'maximally flat': # experimental
			Q = np.zeros([delay+1, delay+1], dtype=np.float64)
			Q[0,0] = 1
			for n in range(1,delay+1):
				Q[0,n] = 2
			for k in range(1,delay+1):
				for n in range(delay+1):
					Q[k,n] = 2*n**(2*k)
					if k%2 != 0: # odd rows
						Q[k,n] = -Q[k,n]
			
			w = sym.symbols('w')
			
			v = np.zeros([delay+1], dtype=np.complex128)
			for k in range(delay+1):
				v[k] = (sym.diff(sym.exp(sym.I*KK*w**2), w, 2*k)).subs(w,0)
			
			cd_filter_coeffs = linalg.solve(Q, v)
			cd_filter_coeffs = np.concatenate([np.flipud(cd_filter_coeffs[1::]), cd_filter_coeffs])
		else:
			raise ValueError("wrong cd filter method")
		
		eps_o = out_of_band_gain(cd_filter_coeffs)
		E = inband_error(cd_filter_coeffs)
		#print(E)
		h = cd_filter_coeffs
		#(np.abs(np.matmul(A,h)-des)**2))
		#np.set_printoptions(threshold=np.inf)
		tmp = (np.abs(np.matmul(A,h)-des)**2)
		#print(A.shape)
		#print(h.shape)
		#print(des.shape)
		#print((np.matmul(A,h)-des).shape)
		
		for i in range(delay):
			absdiff = abs(cd_filter_coeffs[i] - cd_filter_coeffs[cd_filter_length-i-1])
			if(absdiff > 1e-2):
				warnings.warn("filter coefficients are not symmetric: absolute difference = {}".format(absdiff))
		
		return cd_filter_coeffs.reshape([cd_filter_length])
	
	def int_aux(self, n, KK, x):
		# ( integrate exp(j*KK*w^2)exp(j*n*w) dw=-x*pi...x*pi ) / (2*pi*x)
		#
		# eq. (13) in Eghbali et al. (2014), with fixed typo
		# +0j to avoid NANs with sqrt(negative number)
		#	D = exp(-1j*(nn**2/(4*KK) + 3*pi/4))/(4*sqrt(pi*KK+0j)) \
		#			*(special.erf(exp(1j*3*pi/4)*(2*(Omega2/pi)*KK*pi-nn)/(2*sqrt(KK+0j))) \
		#			+ special.erf(exp(1j*3*pi/4)*(2*(Omega2/pi)*KK*pi+nn)/(2*sqrt(KK+0j)))) 
		#
		Omega1 = -pi*x
		Omega2 =  pi*x
		y = exp(-1j*(n**2/(4*KK)+3*pi/4))/(2*(Omega2-Omega1))*sqrt(pi/KK+0j) \
				*(special.erf(exp(-1j*pi/4)*(2*Omega1*KK+n)/(2*sqrt(KK+0j))) \
				- special.erf(exp(-1j*pi/4)*(2*Omega2*KK+n)/(2*sqrt(KK+0j))))
		return y
