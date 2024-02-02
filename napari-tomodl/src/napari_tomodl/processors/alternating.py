import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import phantominator as ph
from skimage.transform import radon, iradon
# import cupy as cp
from time import time
from skimage.metrics import structural_similarity as ssim
import sys
# cambiar para drive
# import DataLoading as dl 

def TwIST(y, A, AT, tau, kwarg, true_img = None):
  '''
  This function solves the regularization problem
     arg min_x = 0.5*|| y - A x ||_2^2 + tau phi( x ), 

  where A is a generic matrix and phi(.) is a regularizarion 
  function  such that the solution of the denoising problem 

      Psi_tau(y) = arg min_x = 0.5*|| y - x ||_2^2 + tau \phi( x ), 
  
  is known. 

  Params:

 y: 1D vector or 2D array (image) of observations

 A: if y and x are both 1D vectors, A can be a 
     k*n (where k is the size of y and n the size of x)
     matrix or a handle to a function that computes
     products of the form A*v, for some vector v.
     In any other case (if y and/or x are 2D arrays), 
     A has to be passed as a handle to a function which computes 
     products of the form A*x; another handle to a function 
     AT which computes products of the form A'*x is also required 
     in this case. The size of x is determined as the size
     of the result of applying AT.

  tau: regularization parameter, usually a non-negative real 
       parameter of the objective  function (see above).

  '''
  # Normalization / True_img comes prevously normalised at fbp reconstruction 

  # Default optional parameters
  stopCriterion = 1
  tolA = 0.01
  debias = 0
  maxiter = 10000
  maxiter_debias = 200
  miniter = 5
  miniter_debias = 5
  init = 0
  enforceMonotone = 1
  compute_mse = 1
  plot_ISNR = 0
  verbose = 1
  alpha = 0
  beta  = 0
  sparse = 1
  tolD = 0.001
  phi_l1 = 0
  psi_ok = 0
  # default eigenvalues 
  lam1=1e-4   
  lamN=1

  # constants ans internal variables
  for_ever = 1
  # maj_max_sv: majorizer for the maximum singular value of operator A
  max_svd = 1   # original set to 1

  # Set the defaults for outputs that may not be computed
  debias_start = 0
  x_debias = []
  
  # Read optional parameters
  for (k,v) in kwarg.items():

    k = k.upper()
    
    if k == 'LAMBDA':
      lam1 = v
    elif k == 'ALPHA':
      alpha = v
    elif k =='BETA':
      beta = v
    elif k =='PSI':
      psi_function = v
    elif k =='PHI':
      phi_function = v
    elif k =='GPU':
      gpu = v
    elif k == 'STOPCRITERION':
      stopCriterion = v
    elif k =='TOLERANCEA':       
      tolA = v
    elif k =='TOLERANCED':
      tolD = v
    elif k == 'DEBIAS':
      debias = v
    elif k == 'MAXITERA':
      maxiter = v
    elif k =='MAXIRERD':
      maxiter_debias = v
    elif k =='MINITERA':
      miniter = v
    elif k =='MINITERD':
      miniter_debias = v
    elif k =='INITIALIZATION':
      if isinstance(v, np.ndarray):       # we have an initial x
        init = 33333      # some flag to be used below
        x = v
      else:
        init = v
    elif k =='MONOTONE':
      enforceMonotone = v
    elif k =='SPARSE':
      sparse = v
    elif k == 'TRUE_X':
      compute_mse = 1
      true = v
      if true.size == y.size:
          plot_ISNR = 1
    elif 'VERBOSE':
      verbose = v
    else:
      # Hmmm, something wrong with the parameter string
      print('Unrecognized option: {}'.format(k))
      return

  # TwIST parameters
  rho0 = (1-lam1/lamN)/(1+lam1/lamN)
  if alpha == 0:
      alpha = 2/(1+np.sqrt(1-rho0**2))

  if  beta == 0:
      beta  = alpha*2/(lam1+lamN)

  if stopCriterion not in [0, 1, 2, 3]:
    print('Unknown stopping criterion')
    return
  # if A is a function handle (callable in python), we have to check presence of AT,
  if callable(A) and not callable(AT):
    print('The function handle for transpose of A is missing')
    return
  
  # if A is a matrix, we find out dimensions of y and x,
  # and create function handles (callables) for multiplication by A and A',
  # so that the code below doesn't have to distinguish between
  # the handle/not-handle cases
    
  # Watch for GPU case!!
  if not callable(A):

    AT = lambda x: np.dot(A.T, x)
    A = lambda x: np.dot(A.T, x)
  #from this point down, A and AT are always function handles.
  
  # Precompute A'*y since it'll be used a lot
  Aty = AT(y)
  # print('ATy', Aty.shape)
  
  # if phi was given, check to see if it is a handle (callable) and that it 
  # accepts two arguments
  if 'psi_function' in locals():
    if callable(psi_function):
      try:  # check if phi can be used, using Aty, which we know has 
            # same size as x
            dummy = psi_function(Aty,tau)
            psi_ok = 1
      except:
        print('Something is wrong with function handle for psi')
        return
    else:
      print('Psi does not seem to be a valid function handle')
      return
  #if nothing was given, use soft thresholding
  else: 
    psi_function = soft(x,tau)
  
  #if psi exists, phi must also exist
  if (psi_ok == 1):
    if 'phi_function' in locals():
      if callable(phi_function):
        # check if phi can be used, using Aty, which we know has 
        # same size as x
        try:
          dummy = phi_function(Aty)
        except:
          print('Something is wrong with function handle for phi')
          return
      else:
        print('Phi does not seem to be a valid function handle')
        return
    else:
      print('If you give Psi you must also give Phi')
  # if no psi and phi were given, simply use the l1 norm.
  else:
    phi_function = lambda x: np.sum(np.abs(x)) 
    phi_l1 = 1

  ## Initialization
  if init == 0:
    if gpu == 1:
      print('Not Implemented Yet!!')
      return
    else:
      x = AT(np.zeros(y.shape))
  elif init == 1:
    if gpu == 1:
      print('Not Implemented Yet!!')
      return
    else:
      x = np.random.randn(AT(np.zeros(y.size)).reshape(y.shape))
  elif init == 2:
    x = Aty
  elif init == 3:
     #initial x was given as a function argument; just check size
    if A(x).shape != y.shape:
      print('Size of initial x is not compatible with A')
  else:
    print('Unknown ''Initialization'' option')

  #now check if tau is an array; if it is, it has to 
  #have the same size as x
  if isinstance(tau, np.ndarray):
    try:
        dummy = np.multiply(x, tau)
    except:
        print('Parameter tau has wrong dimensions; it should be scalar or size(x)')
  
  #if the true_img x was given, check its size
  #if compute_mse and (true_img.size != x.size):  
  #  print('Initial x has incompatible size') 
  #  return

  # if tau is large enough, in the case of phi = l1, thus psi = soft,
  # the optimal solution is the zero vector

  # Watch for GPU implementation!!
  objective = []
  times = []

  if phi_l1:
    max_tau = np.max(np.abs(Aty))

    if (tau >= max_tau) and (psi_ok == 0):
      
      x = np.zeros(Aty.size)
      objective = 0.5*(np.dot(y, y))
      times = 0

  #define the indicator vector or matrix of nonzeros in x
  nz_x = np.float64((x != 0.0))
  num_nz_x = np.sum(nz_x)
  
  # Compute and store initial value of the objective function

  resid =  (y-A(x))
  prev_f = 0.5*(np.dot(resid.flatten().T,resid.flatten())) + tau*phi_function(x)

  # start the clock
  t0 = time()

  times.append(time() - t0)
  objective.append(prev_f)

  cont_outer = 1
  it = 1

  if verbose == 1:
    print(1,'\n Initial objective = {},  nonzeros={} \n'.format(prev_f, num_nz_x))

  # variables controling first and second order iterations
  IST_iters = 0
  TwIST_iters = 0
  
  # initialize
  xm2=x
  xm1=x

  while cont_outer:
    # Gradient
    grad = AT(resid)
      
    while for_ever:

      # IST estimate
      x = psi_function(xm1 + grad/max_svd, tau/max_svd)
      # x = psi_function(xm1 + grad, tau)
      
      if (IST_iters >= 2) or (TwIST_iters != 0):
        # set to zero the past when the present is zero
        # suitable for sparse inducing priors
        if sparse:
          mask = np.float64(x != 0)
          xm1 = np.multiply(xm1, mask)
          xm2 = np.multiply(xm2, mask)
        
        # two-step iteration
        xm2 = (alpha-beta)*xm1 + (1-alpha)*xm2 + beta*x
        # compute residual
        y_new = A(xm2) 
        
        resid =  y-A(xm2)
        f = 0.5*(np.dot(resid.flatten().T, resid.flatten())) + tau*phi_function(xm2)

        if (f > prev_f) and (enforceMonotone):
          TwIST_iters = 0;  # do a IST iteration if monotonocity fails
          
        else:
          TwIST_iters = TwIST_iters+1 # TwIST iterations
          IST_iters = 0
          x = xm2
        
          if TwIST_iters % 10000 == 0:
              max_svd = 0.9*max_svd
              print(max_svd)

          break  # break loop while
      
      else:
           
        y_new = A(x)
        resid = y-A(x)
        f = 0.5*(np.dot(resid.flatten().T, resid.flatten())) + tau*phi_function(x)
        
        if f > prev_f:
          # if monotonicity  fails here  is  because
          # max eig (A'A) > 1. Thus, we increase our guess
          # of max_svd
          max_svd = 2*max_svd
          if verbose:
              print('Incrementing S={}\n'.format(max_svd))
          
          IST_iters = 0
          TwIST_iters = 0
        else:
          TwIST_iters = TwIST_iters + 1
          break  # break loop while        
                
    xm2 = xm1
    xm1 = x

    #update the number of nonzero components and its variation
    nz_x_prev = nz_x
    nz_x = np.int64(x!=0.0)
    num_nz_x = np.sum(nz_x.flatten())
    num_changes_active = (np.sum(nz_x.flatten() != nz_x_prev.flatten()))
    
    #take no less than miniter and no more than maxiter iterations
    if stopCriterion == 0:
      # compute the stopping criterion based on the change
      # of the number of non-zero components of the estimate
      criterion =  num_changes_active
    elif stopCriterion == 1:
      # compute the stopping criterion based on the relative
      # variation of the objective function.
      criterion = np.abs(f-prev_f)/prev_f
    elif stopCriterion == 2:
      #  compute the stopping criterion based on the relative
      #  variation of the estimate.
      criterion = np.linalg.norm(x.flatten()-xm1.flatten())**2/np.linalg.norm(x.flatten())**2
    elif stopCriterion == 3:
      # continue if not yet reached target value tolA
      criterion = f
    else:
      print('Unknwon stopping criterion')
      return
    
    cont_outer = ((it <= maxiter) and (criterion > tolA))
    
    if it <= miniter:
      cont_outer = 1
    
    it= it+ 1
    prev_f = f
    objective.append(f)
    times.append(time()-t0)
    err = true_img - x

    # print out the various stopping criteria
    if verbose:
      if plot_ISNR:
          print('Iteration={}, ISNR={}  objective={}, nz={}, criterion={}\n'.format(it, 10*np.log10(np.sum((y.flatten()-true_img.flatten())**2)/np.sum((x.flatten()-true_img.flatten())**2)), f, num_nz_x, criterion/tolA))
      else:
          print('Iteration={}, objective={}, nz={},  criterion={}\n'.format(it, f, num_nz_x, criterion/tolA))
  
  #--------------------------------------------------------------
  # end of the main loop
  #--------------------------------------------------------------
  # Printout results
  if verbose:

    print('\nFinished the main algorithm!\nResults:\n')
    print('||A x - y ||_2 = {}\n'.format(np.dot(resid.flatten().T,resid.flatten())))
    print('||x||_1 = {}\n'.format(np.sum(np.abs(x.flatten()))))
    print('Objective function = {}\n'.format(f))
    print('Number of non-zero components = {}\n'.format(num_nz_x))
    print('CPU time so far = {}\n'.format(times[it-1]))
    print('\n')

  #--------------------------------------------------------------
  # If the 'Debias' option is set to 1, we try to
  # remove the bias from the l1 penalty, by applying CG to the
  # least-squares problem obtained by omitting the l1 term
  # and fixing the zero coefficients at zero.
  #--------------------------------------------------------------

  if debias:
    print('Not implemented yet!\n')
    return
  

  return x, objective

def ADMM(y, A, AT, Den, alpha, delta, max_iter, 
          phi, tol, warm, invert, true_img = None, 
          alpha_weight = False, alpha_decay = 0.6, verbose = True):
  # % ADMM Reconstruction based in Alternative directions method of multipliers
  # % This function solves the regularization problem 
  # %
  # %     arg min_x = 0.5*|| y - A x ||_2^2 + tau \phi( x )
  # %
  # % where A is a generic matrix and phi(.) is a regularizarion 
  # % function  such that the solution of the denoising problem 
  # %
  # %     Psi_tau(y) = arg min_x = 0.5*|| y - x ||_2^2 + tau \phi( x ), 
  # %
  # % ========================== INPUT PARAMETERS (required) ==================
  # % Parameters    Values description
  # % =========================================================================
  # % y             Observations image.
  # % delta         Regularization penalty parameter.
  # % A             Forward model
  # % AT            Backward operator
  # % Den           Denoise function
  # % alpha         Regularisation parameter of the problem
  # % phi           Norm for regularisation denoise
  # % img           Benchmark image
  # % warm          Warm start. 1 - CG for initial guess
  # % invert        Inversion method : 0 for direct inversion, 1 for CG

  # % based in 'Low-rank and sparse reconstruction in dynamic magnetic resonance imaging 
  # % via proximal splitting methods' 


  assert callable(A) 
  # print("A is callable")  # Assert A is callable
  assert callable(AT) 
  # print("AT is callable")  # Assert AT is callable
  assert callable(Den) 
  # print("Denoiser is callable")  # Assert Denoiser is callable

  # Normalization  
  b = np.zeros(AT(y).shape)
  s = np.zeros(AT(y).shape)
  u = np.zeros(AT(y).shape)

  tol = 1e-4

  if verbose == True:
    print('\n'.join(('ADMM reconstruction', 'Iteration | ISNR | objective | criterion | Phi Norm', '----')))

  objective = []
  error_MSE = []
  error_SSIM = []

  # % Warm Start - If used, initialices the model with a Conjugate Gradient
  # % aproximation

  for i in range(max_iter):
      
      if i == 1:
        
          snew = AT(y)
          unew = np.zeros(AT(y).shape)
          bnew = np.zeros(AT(y).shape)
          
          re= np.linalg.norm(snew.flatten()-s.flatten())/np.linalg.norm(snew.flatten()) #relative error

          s = snew
          u = unew
          b = bnew

      else:
          # % Direct inversion
          if invert == 1:
              
              x_inter = (AT(y)+delta*(b-u))
              snew = (1/delta)*(x_inter-(1/(1+delta))*AT(A(x_inter))); # objective function
          
          else:
          # % Conjugate Gradient
            snew = ConjugateGradient(A, AT, y, b, u, delta, 5)   
          
          bnew = Den(snew + u, alpha)    # denoising step
          unew = u + snew - bnew;        # parameter update

          re=np.linalg.norm(snew.flatten()-s.flatten())/np.linalg.norm(snew.flatten()) # relative error

          s = snew
          u = unew
          b = bnew
      
      if true_img is not None:
        
        ISNR=20*np.log10(np.linalg.norm((AT(y)-true_img).flatten())/np.linalg.norm((s-true_img).flatten()))
      
      y_new = A(snew)
      y_new = (y_new-y_new.min())/(y_new.max()-y_new.min())

      res = y-y_new
      objective.append(0.5*(np.dot(res.flatten(), res.flatten())) +
                       alpha*phi(snew))
      error_MSE.append(np.linalg.norm(true_img-s)**2)
      error_SSIM.append(ssim(true_img, s))
      
      if i>=2:
          crit = abs(objective[i]-objective[i-1])/objective[i-1]; 

          if verbose == True:
            print('{}\t|{}\t|{}\t|{}\t{}'.format(
                i, np.round(10*np.log10(np.sum((AT(y)-true_img)**2)/np.sum((s-true_img)**2)),3), 
                objective[i], crit/tol, alpha*phi(snew)))
          #if crit < tol:
          #  break

      if alpha_weight is True:
        
        alpha = alpha*alpha_decay 

  error_MSE = np.array(error_MSE)/true_img.size
  
  return s, objective, error_MSE, error_SSIM

def ConjugateGradient(A, AT, y, b, u, delta, max_iter = 5):   
    # % Conjugate gradient routine for linear operators - compressed sensing
    # % A - Forward operator
    # % AT - Backward operator
    # % b - denoised variable ADMM
    # % u - ADMM extra variable for augmented Lagrangian
    # % delta - Regularisation penalty parameter
    # % max_iter - Maximum number of iterations - defaults to 5
    
    b_sol = AT(y) + delta*(b - u)
    xn = AT(y)
    
    rn_1 = b_sol - (AT(A(xn))+delta*xn)
    pn = rn_1
    
    for k in range(max_iter):
        
        tn = (AT(A(pn))+delta*pn)    
        alpha = np.dot(rn_1.flatten(), rn_1.flatten())/np.dot(pn.flatten(), tn.flatten())
        
        xn = xn + alpha*pn
        rn_2 = rn_1 - alpha*tn
        
        beta = np.dot(rn_2.flatten(), rn_2.flatten())/np.dot(rn_1.flatten(), rn_1.flatten())
       
        pn = rn_2 + beta*pn   
        rn_1 = rn_2
    
    return xn

def TVdenoise(f, lamb, iters):
  #   %TVDENOISE  Total variation grayscale and color image denoising
  # %   u = TVDENOISE(f,lambda) denoises the input image f.  The smaller
  # %   the parameter lambda, the stronger the denoising.
  # %
  # %   The output u approximately minimizes the Rudin-Osher-Fatemi (ROF)
  # %   denoising model
  # %
  # %       Min  TV(u) + lambda/2 || f - u ||^2_2,
  # %        u
  # %
  # %   where TV(u) is the total variation of u.  If f is a color image (or any
  # %   array where size(f,3) > 1), the vectorial TV model is used,
  # %
  # %       Min  VTV(u) + lambda/2 || f - u ||^2_2.
  # %        u
  # %
  # %   TVDENOISE(...,Tol) specifies the stopping tolerance (default 1e-2).
  # %
  # %   The minimization is solved using Chambolle's method,
  # %      A. Chambolle, "An Algorithm for Total Variation Minimization and
  # %      Applications," J. Math. Imaging and Vision 20 (1-2): 89-97, 2004.
  # %   When f is a color image, the minimization is solved by a generalization
  # %   of Chambolle's method,
  # %      X. Bresson and T.F. Chan,  "Fast Minimization of the Vectorial Total
  # %      Variation Norm and Applications to Color Image Processing", UCLA CAM
  # %      Report 07-25.
  # %
  # %   Example:
  # %   f = double(imread('barbara-color.png'))/255;
  # %   f = f + randn(size(f))*16/255;
  # %   u = tvdenoise(f,12);
  # %   subplot(1,2,1); imshow(f); title Input
  # %   subplot(1,2,2); imshow(u); title Denoised

  # % Pascal Getreuer 2007-2008
  # %  Modified by Jose Bioucas-Dias  & Mario Figueiredo 2010 
  # %  (stopping rule: iters)
  
  # falta el argumento que modifica la tolerancia
  if lamb < 0:
    print('Parameter lambda must be nonnegative.')
    return
  
  dt = 0.25
  N = f.shape
  id = np.append(np.arange(1, N[0]), N[0]-1)
  iu = np.append(0, np.arange(0, N[0]-1))
  ir = np.append(np.arange(1, N[1]), N[1]-1)
  il = np.append(0, np.arange(0, N[1]-1))

  p1 = np.zeros(f.shape)
  p2 = np.zeros(f.shape)
  divp = np.zeros(f.shape)
  lastdivp = np.ones(f.shape)
  
  # just one channel
  for i in range(iters):

    lastdivp = divp
    z = divp - f*lamb
    z1 = z[:,ir] - z
    z2 = z[id,:] - z
    denom = 1 + dt*np.sqrt(z1**2 + z2**2)
    p1 = np.divide((p1 + dt*z1), denom)
    p2 =  np.divide((p2 + dt*z2), denom)
    divp = p1 - p1[:,il] + p2 - p2[iu,:]

  u = f - divp/lamb

  return u

def diffh(x):
  
  h = np.array([[0,0,0],
                [0,1,-1],
                [0,0,0]])

  return sp.signal.convolve2d(x, h, boundary = 'wrap')

def diffv(x):
  
  h = np.array([[0,0,0],
                [0,1,0],
                [0,-1,0]])

  return sp.signal.convolve2d(x, h, boundary = 'wrap')

def TVnorm(x):

  return np.sqrt(np.sum(diffh(x)**2+diffv(x)**2))

def soft(x, T):
  '''
  Soft Thresholding
  '''
  y = np.max(np.abs(x)-T, 0)
  y = np.multiply(np.divide(y, y+T), x)

  return y

def hR2(x, angles, angles2):

  return [radon(x, angles), radon(np.fliplr(np.flipud(x)), angles2)]

# Utilities
# Iterator to check projections
def iter_proj(method, hR, hRT, volume, max_angle, projections, n_z):
  '''
  With method, reconstruct N_z z-slices with a reduced number of projections
  Params:
    - method receives y, hR, hRT and true_img properly set up for the number of 
    projections required.
    - hRT receives sinogram 'y' and projection angles
    - hR receives image 'x' and projection angles
    - n_z provides the z-slices to be taken
  '''

  angle_step = max_angle//projections
  theta_sub, subsampled_vol = dl.subsample(volume, max_angle, angle_step) # Subset of angles and subsampled volume
  
  theta_full = np.linspace(0., max_angle-1, max_angle)  # Set of angles for full projection

  recons_x = []
  objectives = []
  errors_MSE = [] 
  errors_SSIM = [] 
  fbp_x = []
  trues_x = []
  
  for z in n_z:
  
    y_full = volume[:,:, z].T          # Full sinogram
    y_full = normalize(y_sub)

    y_sub = subsampled_vol[:,:, z].T   # subsampled sinogram
    y_sub = normalize(y_sub)

    true_img = hRT(y_full, theta_full)   # use full backprojected image

    # Forward and backward operators
    hR_sub = lambda x: hR(x, theta_sub)
    hRT_sub = lambda y: hRT(y, theta_sub)   
    
    fbp_sub = hRT_sub(y_sub)
    # Iterative method with fewer projections
    x, objective, error_MSE, error_SSIM = method(y_sub, hR_sub, hRT_sub, true_img)   

    recons_x.append(x)
    objectives.append(objective)
    errors_MSE.append(error_MSE)
    errors_SSIM.append(error_SSIM)
    trues_x.append(true_img)
    fbp_x.append(fbp_sub)

  return recons_x, fbp_x, objectives, errors_MSE, errors_SSIM, trues_x

def normalize(x):
  y = x-x.min()
  return y/y.max()
