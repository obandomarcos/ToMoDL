import numpy as np
import scipy as sp

def hessianOperator(f):
  '''
  Calculates Hessian Schatten-Norm forward operator for image, with Neumann (periodic) condition.
  Params: 
    - f (np.ndarray): 1-channel image with shape [n_x, n_y]
  '''
  nx, ny = f.shape  # to retain image shape

  # forward operator Hessian
  fxx = f - 2*np.roll(f, -1, axis = 0) + np.roll(f, -2, axis = 0)
  fyy = f - 2*np.roll(f, -1, axis = 1) + np.roll(f, -2, axis = 1)
  fxy = f + np.roll(f, (-1, -1), axis = (0,1)) - np.roll(f, -1, axis = 0) - np.roll(f, -1, axis = 1)

  # fxy = fyx so transposing shouldn't be an issue
  Hf = np.zeros((nx, ny, 3))
  Hf[:,:,0] = fxx
  Hf[:,:,1] = fxy
  Hf[:,:,2] = fyy

  return Hf

def adjHessianOp(A):
  '''
  Calculates Hessian Schatten-Norm adjoint operator for image, with Neumann (periodic) condition.
  Params: 
    - f (np.ndarray): Nx2x2 Hessian matrix
  ''' 

  Axx = A[:,:,0]
  Axx = Axx - 2*np.roll(Axx, -1, axis = 0) +  np.roll(Axx, -2, axis = 0)

  Axy = 2*A[:,:,1] # third member of equation
  Axy = Axy - np.roll(Axy, -1, axis=0) - np.roll(Axy, -1, axis=1) + np.roll(Axy, (-1,-1), axis=(0,1))

  Ayy = A[:,:,2]
  Ayy = Ayy - 2*np.roll(Ayy, -1, axis=1) + np.roll(Ayy, -2, axis=1) 

  HadjA = Axx + Axy + Ayy
  
  return HadjA

def mixedSchattenNorm(f, p):

  '''
  Mixed l_1 - S_p norm for an argument Phi = [Phi^T_1, Phi^T_2, ..., Phi^T_N]
  '''
  return np.sum(SchattenPNorm(f, p))

def SchattenPNorm(f, p):
  '''
  Hessian space p-norm 
  Params : 
    - f (np.ndarray): 1-channel image
    - p (int):
  '''
  if not isinstance(f, np.ndarray):
    
    print('Not an array!')
    return
  
  if len(f.shape) != 2:

    print('Not in a fit shape!')
    return
    
  hess = hessianOperator(f)
  _, d, _ = np.linalg.svd(hess)

  if not isinstance(p, str):
    
    if p>=1:
      normSh = (np.sum(d**p, axis = 1))**(1/p)

  elif p is 'inf':
    
    print('A otra cosa mariposa')
    # normSh = 0.0
    
    # for d_i in d:
    #   print(d_i)
    #   #normSh = max(np.abs(d_i), normSh)

  return normSh

def denoiseHessian(y,
                  tau,
                  kwarg):
  
  '''
  Hessian variation grayscale denoising, based on Lefkimmiatis solution with Gradient projection or FGP
  Params :
    y (np.ndarray) : Noisy blurred image
    tau (float) : regularization parameter
  Optionals:
    maxiter   :    Number of iterations (Default: 100)
    bounds        Minimize the Objective with a box constraint on the
                  solution (Default: [-inf +inf])
    snorm         Specifies the type of the Hessian Schatten norm.
                  {'spectral'|'nuclear'|'frobenius'|'Sp'}. (Default:
                  'frobenius'). If snorm is set to Sp then the order of
                  the norm has also to be specified.

  '''
  if 'MAX_ITER' in kwarg.keys():
    maxiter = kwarg['MAX_ITER']
  else:
    maxiter = 100
  
  if 'OPTIM' in kwarg.keys():
    optim = kwarg['OPTIM']
  else:
    optim = 'FGP'

  if 'VERBOSE' in kwarg.keys():
    verbose = kwarg['VERBOSE']
  else:
    verbose = True
  
  if 'IMAGE' in kwarg.keys(): 
    img = kwarg['IMAGE']
  else:
    img = None
  
  if 'ORDER' in kwarg.keys(): 
    order = kwarg['ORDER']
  else:
    order = None
  
  if 'SNORM' in kwarg.keys():
    snorm = kwarg['SNORM']
  else:
    snorm = 'frobenius'

  if 'P' in kwarg.keys():
    P = kwarg['P']
  else:
    P = np.zeros((*y.shape, 3))
  
  if 'TOL' in kwarg.keys():
    tol = kwarg['TOL']
  else :
    tol = 1e-4 

  if 'L' not in kwarg.keys():
    L = 64/1.25

  if 'BOUNDS' in kwarg.keys():
    bounds = kwarg['BOUNDS']
  else: 
    bounds = np.array([-np.inf, np.inf])

  if kwarg['SNORM'] == 'Sp' and order == None:
    print('The order of the Sp-norm must be specified!')
    return

  if kwarg['SNORM'] == 'Sp' and order == np.inf:
    print('Try spectral norm')
    return

  if kwarg['SNORM'] == 'Sp' and order == 1:
    print('Try nuclear norm')
    return

  if kwarg['SNORM'] == 'Sp' and order < 1:
    print('Order should be greater or equal to 1!')
    return

  flag = False
  count = 0
  if verbose:
    print('******************************************\n')
    print('**  Denoising with Hessian Regularizer  **\n')
    print('******************************************\n')
    print('#iter     relative-dif   \t fun_val         Duality Gap        ISNR\n')
    print('====================================================================\n')

  # Denoising algorithm
  if optim == 'FGP':
    t = 1
    F = P

    for i in range(0,maxiter):

      K = y - tau*adjHessianOp(F)

      Pnew = F + (1/(tau*L))*hessianOperator(project(K, bounds))    
      Pnew = projectLB(Pnew, snorm, order)
      
      rel_err = np.linalg.norm(Pnew.flatten()-P.flatten())/np.linalg.norm(Pnew.flatten())
      
      if (rel_err<tol):
        count += 1
      else:
        count =  0

      tnew = (1+np.sqrt(1+4*t**2))/2
      F = Pnew + (t-1)/tnew*(Pnew-P)
      P = Pnew
      t = tnew

      if verbose:

        if isinstance(img, np.ndarray):
          
          k = y - tau*adjHessianOp(F)
          x = project(k, bounds)
          fun_val = cost(y, x, tau, snorm, order)
          dual_fun_val = dualcost(y, k, bounds)
          dual_gap = (fun_val - dual_fun_val)
          ISNR = 20*np.log10(np.linalg.norm(y.flatten()-img.flatten())/np.linalg.norm(x.flatten()-img.flatten()))
          
          print('{} \t  {} \t {} \t {} \t  {}\n'.format(i,rel_err,fun_val,dual_gap,ISNR))

        else:
          
          k = y - tau*adjHessianOp(F)
          x = project(k, bounds)
          fun_val = cost(y, x, tau, snorm, order)
          dual_fun_val = dualcost(y, k, bounds)
          dual_gap = (fun_val - dual_fun_val)
          
          print('{} \t  {} \t {} \t {} \n'.format(i,rel_err,fun_val,dual_gap,ISNR))

      if count >= 5:
        flag = True
        iteration = i
        break
  else:
    print('Not implemented yet!')
    return

  if flag == False:
    iteration = maxiter
  
  x = project(y - tau*adjHessianOp(P), bounds)

  return x  

def project(x, bounds):

  lb = bounds[0]
  ub = bounds[1]

  if (lb == -np.inf) and (ub == np.inf): 
    
    Px = x
  
  elif (lb == np.inf) and (ub != np.inf):
    
    x[x>ub] = ub
    Px = x

  elif (ub == np.inf) and (lb != np.inf):
  
    x[x<lb] = lb
    Px = x

  else:
    
    x[x>ub] = ub
    x[x<lb] = lb
    Px = x
  
  return Px

# Projection over S2
# always return and take NxNx3 matrices, but change indices in-Place, ask about this

def projS2(x, rho):

  normF = np.sqrt(np.sum(x**2,axis = 2))
  
  xP = np.empty(x.shape)
  
  xP[:,:,0] = np.where(normF>1, x[:,:,0]/normF, x[:,:,0])
  xP[:,:,1] = np.where(normF>1, x[:,:,1]/normF, x[:,:,1])
  xP[:,:,2] = np.where(normF>1, x[:,:,2]/normF, x[:,:,2])

  return xP
# projection over Sinf
def projSInf(x, rho):
  
  tmp = np.array([[x[:,:,0], x[:,:,1]],[x[:,:,1], x[:,:,2]]]).T
  U, D, Vh = np.linalg.svd(tmp)

  xP = np.empty(x.shape)
  D1 = D.reshape(D.shape[0]*D.shape[1], D.shape[2])
  norm_linf = np.max(np.abs(D1), axis = 1).reshape(D.shape[0], D.shape[1])
  
  # projection of eigenvalues
  # diag flattened
  D1[:,0] = np.multiply(np.sign(D1[:,0]), np.minimum(np.abs(D1[:,0]), rho))
  D1[:,1] = np.multiply(np.sign(D1[:,1]), np.minimum(np.abs(D1[:,1]), rho))

  # reconstruction
  D1 = D1.reshape(D.shape)
  Xp = np.matmul(U[...,:2], D1[..., None] * Vh)

  # norm condition 
  Xp[norm_linf <= rho, :] = tmp[norm_linf <= rho, :]
  # return values

  return np.array([Xp[:,:,0,0], Xp[:,:,0,1], Xp[:,:,1,1]]).T
# projection over S1 - dual from p = inf
def projS1(x, rho):

  tmp = np.array([[x[:,:,0], x[:,:,1]],[x[:,:,1], x[:,:,2]]]).T
  U, D, Vh = np.linalg.svd(tmp)

  norm_l1 = np.sum(np.abs(D), axis = 2)

  gamma = np.where(np.diff(np.abs(D), axis = 2).squeeze() > rho,
                   np.maximum(np.abs(D[:,:,0]), np.abs(D[:,:,1]))-rho,
                   (norm_l1-rho)/2)
  # D projected
  Dp = np.empty(D.shape)
  Dp[:,:, 0] = np.multiply(np.sign(D[:,:,0]), np.maximum(np.abs(D[:,:,0])-gamma, 0))
  Dp[:,:, 1] = np.multiply(np.sign(D[:,:,1]), np.maximum(np.abs(D[:,:,1])-gamma, 0))

  # Reconstruction
  Xp = np.matmul(U[...,:2], Dp[..., None] * Vh)

  # norm condition 
  Xp[norm_l1 <= rho, :] = tmp[norm_l1 <= rho, :]

  return np.array([Xp[:,:,0,0], Xp[:,:,0,1], Xp[:,:,1,1]]).T

def projectLB(A, snorm, order):

  if snorm == 'spectral':
    Ap = projS1(A, 1)

  elif snorm == 'frobenius':
    Ap = projS2(A, 1)

  elif snorm == 'nuclear':
    Ap = projSInf(A, 1)
  
  elif snorm == 'Sp':
    print('Not implemented yet!!')
    return
  else:
    print('Error : Unknown type of norm')
    return
  
  return Ap

def cost(y, f, tau, snorm, order):

  fxx = f - 2*np.roll(f, -1, axis = 0) + np.roll(f, -2, axis = 0)
  fyy = f - 2*np.roll(f, -1, axis = 1) + np.roll(f, -2, axis = 1)
  fxy = f + np.roll(f, (-1, -1), axis = (0,1)) - np.roll(f, -1, axis = 0) - np.roll(f, -1, axis = 1)

  if snorm == 'spectral':
      #Sum of the Hessian spectral radius
      Lf=fxx+fyy                        #Laplacian of the image f
      Of=np.sqrt((fxx-fyy)**2+4*fxy**2)    # Amplitude of the Orientation vector
      Hnorm = np.sum(0.5*(np.abs(Lf)+Of)) # Sum of the Hessian spectral radius
  elif snorm == 'frobenius':
      Hnorm=np.sum(np.sqrt(fxx.flatten()**2+fyy.flatten()**2+2*fxy.flatten()**2))

  elif snorm == 'nuclear':
      Lf=fxx+fyy                      #Laplacian of the image f
      Of=np.sqrt((fxx-fyy)**2+4*fxy**2)  #Amplitude of the Orientation vector
      Hnorm=0.5*np.sum(np.abs(Lf.flatten()+Of.flatten())+np.abs(Lf.flatten()-Of.flatten()))
    
  elif snorm == 'Sp':
      Lf=fxx+fyy                      #Laplacian of the image f
      Of=sqrt((fxx-fyy)**2+4*fxy**2)  # Amplitude of the Orientation vector
      # %e1=0.5*(Lf.flatten()+Of.flatten())
      # %e2=0.5*(Lf.flatten()-Of.flatten())
      Hnorm=np.sum((np.abs(0.5*(Lf.flatten()+Of.flatten()))**(order)+np.abs(0.5*(Lf.flatten()-Of.flatten()))**(order))**(1/order))
  else:
    print('denoiseHS::Unknown type of norm.')
    return
  
  Q = 0.5*np.linalg.norm(y.flatten()-f.flatten())+tau*Hnorm
  
  return Q

def hessNorm(f, snorm, order):

  fxx = f - 2*np.roll(f, -1, axis = 0) + np.roll(f, -2, axis = 0)
  fyy = f - 2*np.roll(f, -1, axis = 1) + np.roll(f, -2, axis = 1)
  fxy = f + np.roll(f, (-1, -1), axis = (0,1)) - np.roll(f, -1, axis = 0) - np.roll(f, -1, axis = 1)

  if snorm == 'spectral':
      #Sum of the Hessian spectral radius
      Lf=fxx+fyy                        #Laplacian of the image f
      Of=np.sqrt((fxx-fyy)**2+4*fxy**2)    # Amplitude of the Orientation vector
      Hnorm = np.sum(0.5*(np.abs(Lf)+Of)) # Sum of the Hessian spectral radius
  elif snorm == 'frobenius':
      Hnorm=np.sum(np.sqrt(fxx.flatten()**2+fyy.flatten()**2+2*fxy.flatten()**2))

  elif snorm == 'nuclear':
      Lf=fxx+fyy                      #Laplacian of the image f
      Of=np.sqrt((fxx-fyy)**2+4*fxy**2)  #Amplitude of the Orientation vector
      Hnorm=0.5*np.sum(np.abs(Lf.flatten()+Of.flatten())+np.abs(Lf.flatten()-Of.flatten()))
    
  elif snorm == 'Sp':
      Lf=fxx+fyy                      #Laplacian of the image f
      Of=sqrt((fxx-fyy)**2+4*fxy**2)  # Amplitude of the Orientation vector
      # %e1=0.5*(Lf.flatten()+Of.flatten())
      # %e2=0.5*(Lf.flatten()-Of.flatten())
      Hnorm=np.sum((np.abs(0.5*(Lf.flatten()+Of.flatten()))**(order)+np.abs(0.5*(Lf.flatten()-Of.flatten()))**(order))**(1/order))

  return Hnorm

def dualcost(y, f, bounds):

  r = f - project(f, bounds)
  
  Q = 0.5*(np.sum(r.flatten()**2)+np.sum(y.flatten()**2)-np.sum(f.flatten()**2))
  
  return Q


