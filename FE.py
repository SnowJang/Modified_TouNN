import numpy as np
import torch


from scipy.sparse import coo_matrix, vstack, bmat
from scipy.sparse.linalg import spsolve
import numpy.matlib
import matplotlib.pyplot as plt
from matplotlib import cm
import cvxopt
import cvxopt.cholmod
#-----------------------#

#%%  structural FE
class StructuralFE:
    def getDMatrix(self):
        E = 1
        nu = 0.3
        k = np.array(
            [1 / 2 - nu / 6, 1 / 8 + nu / 8, -1 / 4 - nu / 12, -1 / 8 + 3 * nu / 8, -1 / 4 + nu / 12, -1 / 8 - nu / 8,
             nu / 6, 1 / 8 - 3 * nu / 8])
        KE = E / (1 - nu ** 2) * np.array([[k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
                                           [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
                                           [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
                                           [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
                                           [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
                                           [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
                                           [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
                                           [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]]);
        return (KE);

    #-----------------------#
    def initializeSolver(self, nelx, nely, forceBC, fixed, penal = 3,Emin = 1e-3, Emax = 1.0):
        self.Emin = Emin;
        self.Emax = Emax;
        self.penal = penal;
        self.nelx = nelx;
        self.nely = nely;
        self.ndof = 2*(nelx+1)*(nely+1)
        self.KE=self.getDMatrix();
        self.fixed = fixed;
        self.free = np.setdiff1d(np.arange(self.ndof),fixed);
        self.f = forceBC;
        self.edofMat=np.zeros((nelx*nely,8),dtype=int)
        for elx in range(nelx):
            for ely in range(nely):
                el = ely+elx*nely
                n1=(nely+1)*elx+ely
                n2=(nely+1)*(elx+1)+ely
                self.edofMat[el,:]=np.array([2*n1+2, 2*n1+3, 2*n2+2, 2*n2+3,2*n2, 2*n2+1, 2*n1, 2*n1+1])

        self.iK = np.kron(self.edofMat,np.ones((8,1))).flatten()
        self.jK = np.kron(self.edofMat,np.ones((1,8))).flatten()
        nodenrs = np.arange(1, (1 + nelx) * (1 + nely) + 1).reshape(1 + nelx, 1 + nely).T
        ## PERIODIC BOUNDARY CONDITIONS
        e0 = np.eye(3)
        ufixed = np.zeros((8, 3))
        U = np.zeros((2 * (nely + 1) * (nelx + 1), 3))
        alldofs = (np.arange(1, 2 * (nely + 1) * (nelx + 1) + 1))
        n1 = np.array([nodenrs[-1, 0], nodenrs[-1, -1], nodenrs[0, -1], nodenrs[0, 0]])
        d1 = np.array([[(2 * n1 - 1)], [2 * n1]]).T.reshape(1, -1).flatten()
        n3 = np.hstack([nodenrs[1:-1, 0].T, nodenrs[-1, 1:-1]])
        d3 = np.array([[(2 * n3 - 1)], [2 * n3]]).T.reshape(1, -1).flatten()
        n4 = np.hstack([nodenrs[1:-1, -1].T, nodenrs[0, 1:-1]])
        d4 = np.array([[(2 * n4 - 1)], [2 * n4]]).T.reshape(1, -1).flatten()
        d2 = np.setdiff1d(alldofs, np.hstack([d1, d3, d4])).flatten()
        for j in range(3):
            ufixed[2, j] = (np.array([[e0[0, j], e0[2, j] / 2], [e0[2, j] / 2, e0[1, j]]]) @ np.array([[nelx], [0]]))[0]
            ufixed[3, j] = (np.array([[e0[0, j], e0[2, j] / 2], [e0[2, j] / 2, e0[1, j]]]) @ np.array([[nelx], [0]]))[1]
            ufixed[6, j] = (np.array([[e0[0, j], e0[2, j] / 2], [e0[2, j] / 2, e0[1, j]]]) @ np.array([[0], [nely]]))[0]
            ufixed[7, j] = (np.array([[e0[0, j], e0[2, j] / 2], [e0[2, j] / 2, e0[1, j]]]) @ np.array([[0], [nely]]))[1]
            ufixed[4, j] = (ufixed[2:4, j] + ufixed[6:8, j])[0]
            ufixed[5, j] = (ufixed[2:4, j] + ufixed[6:8, j])[1]
        wfixed = np.vstack([np.tile(ufixed[2:4, :], reps=[nely - 1, 1]), np.tile(ufixed[6:8, :], reps=[nelx - 1, 1])])
        self.d1 = d1
        self.d2 = d2
        self.d3 = d3
        self.d4 = d4
        self.ufixed = ufixed
        self.wfixed= wfixed

    #-----------------------#
    def solve(self, density):

        self.densityField = density;
        self.u=np.zeros((self.ndof,1))
        # solve
        sK=((self.KE.flatten()[np.newaxis]).T*(self.Emin+(0.01 + density)**self.penal*(self.Emax-self.Emin))).flatten(order='F')
        K = coo_matrix((sK,(self.iK,self.jK)),shape=(self.ndof,self.ndof)).tocsc()
        K = K[self.free,:][:,self.free]
        self.u[self.free,0]=spsolve(K,self.f[self.free,0])

        self.Jelem = (np.dot(self.u[self.edofMat].reshape(self.nelx*self.nely,8),self.KE) * self.u[self.edofMat].reshape(self.nelx*self.nely,8) ).sum(1)

        return self.u, self.Jelem;
    #-----------------------#
    def solve88(self, density,density_2, E_target):
        self.densityField = density;
        self.u=np.zeros((self.ndof,3));
        #sK = (self.KE.T.reshape(-1, 1) @ (Emin + density.T.reshape(-1, 1).T ** (penal * (E0 - Emin)))).T.reshape(64 * nelx * nely, -1).flatten(order='F')
        sK=((self.KE.flatten()[np.newaxis]).T*(self.Emin+ density**self.penal*(self.Emax-self.Emin))).flatten(order='F')
        K = coo_matrix((sK,(self.iK,self.jK)),shape=(self.ndof,self.ndof)).tocsc()
        #K = self.deleterowcol(K,self.fixed,self.fixed).tocoo()
        Kr = bmat([[self.keep(K,self.d2 -1,self.d2-1),self.keep(K,self.d2-1,self.d3-1) + self.keep(K,self.d2-1,self.d4-1)],
                   [self.keep(K,self.d3-1,self.d2-1)+self.keep(K,self.d4-1,self.d2-1),self.keep(K,self.d3-1,self.d3-1)+ self.keep(K,self.d4-1,self.d3-1)+self.keep(K,self.d3-1,self.d4-1)+self.keep(K,self.d4-1,self.d4-1)]]).tocoo()
        #Kr= keep(K,self.d2, self.d2)
        Kr = cvxopt.spmatrix(Kr.data,Kr.row,Kr.col)
        B = -vstack([self.keep(K,self.d2-1,self.d1-1),self.keep(K,self.d3-1,self.d1-1)+self.keep(K,self.d4-1,self.d1-1)]).dot(self.ufixed) - vstack([self.keep(K,self.d2-1,self.d4-1),self.keep(K,self.d3-1,self.d4-1)+self.keep(K,self.d4-1,self.d4-1)]).dot(self.wfixed)
        B = cvxopt.matrix(B)
        cvxopt.cholmod.linsolve(Kr,B)
        self.u[self.d1-1,:]=self.ufixed
        self.u[np.hstack([self.d2, self.d3]) - 1 , :] = np.array(B)
        self.u[self.d4-1,:] = self.u[self.d3-1,:] +self.wfixed
        qe = np.zeros((3,3))
        Q = torch.zeros((3,3))
        dQ = np.zeros((3,3))
        for i in range(3):
            for j in range(3):
                U1 = self.u[:,i]
                U2 = self.u[:,j]
                qe= (((np.dot(U1[self.edofMat],self.KE)*U2[self.edofMat]).sum(axis=1)).reshape(self.nely,self.nelx))/(self.nelx*self.nely)
                Q[i,j] = torch.sum((density_2.reshape(self.nelx,self.nely)*self.Emax)*torch.tensor(qe, dtype=torch.float32))

        self.Jelem = (Q.view(1,-1) - torch.tensor(E_target))
        self.Q = Q.view(1,-1)
        return self.u, self.Jelem, self.Q
    #-----------------------#
    def deleterowcol(self, A, delrow, delcol):
        #Assumes that matrix is in symmetric csc form !
        m = A.shape[0]
        keep = np.delete (np.arange(0, m), delrow)
        A = A[keep, :]
        keep = np.delete (np.arange(0, m), delcol)
        A = A[:, keep]
        return A
    def keep(self, A, row, col):
        A= A[row, :]
        A= A[:, col]
        return A

    #-----------------------#
    def plotFE(self):
         #plot FE results
         fig= plt.figure() # figsize=(10,10)
         plt.subplot(1,2,1);
         im = plt.imshow(self.u[1::2].reshape((self.nelx+1,self.nely+1)).T, cmap=cm.jet,interpolation='none')
         J = ( (self.Emin+self.densityField**self.penal*(self.Emax-self.Emin))*self.Jelem).sum()
         plt.title('U_x , J = {:.2E}'.format(J))
         fig.colorbar(im)
         plt.subplot(1,2,2);
         im = plt.imshow(self.u[0::2].reshape((self.nelx+1,self.nely+1)).T, cmap=cm.jet,interpolation='none')
         fig.colorbar(im)
         plt.title('U_y')
         fig.show()
