import numpy as np

class HMC_Viterbi_Restorator:
    
    def __init__(self, Omega_X, Pi, A, B):
        self.Omega_X = Omega_X
        self.N = len(Omega_X)
        self.Pi = Pi
        self.A = A
        self.B = B
        
    def restore_X(self, Y):
        T = len(Y)
        V, X_path = self.forward(Y)
        X_hat = self.viterbi_path(T, V, X_path)
        return X_hat
    
    def forward(self, Y):
        T = len(Y)
        V = np.zeros((T, self.N))
        X_path = np.zeros((T, self.N))
    
        V[0] = self.compute_V1(Y[0])
        for t in range(T - 1):
            V[t + 1], X_path[t + 1] = self.compute_V_t_plus_1(Y[t + 1], V[t])
        
        return V, X_path
    
    def viterbi_path(self, T, V, X_path):
        X = [None] * T  #X contient les éléments de Omega_X (str)
        X_cp = [None] * T  #X _cp contient les indices des éléments (int), permet de faire la jonction 
        #entre calcul du chemin de viterbi et le résultat de type str dans X
        
        #omega permet d'accéder facilement à un lambda en connaissant son indice
        omega = list(enumerate(self.Omega_X))

        X[T-1] = omega[int(np.argmax(V[-1]))][1]
        X_cp[T-1] = int(np.argmax(V[-1]))
        for i in range(1,T):
            X_cp[T-1-i] = int(X_path[T-i][int(X_cp[T-i])])
            X[T-1-i] = omega[X_cp[T-1-i]][1]

        return X
    
    
    ####################
    ### V and X_path ###
    ####################
    
    def compute_V1(self, y0):
        V1 = np.zeros(self.N)
         #vérification de la nullité des bi
        test = 0
        for idx,i in enumerate(self.Omega_X):
            test += self.B[i].get(y0,0)
        if (test == 0):
            for idx,i in enumerate(self.Omega_X):
                self.B[i][y0] = 1

        for idx , i in enumerate(self.Omega_X):
            V1[idx] = self.Pi[i] * self.B[i].get(y0,0)
        V1 /= np.sum(V1)

        return V1
    
    def compute_V_t_plus_1(self, yt_plus_1, Vt):
        Vt_plus_1 = np.zeros(self.N)
        X_path_t_plus_1 = np.zeros(self.N)
        #vérification de la nullité des bi
        test = 0
        for idx,i in enumerate(self.Omega_X):
            test += self.B[i].get(yt_plus_1,0)
        if (test == 0):
            for idx,i in enumerate(self.Omega_X):
                self.B[i][yt_plus_1] = 1


        for idx , i in enumerate(self.Omega_X):
            L = []
            for jdx, j in enumerate(self.Omega_X):
                L.append(self.B[i].get(yt_plus_1,0) * self.A[j].get(i,0) * Vt[jdx])
            Vt_plus_1[idx] = max(L)
            X_path_t_plus_1[idx] = L.index(max(L))

        Vt_plus_1 /= np.sum(Vt_plus_1)
        return Vt_plus_1, X_path_t_plus_1
    

    
