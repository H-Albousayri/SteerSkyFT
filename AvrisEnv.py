import numpy as np 
import random

class AVRIS():
    def __init__(self, My_BS, Mz_BS, Nx_RIS, Ny_RIS, num_users=3, num_eves=2, train_G=True, mode="Beamforming"):
        super(AVRIS, self).__init__()
        self.My_BS = My_BS
        self.Mz_BS = Mz_BS
        self.Nx_RIS = Nx_RIS
        self.Ny_RIS = Ny_RIS
        
        self.M = My_BS * Mz_BS
        self.N = Nx_RIS * Ny_RIS

        self.K = num_users
        self.K_e = num_eves

        self.train_G = train_G
        self.mode = mode
        ###########################  
        
        self.channel_model = "rician" #OPTIONS = ["los", "rician"]
        
        ###########################
        
        self.Kai = 10000 #for LoS vs NLoS # 40dB value
        self.PLexponent = 4.0
        self.PLexponent_d = 2* self.PLexponent
        
        ###########################
        
        self.direction = 0 
        self.consider_LoS = False
        self.spacing = 10

        ###########################    
        
        self.H_1 = np.zeros(shape=[self.N, self.M], dtype=np.complex128)
        self.H_2 = np.zeros(shape=[self.N, self.K], dtype=np.complex128)
        self.H_2_e = np.zeros(shape=[self.N, self.K_e], dtype=np.complex128)
        
        self.G = np.ones([self.M, self.K], dtype=np.complex128) / np.sqrt(self.M) #* (np.random.random(self.N) + 1j*np.random.random(self.N)) / np.sqrt(2)
        self.Phi = np.eye(self.N, dtype=np.complex128) #* (np.random.random(self.N) + 1j*np.random.random(self.N)) / np.sqrt(2)
        
        ###########################
        
        self.P_max = 100
        self.awgn_var = 10**-12
        
        ###########################
        
        self.xyz_loc_BS = np.array([0., 0., 10.])
        
        ###########################
        
        self.xyz_loc_UAV = np.array([60., 10., 20.])
        self.xyz_loc_UAV[0:2] = np.random.uniform(20, 70, size=2)
        
        ###########################
        
        self.BS_UAV_dis = np.linalg.norm(self.xyz_loc_UAV - self.xyz_loc_BS)
    
        ###########################
        self.xyz_loc_Eve = np.zeros((self.K_e, 3))
        self.xyz_loc_Eve[:, 0:2] = np.random.uniform(20, 70, (self.K_e, 2))
        # self.xyz_loc_Eve = np.array([[60., 60., 0.]])
        
        self.xyz_loc_UE = np.zeros((self.K, 3))
        self.xyz_loc_UE[:, 0] = np.arange(self.K) * self.spacing
        self.xyz_loc_UE[:, 1] = 75
        
        ###########################
        
        self.UAV_UE_dis = np.linalg.norm(self.xyz_loc_UE - self.xyz_loc_UAV, axis=1)
        
        ###########################
        
        self.bit_rates = np.zeros(self.K) 
        self.eve_rates = np.zeros(self.K_e)
        self.scale_eve = 1
        self.reward_scale = 1 
        
        self.is_LoS = np.ones(self.K, dtype=np.bool_)
        self.is_LoS_e = np.ones(self.K_e, dtype=np.bool_)
        
        ###########################

        self.lamda = 0.1 #for 3 GHz system 
        self.d = self.lamda / 2 #perfect element spacing
        self.B_0 = (self.lamda / (4 * np.pi)) ** 2
        
        self.done = True

        if self.mode == "Beamforming":   
            if self.train_G:
                self.action_dim = self.N + self.M*self.K
            else:
                self.action_dim = self.N
            self.state_dim = (self.N * self.M +  self.N * self.K + self.M * self.K +
                                self.M * self.K_e + self.N * self.K_e
            )
        elif self.mode == "Move":
            self.action_dim = 2
            self.state_dim = 3 + 3*self.K
        
        elif self.mode == "All":
            self.action_dim = self.N + self.M*self.K + 2
            self.state_dim = (self.N * self.M +  self.N * self.K + self.M * self.K +
                                self.M * self.K_e + self.N * self.K_e + 2*(self.K + self.K_e) + 1
            )
            

        
        self.name = f"{self.M}x{self.N}_K={self.K}_Ke={self.K_e}_In:{self.state_dim}_Out:{self.action_dim}"
#####################################################
    
    def get_state(self):
        if self.mode == "Beamforming":   
            return np.hstack([np.angle(self.H_1).reshape(-1)/np.pi,
                                    np.angle(self.H_2).reshape(-1)/np.pi,
                                    np.angle(self.H_d).reshape(-1)/np.pi,
                                    np.angle(self.H_2_e).reshape(-1)/np.pi,
                                    np.angle(self.H_d_e).reshape(-1)/np.pi
            ])
            
        if self.mode == "All":   
            return np.hstack([np.angle(self.H_1).reshape(-1)/np.pi,
                                    np.angle(self.H_2).reshape(-1)/np.pi,
                                    np.angle(self.H_d).reshape(-1)/np.pi,
                                    np.angle(self.H_2_e).reshape(-1)/np.pi,
                                    np.angle(self.H_d_e).reshape(-1)/np.pi,
                                    self.BS_UAV_dis*1e-3, 
                                    self.UAV_UE_dis.flatten()*1e-3, 
                                    self.BS_UE_dis.flatten()*1e-3, 
                                    self.UAV_Eve_dis.flatten()*1e-3,
                                    self.BS_Eve_dis.flatten()*1e-3
                                    
            ])
            
        elif self.mode == "Move":
            return np.concatenate([self.xyz_loc_UAV, (self.xyz_loc_UE - self.xyz_loc_UAV).flatten()])
    
            
    def get_steering_vector_BS(self, delta_vec, distance):
        """BS steering vector (yz-plane) for delta_vec (shape [3,] or [K, 3])."""
        delta_y = delta_vec[..., 1]  # Shape: () or [K,]
        delta_z = delta_vec[..., 2]
        # distance = np.linalg.norm(delta_vec, axis=-1) if delta_vec.ndim > 1 else np.linalg.norm(delta_vec)
            
        k_y = 2 * np.pi * delta_y / (self.lamda * distance)  # Shape: () or [K,]
        k_z = 2 * np.pi * delta_z / (self.lamda * distance)
            
        m_y = np.arange(self.My_BS)  # [0, 1, ..., M_y-1]
        m_z = np.arange(self.Mz_BS)
            
        # Compute phase shifts
        a_y = np.exp(-1j * np.outer(k_y, m_y) * self.d)  # Shape: (1, M_y) or (K, M_y)
        a_z = np.exp(-1j * np.outer(k_z, m_z) * self.d)  # Shape: (1, M_z) or (K, M_z)
            
        # Kronecker product for planar array
        if delta_vec.ndim == 1:
            return np.kron(a_y, a_z).reshape(-1, 1)  # [M_y*M_z, 1]
        else:
            # Corrected batched Kronecker product
            return (a_y[:, :, np.newaxis] * a_z[:, np.newaxis, :]).reshape(self.K, -1).T  # [M_y*M_z, K]

#####################################################

    def get_steering_vector_RIS(self, delta_vec, distance, k=None):
        if k == None:
            k = self.K
        delta_x = delta_vec[..., 0]  # Shape: () or [K,]
        delta_y = delta_vec[..., 1]
        # distance = np.linalg.norm(delta_vec, axis=-1) if delta_vec.ndim > 1 else np.linalg.norm(delta_vec)
            
        k_x = 2 * np.pi * delta_x / (self.lamda * distance)
        k_y = 2 * np.pi * delta_y / (self.lamda * distance)
            
        n_x = np.arange(self.Nx_RIS)  # [0, 1, ..., N_x-1]
        n_y = np.arange(self.Ny_RIS)
            
        # Compute phase shifts
        a_x = np.exp(-1j * np.outer(k_x, n_x) * self.d)  # Shape: (1, N_x) or (K, N_x)
        a_y = np.exp(-1j * np.outer(k_y, n_y) * self.d)  # Shape: (1, N_y) or (K, N_y)
            
        # Kronecker product for planar array
        if delta_vec.ndim == 1:
            return np.kron(a_x, a_y).reshape(-1, 1)  # [N_x*N_y, 1]
        else:
            # Corrected batched Kronecker product
            return (a_x[:, :, np.newaxis] * a_y[:, np.newaxis, :]).reshape(k, -1).T  # [N_x*N_y, K]
        
#####################################################
    
    def reset(self):
        self.xyz_loc_Eve[:, 0:2] = np.random.uniform(20, 70, (self.K_e, 2))
        self.xyz_loc_UAV[:2] = np.random.uniform(100, 200, size=(2,))
        
        self.Phi = np.eye(self.N, dtype=np.complex128) * np.exp(1j*np.pi* np.random.uniform(-1, 1, size=self.N))
        
        delta_BS_UAV = self.xyz_loc_UAV - self.xyz_loc_BS
        self.BS_UAV_dis = np.linalg.norm(delta_BS_UAV)
        
        delta_UAV_UE = self.xyz_loc_UE - self.xyz_loc_UAV
        self.UAV_UE_dis = np.linalg.norm(delta_UAV_UE, axis=1)

        delta_UAV_Eve = self.xyz_loc_Eve - self.xyz_loc_UAV
        self.UAV_Eve_dis = np.linalg.norm(delta_UAV_Eve, axis=1)

        delta_BS_UE = self.xyz_loc_UE - self.xyz_loc_BS
        self.BS_UE_dis = np.linalg.norm(delta_BS_UE, axis=1)

        delta_BS_Eve = self.xyz_loc_Eve - self.xyz_loc_BS
        self.BS_Eve_dis = np.linalg.norm(delta_BS_Eve, axis=1)
        
        
        if self.consider_LoS:
            self.update_LoS(delta_UAV_UE, self.UAV_UE_dis, "not_eve")
            self.update_LoS(delta_UAV_Eve, self.UAV_Eve_dis, "eve")
            
        #######################################
        ########## Finding Channel 1 ##########
        #######################################
        
        a_t_BS = self.get_steering_vector_BS(delta_BS_UAV, self.BS_UAV_dis)
        a_r_RIS = self.get_steering_vector_RIS(delta_BS_UAV, self.BS_UAV_dis)
        
        H1_los = a_r_RIS @ a_t_BS.conj().T
        H1_rayleigh = np.random.normal(0, (1/(np.sqrt(2)**2)), size=self.H_1.shape) + 1j * np.random.normal(0, (1/(np.sqrt(2)**2)), size=self.H_1.shape)
        
        if self.channel_model == "rician":
            self.H_1 = (np.sqrt(self.B_0 * self.BS_UAV_dis**(-2))) * ((np.sqrt((self.Kai)/(self.Kai+1)) * H1_los) + (np.sqrt((1)/(self.Kai+1)) * H1_rayleigh))

        elif self.channel_model == "los":
            self.H_1 = (np.sqrt(self.B_0 * self.BS_UAV_dis**(-2))) * H1_los
        
        #######################################
        ########## Finding Channel 2 ##########
        #######################################
        
        a_t_RIS = self.get_steering_vector_RIS(delta_UAV_UE, self.UAV_UE_dis)
        a_t_RIS_e = self.get_steering_vector_RIS(delta_UAV_Eve, self.UAV_Eve_dis, self.K_e)
        
        H2_los = a_t_RIS
        H2_los_e = a_t_RIS_e
        
        H2_rayleigh = (np.random.normal(0, (1/(np.sqrt(2)**2)), size=self.H_2.shape) +
                       1j * np.random.normal(0, (1/(np.sqrt(2)**2)), size=self.H_2.shape))
        
        H2_rayleigh_e = (np.random.normal(0, (1/(np.sqrt(2)**2)), size=H2_los_e.shape) +
                       1j * np.random.normal(0, (1/(np.sqrt(2)**2)), size=H2_los_e.shape))       
        
        if self.channel_model == "rician":
            self.H_2.T[self.is_LoS] = ((np.sqrt(self.B_0 * self.UAV_UE_dis**(-2))) * 
                                       ((np.sqrt((self.Kai)/(self.Kai+1)) * H2_los) 
                                        + (np.sqrt((1)/(self.Kai+1))* H2_rayleigh))).T[self.is_LoS]

            self.H_2_e.T[self.is_LoS_e] = ((np.sqrt(self.B_0 * self.UAV_Eve_dis**(-2))) * 
                                       ((np.sqrt((self.Kai)/(self.Kai+1)) * H2_los_e) 
                                        + (np.sqrt((1)/(self.Kai+1))* H2_rayleigh_e))).T[self.is_LoS_e]
            
        elif self.channel_model == "los":
            self.H_2.T[self.is_LoS] = ((np.sqrt(self.B_0 * self.UAV_UE_dis**(-2))) * H2_los).T[self.is_LoS]
            self.H_2_e.T[self.is_LoS_e] = ((np.sqrt(self.B_0 * self.UAV_Eve_dis**(-2))) * H2_los_e).T[self.is_LoS_e]

        self.H_2.T[~self.is_LoS] = (np.sqrt(self.B_0 * self.UAV_UE_dis**(-self.PLexponent)) * H2_rayleigh).T[~self.is_LoS]
        self.H_2_e.T[~self.is_LoS_e] = (np.sqrt(self.B_0 * self.UAV_Eve_dis**(-self.PLexponent)) * H2_rayleigh_e).T[~self.is_LoS_e]

        ############################################
        ########## Finding Direct Channel ##########
        ############################################

        Hd_rayleigh = (np.random.normal(0, 1/np.sqrt(2), size=(self.K, self.M)) +
                       1j * np.random.normal(0, 1/np.sqrt(2), size=(self.K, self.M)))
        
        Hd_rayleigh_e = (np.random.normal(0, 1/np.sqrt(2), size=(self.K_e, self.M)) +
                       1j * np.random.normal(0, 1/np.sqrt(2), size=(self.K_e, self.M)))
        
        self.H_d = np.sqrt(self.B_0 * self.BS_UE_dis[:, None]**(-self.PLexponent_d)) * Hd_rayleigh
        
        self.H_d_e = np.sqrt(self.B_0 * self.BS_Eve_dis[:, None]**(-self.PLexponent_d)) * Hd_rayleigh_e
        
        #######################################
        #######################################
        

            
        H_eff = self.H_2.conj().T @ self.Phi @ self.H_1 + self.H_d
        H_eff_e = self.H_2_e.conj().T @ self.Phi @ self.H_1 + self.H_d_e
        
        #####################################################

        if self.train_G:
           self.G = np.ones([self.M, self.K], dtype=np.complex128) / np.sqrt(self.M)
        else:
            H_tilde = np.vstack([H_eff, H_eff_e])
            I_tilde = np.vstack([np.eye(self.K), np.zeros((self.K_e, self.K))])
    
            H_tilde_H = H_tilde.conj().T
            self.G = H_tilde_H @ np.linalg.inv(H_tilde @ H_tilde_H) @ I_tilde
            self.G /= (np.linalg.norm(self.G, axis=0, keepdims=True))
            
        #####################################################
        
        Y_pwr = np.abs(H_eff @ self.G)**2
        Desired_pwr = np.diag(Y_pwr)
        Total_power = np.sum(Y_pwr, axis=1) 
        Interference = Total_power - Desired_pwr
        SNIRs = Desired_pwr / (Interference + self.awgn_var) * self.P_max
        self.bit_rates = np.log2(1 + SNIRs) 
        
        #####################################################
        
        Eve_pwr = np.max(np.abs(H_eff_e @ self.G)**2, axis=1)
        SNIRs_e = Eve_pwr / self.awgn_var * self.P_max
        self.eve_rates = np.log2(1 + SNIRs_e)
        
        #####################################################
        secrecy_rate = np.sum(self.bit_rates) - self.scale_eve * np.sum(self.eve_rates)
        #####################################################
        
        self.state = self.get_state()
                        
        return self.state

#####################################################
    
    def update_LoS(self, delta, dis, which):
        C = 0.6
        Y = 0.11
        Th0 = 15
        Th_k = np.abs(np.arcsin(delta[..., 2] / (dis))) * (180/np.pi)
        Th_k = np.maximum(Th_k, Th0)
        Prob_LoS = C * ((Th_k - Th0) ** Y)
        
        if which == "eve":
            self.is_LoS_e = np.random.binomial(1, Prob_LoS).astype(dtype=np.bool_)
        else:
            self.is_LoS = np.random.binomial(1, Prob_LoS).astype(dtype=np.bool_)
        
###################################################
    def step(self, action):
        if self.mode == "Move":
            dx, dy = action[self.N+self.M*self.K:]
            self.xyz_loc_UAV[0:2] += dx, dy
        
        if self.mode == "All":
            self.Phi = np.diag(np.exp(1j * action[:self.N] * np.pi))
            self.G = np.exp(1j * action[self.N:self.N+self.M*self.K] * np.pi).reshape(self.M, self.K) / np.sqrt(self.M)
            dx, dy = action[self.N+self.M*self.K:]
            self.xyz_loc_UAV[0:2] += dx, dy
        
        if self.mode == "Beamforming" and self.train_G:
            self.Phi = np.diag(np.exp(1j * action[:self.N] * np.pi))
            self.G = np.exp(1j * action[self.N:self.N+self.M*self.K] * np.pi).reshape(self.M, self.K) / np.sqrt(self.M)
        
        delta_BS_UAV = self.xyz_loc_UAV - self.xyz_loc_BS
        self.BS_UAV_dis = np.linalg.norm(delta_BS_UAV)
        
        delta_UAV_UE = self.xyz_loc_UE - self.xyz_loc_UAV
        self.UAV_UE_dis = np.linalg.norm(delta_UAV_UE, axis=1)

        delta_UAV_Eve = self.xyz_loc_Eve - self.xyz_loc_UAV
        self.UAV_Eve_dis = np.linalg.norm(delta_UAV_Eve, axis=1)

        delta_BS_UE = self.xyz_loc_UE - self.xyz_loc_BS
        self.BS_UE_dis = np.linalg.norm(delta_BS_UE, axis=1)

        delta_BS_Eve = self.xyz_loc_Eve - self.xyz_loc_BS
        self.BS_Eve_dis = np.linalg.norm(delta_BS_Eve, axis=1)
        
        if self.consider_LoS:
            self.update_LoS(delta_UAV_UE, self.UAV_UE_dis, "not_eve")
            self.update_LoS(delta_UAV_Eve, self.UAV_Eve_dis, "eve")
            
        # print(f"UE LoS: {self.is_LoS} | Eve LoS {self.is_LoS_e}")
        
        #######################################
        ########## Finding Channel 1 ##########
        #######################################
        
        a_t_BS = self.get_steering_vector_BS(delta_BS_UAV, self.BS_UAV_dis)
        a_r_RIS = self.get_steering_vector_RIS(delta_BS_UAV, self.BS_UAV_dis)
        
        H1_los = a_r_RIS @ a_t_BS.conj().T
        H1_rayleigh = np.random.normal(0, (1/(np.sqrt(2)**2)), size=self.H_1.shape) + 1j * np.random.normal(0, (1/(np.sqrt(2)**2)), size=self.H_1.shape)
        
        if self.channel_model == "rician":
            self.H_1 = (np.sqrt(self.B_0 * self.BS_UAV_dis**(-2))) * ((np.sqrt((self.Kai)/(self.Kai+1)) * H1_los) + (np.sqrt((1)/(self.Kai+1)) * H1_rayleigh))

        elif self.channel_model == "los":
            self.H_1 = (np.sqrt(self.B_0 * self.BS_UAV_dis**(-2))) * H1_los
        
        #######################################
        ########## Finding Channel 2 ##########
        #######################################
        
        a_t_RIS = self.get_steering_vector_RIS(delta_UAV_UE, self.UAV_UE_dis)
        a_t_RIS_e = self.get_steering_vector_RIS(delta_UAV_Eve, self.UAV_Eve_dis, self.K_e)
        
        H2_los = a_t_RIS
        H2_los_e = a_t_RIS_e
        
        H2_rayleigh = (np.random.normal(0, (1/(np.sqrt(2)**2)), size=self.H_2.shape) +
                       1j * np.random.normal(0, (1/(np.sqrt(2)**2)), size=self.H_2.shape))
        
        H2_rayleigh_e = (np.random.normal(0, (1/(np.sqrt(2)**2)), size=H2_los_e.shape) +
                       1j * np.random.normal(0, (1/(np.sqrt(2)**2)), size=H2_los_e.shape))       
        
        if self.channel_model == "rician":
            self.H_2.T[self.is_LoS] = ((np.sqrt(self.B_0 * self.UAV_UE_dis**(-2))) * 
                                       ((np.sqrt((self.Kai)/(self.Kai+1)) * H2_los) 
                                        + (np.sqrt((1)/(self.Kai+1))* H2_rayleigh))).T[self.is_LoS]

            self.H_2_e.T[self.is_LoS_e] = ((np.sqrt(self.B_0 * self.UAV_Eve_dis**(-2))) * 
                                       ((np.sqrt((self.Kai)/(self.Kai+1)) * H2_los_e) 
                                        + (np.sqrt((1)/(self.Kai+1))* H2_rayleigh_e))).T[self.is_LoS_e]
            
        elif self.channel_model == "los":
            self.H_2.T[self.is_LoS] = ((np.sqrt(self.B_0 * self.UAV_UE_dis**(-2))) * H2_los).T[self.is_LoS]
            self.H_2_e.T[self.is_LoS_e] = ((np.sqrt(self.B_0 * self.UAV_Eve_dis**(-2))) * H2_los_e).T[self.is_LoS_e]

        self.H_2.T[~self.is_LoS] = (np.sqrt(self.B_0 * self.UAV_UE_dis**(-self.PLexponent)) * H2_rayleigh).T[~self.is_LoS]
        self.H_2_e.T[~self.is_LoS_e] = (np.sqrt(self.B_0 * self.UAV_Eve_dis**(-self.PLexponent)) * H2_rayleigh_e).T[~self.is_LoS_e]

        ############################################
        ########## Finding Direct Channel ##########
        ############################################

        Hd_rayleigh = (np.random.normal(0, 1/np.sqrt(2), size=(self.K, self.M)) +
                       1j * np.random.normal(0, 1/np.sqrt(2), size=(self.K, self.M)))
        
        Hd_rayleigh_e = (np.random.normal(0, 1/np.sqrt(2), size=(self.K_e, self.M)) +
                       1j * np.random.normal(0, 1/np.sqrt(2), size=(self.K_e, self.M)))
        
        self.H_d = np.sqrt(self.B_0 * self.BS_UE_dis[:, None]**(-self.PLexponent_d)) * Hd_rayleigh
        
        self.H_d_e = np.sqrt(self.B_0 * self.BS_Eve_dis[:, None]**(-self.PLexponent_d)) * Hd_rayleigh_e
        
        #######################################
        #######################################
            
        H_eff = self.H_2.conj().T @ self.Phi @ self.H_1 + self.H_d
        H_eff_e = self.H_2_e.conj().T @ self.Phi @ self.H_1 + self.H_d_e
        
        #####################################################

        if not self.train_G and self.mode != "Move":
            H_tilde = H_eff
            I_tilde = np.eye(self.K)
            H_tilde_H = H_tilde.conj().T
            self.G = H_tilde_H @ np.linalg.inv(H_tilde @ H_tilde_H) @ I_tilde
            self.G /= (np.linalg.norm(self.G, axis=0, keepdims=True))
            
        #####################################################
        
        Y_pwr = np.abs(H_eff @ self.G)**2
        Desired_pwr = np.diag(Y_pwr)
        Total_power = np.sum(Y_pwr, axis=1) 
        Interference = Total_power - Desired_pwr
        SNIRs = Desired_pwr / (Interference + self.awgn_var) * self.P_max
        self.bit_rates = np.log2(1 + SNIRs) 
        
        #####################################################
        
        Eve_pwr = np.sum(np.abs(H_eff_e @ self.G)**2, axis=1)
        SNIRs_e = Eve_pwr / self.awgn_var * self.P_max
        self.eve_rates = np.log2(1 + SNIRs_e)
        
        #####################################################
        secrecy_rate = np.sum(self.bit_rates) - self.scale_eve * np.sum(self.eve_rates)
        #####################################################         
        
        reward = self.reward_scale* secrecy_rate

        self.xyz_loc_Eve[:, 0:2] = np.random.uniform(20, 70, (self.K_e, 2))
        delta_BS_Eve = self.xyz_loc_Eve - self.xyz_loc_BS
        self.BS_Eve_dis = np.linalg.norm(delta_BS_Eve, axis=1)
        delta_UAV_Eve = self.xyz_loc_Eve - self.xyz_loc_UAV
        self.UAV_Eve_dis = np.linalg.norm(delta_UAV_Eve, axis=1)
        a_t_RIS_e = self.get_steering_vector_RIS(delta_UAV_Eve, self.UAV_Eve_dis, self.K_e)
        
        H2_los_e = a_t_RIS_e
        H2_rayleigh_e = (np.random.normal(0, (1/(np.sqrt(2)**2)), size=H2_los_e.shape) +
                       1j * np.random.normal(0, (1/(np.sqrt(2)**2)), size=H2_los_e.shape))       
        
        if self.channel_model == "rician":
            self.H_2_e.T[self.is_LoS_e] = ((np.sqrt(self.B_0 * self.UAV_Eve_dis**(-2))) * 
                                       ((np.sqrt((self.Kai)/(self.Kai+1)) * H2_los_e) 
                                        + (np.sqrt((1)/(self.Kai+1))* H2_rayleigh_e))).T[self.is_LoS_e]
            
        elif self.channel_model == "los":
            self.H_2_e.T[self.is_LoS_e] = ((np.sqrt(self.B_0 * self.UAV_Eve_dis**(-2))) * H2_los_e).T[self.is_LoS_e]

        self.H_2_e.T[~self.is_LoS_e] = (np.sqrt(self.B_0 * self.UAV_Eve_dis**(-self.PLexponent)) * H2_rayleigh_e).T[~self.is_LoS_e]

        self.state = self.get_state()
        
        return self.state, reward, False, None