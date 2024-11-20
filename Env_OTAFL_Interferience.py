import gym
from gym import spaces
import numpy as np
#import cv2
import random
import time
from collections import deque
import math

def generate_rayleigh_coefficient(num_coefficients, scale=1):
    # Generate complex Gaussian random numbers with zero mean and unit variance
    gaussian_real = np.round(np.random.normal(loc=0, scale=scale, size=num_coefficients), 4)
    gaussian_imag = np.round(np.random.normal(loc=0, scale=scale, size=num_coefficients), 4)
    
    # Form complex numbers
    complex_numbers = gaussian_real + 1j * gaussian_imag
    
    # Calculate the magnitude of complex numbers
    rayleigh_coefficients = np.abs(complex_numbers)
    
    return complex_numbers[0] 


def channel_gain(nk=2, nris=3):
    hd_ss = []
    hd_sp = []
    hk_sr = []
    hd_ps = []
    hk_pr = []
    G_rs = []
    G_rp = []

    for i in range(nk):
        hd_ss.append(generate_rayleigh_coefficient(1, scale=1))
        hd_sp.append(generate_rayleigh_coefficient(1, scale=1))
   
    for i in range(nk):
        a = []
        for j in range(nris):
            a.append(generate_rayleigh_coefficient(1, scale=1))
        hk_sr.append(a)
    hk_sr =hk_sr

    hd_ps.append(generate_rayleigh_coefficient(1, scale=1))

    for j in range(nris):
        hk_pr.append(generate_rayleigh_coefficient(1, scale=1))
        G_rs.append(generate_rayleigh_coefficient(1, scale=1))
        G_rp.append(generate_rayleigh_coefficient(1, scale=1))


    # print("hk_pr",np.array(hk_pr))

    # print(np.array(hd_ss).shape,np.array(hk_sr).shape ,"\n")
    return np.array(hd_ss),np.array(hd_sp), np.array(hk_sr), np.array(hd_ps), np.array(hk_pr), np.array(G_rs), np.array(G_rp)  

def channel_gain(nk=2, nris=3):
    hd_ss = []
    hd_sp = []
    hk_sr = []
    hd_ps = []
    hk_pr = []
    G_rs = []
    G_rp = []

    for i in range(nk):
        hd_ss.append(generate_rayleigh_coefficient(1, scale=1))
        hd_sp.append(generate_rayleigh_coefficient(1, scale=1))
   
    for i in range(nk):
        a = []
        for j in range(nris):
            a.append(generate_rayleigh_coefficient(1, scale=1))
        hk_sr.append(a)
    

    hd_ps.append(generate_rayleigh_coefficient(1, scale=1))

    for j in range(nris):
        hk_pr.append(np.array(generate_rayleigh_coefficient(1, scale=1)))
        G_rs.append(generate_rayleigh_coefficient(1, scale=1))
        G_rp.append(np.array(generate_rayleigh_coefficient(1, scale=1)))


    # print("hk_pr",np.array(hk_pr))

    # print("np.array(hd_ss):",np.array(hd_ss), "\n np.array(hd_sp)",np.array(hd_sp),
    #      "\n np.array(hk_sr)",np.array(hk_sr),"\n np.array(hd_ps)",np.array(hd_ps),
    #      "\n ,np.array(hk_pr),",np.array(hk_pr), "\n np.array(G_rs)",np.array(G_rs), "\n np.array(G_rp)",np.array(G_rp)  ,"\n")


    # print(np.array([hd_ss]).shape,np.array([hd_sp]).shape, np.array(hk_sr).shape, np.array([hd_ps]).shape,
    #       np.array([hk_pr]).shape, np.array([G_rs]).shape, np.array([G_rp]).shape)
    return np.array([hd_ss]),np.array([hd_sp]), np.array(hk_sr), np.array([hd_ps]), np.array([hk_pr]), np.array([G_rs]), np.array([G_rp])  


# pathloss_coef
def generate_path_loss(nk,dk, exp_coef=1, l=1):
    path_loss = []
    for i in range(nk):
        path_loss.append(l/((dk[i])**exp_coef)) 
    # print("pathoriginal",path_loss)
    return np.round(path_loss, 10)

# chnnel quef
def generate_channel_coef(nk, hd, hr, G, diag, path_loss_d, path_loss_r):
    channel_coef = []
    # print("\n hd", hd.reshape(1,2))
    for i in range(nk):
        path_loss_r[i] =0
        if nk > 1:
            # print(hr[i].shape)
            # print(hd.shape, np.array(path_loss_d).shape, path_loss_r.shape,hr.shape )
            # print(G.shape,diag.shape,hr[i].reshape(3,1).shape )
            # print("if")
            # print("path:", path_loss_d[1],path_loss_r)
            # print("\n i",i,np.sqrt(path_loss_d[i]),)
            # print("\n i", hd[0][i],)
            # print("\n i",np.sqrt(path_loss_r[i]))
            # print("\n i",hr[i].reshape(3,1) )
            h = np.sqrt(path_loss_d[i]) * hd[0][i] + np.sqrt(path_loss_r[i]) * ((G) @ diag @ (hr[i].reshape(6,1)))
            # print("\n nk",nk,path_loss_d,hd.size,path_loss_r,hr.size )
        else:
        #    print(hr.shape,diag.shape,G.T.shape ) 
        #    print("else")
           h = np.sqrt(path_loss_d) * hd + np.sqrt(path_loss_r[i]) * ((G) @ diag @ (hr.reshape(6,1)))
           
        channel_coef.append(h)
    return channel_coef

def channel_coef(nk,nkp,hd_ss, hd_sp, hk_sr, hd_ps, hk_pr, G_rs, G_rp, teta, dk_ss, dk_srs, dk_sp, dk_srp, dk_prs, dk_ps, l=1, exp_coef_d=1, exp_coef_r=1):
    exp_teta = [np.exp(1j*val) for val in teta]
    diag = np.diag(exp_teta)



    # dk_ss, dk_srs, dk_sp, dk_srp, dk_prs, dk_ps = distance()

    path_loss_ss = generate_path_loss(nk,dk_ss, exp_coef_d, l)
    path_loss_srs = generate_path_loss(nk,dk_srs, exp_coef_r, l)

    path_loss_sp = generate_path_loss(nk,dk_sp, exp_coef_d, l)
    path_loss_srp = generate_path_loss(nk,dk_srp, exp_coef_r, l)

    path_loss_ps = generate_path_loss(nkp,dk_ps, exp_coef_d, l)
    path_loss_prs = generate_path_loss(nkp,dk_prs, exp_coef_r, l)

    # print("SSSSSSS")
    channel_coeff_ss = generate_channel_coef(nk, hd_ss, hk_sr, G_rs, diag, path_loss_ss, path_loss_srs)
    # print("\n hd_ss,",hd_ss,"\n hk_sr",hk_sr, "\n G_rs=", G_rs,"\ndiag=", diag, "\npath_loss_ss", path_loss_ss,"\npath_loss_srs", path_loss_srs)
    # print("\nchannel_coeff_ss[0]",channel_coeff_ss[0])
    # print("sssssp")
    channel_coeff_sp = generate_channel_coef(nk,hd_sp, hk_sr, G_rp, diag, path_loss_sp, path_loss_srp)
    # print("pssssss")
    channel_coeff_ps = generate_channel_coef(nkp,hd_ps, hk_pr, G_rs, diag, path_loss_ps, path_loss_prs)

    return channel_coeff_ss, channel_coeff_sp, channel_coeff_ps 

def calculate_distance(ss, orginal):
    d = []
    for i in range(math.floor(ss.size/2)):
        d.append(math.floor(np.sqrt((ss[i][0]-orginal[0])**2 +(ss[i][1]-orginal[1])**2 )))
    return np.array(d)

def distance():
    # ss = np.array([[70,20],[80,10],[55,40],[40,10],[40,30],[40,50],[70,30],[80,60],[60,20],[90,10],
    #                [66,2],[50,10],[60,20],[50,10],[40,40],[80,11],[100,20],[30,50],[90,20],[100,10]])
    ss=np.array([[70,20],[80,10],[55,40],[78,10],[90,30]])
    pu = np.array([[0,50]])
    
    dk_ss = calculate_distance(ss, np.array([100,0]))

    dk_srs = []
    dk_srp = []
    for i in range(math.floor(ss.size/2)):
        dk_srs.append(calculate_distance(ss, [50,0])[i] + 25)
        dk_srp.append(calculate_distance(ss, [50,0])[i] + 75)
    
    dk_prs = np.array([calculate_distance(pu, [50,0])[0] + 25])
    dk_sp = calculate_distance(ss, [0,0])
    dk_ps = calculate_distance(pu, [100,0]) 

    return dk_ss, np.array(dk_srs), np.array(dk_sp), np.array(dk_srp), dk_prs, dk_ps

class SnekEnv(gym.Env):
    def __init__(self):
        super(SnekEnv, self).__init__()
        self.nk = 5
        self.nris = 6
        self.hd_ss, self.hd_sp, self.hk_sr, self.hd_ps, self.hk_pr, self.G_rs, self.G_rp = channel_gain(self.nk, self.nris)
        self.teta = [1, 1, 1,1,1,1]
        self.dk_ss, self.dk_srs, self.dk_sp, self.dk_srp, self.dk_prs, self.dk_ps = distance()
        self.l =10**(30 / 10) / 1000
        self.exp_coef_d = 3.5
        self.exp_coef_r = 2.5
        self.rho = 0.7
        self.noise_ss = np.random.uniform()
        self.noise_sp = np.random.uniform()
        self.penalty = 10
        self.bw = 1000
        self.nois_power_ss = 10**(-117 / 10) / 1000
        self.nois_power_sp = 10**(-113 / 10) / 1000
        self.pu_power = 10**(5 / 10) / 1000
        self.k = 5
        self.tershold = 10**(25 / 10) / 1000
        self.power_max =10**(36 / 10) / 1000
        self.eta_max = 5

        self.ns=5
        self.np=1

        self.step_num=0
        self.step_num_1 = 0

        self.dk_ss, self.dk_srs, self.dk_sp, self.dk_srp, self.dk_prs, self.dk_ps = distance()

        self.actionspace = self.ns + self.nris + self.ns + 1
        self.action_space = spaces.Box(low=0.1, high=1, shape=(self.actionspace,))
        self.obsevationspace = self.ns + self.ns + self.ns * self.nris + self.np + self.nris + self.nris + self.nris
        self.observation_space = spaces.Box(low=0.1, high=1, shape=(self.obsevationspace,))

    def uncertainty(self,ris,nk, hk):
        hk_hat = []
        # print(np.array(hk).shape,   nk,"\n")
        # print(hk,   nk,"\n")
        if ris ==0:
            for i in range(nk):
                # print(hk[0][i]) 
                hk_hat.append(self.rho * hk[0][i] + np.sqrt(1 - self.rho**2) * (np.random.uniform()))
            # print(np.array(hk_hat).shape,"\n")
        else:
             for i in range(nk):
                temp=[]
                # print("ris:",ris,hk)
                for element in hk[i]:
                    temp.append(self.rho * element + np.sqrt(1 - self.rho**2) * (np.random.uniform()))
                hk_hat.append(temp)
        hk_hat = np.array(hk_hat).reshape(hk.shape)    
        # print("\n",hk.shape,np.array(hk_hat).shape,hk,np.array(hk_hat) )
        return np.array(hk_hat)

    def MSE(self, eta, hk_ss, hk_ss_hat, indicator, pk, hk_ps, ppu, noise_power):
        item_1 = 0
        for i in range(np.array(hk_ss).size):
            # print(eta,hk_ss[i],)
            # print("hk:",hk_ss[i],"eta",eta,"self.k",self.k)
            item_1 += (np.abs((1/self.k) - 1/np.sqrt(eta) * (((hk_ss_hat[i] * hk_ss[i].conjugate()).real) / np.abs(hk_ss[i])**2) * indicator[i] * pk[i])**2)
            # print("1/k:",np.abs(1/self.k),"1/eta:",1/np.sqrt(eta), "h*h1/||^2:",(((hk_ss_hat[i] * hk_ss[i].conjugate()).real) / np.abs(hk_ss[i])**2),"pk[i]",pk[i])  
            # print("i:",i,"self.k:",self.k, "eta",eta, "hk_ss_hat[i]:",hk_ss_hat[i], "hk_ss[i]",hk_ss[i],"pk[i]:",pk[i], "MSE_part:",item_1 )
        a = np.abs(item_1) + (1/eta) * np.abs(hk_ps[0] * ppu)**2 + 1/eta * noise_power
        # print("item_1:",item_1 , "item_2:",((1/eta) * np.abs(hk_ps[0] * ppu)**2 + 1/eta * noise_power), "MSE:",a )
        return a

    def Interference(self, hk_ss, hk_sp_hat, indicator, pk, noise_power):
        item = 0
        for i in range(np.array(hk_ss).size):
            item += np.abs((hk_sp_hat[i] * hk_ss[i].conjugate()) / (np.abs(hk_ss[i])**2) * indicator[i] * pk[i])**2
        myinterference = item + noise_power
        return myinterference

    def step(self, action):
        self.step_num +=1
        self.step_num_1 +=1
        self.pk = np.round(np.array(action[0:self.ns]),5) * self.power_max
        if(self.step_num_1 > 10000):
            self.indicator = np.abs(np.round(np.array(action[self.ns:self.ns*2])))
        else:
            self.indicator = np.ones(self.ns)
        self.teta = np.round(np.array(action[2*self.ns:self.ns*2+self.nris]) * 2 * math.pi,5)
        self.eta = np.array(action[-1]) * self.eta_max
        self.eta = np.round(np.abs(self.eta) * self.ns,5)

        if np.any(self.pk == 0):
            self.pk[self.pk == 0] = 0.01

        if np.any(self.pk < 0):
            self.pk[self.pk < 0] = np.abs(self.pk[self.pk < 0])

        if np.any(self.pk == 0):
            print('k: ', self.k+1, 'pk:  ', self.pk, 'teta:    ', self.teta)
        if np.any(self.pk < 0):
            print('k: ', self.k+1, 'pk:  ', self.pk, 'teta:    ', self.teta)

        if np.any((self.indicator != 0) & (self.indicator != 1)):
            print("indicator",self.indicator)
        
        
        if (self.eta ==0):
            self.eta =0.01
        

        # new_action = np.concatenate((np.array(self.pk).flatten(),np.array(self.indicator).flatten(),np.array(self.teta).flatten(),
                                        # np.array(self.eta).flatten()))
        self.channel_coeff_ss, self.channel_coeff_sp, self.channel_coeff_ps = channel_coef(self.ns,self.np,self.hd_ss, self.hd_sp, self.hk_sr, self.hd_ps, self.hk_pr, self.G_rs, self.G_rp, self.teta, self.dk_ss, self.dk_srs, self.dk_sp, self.dk_srp, self.dk_prs, self.dk_ps, self.l, self.exp_coef_d, self.exp_coef_r)

        self.hd_ss_hat = self.uncertainty(0,self.ns,self.hd_ss)
        self.hd_sp_hat = self.uncertainty(0,self.ns,self.hd_sp)
        self.hk_sr_hat = self.uncertainty(1,self.ns, self.hk_sr)
        self.hd_ps_hat = self.uncertainty(0,self.np, self.hd_ps)
        self.hk_pr_hat = self.uncertainty(1,self.np , self.hk_pr)
        self.G_rs_hat = self.uncertainty(0,self.nris, self.G_rs)
        self.G_rp_hat = self.uncertainty(0,self.nris,self.G_rp)

        self.channel_coeff_ss_hat, self.channel_coeff_sp_hat, self.channel_coeff_ps_hat = channel_coef(self.ns,self.np,self.hd_ss_hat, self.hd_sp_hat, self.hk_sr_hat, self.hd_ps_hat, self.hk_pr_hat, self.G_rs_hat, self.G_rp_hat, self.teta, self.dk_ss, self.dk_srs, self.dk_sp, self.dk_srp, self.dk_prs, self.dk_ps, self.l, self.exp_coef_d, self.exp_coef_r)

        self.mse_vlue = self.MSE(self.eta, self.channel_coeff_ss, self.channel_coeff_ss_hat, self.indicator, self.pk, self.channel_coeff_ps, self.pu_power, self.nois_power_ss)
        # (self, hk_ss, hk_sp_hat, indicator, pk, noise_power)
        self.interfer_tershold = self.Interference(self.channel_coeff_ss, self.channel_coeff_sp_hat, self.indicator,self.pk ,self.nois_power_sp)
        penalty_indicator = 0
        # print("self.interfer_tershold",self.interfer_tershold)
        # print("self.interfer_tershold:\n",self.interfer_tershold)
        if self.interfer_tershold > self.tershold:
            penalty_indicator = 1
        # print("self.MSE and penalty_indicator",self.mse_vlue,penalty_indicator)
        self.reward = -self.mse_vlue - penalty_indicator * self.penalty
        
        self.done = False
        # self.step_num +=1
        if self.step_num >5:
            self.done = True
            self.setp_num =0 
            # print("self.pk:   ", self.pk[1:5], "self.indicator:" , self.indicator[1:5], "self.teta", self.teta[1:5],"self.eta",self.eta ,"\n")
    

        info = {}

        # self.hd_ss, self.hd_sp, self.hk_sr, self.hd_ps, self.hk_pr, self.G_rs, self.G_rp = channel_gain(self.ns, self.nris)

        

        hd_ss = np.round(np.array(self.hd_ss).flatten(),2)
        hd_sp= np.round(np.array(self.hd_sp).flatten(),2)
        hk_sr=np.round(np.array(self.hk_sr).flatten(),2)
        hd_ps = np.round(np.array(self.hd_ps).flatten(),2)
        hk_pr= np.round(np.array(self.hk_pr).flatten(),2)
        G_rs=np.round(np.array(self.G_rs).flatten(),2)
        G_rp=np.round(np.array(self.G_rp).flatten(),2)

        # print("\n",self.step_num_1,hd_ss, hd_sp, hk_sr, hd_ps, hk_pr, G_rs, G_rp)
        new_observation = np.concatenate((hd_ss, hd_sp, hk_sr, hd_ps, hk_pr, G_rs, G_rp))

        
        return new_observation, np.array(self.reward), self.done, info

    def reset(self):
        self.hd_ss, self.hd_sp, self.hk_sr, self.hd_ps, self.hk_pr, self.G_rs, self.G_rp = channel_gain(self.nk, self.nris)
        
       
        # print(np.array(np.ravel(self.hd_ss)).shape, np.array(self.hd_sp.shape), np.array(self.hk_sr).shape, np.array(self.hd_ps).shape,
        #      np.array(self.hk_pr).shape, np.array(self.G_rs).shape, np.array(self.G_rp).shape)
        hd_ss = np.round(np.array(self.hd_ss).flatten(),2)
        hd_sp= np.round(np.array(self.hd_sp).flatten(),2)
        hk_sr=np.round(np.array(self.hk_sr).flatten(),2)
        hd_ps = np.round(np.array(self.hd_ps).flatten(),2)
        hk_pr= np.round(np.array(self.hk_pr).flatten(),2)
        G_rs=np.round(np.array(self.G_rs).flatten(),2)
        G_rp=np.round(np.array(self.G_rp).flatten(),2)
        
        # print("hd_ss=", np.array(self.hd_ss),"\n hd_sp=",np.array(self.hd_sp))
        
        
        
        
        # print("\n",hd_ss, hd_sp, hk_sr, hd_ps, hk_pr, G_rs, G_rp)
        observation = np.concatenate((hd_ss, hd_sp, hk_sr, hd_ps, hk_pr, G_rs, G_rp))
        return observation
