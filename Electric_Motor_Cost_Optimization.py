# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 11:40:41 2021

Copyright TheAmirHK
"""
import numpy as np
import sys
import time 
from func_timeout import func_timeout, FunctionTimedOut
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import math as m
import pickle


# In[] Electric motor and processing information 
L=[-80, -120, 60, 80, 100, 140, 50, 30, -20, 85, -50]
s = [0.0566, 0.0133, 0.0100,0.0166, 0.0300, 0.0208,0.0208, 0.0133, 0.009]


Components_Manufacturing_cost= (5, 8, 10, 3, 2.5, 2.95, 2.95, 3.15, 4)  
Components_Scraping_cost = (.5,.5,.5) 
Components_Inspection_cost = (1,1.5,1)
Components_Reworking_cost = (1,1,1) 
Components_Inventory_cost = (2,2,2) 

Product_Scraping_cost = 10
Product_Inspection_cost = 0.5
Product_Assembly_cost = 3


Nlinear = 100 #Linearization parameter on non-linear surfaces
gap = 0.05 #Gap between surfaces
factor = 6
nmc = 1000 #Number of Monte-Carlo simulation for dynamic behaviour of the case
la = 0
al=0.0027 #Error type I
be=0.00005 #Error type II

Q = 100  # Number of Demands
Number_of_Components = 3
Partitions = 1  # Number of Partitions
Rework = 1
start_time = time.time()
# In[] Genetic algorithm

class geneticalgorithm():

    #############################################################
    def __init__(self, function, dimension, variable_type='bool', \
                 variable_boundaries=None,\
                 variable_type_mixed=None, \
                 function_timeout=100,\
                 algorithm_parameters={'max_num_iteration': 100,\
                                       'population_size':100,\
                                       'mutation_probability':0.04,\
                                       'elit_ratio': 0.01,\
                                       'crossover_probability': 0.7,\
                                       'parents_portion': 0.2,\
                                       'crossover_type':'uniform',\
                                       'max_iteration_without_improv':None},\
                     convergence_curve=True,\
                         progress_bar=True):

        self.__name__= geneticalgorithm
        #############################################################
        # input function
        assert (callable(function)),"function must be callable"     
        
        self.f=function
        #############################################################
        #dimension
        
        self.dim=int(dimension)
        
        #############################################################
        # input variable type
        
        assert(variable_type=='bool' or variable_type=='int' or\
               variable_type=='real'), \
               "\n variable_type must be 'bool', 'int', or 'real'"
       #############################################################
        # input variables' type (MIXED)     

        if variable_type_mixed is None:
            
            if variable_type=='real': 
                self.var_type=np.array([['real']]*self.dim)
            else:
                self.var_type=np.array([['int']]*self.dim)            

 
        else:
            assert (type(variable_type_mixed).__module__=='numpy'),\
            "\n variable_type must be numpy array"  
            assert (len(variable_type_mixed) == self.dim), \
            "\n variable_type must have a length equal dimension."       

            for i in variable_type_mixed:
                assert (i=='real' or i=='int'),\
                "\n variable_type_mixed is either 'int' or 'real' "+\
                "ex:['int','real','real']"+\
                "\n for 'boolean' use 'int' and specify boundary as [0,1]"
                

            self.var_type=variable_type_mixed
        #############################################################
        # input variables' boundaries 

            
        if variable_type!='bool' or type(variable_type_mixed).__module__=='numpy':
                       
            assert (type(variable_boundaries).__module__=='numpy'),\
            "\n variable_boundaries must be numpy array"
        
            assert (len(variable_boundaries)==self.dim),\
            "\n variable_boundaries must have a length equal dimension"        
        
        
            for i in variable_boundaries:
                assert (len(i) == 2), \
                "\n boundary for each variable must be a tuple of length two." 
                assert(i[0]<=i[1]),\
                "\n lower_boundaries must be smaller than upper_boundaries [lower,upper]"
            self.var_bound=variable_boundaries
        else:
            self.var_bound=np.array([[0,1]]*self.dim)
 
        ############################################################# 
        #Timeout
        self.funtimeout=float(function_timeout)
        ############################################################# 
        #convergence_curve
        if convergence_curve==True:
            self.convergence_curve=True
        else:
            self.convergence_curve=False
        ############################################################# 
        #progress_bar
        if progress_bar==True:
            self.progress_bar=True
        else:
            self.progress_bar=False
        ############################################################# 
        ############################################################# 
        # input algorithm's parameters
        
        self.param=algorithm_parameters
        
        self.pop_s=int(self.param['population_size'])
        
        assert (self.param['parents_portion']<=1\
                and self.param['parents_portion']>=0),\
        "parents_portion must be in range [0,1]" 
        
        self.par_s=int(self.param['parents_portion']*self.pop_s)
        trl=self.pop_s-self.par_s
        if trl % 2 != 0:
            self.par_s+=1
               
        self.prob_mut=self.param['mutation_probability']
        
        assert (self.prob_mut<=1 and self.prob_mut>=0), \
        "mutation_probability must be in range [0,1]"
        
        
        self.prob_cross=self.param['crossover_probability']
        assert (self.prob_cross<=1 and self.prob_cross>=0), \
        "mutation_probability must be in range [0,1]"
        
        assert (self.param['elit_ratio']<=1 and self.param['elit_ratio']>=0),\
        "elit_ratio must be in range [0,1]"                
        
        trl=self.pop_s*self.param['elit_ratio']
        if trl<1 and self.param['elit_ratio']>0:
            self.num_elit=1
        else:
            self.num_elit=int(trl)
            
        assert(self.par_s>=self.num_elit), \
        "\n number of parents must be greater than number of elits"
        
        if self.param['max_num_iteration']==None:
            self.iterate=0
            for i in range (0,self.dim):
                if self.var_type[i]=='int':
                    self.iterate+=(self.var_bound[i][1]-self.var_bound[i][0])*self.dim*(100/self.pop_s)
                else:
                    self.iterate+=(self.var_bound[i][1]-self.var_bound[i][0])*50*(100/self.pop_s)
            self.iterate=int(self.iterate)
            if (self.iterate*self.pop_s)>10000000:
                self.iterate=10000000/self.pop_s
        else:
            self.iterate=int(self.param['max_num_iteration'])
        
        self.c_type=self.param['crossover_type']
        assert (self.c_type=='uniform' or self.c_type=='one_point' or\
                self.c_type=='two_point'),\
        "\n crossover_type must 'uniform', 'one_point', or 'two_point' Enter string" 
        
        
        self.stop_mniwi=False
        if self.param['max_iteration_without_improv']==None:
            self.mniwi=self.iterate+1
        else: 
            self.mniwi=int(self.param['max_iteration_without_improv'])

        
        ############################################################# 
    def run(self):
        
        
        ############################################################# 
        # Initial Population
        
        self.integers=np.where(self.var_type=='int')
        self.reals=np.where(self.var_type=='real')
        
        
        
        pop=np.array([np.zeros(self.dim+1)]*self.pop_s)
        solo=np.zeros(self.dim+1)
        var=np.zeros(self.dim)       
        
        for p in range(0,self.pop_s):
         
            for i in self.integers[0]:
                var[i]=np.random.randint(self.var_bound[i][0],\
                        self.var_bound[i][1]+1)  
                solo[i]=var[i].copy()
            for i in self.reals[0]:
                var[i]=self.var_bound[i][0]+np.random.random()*\
                (self.var_bound[i][1]-self.var_bound[i][0])    
                solo[i]=var[i].copy()


            obj=self.sim(var)            
            solo[self.dim]=obj
            pop[p]=solo.copy()

        #############################################################

        #############################################################
        # Report
        self.report=[]
        self.report2=[]
        self.test_obj=obj
        self.best_variable=var.copy()
        self.best_function=obj
        ##############################################################   
                        
        t=1
        counter=0
        while t<=self.iterate:
            
            if self.progress_bar==True:
                self.progress(t,self.iterate,status="GA is running...")
            #############################################################
            #Sort
            pop = pop[pop[:,self.dim].argsort()]

                
            
            if pop[0,self.dim]<self.best_function:
                counter=0
                self.best_function=pop[0,self.dim].copy()
                self.best_variable=pop[0,: self.dim].copy()
            else:
                counter+=1
                
 
            #############################################################
            # Report
            self.report.append(pop[0,self.dim])
            self.report2.append(pop[0,: self.dim])
            ##############################################################         
            # Normalizing objective function 
            
            normobj=np.zeros(self.pop_s)
            
            minobj=pop[0,self.dim]
            if minobj<0:
                normobj=pop[:,self.dim]+abs(minobj)
                
            else:
                normobj=pop[:,self.dim].copy()
    
            maxnorm=np.amax(normobj)
            normobj=maxnorm-normobj+1

            #############################################################        
            # Calculate probability
            
            sum_normobj=np.sum(normobj)
            prob=np.zeros(self.pop_s)
            prob=normobj/sum_normobj
            cumprob=np.cumsum(prob)
  
            #############################################################        
            # Select parents
            par=np.array([np.zeros(self.dim+1)]*self.par_s)
            
            for k in range(0,self.num_elit):
                par[k]=pop[k].copy()
            for k in range(self.num_elit,self.par_s):
                index=np.searchsorted(cumprob,np.random.random())
                par[k]=pop[index].copy()
                
            ef_par_list=np.array([False]*self.par_s)
            par_count=0
            while par_count==0:
                for k in range(0,self.par_s):
                    if np.random.random()<=self.prob_cross:
                        ef_par_list[k]=True
                        par_count+=1
                 
            ef_par=par[ef_par_list].copy()
    
            #############################################################  
            #New generation
            pop=np.array([np.zeros(self.dim+1)]*self.pop_s)
            
            for k in range(0,self.par_s):
                pop[k]=par[k].copy()
                
            for k in range(self.par_s, self.pop_s, 2):
                r1=np.random.randint(0,par_count)
                r2=np.random.randint(0,par_count)
                pvar1=ef_par[r1,: self.dim].copy()
                pvar2=ef_par[r2,: self.dim].copy()
                
                ch=self.cross(pvar1,pvar2,self.c_type)
                ch1=ch[0].copy()
                ch2=ch[1].copy()
                
                ch1=self.mut(ch1)
                ch2=self.mutmidle(ch2,pvar1,pvar2)               
                solo[: self.dim]=ch1.copy()                
                obj=self.sim(ch1)
                solo[self.dim]=obj
                pop[k]=solo.copy()                
                solo[: self.dim]=ch2.copy()                
                obj=self.sim(ch2)               
                solo[self.dim]=obj
                pop[k+1]=solo.copy()
 
        #############################################################       
            t+=1
            if counter > self.mniwi:
                pop = pop[pop[:,self.dim].argsort()]
                if pop[0,self.dim]>=self.best_function:
                    t=self.iterate
                    if self.progress_bar==True:
                        self.progress(t,self.iterate,status="GA is running...")
                    time.sleep(2)
                    t+=1
                    self.stop_mniwi=True
         
   
        #############################################################
        #Sort
        pop = pop[pop[:,self.dim].argsort()]
        
 
        
        if pop[0,self.dim]<self.best_function:
                
            self.best_function=pop[0,self.dim].copy()
            self.best_variable=pop[0,: self.dim].copy()
            

            
 
        #############################################################
        # Report
        self.report.append(pop[0,self.dim])
        self.report2.append(pop[0,: self.dim])

        
        D = open("Pareto_variables.dat", "wb")
        E = open("Pareto_variables_cost.dat", "wb")
        pickle.dump(self.report , E) 
        pickle.dump(self.report2 , D) 
 
        
        self.output_dict={'variable': self.best_variable, 'function':\
                          self.best_function}
        if self.progress_bar==True:
            show=' '*100
            sys.stdout.write('\r%s' % (show))
#        sys.stdout.write('\r The best solution found:\n %s' % (self.best_variable))
        sys.stdout.write('\n\n Production cost:\n %0.2f\n ' % (self.best_function))
        
        Solutions = np.append( self.best_variable, self.best_function)
        F = open("results.dat", "wb")
        pickle.dump(Solutions , F)

        
        def Body_conformity_rate (process_deviation,tolerance_2_b,tolerance_2_c):
            sigma = (process_deviation/factor)**2
            tolerance_2_b =   tolerance_2_b 
            tolerance_2_c =   tolerance_2_c
        
            b_coefficient = np.abs((L[10]-L[0]))*sigma
            c_coefficient = np.abs((L[8]-L[0]))*sigma
            cov = [[b_coefficient,0],[0,c_coefficient]]
            body_conformity_rate = (multivariate_normal.cdf([ tolerance_2_b,tolerance_2_c ], [0,0] , cov) - multivariate_normal.cdf([ -tolerance_2_b,-tolerance_2_c ] , [0,0] , cov ))
            body_conformity_rate = body_conformity_rate*(1-al)+(1-body_conformity_rate)*be
            Component_Rework_ = (1-body_conformity_rate)*Rework/2
            body_conformity_rate = body_conformity_rate + body_conformity_rate*Component_Rework_
            return body_conformity_rate
        
        def Shaft_conformity_rate (process_deviation, tolerance_3_c,tolerance_3_d,tolerance_3_e ):
            sigma = (process_deviation/factor)**2
        
            tolerance_3_c = tolerance_3_c
            tolerance_3_d = tolerance_3_d
            tolerance_3_e = tolerance_3_e
        
        
            c_coefficient = np.abs((L[8]-L[0]))*sigma
            d_coefficient = np.abs((L[2]))*sigma
            e_coefficient = np.abs((L[4]-L[3]))*sigma
           
            cov = [[c_coefficient,0,0], [0,d_coefficient,0],[0,0,e_coefficient]] + np.full((3, 3), 1e-10)
            shaft_conformity_rate = (multivariate_normal.cdf([tolerance_3_c,tolerance_3_d,tolerance_3_e], [0,0,0] , cov) - multivariate_normal.cdf([-tolerance_3_c,-tolerance_3_d,-tolerance_3_e] , [0,0,0] , cov ))
            shaft_conformity_rate = shaft_conformity_rate*(1-al)+(1-shaft_conformity_rate)*be
            Component_Rework_ = (1-shaft_conformity_rate)*Rework/2
            shaft_conformity_rate = shaft_conformity_rate + shaft_conformity_rate*Component_Rework_
            return shaft_conformity_rate
    
    
        def Housing_conformity_rate (process_deviation,tolerance_1_b,tolerance_1_d,tolerance_1_e ):
            sigma = (process_deviation/factor)**2
        
            tolerance_1_b = tolerance_1_b
            tolerance_1_d = tolerance_1_d
            tolerance_1_e = tolerance_1_e
            
            b_coefficient = np.abs((L[10]-L[0]))*sigma
            d_coefficient = np.abs((L[2]))*sigma
            e_coefficient = np.abs((L[3]-L[4]))*sigma
        
            cov = [[b_coefficient,0,0],[0,d_coefficient,0],[0,0,e_coefficient]] + np.full((3, 3), 1e-10)
            housing_conformity_rate = (multivariate_normal.cdf([tolerance_1_b,tolerance_1_d,tolerance_1_e], [0,0,0] , cov) - multivariate_normal.cdf([-tolerance_1_b,-tolerance_1_d,-tolerance_1_e] , [0,0,0] , cov ))
            housing_conformity_rate = housing_conformity_rate*(1-al)+(1-housing_conformity_rate)*be
            Component_Rework_ = (1-housing_conformity_rate)*Rework/2
            housing_conformity_rate = housing_conformity_rate + housing_conformity_rate*Component_Rework_
            return housing_conformity_rate
    
        print ("Housing conformity rate =%", "%.2f" %(Housing_conformity_rate (s[int(Solutions[0])],Solutions[3],Solutions[4],Solutions[5]) *100))      
        print ("Body conformity rate = %", "%.2f" %(Body_conformity_rate (s[int(Solutions[1])],Solutions[6],Solutions[7]) *100))    
        print ("Shaft conformity rate = %", "%.2f" %(Shaft_conformity_rate (s[int(Solutions[2])],Solutions[8],Solutions[9],Solutions[8]) *100))      
        print ("Housing allocated resource = M1", "%s" %(int(Solutions[0])+1))      
        print ("Body allocated resource = M2",  "%.0f" %(int(Solutions[1])-2))    
        print ("Shaft allocated resource = M3", "%.0f" %(int(Solutions[2])-5))
        print ("Housing allocated tolerances F11 = %.3f  F12 = %.3f F13 = %.3f" %(Solutions[3],Solutions[4],Solutions[5]))      
        print ("Body allocated tolerances F21 = %.3f  F22 = %.3f" %(Solutions[6],Solutions[7]))    
        print ("Shaft allocated tolerances F31 = %.3f  F32 = %.3f F33 = %.3f" %(Solutions[8],Solutions[9],Solutions[8]))           
        
        Prob = np.zeros(3)
        Prob_list = []
        for i in range (len(self.report2)):
            Prob = [Housing_conformity_rate (s[int(self.report2[i][0])],self.report2[i][3],self.report2[i][4],self.report2[i][5]),\
                Body_conformity_rate (s[int(self.report2[i][1])],self.report2[i][6],self.report2[i][7]),\
                Shaft_conformity_rate (s[int(self.report2[i][2])],self.report2[i][8],self.report2[i][9],self.report2[i][8])]
            Prob_list.append(Prob)

        PROB = open("Probabilities.dat", "wb")        
        pickle.dump(Prob_list , PROB) 
         
        def Assembly_functionality ():
            sigma_1 = s[int(Solutions[0])]/factor
            sigma_2 = s[int(Solutions[1])]/factor
            sigma_3 = s[int(Solutions[2])]/factor
            GI = []
                    
            #     1b   2b    2c    3c   1d  3d    1e  3e  
            mD = [65, 64.98, 30, 29.95, 50, 49.9, 30, 29.95]
            gap_1b_2b = (np.abs(mD[1]-mD[0]) + Solutions[3] + Solutions[6])/2 
            gap_3c_2c = (np.abs(mD[3]-mD[2]) + Solutions[7] + Solutions[8] )/2 
            gap_1e_3e = (np.abs(mD[7]-mD[6]) + Solutions[5] + Solutions[8])/2 
            gap_1d_3d = (np.abs(mD[5]-mD[4]) + Solutions[4] + Solutions[9] )/2
            
            #al 2b1b	be 2b1b	ga 2b1b	u 2b1b	v 2b1b	w 2b1b	al 3c2c	be 3c2c	ga 3c2c	u 3c2c	v 3c2c	w 3c2c	al 3e1e	be 3e1e	ga 3e1e	u 3e1e	v 3e1e	w 3e1e	u 3d1d	v 3d1d#    
            for k in range(Nlinear) :  
                t = (2*(k+1)*m.pi)/Nlinear
                GI.append([ -L[8]*m.cos(t), -L[8]*m.sin(t), 0., -m.sin(t), m.cos(t), 0., 0, 0, 0., 0, 0,
                               0., 0., 0., 0., 0., 0., 0., 0., 0.])    
                GI.append([ -L[0]*m.cos(t), -L[0]*m.sin(t), 0., -m.sin(t), m.cos(t), 0., 0, 0, 0., 0, 0,
                               0., 0., 0., 0., 0., 0., 0., 0., 0.]) 
                GI.append([ 0., 0., 0., 0., 0., 0., -L[3]*m.cos(t), -L[3]*m.sin(t), 0., -m.sin(t),
                               m.cos(t), 0., 0, 0, 0., 0, 0, 0., 0., 0.])    
                GI.append([ 0., 0., 0., 0., 0., 0., -L[4]*m.cos(t), -L[4]*m.sin(t), 0., -m.sin(t),
                               m.sin(t), 0., 0, 0, 0., 0, 0, 0., 0., 0.])   
                GI.append([ 0, 0, 0., 0, 0, 0., 0., 0., 0., 0.,
                               0., 0., -L[10]*m.cos(t), -L[10]*m.sin(t), 0., -m.sin(t), m.cos(t), 0., 0., 0.])     
                GI.append([ 0, 0, 0., 0, 0, 0., 0., 0., 0., 0.,
                               0., 0., -L[8]*m.cos(t), -L[8]*m.sin(t), 0., -m.sin(t), m.cos(t), 0., 0., 0.])
                
            
            GI.append([ 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1, 0.])    
            GI.append([ 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1])
            
            b1_b2_coefficient = np.sqrt(sum((GI[i][j])**2 for i in range(Nlinear*6+2) for j in range(6)))/36
            c2_c3_coefficient = np.sqrt(sum((GI[i][j])**2 for i in range(Nlinear*6+2) for j in range(6,12,1)))/36
            e1_e3_coefficient = np.sqrt(sum((GI[i][j])**2 for i in range(Nlinear*6+2) for j in range(12,18,1)))/36
            d1_d3_coefficient = np.sqrt(sum((GI[i][j])**2 for i in range(Nlinear*6+2) for j in range(18,20,1)))/36
            
            cov = [[b1_b2_coefficient*(sigma_1*sigma_2), 0, 0, 0], [0, c2_c3_coefficient*(sigma_2*sigma_3),0,0], [0, 0,e1_e3_coefficient*(sigma_1*sigma_3),0], [0, 0,0, d1_d3_coefficient*(sigma_1*sigma_3)]]
            assembly_functionality = (multivariate_normal.cdf([gap_1b_2b,gap_3c_2c,gap_1e_3e,gap_1d_3d], [0.0001,0.0001,0.0001,gap] , cov) - multivariate_normal.cdf([-gap_1b_2b,-gap_3c_2c,-gap_1e_3e,-gap_1d_3d] , [0.0001,0.0001,0.0001,gap] , cov ))
            assembly_functionality = assembly_functionality*(1-al)+(1-assembly_functionality)*be

            return  assembly_functionality
        pickle.dump(Assembly_functionality () , F)
        print ("Assembly conformity rate =%", "%.2f" %(Assembly_functionality ()*100))       
        
        sys.stdout.flush() 
        re=np.array(self.report)
        if self.convergence_curve==True:
            plt.plot(re)
            plt.xlabel('Iteration')
            plt.ylabel('Production cost')
            plt.title('Genetic Algorithm')
            plt.show()
        
        if self.stop_mniwi==True:
            sys.stdout.write('\nWarning: GA is terminated due to the'+\
                             ' maximum number of iterations without improvement was met!')
##############################################################################         
##############################################################################         
    def cross(self,x,y,c_type):
         
        ofs1=x.copy()
        ofs2=y.copy()
        

        if c_type=='one_point':
            ran=np.random.randint(0,self.dim)
            for i in range(0,ran):
                ofs1[i]=y[i].copy()
                ofs2[i]=x[i].copy()
  
        if c_type=='two_point':
                
            ran1=np.random.randint(0,self.dim)
            ran2=np.random.randint(ran1,self.dim)
                
            for i in range(ran1,ran2):
                ofs1[i]=y[i].copy()
                ofs2[i]=x[i].copy()
            
        if c_type=='uniform':
                
            for i in range(0, self.dim):
                ran=np.random.random()
                if ran <0.5:
                    ofs1[i]=y[i].copy()
                    ofs2[i]=x[i].copy() 
                   
        return np.array([ofs1,ofs2])
###############################################################################  
    
    def mut(self,x):
        
        for i in self.integers[0]:
            ran=np.random.random()
            if ran < self.prob_mut:
                
                x[i]=np.random.randint(self.var_bound[i][0],\
                 self.var_bound[i][1]+1) 
                    
        

        for i in self.reals[0]:                
            ran=np.random.random()
            if ran < self.prob_mut:   

               x[i]=self.var_bound[i][0]+np.random.random()*\
                (self.var_bound[i][1]-self.var_bound[i][0])    
            
        return x
###############################################################################
    def mutmidle(self, x, p1, p2):
        for i in self.integers[0]:
            ran=np.random.random()
            if ran < self.prob_mut:
                if p1[i]<p2[i]:
                    x[i]=np.random.randint(p1[i],p2[i])
                elif p1[i]>p2[i]:
                    x[i]=np.random.randint(p2[i],p1[i])
                else:
                    x[i]=np.random.randint(self.var_bound[i][0],\
                 self.var_bound[i][1]+1)
                        
        for i in self.reals[0]:                
            ran=np.random.random()
            if ran < self.prob_mut:   
                if p1[i]<p2[i]:
                    x[i]=p1[i]+np.random.random()*(p2[i]-p1[i])  
                elif p1[i]>p2[i]:
                    x[i]=p2[i]+np.random.random()*(p1[i]-p2[i])
                else:
                    x[i]=self.var_bound[i][0]+np.random.random()*\
                (self.var_bound[i][1]-self.var_bound[i][0]) 
        return x
###############################################################################     
    def evaluate(self):
        return self.f(self.temp)
###############################################################################    
    def sim(self,X):
        self.temp=X.copy()
        obj=None
        try:
            obj=func_timeout(self.funtimeout,self.evaluate)
        except FunctionTimedOut:
            print("given function is not applicable")
        assert (obj!=None), "After "+str(self.funtimeout)+" seconds delay "+\
                "func_timeout: the given function does not provide any output"
        return obj

###############################################################################
    def progress(self, count, total, status=''):
        bar_len = 50
        filled_len = int(round(bar_len * count / float(total)))

        percents = round(100.0 * count / float(total), 1)
        bar = '|' * filled_len + '_' * (bar_len - filled_len)

        sys.stdout.write('\r%s %s%s %s' % (bar, percents, '%', status))
        sys.stdout.flush()     
###############################################################################            
###############################################################################
            
def f(X):
    
    Component_Operation_ = np.zeros(Number_of_Components)
    for i in range (Number_of_Components):
        Component_Operation_[i]=int(X[i]) 
    
    
    def Body_conformity_rate (process_deviation,tolerance_2_b,tolerance_2_c):
        sigma = (process_deviation/factor)**2
        tolerance_2_b =   tolerance_2_b 
        tolerance_2_c =   tolerance_2_c
    
        b_coefficient = np.abs((L[10]-L[0]))*sigma
        c_coefficient = np.abs((L[8]-L[0]))*sigma
        cov = [[b_coefficient,0],[0,c_coefficient]]
        body_conformity_rate = (multivariate_normal.cdf([ tolerance_2_b,tolerance_2_c ], [0,0] , cov) - multivariate_normal.cdf([ -tolerance_2_b,-tolerance_2_c ] , [0,0] , cov ))
        body_conformity_rate = body_conformity_rate*(1-al)+(1-body_conformity_rate)*be

        return body_conformity_rate, tolerance_2_b,tolerance_2_c, process_deviation
    
    def Shaft_conformity_rate (process_deviation, tolerance_3_c,tolerance_3_d,tolerance_3_e ):
        sigma = (process_deviation/factor)**2
    
        tolerance_3_c = tolerance_3_c
        tolerance_3_d = tolerance_3_d
        tolerance_3_e = tolerance_3_e
    
    
        c_coefficient = np.abs((L[8]-L[0]))*sigma
        d_coefficient = np.abs((L[2]))*sigma
        e_coefficient = np.abs((L[4]-L[3]))*sigma
       
        cov = [[c_coefficient,0,0], [0,d_coefficient,0],[0,0,e_coefficient]] + np.full((3, 3), 1e-10)
        shaft_conformity_rate = (multivariate_normal.cdf([tolerance_3_c,tolerance_3_d,tolerance_3_e], [0,0,0] , cov) - multivariate_normal.cdf([-tolerance_3_c,-tolerance_3_d,-tolerance_3_e] , [0,0,0] , cov ))
        shaft_conformity_rate = shaft_conformity_rate*(1-al)+(1-shaft_conformity_rate)*be
        return shaft_conformity_rate, tolerance_3_c,tolerance_3_d,tolerance_3_e, process_deviation
    
    
    def Housing_conformity_rate (process_deviation,tolerance_1_b,tolerance_1_d,tolerance_1_e ):
        sigma = (process_deviation/factor)**2
    
        tolerance_1_b = tolerance_1_b
        tolerance_1_d = tolerance_1_d
        tolerance_1_e = tolerance_1_e
        
        b_coefficient = np.abs((L[10]-L[0]))*sigma
        d_coefficient = np.abs((L[2]))*sigma
        e_coefficient = np.abs((L[3]-L[4]))*sigma
    
        cov = [[b_coefficient,0,0],[0,d_coefficient,0],[0,0,e_coefficient]] + np.full((3, 3), 1e-10)
        housing_conformity_rate = (multivariate_normal.cdf([tolerance_1_b,tolerance_1_d,tolerance_1_e], [0,0,0] , cov) - multivariate_normal.cdf([-tolerance_1_b,-tolerance_1_d,-tolerance_1_e] , [0,0,0] , cov ))
        housing_conformity_rate = housing_conformity_rate*(1-al)+(1-housing_conformity_rate)*be
        return housing_conformity_rate,tolerance_1_b,tolerance_1_d,tolerance_1_e, process_deviation
    
    
    housing_conformity_rate,t_1_b,t_1_d,t_1_e,process_deviation_1 = Housing_conformity_rate (s[int(X[0])],X[3],X[4],X[5])
    body_conformity_rate, t_2_b,t_2_c,process_deviation_2 = Body_conformity_rate (s[int(X[1])],X[6],X[7]) 
    shaft_conformity_rate, t_3_c,t_3_d,t_3_e,process_deviation_3  = Shaft_conformity_rate (s[int(X[2])],X[8],X[9],X[8])
    
    def Assembly_functionality ():
        sigma_1 = s[int(X[0])]/factor
        sigma_2 = s[int(X[1])]/factor
        sigma_3 = s[int(X[2])]/factor
        GI = []

    
        #     1b   2b    2c    3c   1d  3d    1e  3e  
        mD = [65, 64.98, 30, 29.95, 50, 49.9, 30, 29.95]
        gap_1b_2b = (np.abs(mD[1]-mD[0]) + X[3] + X[6])/2 
        gap_3c_2c = (np.abs(mD[3]-mD[2]) + X[7] + X[8] )/2 
        gap_1e_3e = (np.abs(mD[7]-mD[6]) + X[5] + X[8])/2 
        gap_1d_3d = (np.abs(mD[5]-mD[4]) + X[4] + X[9] )/2
        
       
        #al 2b1b	be 2b1b	ga 2b1b	u 2b1b	v 2b1b	w 2b1b	al 3c2c	be 3c2c	ga 3c2c	u 3c2c	v 3c2c	w 3c2c	al 3e1e	be 3e1e	ga 3e1e	u 3e1e	v 3e1e	w 3e1e	u 3d1d	v 3d1d#    
        for k in range(Nlinear) :  
            t = (2*(k+1)*m.pi)/Nlinear
            GI.append([ -L[8]*m.cos(t), -L[8]*m.sin(t), 0., -m.sin(t), m.cos(t), 0., 0, 0, 0., 0, 0,
                           0., 0., 0., 0., 0., 0., 0., 0., 0.])    
            GI.append([ -L[0]*m.cos(t), -L[0]*m.sin(t), 0., -m.sin(t), m.cos(t), 0., 0, 0, 0., 0, 0,
                           0., 0., 0., 0., 0., 0., 0., 0., 0.]) 
            GI.append([ 0., 0., 0., 0., 0., 0., -L[3]*m.cos(t), -L[3]*m.sin(t), 0., -m.sin(t),
                           m.cos(t), 0., 0, 0, 0., 0, 0, 0., 0., 0.])    
            GI.append([ 0., 0., 0., 0., 0., 0., -L[4]*m.cos(t), -L[4]*m.sin(t), 0., -m.sin(t),
                           m.sin(t), 0., 0, 0, 0., 0, 0, 0., 0., 0.])   
            GI.append([ 0, 0, 0., 0, 0, 0., 0., 0., 0., 0.,
                           0., 0., -L[10]*m.cos(t), -L[10]*m.sin(t), 0., -m.sin(t), m.cos(t), 0., 0., 0.])     
            GI.append([ 0, 0, 0., 0, 0, 0., 0., 0., 0., 0.,
                           0., 0., -L[8]*m.cos(t), -L[8]*m.sin(t), 0., -m.sin(t), m.cos(t), 0., 0., 0.])
            
        
        GI.append([ 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1, 0.])    
        GI.append([ 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1])
        
        b1_b2_coefficient = np.sqrt(sum((GI[i][j])**2 for i in range(Nlinear*6+2) for j in range(6)))/36
        c2_c3_coefficient = np.sqrt(sum((GI[i][j])**2 for i in range(Nlinear*6+2) for j in range(6,12,1)))/36
        e1_e3_coefficient = np.sqrt(sum((GI[i][j])**2 for i in range(Nlinear*6+2) for j in range(12,18,1)))/36
        d1_d3_coefficient = np.sqrt(sum((GI[i][j])**2 for i in range(Nlinear*6+2) for j in range(18,20,1)))/36
        
        cov = [[b1_b2_coefficient*(sigma_1*sigma_2), 0, 0, 0], [0, c2_c3_coefficient*(sigma_2*sigma_3),0,0], [0, 0,e1_e3_coefficient*(sigma_1*sigma_3),0], [0, 0,0, d1_d3_coefficient*(sigma_1*sigma_3)]]
        assembly_functionality = (multivariate_normal.cdf([gap_1b_2b,gap_3c_2c,gap_1e_3e,gap_1d_3d], [0.0001,gap,gap,gap] , cov) - multivariate_normal.cdf([-gap_1b_2b,-gap_3c_2c,-gap_1e_3e,-gap_1d_3d] , [0.0001,gap,gap,gap] , cov ))
        return  assembly_functionality

    Assembly_Conformity_with_Failure = Assembly_functionality ()

    gp = [body_conformity_rate,shaft_conformity_rate, housing_conformity_rate] # before reworking
    
    Component_Rework_ = np.zeros(Number_of_Components)
    sr = np.zeros(Number_of_Components)
    Conformity_with_reworking = np.zeros(Number_of_Components)
    for i in range (Number_of_Components):
        Component_Rework_[i] = (1-gp[i])*Rework/2
        sr[i]= (1-(gp[0]+Component_Rework_[0]))
        Conformity_with_reworking[i] = gp[i] + gp[i]*Component_Rework_[i]
    
    
    Manufacturing_cost = sum((Components_Manufacturing_cost[int(X[i])])/Conformity_with_reworking[i] for i in range(Number_of_Components))     
    Scrap_cost = sum (Components_Scraping_cost[i]*sr[i]/Conformity_with_reworking[i] for i in range (Number_of_Components))
    Inspection_cost = sum (Components_Inspection_cost[i]/Conformity_with_reworking[i] for i in range (Number_of_Components))
    Reworking_cost = sum (Component_Rework_[i]*Rework*Components_Reworking_cost[i]/Conformity_with_reworking[i] for i in range (Number_of_Components))
    CIP = (Product_Inspection_cost *Q /Assembly_Conformity_with_Failure)        
    CA = (Product_Assembly_cost*Q /(Assembly_Conformity_with_Failure)) 
    CSP = Product_Scraping_cost*Q*(1-Assembly_Conformity_with_Failure)    
    pen = Manufacturing_cost + Scrap_cost + Inspection_cost + Reworking_cost+ CIP + CA + CSP
    
    mD = [65, 64.98, 30, 29.95, 50, 49.9, 30, 29.95]
    
    pen1 = 0
    if ((np.abs(mD[1]-mD[0]) + X[3] + X[6]) > (2)*gap) \
    or ((np.abs(mD[3]-mD[2]) + X[7] + X[8]) > (2)*gap) \
    or ((np.abs(mD[7]-mD[6]) + X[5] + X[8]) > (2)*gap) \
    or ((np.abs(mD[5]-mD[4]) + X[4] + X[9]) > (3)*gap):        
        pen1= 5e50
    
          
    return pen + pen1


varbound=np.array([[0,2],[3,5],[6,8],[0.001,0.1],[0.001,0.1],[0.001,0.1],[0.001,0.1],[0.001,0.1],[0.001,0.1],[0.001,0.1]])

vartype=np.array([['int'],['int'],['int'],['real'],['real'],['real'],['real'],['real'],['real'],['real']])

model=geneticalgorithm(function=f,dimension=10,variable_type_mixed=vartype,variable_boundaries=varbound)

model.run()

print("--- Optimization RunTime = %s seconds ---" % int(round((time.time() - start_time))))


######################################################################## plot Body deviation distribution
#Parameters to set
sigma = (results[1]/factor)**2
mu_x = 0
variance_x =  np.abs((L[10]-L[0]))*sigma

mu_y = 0
variance_y = np.abs((L[8]-L[0]))*sigma

#Create grid and multivariate normal
x = np.linspace(-0.1,0.1,1000)
y = np.linspace(-0.1,0.1,1000)
X, Y = np.meshgrid(x,y)
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X; pos[:, :, 1] = Y
rv = multivariate_normal([mu_x, mu_y], [[variance_x, 0], [0, variance_y]])

#Make a 3D plot
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, rv.pdf(pos),cmap='viridis',linewidth=0)
ax.set_xlabel('Deviation D22 axis')
ax.set_ylabel('Deviation D21 axis')
ax.set_zlabel('Body conformity axis')


# Plot 2D contour
pos = np.dstack((X, Y))
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
plt.axhline(y=results[7],linewidth = 1, linestyle ="--",color ='white',label="Tolerances on the Body")
plt.axhline(y=-results[7],linewidth = 1, linestyle ="--",color ='white')

plt.axvline(x=results[6],linewidth = 1, linestyle ="--",color ='white')
plt.axvline(x=-results[6],linewidth = 1, linestyle ="--",color ='white')

plt.legend(loc="upper right")

cmap=ax2.contourf(X, Y, rv.pdf(pos))
fig2.colorbar(cmap)


ax2.set_xlabel('Deviation D22 axis')
ax2.set_ylabel('Deviation D21 axis')

plt.show()


#################################################################################
from lpsolve55 import *
import numpy as np

import openturns as ot



def create_lp(dimension, nb_Cc, nb_Ci, nb_Cieq):
    
    # compute the total number of constraints
    nb_eq = nb_Cc + nb_Ci + nb_Cieq
    # create the lp object
    lp = lpsolve('make_lp', nb_eq, dimension)
    # set the verbose so as to return nothing
    lpsolve('set_verbose', lp, NEUTRAL)
    # tell that this is minimization problem
    lpsolve('set_maxim', lp)
    # set the type of the constraints in a vector
    e = []
    e.extend([0]*nb_Cc)
    e.extend([-1]*nb_Ci)
    
    for i in range(nb_eq): 
        if e[i] < 0: 
              con_type = 'LE'
        elif e[i] == 0: 
               con_type = 'EQ' 
        else: 
               con_type = 'GE' 
        lpsolve('set_constr_type', lp, i + 1, con_type)  
    
    # set large bounds for the gaps
    lpsolve('set_bounds', lp, [-100]*dimension, [100]*dimension)

    return lp

def solve_lp(f, a, b, criterion):
   
    lpsolve('set_mat', lp, a)
    lpsolve('set_rh_vec', lp, b)
    lpsolve('set_obj_fn', lp, f)
    result = lpsolve('solve', lp)
    
    # optimal solution found
    if result == 0:
        [obj, x, duals, ret] = lpsolve('get_solution', lp)
        
        # compute the functional condition
#        print(x)
        
        fc = criterion -obj
        status = 0        

    # infeasible problem
    else:
        fc = np.Infinity
        status = 1

    return fc, status

def compute_mod(X, Nlinear,j):
    """
    Compute the compatibility equations and interface constraints
    for a given vector of realizations and number of linearizations.
    
    """
    d = X[:8]
    E = X[8:]
    
    T1a1 = [0, 0, 0, 0, 0, 0]
    T1b1 = [E[0], E[1],   0, 0, 0, 0]
    T1d1 = [E[2], E[3],   0, E[16], E[17], 0]
    T1e1 = [E[4], E[5],   0, E[18], E[19], 0]
    T2a2 = [0, 0, 0, 0, 0, 0]
    T2b2 = [E[6], E[7], 0, 0, 0, 0]
    T2c2 = [E[8], E[9], 0, E[20], E[21], 0]
    T3c3 = [E[10], E[11], 0, E[22], E[23], 0]
    T3d3 = [E[12], E[13], 0, E[24], E[25], 0]
    T3e3 = [E[14], E[15], 0, E[26], E[27], 0]
    
    L=[-80, -120, 60, 80, 100, 140, 0, 30, -20, 85, -50]
    
#    """Gaps Variables : C1a2a, u1a2a, v1a2a, A2b1b, B2b1b, C2b1b, u2b1b, v2b1b, w2b1b, A3c2c, B3c2c, C3c2c,
#    u3c2c, v3c2c, w3c2c, A3e1e, B3e1e, C3e1e, u3e1e, v3e1e, w3e1e, u3d1d, v3d1d """
#        
    G = []; D = []
    
    # Boucle 1/2
    # Matrice rassemblant les coefficients multipliant les valeurs des torseurs jeux dans les équations de compatibilité
    G.append([ 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    G.append([ 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    G.append([ 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    G.append([ 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    G.append([ 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    G.append([ 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    
    
    G.append([ 0., 0., 0., -1., 0., 0., 0., 0., 0., -1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.])
    G.append([ 0., 0., 0., 0., -1., 0., 0., 0., 0., 0., -1., 0., 0., 0., 0., 0., 1., 0., 0., 0.])
    G.append([ 0., 0., 0., 0., 0., -1., 0., 0., 0., 0., 0., -1., 0., 0., 0., 0., 0., 1., 0., 0.])
    G.append([ -1., 0., 0., 0., 0., 0., -1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.])
    G.append([ 0., -1., 0., 0., 0., 0., 0., -1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.])
    G.append([ 0., 0., -1., 0., 0., 0., 0., 0., -1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.])
    
    
    G.append([0., 0., 0., 0., 0., 0., 0., -L[7], 0., -1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.])
    G.append([0., 0., 0., 0., 0., 0., L[7], 0., -L[6], 0., -1., 0., 0., 0., 0., 0., 0., 0., 0., 1.])

   # Le vecteur contenant les valeurs des torseurs écarts ligne par ligne des équations de compatibilité

    D = [-(-T2b2[3]+T1b1[3]), -T1b1[4],  -(-T1a1[5] + T2a2[5]), -(-T1a1[0] + T2a2[0] + T1b1[0]), \
    -(-T1a1[1] +T2a2[1] - T2b2[1] + T1b1[1]),-(-T2b2[2]),\

    
    -(-T1b1[3] + T2b2[3] - T2c2[3] + T3c3[3] - T3e3[3] + T1e1[3]),
      -(-T1b1[4] + T2b2[4] -T2c2[4] + T3c3[4] - T3e3[4] + T1e1[4]),\
      0., -(-T1b1[0] + T2b2[0] -T2c2[0] + T3c3[0] - T3e3[0] + T1e1[0]), \
            -(-T1b1[1] + T2b2[1] - T2c2[1] + T3c3[1] - T3e3[1] + T1e1[1]),\
      0, 
      
      
    -(-L[7]*T1a1[1] + L[7]*T2a2[1] -T2c2[3] - L[7]*T2c2[1] + T3c3[3] + L[7]*T3c3[1] - T3d3[3] + T1d1[3]),\
    -(L[7]*T1a1[0] -L[7]*T2a2[0] - T2c2[4] + L[7]*T2c2[0] + T3c3[4] + L[7]*T3c3[0] - T3d3[4] + T1d1[4])]


    """ Les écarts de diamètres et leur tolérance à considérer dans les équations """ 
    DI1 = (d[2]-d[3])/2 # liaison 3c/2c
    DI2 = (d[6]-d[7])/2 # liaison 3e/1e 
    DI3 = (d[0]-d[1])/2 # liaison 2b/1b
    
    """ GI estla matrice contenant les coefficients multipliant les paramètres des torseurs au niveau
    des équations d'interface """
    GI = []
    for k in range(Nlinear) :  
        t = (2*(k+1)*m.pi)/Nlinear
        GI.append([ 0., 0., 0., 0., 0., 0., -L[8]*m.sin(t), L[8]*m.cos(t), 0., m.cos(t), m.sin(t),
                   0., 0., 0., 0., 0., 0., 0., 0., 0.])    
        GI.append([ 0., 0., 0., 0., 0., 0., -L[0]*m.sin(t), L[0]*m.cos(t), 0., m.cos(t), m.sin(t),
                    0., 0., 0., 0., 0., 0., 0., 0., 0.]) 
        GI.append([ 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                   0., 0., -L[3]*m.sin(t), L[3]*m.cos(t), 0., m.cos(t), m.sin(t), 0., 0., 0.])    
        GI.append([ 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                    0., 0., -L[4]*m.sin(t), L[4]*m.cos(t), 0., m.cos(t), m.sin(t), 0., 0., 0.])     
        GI.append([ -L[10]*m.sin(t), L[10]*m.cos(t), 0., m.cos(t), m.sin(t), 0., 0., 0., 0., 0.,
                   0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])  
        GI.append([ -L[8]*m.sin(t), L[8]*m.cos(t), 0., m.cos(t), m.sin(t), 0., 0., 0., 0., 0.,
                   0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])                       
                  
    GI.append([ 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.])    
    GI.append([ 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,-1., 0., 0., 0., 0., 0., 0., 0., 0.])    
    GI.append([ 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.])    
    GI.append([ 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,-1., 0., 0.])    
    
    IL = 0.1
    
    DI = [DI1, DI1, DI2, DI2, DI3, DI3] * (Nlinear)
    DI4 = [IL, IL, IL, IL]
    G.extend(GI)
    D.extend(DI)
    D.extend(DI4)

    """ the objective function"""
    if j == 1:
#    f1 represents the vector representing the gaps coefficients
        Objf = [ 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0, 0., 0., 1., 0.] # u
 
    elif j == 2 :
   
        Objf = [ 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0, 0., 0., 0., 1.] # v

    return Objf, G, D

# In[] Electric motor simulation

# Parameters of the model and simulation
ot.RandomGenerator.SetSeed(0)
# number of Monte Carlo sampling

# number of geometrical deviations = random variables
dimX  = 36
# means
mD = np.hstack([np.array([65, 64.98, 30, 29.95, 50, 49.9, 30, 29.95]), np.zeros(28)])

# standard deviations

l = [30, 60, 80, 60] # les longueurs des pièces là où les variables de torsuers sont recherchées


Cp = np.zeros (7)
for i in range (7):
    if  i<3 :
        Cp[i] = results[i+3]/(s[int(results[0])])
    elif i >2 and i<5:
        Cp[i] = results[i+3]/(s[int(results[1])])
    elif i > 4 and i < 7:
        Cp[i] = results[i+3]/(s[int(results[2])])
        
#Cp = 1.66 # l'indice de capabilité

# alésage 1b 
IT1b = results[3]
So1 = IT1b/(6*np.sqrt(2)*l[0]*Cp[0]) # l'écart-type des orientations ou des rotations
Sd1 = IT1b/(6*Cp[0]) # l'écart-type concernant le diamètre 


# alésage 1d
IT1d = results[4]
So2 = IT1d/(3*np.sqrt(2)*l[3]*6*Cp[1]) # l'écart-type des orientations ou des rotations
St2 = (2*IT1d)/(3*np.sqrt(2)*6*Cp[1]) # l'écart-type des translations 
Sd2 = IT1d/(6*Cp[1])  # l'écart-type concernant le diamètre

# alésage 1e
IT1e = results[5]
So3 = IT1e/(l[1]*np.sqrt(2)*3*6*Cp[2]) # l'écart-type des orientations ou des rotations
St3 = (2*IT1e)/(3*np.sqrt(2)*6*Cp[2]) # l'écart-type des translations 
Sd3 = IT1e/(6*Cp[2]) # l'écart-type concernant le diamètre

# alésage 2b
IT2b = results[6]
So4 = IT2b/(6*np.sqrt(2)*l[0]*Cp[3]) # l'écart-type des orientations ou des rotations
Sd4 = IT2b/(6*Cp[3]) # l'écart-type concernant le diamètre

# alésage 2c
IT2c = results[7]
So5 = IT2c/(l[1]*np.sqrt(2)*3*6*Cp[4]) # l'écart-type des orientations ou des rotations
St5 = (2*IT2c)/(3*np.sqrt(2)*6*Cp[4]) # l'écart-type des translations 
Sd5 = IT2c/(6*Cp[4]) # l'écart-type concernant le diamètre

# alésage 3c/3e
IT3c = results[8]
So6 = IT3c/(3*l[2]*np.sqrt(2)*6*Cp[5]) # l'écart-type des orientations ou des rotations
St6 = (2*IT3c)/(3*np.sqrt(2)*6*Cp[5]) # l'écart-type des translations 
Sd6 = IT3c/(6*Cp[5])  # l'écart-type concernant le diamètre


# alésage 3d
IT3d = results[9]
So7 = IT3d/(3*np.sqrt(2)*l[3]*6*Cp[6]) # l'écart-type des orientations ou des rotations
St7 = (2*IT3d)/(3*np.sqrt(2)*6*Cp[6]) # l'écart-type des translations 
Sd7 = IT3d/(6*Cp[6])  # l'écart-type concernant le diamètre



sD =[Sd1, Sd2, Sd3, Sd4, Sd5, Sd6, Sd6, Sd7, 
     
    So1, So1, So2, So2, So3, So3, So4, So4, So5, So5, So6, So6, So6, So6, So7, So7, 
    
    St2, St2, St3, St3, St5, St5, St6, St6, St6, St6, St7, St7]
              
# generate the sample
sample = ot.Normal(mD, sD, ot.CorrelationMatrix(dimX)).getSample(nmc)

# computing loop for the Monte Carlo simulation

# initilization
fc1 = np.zeros(nmc)
fc2 = np.zeros(nmc)
status1 = np.zeros(nmc)
status2 = np.zeros(nmc)

# create lp problem
lp = create_lp(dimension = 20, nb_Cc = 14, nb_Ci =6*Nlinear+4, nb_Cieq=0)

for i, X in enumerate(sample): 

    f1, a, b = compute_mod(X, Nlinear, 1)
    f2, c, d = compute_mod(X, Nlinear, 2)
    
    fc2[i], status2[i] = solve_lp(f2, c, d, gap)
    fc1[i], status1[i] = solve_lp(f1, a, b, gap)

lpsolve('delete_lp', lp)

# compute the probability of failure
pnf1 = (np.sum( fc1< 0 ).astype(float))/nmc
pna1 = (np.sum(status1 == 1).astype(float))/nmc

pnf2 = (np.sum(fc2< 0).astype(float))/nmc
pna2 = (np.sum(status2 == 1).astype(float))/nmc

Assembly_Conformity = 1 - max(pna1, pna2)
Assembly_Conformity_with_Failure = Assembly_Conformity*(1-al)+(1-Assembly_Conformity)*be
print("Assembly_Conformity_with_Failure =%", "%.2f" %(Assembly_Conformity_with_Failure*100) )

Functionality_Conformity = 1 - max(pnf1, pnf2)
Functionality_Conformity_with_Failure = Functionality_Conformity*(1-al)+(1-Functionality_Conformity)*be
print("Functionality_Conformity_with_Failure =%", "%.2f" %(Functionality_Conformity_with_Failure*100) )

### Assebled product cost extention 
CIP = (Product_Inspection_cost *Q /Assembly_Conformity_with_Failure*Functionality_Conformity_with_Failure)
CA = (Product_Assembly_cost*Q /(Assembly_Conformity_with_Failure))
CSP = Product_Scraping_cost*Q*(1-Assembly_Conformity_with_Failure)/(Assembly_Conformity_with_Failure)\
+ Product_Scraping_cost*Q*(Assembly_Conformity_with_Failure*(1-Functionality_Conformity_with_Failure))/(Assembly_Conformity_with_Failure*Functionality_Conformity_with_Failure)

Manufacturing_cost = results[10] + CIP + CA + CSP 
print("OptiSimu roduction cost  = %.2f" %Manufacturing_cost)
print("--- OptiSimu RunTime = %s seconds ---" % int(round((time.time() - start_time))))



   
