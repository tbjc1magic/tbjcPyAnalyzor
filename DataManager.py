import pandas as pd
import numpy as np


    

class RunManager():
    
    
    def __init__(self, datafile, calibration_inputs):
        
        
        self._load_data(datafile)
        self._load_cal_coeff(calibration_inputs)

        
        ### calibrate XN-XF
        self.XN, self.XF = self._calibrate_XNXF(self.XN, self.XF, self.XNXF_cal)
        ### calculate X
        self.X = self._calculateX(self.E, self.XN, self.XF)
        ### calcualte Z
        self.Z = self._calculateZ(self.X, self.z_cal['positions'], self.z_cal['shift'])
        ### rouge callibrate E
        self.E = self._calibrateE(self.E, self.E_cal)
        
        
        ### eventize the data
        self._eventize()
        
        ### calibrate TDC
        self.TDC = self._calibrateTDC(self.Aux['Array RF'], self.X, self.trigger_ch, self.TDC_cal)
        
        ### rouge callibrate RF1, RF2
        self.RF1, self.RF2 = self._calibrateRF(self.TDC, self.RF_cal)
        
        ### Finer calibration of E
        self.fE = self._fineCalibrationE(self.E.copy(), self.Z, self.trigger_ch, self.FinerE_cal)
        
        ### calibrate TAC
        self.TAC = self._calibrateTAC(self.Aux['TAC'], self.TAC_cal, self.trigger_ch)
        
        
        attrs = ['trigger_ch', 'Z', 'fE', 'E', 'X', 'RF1', 'RF2', 'TAC']

        ### convert to dataframe ###
        data = pd.DataFrame({n:getattr(self,n) for n in attrs})

        for attr in attrs: del self.__dict__[attr]
        
        self.data = pd.concat([data, self.Aux[['IC0','IC1','IC2','IC3']]], axis=1)
        
        del self.Aux
    
    def _eventize(self):
        
        del self.XN; del self.XF
        
        trigger_ch = np.argmax(self.E.values, axis=1)
        self.trigger_ch = trigger_ch
        events_num = range(self.X.shape[0])
        self.X = self.X.values[events_num, trigger_ch]
        self.Z = self.Z.values[events_num, trigger_ch]
        self.E = self.E.values[events_num, trigger_ch]
    
    
    
    
    def _load_data(self, fname):    
        with pd.HDFStore(fname) as s:
            self.E = s['E']
            self.XF = s['XF']
            self.XN = s['XN']
            self.Aux = s['AuxData']
    
    def _load_cal_coeff(self, calibration_inputs):
        
        self.z_cal = calibration_inputs['z_cal']


        XNXF_cal = pd.read_table(calibration_inputs['XNXF_cal'], 
                                     delimiter=' ', header=None, index_col=0)
        XNXF_cal.columns = [ 'xx', 'yy', 'bb']
        self.XNXF_cal = XNXF_cal

        E_cal = pd.read_table(calibration_inputs['E_cal'], 
                                     delimiter=' ', header=None, index_col=0)
        E_cal.columns = [ 'k', 'b']
        self.E_cal = E_cal
            
        TAC_cal = pd.read_table(calibration_inputs['TAC_cal'], 
                             delimiter='\s+', header=None, index_col=0)
        TAC_cal.columns = [ 'shift']
        self.TAC_cal = TAC_cal
        
        TDC_cal = pd.read_table(calibration_inputs['TDC_cal'], delimiter='\s+', header=None, index_col=0)
        self.TDC_cal = TDC_cal
        
        self.RF_cal = calibration_inputs['RF_cal']
            
        self.FinerE_cal = pd.read_csv('data/FinerCD2.dat', 
                                      delimiter=',',
                                      header=None, 
                                      names=['k','b'], 
                                      index_col=0)
      
    
    def _calibrate_XNXF(self, XN, XF, XNXF_cal):
        data = XNXF_cal.yy.values*(XN.values+XNXF_cal.bb.values/(XNXF_cal.xx.values+XNXF_cal.yy.values))
        XN = pd.DataFrame(data=data, columns=XN.columns, index=XN.index)

        data = XNXF_cal.xx.values*(XF.values+XNXF_cal.bb.values/(XNXF_cal.xx.values+XNXF_cal.yy.values))
        XF = pd.DataFrame(data=data, columns=XF.columns, index=XF.index)

        return XN, XF


    def _calculateX(self, E, XN, XF):
        data = np.where((E>0)&(XN+XF>0),  
                     (XN.values>XF.values)*(1-XN.values/np.clip(E.values,1e-5,np.inf)) 
                     + (XN.values<=XF.values)*(XF.values/np.clip(E.values,1e-5,np.inf)),
                    -1000
                    )

        data = np.where((data>0.05)&(data<0.95), data, -1000)

        return pd.DataFrame(data=data, columns=XN.columns, index=XN.index)


    def _calculateZ(self, X, positions, shift):
        positions = np.tile(positions[::-1],4)

        data = np.where(X<-999, -1000,  shift+positions+X*5)
        return pd.DataFrame(data=data, columns=X.columns, index=X.index)

    def _calibrateE(self, E, E_cal):

        data = E.values*E_cal.values.T[0] + E_cal.values.T[1]
        return pd.DataFrame(data=data, columns=E.columns)

    def _calibrateTAC(self, TAC, TAC_cal, trigger_ch):
        data = TAC.values + 1900 - TAC_cal.loc[trigger_ch].values[:,0]
        return pd.Series(data,name='TAC')

    def _calibrateTDC(self, TDC, X, trigger_ch, TDC_cal):

        xTDCcurve_param = TDC_cal.loc[trigger_ch].values
        tmp = 0

        for i in range(5):
            tmp = tmp+xTDCcurve_param[:,i]*np.power(X, i)

        return pd.Series(data=(TDC - tmp).values,name='TDC')

    def _calibrateRF(self, TDC, RF_shift):    
        return TDC+1500, TDC+1500+RF_shift['RF_shift'] 


    def _fineCalibrationE(self, E, Z, trigger_ch, FinerE_cal):
         raise(NotImplementedError) 

    
class DataManager():
    
    def __init__(self, run_list, calibration_inputs, RunManagerClass):
        
        dfs = []
        
        for fname in run_list:
            rm = RunManagerClass(fname, calibration_inputs)
            dfs.append(rm.data)
            del rm
            
        self.data = pd.concat(dfs).reset_index(drop=True)
        